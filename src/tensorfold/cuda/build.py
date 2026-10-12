"""Build the CUDA extensions for the GPU present, whatever architecture list the container sets, saying when a build runs or waits on a lock."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import shutil
import sys
import threading
from typing import Any

MIN_CAPABILITY = (8, 9)         # FP8 MMA and e4m3 conversions (Ada); kernels with clusters use them from 9.0
CLUSTERS = (9, 0)               # extensions built only on thread-block clusters (NVFP4) need Hopper or newer

def _env_floor() -> tuple[int, int]:
    """TENSORFOLD_MIN_CAPABILITY=8.0 lowers the FP8-free floor at import, before ``need`` binds; never raises it."""
    raw = os.environ.get("TENSORFOLD_MIN_CAPABILITY", "").strip()
    if not raw:
        return MIN_CAPABILITY
    major, minor = (int(part) for part in raw.split("."))
    lowered = (major, minor)
    return lowered if lowered < MIN_CAPABILITY else MIN_CAPABILITY


UPSTREAM_FLOOR = MIN_CAPABILITY   # the stock Ada floor, kept for the refusal hint
MIN_CAPABILITY = _env_floor()
FLASHNEXT_FLOOR = (12, 0)       # Flash Next serves CUDA on sm_120/sm_121 cards only; anything below is refused by name
# stop first: when the lock goes, a waiting start imports whatever module is there without building, even an old one
HINT = "if no other build is running, a killed build left it: stop this start, delete the lock and start again"
LOCK_WAIT_SECONDS = 60.0        # a start still waiting on the same lock this long says so again


def arch_flags(need: tuple[int, int] = MIN_CAPABILITY, arch_specific: bool = False) -> list[str]:
    """nvcc flags for this GPU alone (``arch_specific``: its ``a`` target); a GPU under ``need`` is refused by name."""

    import torch
    if arch_specific and need < CLUSTERS:
        need = CLUSTERS        # the ``a`` target exists only for cluster (NVFP4) builds; no floor may lower it

    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < need:
        why = "thread-block clusters" if need >= CLUSTERS else "FP8 MMA"
        hint = ("; Ampere W4A16/EXL3 is available with TENSORFOLD_MIN_CAPABILITY=8.0 (FP8 and NVFP4 stay refused)"
                if MIN_CAPABILITY >= UPSTREAM_FLOOR else "") if why == "FP8 MMA" else ""
        raise RuntimeError(f"TensorFold's CUDA kernels need compute capability {need[0]}.{need[1]} or newer ({why}"
                           f"{' for these weights' if need > MIN_CAPABILITY else ''}); this GPU "
                           f"({torch.cuda.get_device_name()}) is {major}.{minor}{hint}")
    a = "a" if arch_specific else ""
    return [f"-gencode=arch=compute_{major}{minor}{a},code=sm_{major}{minor}{a}"]


def refuse_old_gpu(need: tuple[int, int] = MIN_CAPABILITY) -> None:
    """At startup, before any weight loads: a GPU older than ``need`` is refused by name (no GPU: skipped)."""

    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        try:
            arch_flags(need)
        except RuntimeError as exc:
            raise ValueError(str(exc)) from None


def refuse_small_gpu(need: tuple[int, int] = FLASHNEXT_FLOOR) -> None:
    """At startup, before any weight loads: a GPU below Flash Next's CUDA floor is refused by name (no GPU: skipped)."""

    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) < need:
            raise ValueError("TensorFold's Flash Next CUDA kernels run on sm_120 and sm_121 GPUs only "
                             "(DGX Spark GB10, RTX 50 and RTX PRO 6000); this GPU "
                             f"({torch.cuda.get_device_name()}) is {major}.{minor}. Serve Flash Next on CUDA on one "
                             "of those, on the MLX backend, or with another model.")


def _unbundled_math_includes() -> list[str]:
    """Where torch's ``ATen/cuda`` finds the math headers when the toolkit has none: CUDA 13 moved them into the ``nvidia`` wheels.

    ``ATen/cuda/CUDAContextLight.h`` includes ``<cusparse.h>`` and ``<cusolverDn.h>``, which CUDA 13's toolkit image no
    longer ships, so any kernel that includes ``ATen/cuda/CUDAContext.h`` (for ``at::cuda::getCurrentCUDAStream``) fails
    to compile without them. Only headers the toolkit **lacks** are linked into a shim directory: the wheel also
    carries a whole newer CUDA tree (nvcc 13.3 beside a 13.0 toolkit), and putting that on the include path shadows
    the toolkit's own ``crt/`` headers and breaks nvcc's generated stubs (``__cudaLaunch`` undeclared). Empty when the
    toolkit has everything, so a complete installation builds unchanged.
    """

    import glob
    from pathlib import Path

    toolkit = [Path(os.environ.get("CUDA_HOME") or "/usr/local/cuda") / "include", Path("/usr/local/cuda/include")]
    have = {p.name for base in toolkit if base.is_dir() for p in base.glob("*.h")}
    if {"cusparse.h", "cusolverDn.h"} <= have:
        return []
    found: list[Path] = []
    for root in ("/usr/local/lib", "/usr/lib", "/usr/local/cuda"):
        for pattern in ("python*/dist-packages/nvidia/*/include", "python*/site-packages/nvidia/*/include",
                        "python*/dist-packages/nvidia/*/include/cuda"):
            for path in glob.glob(os.path.join(root, pattern)):
                found += [p for p in Path(path).glob("*.h") if p.name not in have]
    if not found:
        return []
    shim = Path(os.environ.get("TORCH_EXTENSIONS_DIR") or "/tmp") / "tensorfold_unbundled"
    shim.mkdir(parents=True, exist_ok=True)
    for source in found:
        target = shim / source.name
        if not target.exists():
            target.symlink_to(source)
    return [str(shim)]


@lru_cache(maxsize=1)
def _nvcc_host_limit() -> int | None:
    """The newest g++ nvcc accepts, read from its own ``crt/host_config.h`` (None: the header names no limit)."""

    import re

    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    try:
        text = (Path(home) / "include" / "crt" / "host_config.h").read_text()
    except OSError:
        return None
    found = re.search(r"unsupported GNU version! gcc versions later than (\d+)", text)
    return int(found.group(1)) if found else None


@lru_cache(maxsize=1)
def _host_compiler() -> str | None:
    """A ``-ccbin`` for nvcc when this container's default g++ is newer than nvcc's check allows.

    CUDA 13 stops at gcc 15 and this image ships gcc 16, which nvcc refuses before compiling anything (the kernels
    would build with ``-allow-unsupported-compiler``, but a supported g++ of the same install is the honest choice).
    """

    import re
    import shutil
    import subprocess

    limit = _nvcc_host_limit()
    if limit is None:
        return None
    cxx = os.environ.get("CXX") or shutil.which("g++") or ""
    if "clang" in cxx:
        return None
    try:
        out = subprocess.run([cxx, "-dumpfullversion"], capture_output=True, text=True, timeout=10).stdout.strip()
        parts = re.findall(r"\d+", out)
        major = int(parts[0]) if parts else 0
    except (OSError, subprocess.SubprocessError):
        return None
    if 0 < major <= limit:
        return None
    for candidate in range(limit, 9, -1):
        found = shutil.which(f"g++-{candidate}")
        if found:
            return found
    return None


def load(name: str, sources: str | list[str], need: tuple[int, int] = MIN_CAPABILITY, *, arch_specific: bool = False,
         **kwargs: Any) -> Any:
    """torch's JIT ``load`` for this GPU only (NVIDIA's containers list every architecture back to sm_80), with a line when it compiles or waits on a lock."""

    from torch.utils import cpp_extension

    kwargs["extra_cuda_cflags"] = [*kwargs.get("extra_cuda_cflags", []), *arch_flags(need, arch_specific)]
    host = _host_compiler()
    if host:
        kwargs["extra_cuda_cflags"].append(f"-ccbin={host}")
    math = _unbundled_math_includes()
    if math:                                            # torch's ATen/cuda needs cusparse.h; CUDA 13 hides it in the wheels
        kwargs["extra_include_paths"] = [*math, *kwargs.get("extra_include_paths", [])]
    links = _toolkit()
    if links:
        kwargs["extra_ldflags"] = [*kwargs.get("extra_ldflags", []), *links]
    held = _announce(cpp_extension, name, sources, kwargs.get("build_directory"))
    timer = None
    if held is not None:
        timer = threading.Timer(LOCK_WAIT_SECONDS, _still_waiting, held)
        timer.daemon = True
        timer.start()
    try:
        return cpp_extension.load(name=name, sources=sources, **kwargs)
    finally:
        if timer is not None:
            timer.cancel()


def _announce(cpp_extension: Any, name: str, sources: str | list[str],
              directory: str | None) -> tuple[str, tuple[int, int]] | None:
    """Say whether ``name`` builds or waits on a lock; return (lock path, (inode, mtime)) when it waits."""

    try:
        directory = directory or cpp_extension._get_build_directory(name, verbose=False)   # private in torch
    except Exception:  # noqa: BLE001 - no lookup in this torch: build without the lines
        return None
    lock = os.path.join(directory, "lock")          # torch's FileBaton for this extension
    try:
        seen = os.stat(lock)
    except OSError:
        seen = None
    if seen is not None:
        _say(f"CUDA extension {name} waits on the build lock {lock}; {HINT}")
        return lock, (seen.st_ino, seen.st_mtime_ns)
    if _needs_build(os.path.join(directory, name + getattr(cpp_extension, "LIB_EXT", ".so")), sources):
        _say(f"building CUDA extension {name} (first start after an install or update; later starts reuse it)")
    return None


def _needs_build(module: str, sources: str | list[str]) -> bool:
    """No built module, or a source newer than it (ninja then compiles again)."""

    try:
        built = os.path.getmtime(module)
    except OSError:
        return True
    for source in [sources] if isinstance(sources, str) else sources:
        try:
            if os.path.getmtime(source) > built:
                return True
        except OSError:
            continue
    return False


def _still_waiting(lock: str, identity: tuple[int, int]) -> None:
    """Repeat the hint while the lock seen before the build is still the one there."""

    try:
        now = os.stat(lock)
    except OSError:
        return
    if (now.st_ino, now.st_mtime_ns) == identity:
        _say(f"still waiting after {LOCK_WAIT_SECONDS:g} s on the build lock {lock}; {HINT}")


def _toolkit() -> list[str]:
    """The pip route's tools: the venv's ninja on PATH, NVIDIA's pip toolkit, and the link flags it needs."""

    venv = Path(sys.executable).parent              # an inactive venv: ninja is beside its python, not on PATH
    tool = "ninja.exe" if os.name == "nt" else "ninja"              # the venv's ninja, named as Windows names it
    if shutil.which("ninja") is None and (venv / tool).is_file():
        os.environ["PATH"] = f"{venv}{os.pathsep}{os.environ.get('PATH', '')}"
    return list(_pip_flags())


@lru_cache(maxsize=1)
def _pip_flags() -> tuple[str, ...]:
    """Once a process: NVIDIA's pip toolkit when torch found none (the module keeps it after the first call)."""

    import torch
    from torch.utils import cpp_extension

    return tuple(pip_toolkit(cpp_extension, torch))


def pip_toolkit(cpp_extension: Any, torch: Any) -> list[str]:
    """No toolkit found (CUDA_HOME, nvcc, /usr/local/cuda): NVIDIA's pip one beside torch, its cudart linked by name."""

    if cpp_extension.CUDA_HOME is not None or not getattr(torch.version, "cuda", None):
        return []
    home = Path(torch.__file__).resolve().parents[1] / "nvidia" / f"cu{torch.version.cuda.split('.')[0]}"
    if os.name == "nt":                       # a Windows wheel: DLLs beside torch and in NVIDIA's cu13 folder
        search = [folder for folder in (home / "bin", home / "lib", Path(torch.__file__).resolve().parent / "lib")
                  if folder.is_dir()]
        for folder in search:                 # a built .pyd loads cudart64_13.dll by name at import
            try:
                os.add_dll_directory(str(folder))
            except (AttributeError, OSError):
                pass                          # old CPython or a refused folder: PATH is the fallback
            os.environ["PATH"] = f"{folder}{os.pathsep}{os.environ.get('PATH', '')}"
        return [f"/LIBPATH:{folder}" for folder in search if (folder / "cudart.lib").is_file()]
    if not (home / "bin" / "nvcc").is_file():
        return []
    cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"] = str(home)    # torch reads its CUDA_HOME at every build
    os.environ["PATH"] = f"{home / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
    _say(f"CUDA compiler: NVIDIA's pip toolkit for CUDA {torch.version.cuda} at {home}")
    versioned = sorted((home / "lib").glob("libcudart.so.*"))
    if (home / "lib" / "libcudart.so").exists() or not versioned:
        return []
    root = os.environ.get("TORCH_EXTENSIONS_DIR") or cpp_extension.get_default_build_root()
    links = Path(root) / "tensorfold_cudart"                         # -lcudart wants the bare name
    links.mkdir(parents=True, exist_ok=True)
    bare = links / "libcudart.so"
    if not bare.is_symlink() or bare.resolve() != versioned[0].resolve():
        bare.unlink(missing_ok=True)
        try:
            bare.symlink_to(versioned[0])
        except FileExistsError:                                      # another start made the same link first
            pass
    return [f"-L{links}"]


def _say(text: str) -> None:
    """One ``[tensorfold]`` line, flushed like the CLI's."""

    print(f"[tensorfold] {text}", flush=True)


__all__ = ["CLUSTERS", "MIN_CAPABILITY", "arch_flags", "load", "pip_toolkit"]
