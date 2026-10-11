"""EXL3 linear layers on CUDA (``linear.cu``), any codebook and width: y = x @ W + bias for 1 to 128 rows, row-invariant, no cuBLAS."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from . import format as fmt

CODEBOOK_IDS = {"3inst": 0, "mcg": 1, "mul1": 2}
ROWS = 128                 # rows a call takes; a caller with more dispatches tiles (never a row of its own)
# A pair call rotates both parts' input in one launch (plan K1-b2); off keeps two rot_in launches for an A/B.
PAIR_ROT2 = True


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_exl3_linear_v4", sources=[str(here / "linear.cpp"), str(here / "linear.cu")],
                extra_include_paths=[str(here)], extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
                verbose=False)


def k2_of(bits: float) -> int:
    return int(2 * fmt.check_bits(bits))


def plan(k: int, n: int, blocks: int = 192, min_tiles: int = 8) -> tuple[int, int]:
    """(K splits, warps a program) for a K x N layer: the shape's alone, so a layer keeps one reduction for every row count."""

    kt, nb = k // 16, n // 128
    sk, wk = (1, 8) if nb >= 64 else (1, 4)
    if nb < 64:                                   # a narrow layer needs K splits to fill the SMs at all
        while nb * sk < blocks and sk < 64:
            sk *= 2
    while (wk > 4 or sk > 1) and (kt % (sk * wk) or kt // (sk * wk) < min_tiles):
        if wk > 4:
            wk = 4
        elif sk > 1:
            sk //= 2
        else:
            break
    return sk, wk


CANDIDATE_SK = (1, 2, 4, 8, 16, 32, 64)     # K splits to try a shape
TUNED: dict[tuple[int, int, float], tuple[int, int]] = {}

# Decided splits, distributed instead of re-measured (plan §5.1 items 5-7): probe/tune_shapes.py sweeps every real
# rank shape of a checkpoint under the four ranks' load and records one (K splits, warps) a shape. A run installs that
# manifest, so all ranks use one arithmetic revision, and an unknown shape is swept on its own (and reported as such)
# rather than silently racing the ranks to different winners.
MANIFEST: dict[tuple[int, int, float, str], tuple[int, int]] = {}
MANIFEST_INFO: dict = {}


def install_manifest(path: str | Path) -> dict:
    """Read a ``tune_shapes`` manifest into ``MANIFEST``; returns what was installed (and what it did not cover)."""

    import json

    doc = json.loads(Path(path).read_text())
    MANIFEST.clear()
    for got in doc.get("shapes", {}).values():
        MANIFEST[(int(got["k"]), int(got["n"]), float(got["bits"]), got["codebook"])] = tuple(got["chosen"])
    MANIFEST_INFO.clear()
    MANIFEST_INFO.update(path=str(path), shapes=len(MANIFEST), m=doc.get("m"), margin=doc.get("margin"),
                         source=doc.get("source", {}), checkpoint_plan_sha256=doc.get("checkpoint_plan_sha256"),
                         disagreements=doc.get("disagreements", []),
                         worst_confirmation_loss_pct=doc.get("worst_confirmation_loss_pct"))
    return dict(MANIFEST_INFO)


def manifest_split(layer: "Exl3Linear") -> tuple[int, int] | None:
    """The recorded split for this shape, or None when the manifest does not cover it."""

    if not MANIFEST:
        return None
    return MANIFEST.get((layer.k, layer.n, float(layer.bits), layer.codebook))


def candidates(k: int, n: int) -> list[tuple[int, int]]:
    """(K splits, warps) a K x N layer may use, the planned one first, then longer grids.

    ``plan`` sizes a program's work for a prompt-sized M, which leaves a one-row decode with too few programs to hide
    DRAM latency; a longer grid is what buys throughput there (measured: 54 us at 8 programs, 9 us at 512, for the same
    tensors).
    """

    out = [plan(k, n)]
    for wk in (4, 8, 2):
        for sk in CANDIDATE_SK:
            tiles = (k // 16) // (sk * wk)
            if (k // 16) % (sk * wk) or tiles < 1:
                continue
            if (sk, wk) not in out:
                out.append((sk, wk))
    return out


def tune(layer: "Exl3Linear", m: int = 3, rounds: int = 10, margin: float = 0.98) -> tuple[int, int]:
    """This layer's split: the installed manifest's entry, else measure once a shape and cache by (k, n, bits).

    A 2% margin keeps a near-tie on the earlier candidate. A manifest entry is one decided setting for every rank and
    every row count; without one, four ranks measuring the same shape can settle on different winners, which is a
    different K partition (a different arithmetic revision) per rank, so those shapes are recorded as uncovered.
    """

    key = (layer.k, layer.n, layer.bits)
    found = TUNED.get(key)
    if found is not None:
        layer.split = found
        return found
    decided = manifest_split(layer)
    if decided is not None:
        layer.split = TUNED[key] = decided
        return decided
    if MANIFEST:
        missing = MANIFEST_INFO.setdefault("uncovered", [])
        if key not in missing:
            missing.append(list(key))
    dev = layer.words.device
    x = (torch.randn(m, layer.k, dtype=torch.float32, device=dev) * 0.05).to(torch.bfloat16)
    out = torch.empty((m, layer.n), dtype=torch.bfloat16, device=dev)
    xh = torch.empty((m, layer.k), dtype=torch.float16, device=dev)
    z = torch.empty((CANDIDATE_SK[-1] * m * layer.n,), dtype=torch.float32, device=dev)
    counters = torch.zeros((8 * (layer.n // 128),), dtype=torch.int32, device=dev)
    best, best_secs = layer.split, float("inf")
    for cand in candidates(layer.k, layer.n):
        layer.split = cand
        try:
            layer(x, out=out, xh=xh, z=z, counters=counters)
        except Exception:                     # noqa: BLE001  a split the kernel refuses is simply not a candidate
            continue
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(rounds):
            layer(x, out=out, xh=xh, z=z, counters=counters)
        end.record()
        torch.cuda.synchronize()
        secs = start.elapsed_time(end) / rounds
        if secs < best_secs * margin:
            best, best_secs = cand, secs
    layer.split = best
    TUNED[key] = best
    del x, out, xh, z, counters
    return best


def strips(words: torch.Tensor) -> torch.Tensor:
    """Trellis words [K/16, N/16, W] -> [N/128, K/16, 8, W]: each 128-column block's tiles in k order (a copy)."""

    kt, nt, w = words.shape
    return words.view(kt, nt // 8, 8, w).permute(1, 0, 2, 3).contiguous()


@dataclass
class Exl3Linear:
    """One EXL3 layer on the GPU: trellis words (``layout``), fp16 scales, optional fp16 bias."""

    words: torch.Tensor           # int32, [N/128, K/16, 8, 8 * bits] ("strips") or [K/16, N/16, 8 * bits] ("stored")
    suh: torch.Tensor             # fp16 [K]
    svh: torch.Tensor             # fp16 [N]
    bias: torch.Tensor | None     # fp16 [N]
    bits: float
    codebook: str
    k: int
    n: int
    layout: str = "strips"
    split: tuple[int, int] | None = None     # (K splits, warps a program); plan(k, n) when None, fixed thereafter
    kernel_mode: int = 0                     # 1: bit-tested transposed MMA for M<=8

    @classmethod
    def from_tensors(cls, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, codebook: str,
                     bias: torch.Tensor | None = None, device: str | torch.device = "cuda",
                     layout: str = "strips") -> "Exl3Linear":
        """From a group's tensors: trellis int16 [K/16, N/16, 16 * bits], suh/svh fp16 (or packed su/sv sign words), codebook name."""

        bits = fmt.bits_of(trellis.shape)
        if codebook not in CODEBOOK_IDS:
            raise ValueError(f"unknown EXL3 codebook {codebook!r}")
        if not float(bits).is_integer() and codebook != "mul1":
            raise ValueError(f"{bits}-bit EXL3 tiles need the mul1 codebook")
        k, n = 16 * trellis.shape[0], 16 * trellis.shape[1]
        if k % 128 or n % 128:
            raise ValueError(f"K={k} and N={n} must be multiples of 128")

        def scales(t: torch.Tensor, size: int) -> torch.Tensor:
            if t.dtype == torch.int16 and t.numel() * 16 == size:
                t = torch.from_numpy(fmt.unpack_signs(t))
            if t.dtype != torch.float16 or t.numel() != size:
                raise ValueError(f"scales must be fp16 [{size}] or int16 sign words [{size // 16}]")
            return t.reshape(size).to(device).contiguous()

        words = trellis.to(device).contiguous().view(torch.int32)
        if layout == "strips":
            words = strips(words)
        elif layout != "stored":
            raise ValueError(f"layout must be 'strips' or 'stored', got {layout!r}")
        b = None if bias is None else bias.to(device=device, dtype=torch.float16).contiguous()
        return cls(words, scales(suh, k), scales(svh, n), b, bits, codebook, k, n, layout)

    @classmethod
    def load(cls, model_dir: str | Path, prefix: str, device: str | torch.device = "cuda",
             layout: str = "strips") -> "Exl3Linear":
        """Layer ``prefix`` (e.g. "model.layers.0.self_attn.q_proj") of an EXL3 checkpoint folder."""

        from safetensors import safe_open

        root = Path(model_dir)
        index = root / "model.safetensors.index.json"
        if index.exists():
            import json

            weight_map = json.loads(index.read_text())["weight_map"]
            rel = weight_map.get(prefix + ".trellis")
            if rel is None:  # a group outside the index's coverage: find the shard that holds it
                for part in sorted(root.glob("*.safetensors")):
                    with safe_open(str(part), framework="pt") as g:
                        if f"{prefix}.trellis" in set(g.keys()):
                            rel = part.name
                            break
            if rel is None:
                raise ValueError(f"{prefix}.trellis is in no file of {root}")
            file = root / rel
        else:
            file = root / "model.safetensors"
        with safe_open(str(file), framework="pt") as f:
            names = set(f.keys())

            def get(part: str) -> torch.Tensor | None:
                return f.get_tensor(f"{prefix}.{part}") if f"{prefix}.{part}" in names else None

            parts = {p: get(p) for p in fmt.PARTS}
        codebook = "mul1" if parts["mul1"] is not None else "mcg" if parts["mcg"] is not None else "3inst"
        suh = parts["suh"] if parts["suh"] is not None else parts["su"]
        svh = parts["svh"] if parts["svh"] is not None else parts["sv"]
        return cls.from_tensors(parts["trellis"], suh, svh, codebook, parts["bias"], device, layout)

    @property
    def k2(self) -> int:
        return k2_of(self.bits)

    @property
    def strides(self) -> tuple[int, int]:
        """(words between k tiles, words between 128-column blocks) in ``words``."""

        tw = 4 * self.k2
        if self.layout == "strips":
            return 8 * tw, (self.k // 16) * 8 * tw
        return (self.n // 16) * tw, 8 * tw

    def nbytes(self) -> int:
        return self.words.numel() * 4

    def __post_init__(self) -> None:
        if self.split is None:
            self.split = plan(self.k, self.n)

    @property
    def counters(self) -> torch.Tensor:
        c = getattr(self, "_counters", None)
        if c is None:
            c = torch.zeros((8 * (self.n // 128),), dtype=torch.int32, device=self.words.device)
            self._counters = c
        return c

    def workspace(self, device: torch.device, rows: int = ROWS) -> tuple[torch.Tensor, torch.Tensor | None,
                                                                         torch.Tensor]:
        """This layer's per-call scratch for one slot to own: the rotated input, the split-K partials, the counters.

        One execution slot (a sequence's buffers, its graph instances) owns them, so two streams never share a
        projection's mutable scratch; the kernel clears each counter block it completes, so the tensors are reusable
        as they are.
        """

        rows = min(rows, ROWS)
        sk, _ = self.split
        xh = torch.empty((rows, self.k), dtype=torch.float16, device=device)
        z = torch.empty((sk * rows * self.n,), dtype=torch.float32, device=device) if sk > 1 else None
        return xh, z, torch.zeros((8 * (self.n // 128),), dtype=torch.int32, device=device)

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None, out_dtype: torch.dtype | None = None,
                 xh: torch.Tensor | None = None, z: torch.Tensor | None = None,
                 counters: torch.Tensor | None = None) -> torch.Tensor:
        """y [M, N] = x [M, K] @ W + bias for M = 1..128 (``ROWS``); scratch ``xh`` fp16 [M, K], ``z`` fp32 (SK > 1)
        and ``counters`` allocated when not given (a slot passes its own through ``workspace``)."""

        if x.dim() != 2 or x.shape[1] != self.k or not 1 <= x.shape[0] <= ROWS:
            raise ValueError(f"x must be [1..{ROWS}, {self.k}], got {tuple(x.shape)}")
        x = x.contiguous()
        m = x.shape[0]
        if out is None:
            out = torch.empty((m, self.n), dtype=out_dtype or x.dtype, device=x.device)
        sk, wk = self.split
        need = sk * m * self.n
        if xh is None or xh.shape[0] < m:
            xh = torch.empty((m, self.k), dtype=torch.float16, device=x.device)
        elif xh.shape[0] > m:
            xh = xh[:m]                       # a slot's scratch is sized for a whole tile, not this call's rows
        if sk > 1:
            if z is None or z.numel() < need:
                z = torch.empty((need,), dtype=torch.float32, device=x.device)
            elif z.numel() > need:
                z = z[:need]
        sk_stride, nb_stride = self.strides
        ext = _ext()
        ext.rot_in(x, self.suh, xh)
        ext.linear(xh, self.words, sk_stride, nb_stride, self.svh, self.bias, out, z if sk > 1 else None,
                   self.counters if counters is None else counters, self.k2, CODEBOOK_IDS[self.codebook], sk, wk, self.kernel_mode)
        return out

    def unpack(self, out: torch.Tensor | None = None) -> torch.Tensor:
        """W_q [K, N] fp16 decoded on the GPU from either layout, into ``out`` when given."""

        w = out if out is not None else torch.empty((self.k, self.n), dtype=torch.float16, device=self.words.device)
        _ext().unpack(self.words, w, *self.strides, self.k2, CODEBOOK_IDS[self.codebook])
        return w


def pair_compatible(a: "Exl3Linear", b: "Exl3Linear", k: int) -> bool:
    """Whether two projections can share one launch: same input shape and width, same output layout.

    The pair kernel gives each part its own weights, scales, bias, output, split-K scratch and counters and only
    shares the launch and the SMs, so the arithmetic of a part is exactly a launch of its own. What it cannot share
    is the template: the trellis width/codebook (the decode is compiled) and the program's warp count (the block
    size), so those have to match. The two K splits may differ.
    """

    return (a.k == b.k == k and a.k2 == b.k2 and a.k2 in (6, 8, 10, 12) and a.codebook == b.codebook == "mul1"
            and a.kernel_mode == b.kernel_mode and a.layout == b.layout and a.split is not None and b.split is not None and a.split[1] == b.split[1])


def pair_call(a: "Exl3Linear", b: "Exl3Linear", x: torch.Tensor, out_a: torch.Tensor, out_b: torch.Tensor,
              ws_a: tuple | None = None, ws_b: tuple | None = None) -> None:
    """One launch for two EXL3 projections of ``x`` (plan K1-b's multi-projection); callers must check ``pair_compatible``."""

    m = x.shape[0]
    xh_a, z_a, c_a = (None, None, None) if ws_a is None else ws_a
    xh_b, z_b, c_b = (None, None, None) if ws_b is None else ws_b
    if xh_a is None or xh_a.shape[0] < m:
        xh_a = torch.empty((m, a.k), dtype=torch.float16, device=x.device)
    else:
        xh_a = xh_a[:m]
    if xh_b is None or xh_b.shape[0] < m:
        xh_b = torch.empty((m, b.k), dtype=torch.float16, device=x.device)
    else:
        xh_b = xh_b[:m]
    ext = _ext()
    if PAIR_ROT2:                                # one launch: the parts share the input rows but not their suh
        ext.rot_in2(x, a.suh, xh_a, b.suh, xh_b)
    else:
        ext.rot_in(x, a.suh, xh_a)
        ext.rot_in(x, b.suh, xh_b)
    sk_a, wk = a.split
    sk_b, _ = b.split
    z_a = None if sk_a == 1 else (z_a if z_a is not None and z_a.numel() >= sk_a * m * a.n
                                  else torch.empty((sk_a * m * a.n,), dtype=torch.float32, device=x.device))
    z_b = None if sk_b == 1 else (z_b if z_b is not None and z_b.numel() >= sk_b * m * b.n
                                  else torch.empty((sk_b * m * b.n,), dtype=torch.float32, device=x.device))
    ext.linear2(xh_a, a.words, *a.strides, a.svh, a.bias, out_a, z_a,
                a.counters if c_a is None else c_a, xh_b, b.words, *b.strides, b.svh, b.bias, out_b, z_b,
                b.counters if c_b is None else c_b, a.k2, CODEBOOK_IDS[a.codebook], sk_a, sk_b, wk, a.kernel_mode)


def unpack_cuda(trellis: torch.Tensor, codebook: str) -> torch.Tensor:
    """W_q [K, N] fp16 from trellis int16 [K/16, N/16, 16 * bits] on the GPU (``decode.cuh``)."""

    bits = fmt.bits_of(trellis.shape)
    words = trellis.cuda().contiguous().view(torch.int32)
    k, n = 16 * trellis.shape[0], 16 * trellis.shape[1]
    w = torch.empty((k, n), dtype=torch.float16, device=words.device)
    tw = 4 * k2_of(bits)
    _ext().unpack(words, w, (n // 16) * tw, 8 * tw, k2_of(bits), CODEBOOK_IDS[codebook])
    return w
