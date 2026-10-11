"""GLM-5.3 (``model_type`` ``glm_moe_dsa``) for the ``glm5_next`` CUDA family: its HF-native layout, its settings, and the tensors a rank reads.

GLM-5.3 differs from GLM-5.3-Flash (``glm5_next``) in four ways the engine has to know about:

* every layer is sparse MLA (no Kimi delta attention), all of it with a **rotary** key part
  (``qk_rope_head_dim`` 64) where Flash is NoPE;
* plain residuals (no manifold-constrained hyper-connections: no ``hc_*`` tensors);
* a DSA indexer on one layer in four (``index_topk_freq``), the other three **sharing** the block's
  selection (``indexer_types``: 21 ``full``, 57 ``shared``), and one on the MTP layer;
* EXL3 on every linear, mixed 3/4/5/6 bits with the ``mul1`` codebook (Flash is 4-bit ``mcg`` routed
  experts with BF16 elsewhere).

This module is pure Python (no torch): the CUDA loader and the engine's settings both build on it, so
the plan can be checked against a checkpoint's headers before any GPU is touched.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_TYPE = "glm_moe_dsa"
PREFIX = "model."                    # model.layers.*, model.embed_tokens.weight, model.norm.weight (lm_head.* bare)
# the model-type this family reads beside its own, and the checkpoint the CUDA engine was qualified on
TITLE = "GLM-5.3"
MODELS = ("dealignai/GLM-5.3-UNCENSORED-FP8 (EXL3 exports)",)
# one layer in four recomputes the DSA top-k; the others attend to the block's own selection
INDEX_STRIDE = 4


def is_full(raw: dict[str, Any]) -> bool:
    """Whether a config.json is full GLM-5.3 rather than GLM-5.3-Flash."""

    text = raw.get("text_config") or raw
    return str(text.get("model_type") or raw.get("model_type") or "") == MODEL_TYPE


def fields(raw: dict[str, Any]) -> dict[str, Any]:
    """The settings the CUDA engine reads, in the shape ``cuda.weights.Config`` fills from them."""

    t = dict(raw.get("text_config") or raw)
    lin = dict(t.get("linear_attn_config") or {})          # absent: no delta-attention layers
    n = int(t["num_hidden_layers"])
    if int(t.get("n_group", 1)) != 1 or int(t.get("topk_group", 1)) != 1:
        raise ValueError("GLM-5.3: grouped expert selection (n_group > 1) is not implemented")
    kinds = list(t.get("layer_types") or ["full_attention"] * n)    # every layer sparse MLA; never linear_attention
    dense = int(t.get("first_k_dense_replace", 3))
    mlp_kinds = list(t.get("mlp_layer_types") or ["dense"] * dense + ["sparse"] * (n - dense))
    eos = t.get("eos_token_id", raw.get("eos_token_id"))
    return {
        "hidden": int(t["hidden_size"]), "layers": n, "vocab": int(t["vocab_size"]),
        "eps": float(t["rms_norm_eps"]), "heads": int(t["num_attention_heads"]),
        "q_lora": int(t["q_lora_rank"]), "kv_lora": int(t["kv_lora_rank"]),
        "qk_nope": int(t["qk_nope_head_dim"]), "qk_rope": int(t.get("qk_rope_head_dim", 0)),
        "v_dim": int(t["v_head_dim"]),
        "rope_theta": float((t.get("rope_parameters") or {}).get("rope_theta", t.get("rope_theta", 10000.0))),
        "rope_interleave": bool(t.get("rope_interleave", True)),
        "indexer_interleave": bool(t.get("indexer_rope_interleave", t.get("rope_interleave", True))),
        "lin_heads": int(lin.get("num_heads", 64)), "lin_dim": int(lin.get("head_dim", 128)),
        "conv": int(lin.get("short_conv_kernel_size", 4)), "lower": float(lin.get("gate_lower_bound", -5.0)),
        "experts": int(t["n_routed_experts"]), "top_k": int(t["num_experts_per_tok"]),
        "moe_width": int(t["moe_intermediate_size"]),
        "shared_width": int(t["moe_intermediate_size"]) * int(t.get("n_shared_experts", 1) or 1),
        "dense_width": int(t["intermediate_size"]), "routed_scale": float(t["routed_scaling_factor"]),
        "norm_topk": bool(t.get("norm_topk_prob", True)),
        # no hc_mult: one residual stream, the plain path (see cuda/mtp.py's layer for the kernel sequence)
        "streams": 1, "hc_iters": 0, "hc_eps": 0.0,
        "index_heads": int(t.get("index_n_heads", 32)), "index_dim": int(t.get("index_head_dim", 128)),
        # full GLM-5.3 scores every visible token: the checkpoint has no pooling tensors (no
        # index_kpool*, no ape), so one pool is one token and the reference's dense/sparse boundary
        # (``visible > index_topk``) is ``dense_limit = index_topk``.
        "index_topk": int(t.get("index_topk", 2048)), "kpool": int(t.get("index_kpool", 1)),
        # swiglu_limit absent: the checkpoint clamps nothing, which the kernels read as an infinite limit (zero would
        # clamp every gate and up value to zero, and Flash's default of 10.0 would change this model's bits)
        "limit": float(t["swiglu_limit"]) if t.get("swiglu_limit") else math.inf,
        "kinds": ["kda" if k == "linear_attention" else "dsa" for k in kinds],
        "mlp_kinds": ["moe" if k == "sparse" else "dense" for k in mlp_kinds],
        "indexer_types": list(t.get("indexer_types") or []),
        "index_stride": int(t.get("index_topk_freq", INDEX_STRIDE)),
        "index_skip": int(t.get("index_skip_topk_offset", 0)),
        "eos": tuple(int(e) for e in eos) if isinstance(eos, list) else (int(eos),),
        "mtp_layers": int(t.get("num_nextn_predict_layers", 0)),
    }


@dataclass(frozen=True)
class Schedule:
    """Which layers are sparse MLA, which hold the indexer, and where each one takes its top-k selection from."""

    kinds: list[str]                 # "dsa" per layer (0 .. layers - 1); the MTP layer is not listed
    mlp: list[str]                   # "dense" or "moe"
    full: list[int]                  # layers whose own indexer weights are loaded (indexer_types == "full")
    source: list[int]                # per layer: the layer whose indexer selection it may reuse (itself when full)
    mtp_indexer: bool                # the MTP layer (index == layers) loads its own indexer

    @property
    def layers(self) -> int:
        return len(self.kinds)

    def owns_indexer(self, i: int) -> bool:
        return i == self.layers and self.mtp_indexer or i < self.layers and i in self.full

    def indexer_of(self, i: int) -> int:
        """The layer whose top-k selection layer ``i`` reuses (its own when it holds an indexer)."""

        return i if self.owns_indexer(i) else self.source[i]


def schedule(f: dict[str, Any]) -> Schedule:
    """The per-layer plan: what the config's indexer_types says, cross-checked against the block arithmetic."""

    n, stride = int(f["layers"]), int(f["index_stride"])
    kinds, mlp = list(f["kinds"]), list(f["mlp_kinds"])
    stated = list(f.get("indexer_types") or [])
    if len(stated) == n:
        full = [i for i, k in enumerate(stated) if k == "full"]
    else:                                    # derive it: one layer of every block of `stride` (the block's first)
        full = [i for i in range(n) if i % stride == 0]
    source, at = [], -1
    for i in range(n):
        if i in full:
            at = i
        source.append(i if at < 0 else at)
    return Schedule(kinds, mlp, full, source, bool(f["mtp_layers"]))


# -- the tensors a rank reads ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Plan:
    """Every tensor of one rank: EXL3 group prefixes and plain names, each with the field it fills."""

    exl3: dict[str, str]              # prefix (e.g. model.layers.0.self_attn.q_a_proj) -> field
    plain: dict[str, str]             # full tensor name (e.g. model.layers.0.self_attn.kv_b_proj.weight) -> field

    @property
    def names(self) -> set[str]:
        return set(self.exl3) | set(self.plain)

    def layers_of(self, field: str) -> list[int]:
        return sorted({int(k.split(".")[2]) for k, v in {**self.exl3, **self.plain}.items() if v.startswith(field)})


ATTENTION = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj")
INDEXER = ("indexer.wk", "indexer.wq_b", "indexer.weights_proj", "indexer.k_norm.weight", "indexer.k_norm.bias")
DENSE = ("gate_proj", "up_proj", "down_proj")
MOE_ROUTED = ("gate_proj", "up_proj", "down_proj")


def plan(f: dict[str, Any], *, mtp: bool = True) -> Plan:
    """The tensors one rank of the engine reads, GLM-5.3's HF-native layout (``model.layers.*``)."""

    sch = schedule(f)
    n = int(f["layers"])
    exl3: dict[str, str] = {}
    plain: dict[str, str] = {"model.embed_tokens.weight": "embed", "lm_head.weight": "head",
                             "model.norm.weight": "norm"}
    for i in list(range(n)) + ([n] if mtp and sch.mtp_indexer else []):
        p = f"{PREFIX}layers.{i}."
        mtp_layer = i == n
        plain[p + "input_layernorm.weight"] = f"layers.{i}.in_norm"
        plain[p + "post_attention_layernorm.weight"] = f"layers.{i}.post_norm"
        a = p + "self_attn."
        for proj in ATTENTION:
            exl3[a + proj] = f"layers.{i}.dsa.{proj}"
        plain[a + "q_a_layernorm.weight"] = f"layers.{i}.dsa.q_norm"
        plain[a + "kv_a_layernorm.weight"] = f"layers.{i}.dsa.kv_norm"
        plain[a + "kv_b_proj.weight"] = f"layers.{i}.dsa.kv_b"
        if sch.owns_indexer(i):
            ix = a + "indexer."
            exl3[ix + "wq_b"] = f"layers.{i}.dsa.index.wq_b"
            for part in ("wk", "weights_proj"):
                plain[ix + part + ".weight"] = f"layers.{i}.dsa.index.{part}"
            for part in ("weight", "bias"):
                plain[ix + "k_norm." + part] = f"layers.{i}.dsa.index.k_norm_{part}"
        if mtp_layer:
            plain[p + "enorm.weight"] = f"layers.{i}.enorm"
            plain[p + "hnorm.weight"] = f"layers.{i}.hnorm"
            exl3[p + "eh_proj"] = f"layers.{i}.eh"
            plain[p + "shared_head.norm.weight"] = f"layers.{i}.shared_head"
        if not mtp_layer and sch.mlp[i] == "dense":      # the MTP layer is always a sparse MoE
            for proj in DENSE:
                exl3[p + "mlp." + proj] = f"layers.{i}.mlp.{proj}"
            continue
        plain[p + "mlp.gate.weight"] = f"layers.{i}.moe.router"
        plain[p + "mlp.gate.e_score_correction_bias"] = f"layers.{i}.moe.bias"
        for proj in MOE_ROUTED:
            exl3[p + "mlp.shared_experts." + proj] = f"layers.{i}.moe.shared.{proj}"
            for e in range(int(f["experts"])):
                exl3[p + f"mlp.experts.{e}.{proj}"] = f"layers.{i}.moe.experts.{proj}"
    return Plan(exl3, plain)


# -- checking a checkpoint against the plan ----------------------------------------------------------------------
def check_plan(model_dir: str | Path, f: dict[str, Any], *, mtp: bool = True) -> dict[str, Any]:
    """Compare the plan with a checkpoint's safetensors headers: what is missing, and what nothing claims.

    Reads headers only (``tensorfold.cuda.exl3.format.scan``), so it runs on any machine, before a GPU is picked.
    """

    from tensorfold.cuda.exl3 import format as fmt                 # numpy only

    ck = fmt.scan(model_dir)
    want = plan(f, mtp=mtp)
    groups, plains = set(ck.groups), set(ck.plain)
    missing = sorted((set(want.exl3) - groups) | (set(want.plain) - plains))
    unclaimed_groups = sorted(g for g in groups if g not in want.exl3)
    unclaimed_plain = sorted((p for p in plains if p not in want.plain and not p.startswith("model.visual.")))
    widths: dict[str, dict[float, int]] = {}
    for prefix, bits in ((g, ck.groups[g].bits) for g in groups):
        widths.setdefault(prefix.rsplit(".", 1)[-1], {}).setdefault(bits, 0)
        widths[prefix.rsplit(".", 1)[-1]][bits] += 1
    return {"model_dir": str(model_dir), "exl3_groups": len(groups), "plain": len(plains), "bad_groups": ck.bad,
            "missing": missing, "unclaimed_exl3": unclaimed_groups, "unclaimed_plain": unclaimed_plain,
            "bits_by_part": {k: dict(sorted(v.items())) for k, v in sorted(widths.items())}}


def read(model_dir: str | Path) -> dict[str, Any]:
    """A checkpoint folder's GLM-5.3 settings (raises when it is another architecture)."""

    raw = json.loads((Path(model_dir) / "config.json").read_text())
    if not is_full(raw):
        raise ValueError(f"{model_dir} is model_type "
                         f"{raw.get('model_type') or raw.get('text_config', {}).get('model_type')!r}, not {MODEL_TYPE}")
    return fields(raw)


__all__ = ["ATTENTION", "INDEX_STRIDE", "MODEL_TYPE", "MODELS", "PREFIX", "TITLE", "Plan", "Schedule",
           "check_plan", "fields", "is_full", "plan", "read", "schedule"]
