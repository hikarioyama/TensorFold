"""Load one rank of GLM-5.3 (``glm_moe_dsa``): EXL3 on every linear at the width each tensor states, a BF16 head.

The checkpoint is HF-native (``model.layers.*``) with ExLlamaV3's EXL3 groups beside it, so this loader reads the
group metadata out of the safetensors headers (``tensorfold.cuda.exl3.format``) rather than a layout table: each
tensor's bit width comes from its trellis' last dimension and its codebook from the group's marker tensor, and one
checkpoint mixes 3, 4, 5 and 6 bits across layers, projections and even the experts of one MoE layer.

The split across ranks is ``glm5_next.cuda.split.rule_full``: whole heads (``q_b_proj`` rows, ``kv_b_proj`` rows and
``o_proj`` columns), whole experts' width (``gate``/``up`` tile columns and ``down`` tile rows, whose fp32 partials
the engine reduces in rank order), and a whole vocabulary block each (the BF16 head). Every dtype is read as the
checkpoint stores it — the router is F16 and the indexer's norms are F16 — because the reference upcasts those to
fp32, and rounding them through BF16 first would change which experts a token picks.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from tensorfold.cuda.exl3 import format as fmt
from tensorfold.cuda.exl3 import experts as exl3_experts
from tensorfold.cuda.exl3.linear import ROWS, Exl3Linear, pair_call, pair_compatible, tune

from .. import glm53
from . import latent
from .qmm import B16, make_b16
from .split import RankReader
from .weights import Config, DSAW, IndexW, LayerW, MLPW, MoEW, MTPW, Weights


DECODE_ROWS = 3            # rows the split-K choice is tuned for: a pending token and two MTP drafts
# K1-b: a fused projection writes its parts straight into the output's columns (the kernel takes the row stride); the
# per-part scratch + copy it replaced stays behind TF_GLM_CONCAT_DIRECT=0 so a run can fall back and A/B it.
CONCAT_DIRECT = os.environ.get("TF_GLM_CONCAT_DIRECT", "1").strip() != "0"
# K1-b's second half: the two parts of a fused projection (q_a|kv_a, gate|up) share one launch when their width,
# codebook and warp count match (TF_GLM_CONCAT_PAIR=0 turns it off; the per-part launches stay the fallback).
CONCAT_PAIR = os.environ.get("TF_GLM_CONCAT_PAIR", "1").strip() != "0"
# Layout-only candidate: keep legacy as the default and preserve the manifest's reduction choices.
LINEAR_MODE = os.environ.get("TF_GLM_LINEAR_MODE", "legacy")
LINEAR_MODES = {"legacy": 0, "transpose": 1, "transpose_cooperative": 3, "transpose_narrow": 7}
if LINEAR_MODE not in LINEAR_MODES:
    raise ValueError("TF_GLM_LINEAR_MODE: legacy, transpose, transpose_cooperative or transpose_narrow")


@dataclass
class Concat:
    """Several EXL3 layers into one output's column ranges: the fused q_a|kv_a and gate|up the engine's buffers read.

    Each part writes its own columns of the caller's output directly: the EXL3 kernel takes the output's row stride
    from the tensor it is given, so a part's ``out[:, at:at + n]`` view is written in place and the copy through
    per-part scratch is gone (plan K1-b). The views are disjoint, so parts may run on one stream or beside each
    other (``side``). A part's own workspace (rotated input, split-K partials, counters) still belongs to the
    *caller's slot* (``ws``), never to the weights: two streams running the same projection must not share it.

    """
    layers: list[Exl3Linear]
    rows: int = ROWS

    def make_scratch(self, device: torch.device, rows: int | None = None) -> list[torch.Tensor]:
        """The part outputs the copy fallback (``CONCAT_DIRECT`` off) writes into first."""

        rows = self.rows if rows is None else min(rows, ROWS)
        return [torch.empty((rows, layer.n), dtype=torch.bfloat16, device=device) for layer in self.layers]

    @property
    def k(self) -> int:
        return self.layers[0].k

    @property
    def n(self) -> int:
        return sum(layer.n for layer in self.layers)

    def nbytes(self) -> int:
        return sum(layer.nbytes() for layer in self.layers)

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None, out_dtype: torch.dtype | None = None,
                 scratch: list[torch.Tensor] | None = None, ws: list[tuple] | None = None,
                 side: tuple | None = None, **kwargs: Any) -> torch.Tensor:
        """Every part projects the same rows into its own columns: with ``side`` (stream, in-event, out-event) the
        parts past the first run beside the first, which is what a decode step wants (each call is latency-bound)."""

        rows = x.shape[0]
        if rows > ROWS:
            raise ValueError(f"Concat: {rows} rows past one EXL3 call's {ROWS}")
        if out is None:
            out = torch.empty((rows, self.n), dtype=out_dtype or torch.bfloat16, device=x.device)

        pair = (CONCAT_DIRECT and CONCAT_PAIR and len(self.layers) == 2
                and pair_compatible(self.layers[0], self.layers[1], x.shape[1]))

        def part_pair() -> None:
            a_, b_ = self.layers
            ws_a, ws_b = (None, None) if ws is None else (ws[0], ws[1])
            pair_call(a_, b_, x, out[:, at[0]:at[0] + a_.n], out[:, at[1]:at[1] + b_.n], ws_a, ws_b)

        def part(i: int) -> None:
            layer = self.layers[i]
            xh, z, counters = (None, None, None) if ws is None else ws[i]
            if CONCAT_DIRECT:
                layer(x, out=out[:, at[i]:at[i] + layer.n], out_dtype=out_dtype, xh=xh, z=z, counters=counters)
                return
            slot = (scratch[i] if scratch is not None else torch.empty(
                (rows, layer.n), dtype=torch.bfloat16, device=x.device))[:rows]
            layer(x, out=slot, out_dtype=out_dtype, xh=xh, z=z, counters=counters)
            out[:, at[i]:at[i] + layer.n].copy_(slot)

        at = [0]
        for layer in self.layers[:-1]:
            at.append(at[-1] + layer.n)
        if pair:
            part_pair()
            return out
        if side is not None and len(self.layers) > 1:
            stream, ev_in, ev_out = side
            main = torch.cuda.current_stream()
            ev_in.record(main)                          # x, the weights and both parts' scratch are ready
            stream.wait_event(ev_in)
            with torch.cuda.stream(stream):
                for i in range(1, len(self.layers)):
                    part(i)
                ev_out.record(stream)
            part(0)
            main.wait_event(ev_out)                    # the caller sees every column written
        else:
            for i in range(len(self.layers)):
                part(i)
        return out


def inv_freq(dim: int, theta: float, device: torch.device | str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The rotary frequencies of a ``dim``-wide head part: theta ** (-2i / dim), i = 0 .. dim/2 - 1."""

    i = torch.arange(0, dim, 2, dtype=torch.float64)
    return (1.0 / (theta ** (i / dim))).to(dtype).to(device).contiguous()


def install_tuning(model_dir: str | Path, ck: fmt.Checkpoint) -> dict:
    """Install the recorded split-K manifest for this checkpoint, if one is reachable (plan §5.1 items 5-7).

    Search order: ``TF_GLM_TUNING``, ``notes/tuning_manifest.json`` under the working directory, then under the
    project the package sits in. A manifest whose ``checkpoint_plan_sha256`` names a different group plan is refused
    (its splits would describe other shapes) and the load falls back to measuring, which is reported as uncovered.
    The result goes into ``Weights.meta['tuning']`` so a run's log states exactly what it used.
    """

    from tensorfold.cuda.exl3 import linear as exl3_linear

    if exl3_linear.MANIFEST:
        return dict(exl3_linear.MANIFEST_INFO, installed=True, reused=True)
    names = [os.environ.get("TF_GLM_TUNING", "")]
    try:
        names.append(str(Path(__file__).resolve().parents[6] / "notes" / "tuning_manifest.json"))
    except IndexError:                                    # installed outside a project tree: only the cwd matters
        pass
    names.append(str(Path.cwd() / "notes" / "tuning_manifest.json"))
    path = next((Path(n) for n in names if n and Path(n).is_file()), None)
    if path is None:
        return {"installed": False, "reason": "no manifest found"}
    info = exl3_linear.install_manifest(path)
    plan = fmt.plan_hash(ck)
    if info.get("checkpoint_plan_sha256") not in (None, plan):
        exl3_linear.MANIFEST.clear()
        exl3_linear.MANIFEST_INFO.clear()
        return {"installed": False, "reason": "checkpoint plan mismatch", "path": str(path),
                "manifest_plan": info.get("checkpoint_plan_sha256"), "checkpoint_plan": plan}
    return dict(info, installed=True, checkpoint_plan=plan, covered=len(exl3_linear.MANIFEST))


def private_head(tensor: torch.Tensor, rank: int, world: int, vocab: int, device) -> B16:
    """Slice before upload and own exactly one shard, even with BF16/CPU or old replicated rank files."""

    if world < 1 or not 0 <= rank < world or vocab % world:
        raise ValueError("head vocabulary must split evenly over valid ranks")
    rows = vocab // world
    if tensor.shape[0] == vocab:
        tensor = tensor[rank * rows:(rank + 1) * rows]
    elif tensor.shape[0] != rows:
        raise ValueError(f"head has {tensor.shape[0]} rows, expected {vocab} or {rows}")
    return make_b16(tensor.to(device=device, dtype=torch.bfloat16, copy=True).contiguous())


def tokenizer_vocab(model_dir: str | Path, physical: int) -> int:
    """Keep the physical head extent separate from the tokenizer's contiguous valid ID range."""

    from tokenizers import Tokenizer

    ids = set(Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json")).get_vocab().values())
    if not ids or min(ids) != 0 or max(ids) + 1 != len(ids) or len(ids) > physical:
        raise ValueError("GLM tokenizer IDs must be contiguous and fit the physical vocabulary")
    return len(ids)


def _configure_linear(layer: Exl3Linear) -> Exl3Linear:
    """Tune legacy arithmetic first, then select the layout before any graph is built.

    Even without a manifest, this option must not retune logical split-K or warp sums.
    """
    layer.kernel_mode = 0
    tune(layer, m=DECODE_ROWS)
    layer.kernel_mode = LINEAR_MODES[LINEAR_MODE]
    return layer


def load(model_dir: str | Path, *, rank: int, device: str = "cuda", mtp: bool = True, world: int = 4) -> Weights:
    """One of ``world`` ranks of a GLM-5.3 checkpoint, MTP layer included unless ``mtp`` is False."""

    cfg = Config.read(model_dir)
    if not cfg.full:
        raise ValueError("weights53 reads full GLM-5.3 (glm_moe_dsa); use weights.load for GLM-5.3-Flash")
    if cfg.heads % world or cfg.experts % world:
        raise ValueError(f"heads {cfg.heads} and experts {cfg.experts} must divide over {world} ranks")
    valid_vocab = tokenizer_vocab(model_dir, cfg.vocab)
    fields = glm53.read(model_dir)
    sched = glm53.schedule(fields)
    dev = torch.device(device)
    ck = fmt.scan(model_dir, read_markers=False)     # headers only: every group's widths, shapes and codebook
    tuning = install_tuning(model_dir, ck)           # one decided split per shape for every rank, when recorded
    rd = RankReader(model_dir, rank, ranks=world)
    HL = cfg.heads // world
    MTP_LAYER = cfg.layers

    def meta(prefix: str) -> fmt.Exl3Tensor:
        found = ck.groups.get(prefix)
        if found is None:
            raise KeyError(f"{prefix}: not an EXL3 group of {model_dir} ({ck.bad.get(prefix, 'absent')})")
        return found

    def names(prefix: str) -> list[str]:
        m = meta(prefix)
        out = [prefix + ".trellis", f"{prefix}.{m.in_scales}", f"{prefix}.{m.out_scales}"]
        return out + ([prefix + ".bias"] if m.bias else [])

    def group(prefix: str) -> Exl3Linear:
        m = meta(prefix)
        layer = Exl3Linear.from_tensors(rd.get(prefix + ".trellis"), rd.get(f"{prefix}.{m.in_scales}"),
                                        rd.get(f"{prefix}.{m.out_scales}"),
                                        m.codebook, rd.get(prefix + ".bias") if m.bias else None, device=dev)
        # a decode row wants programs, not work a program: measure this shape once (cached by shape, the same winner
        # on every rank) and keep the split that streams fastest at a verify window's row count
        return _configure_linear(layer)

    def t(name: str) -> torch.Tensor:
        """A stored tensor at the dtype the checkpoint keeps it in, on this rank's slice."""

        return rd.get(name).contiguous().to(dev)

    def bf16(name: str) -> torch.Tensor:
        return t(name).to(torch.bfloat16)

    def triples(layer: int, proj: str, experts: Iterable[int]) -> list[tuple]:
        """One projection's per-expert (trellis, suh, svh) triples, trellises kept in the checkpoint's own bytes."""

        root = f"model.layers.{layer}.mlp.experts."
        out = []
        for e in experts:
            p = f"{root}{e}.{proj}."
            m = meta(p[:-1])
            out.append((t(p + "trellis"), t(p + m.in_scales), t(p + m.out_scales)))
        return out

    def mlp(prefix: str) -> MLPW:
        """A dense or shared MLP at the rank's width: gate and up fused, down the other way."""

        parts = [group(prefix + "gate_proj"), group(prefix + "up_proj")]
        return MLPW(Concat(parts), group(prefix + "down_proj"), parts[0].n)

    def moe(layer: int) -> MoEW:
        p = f"model.layers.{layer}.mlp."
        experts = exl3_experts.prepare(triples(layer, "gate_proj", range(cfg.experts)),
                                       triples(layer, "up_proj", range(cfg.experts)),
                                       triples(layer, "down_proj", range(cfg.experts)),
                                       meta(p + "experts.0.gate_proj").codebook, device=dev)
        # the checkpoint keeps the router in F16 and the reference's kernel multiplies half activations by it with an
        # fp32 accumulator (moe_router_dtype float32 in the config is the training dtype; rounding the stored F16 to
        # bf16 first would drop three mantissa bits and can change which experts a token picks).
        return MoEW(t(p + "gate.weight"), t(p + "gate.e_score_correction_bias"), experts,
                    mlp(p + "shared_experts."))

    def indexer(layer: int) -> IndexW:
        p = f"model.layers.{layer}.self_attn.indexer."
        return IndexW(make_b16(t(p + "wk.weight")), group(p + "wq_b"), t(p + "k_norm.weight"), t(p + "k_norm.bias"),
                      None, None, make_b16(t(p + "weights_proj.weight")))

    def dsa(layer: int) -> DSAW:
        p = f"model.layers.{layer}.self_attn."
        proj = Concat([group(p + "q_a_proj"), group(p + "kv_a_proj_with_mqa")])
        rows = torch.arange(HL * (cfg.qk_nope + cfg.v_dim), device=dev).view(HL, cfg.qk_nope + cfg.v_dim)
        krows, vrows = rows[:, :cfg.qk_nope].reshape(-1), rows[:, cfg.qk_nope:].reshape(-1)
        w = t(p + "kv_b_proj.weight").to(torch.bfloat16)
        return DSAW(proj, t(p + "q_a_layernorm.weight"), t(p + "kv_a_layernorm.weight"), group(p + "q_b_proj"),
                    make_b16(w[krows]), make_b16(w[vrows]), group(p + "o_proj"), HL,
                    indexer(layer) if sched.owns_indexer(layer) else None,
                    latent.AbsorbW.from_rows(w[krows].float(), w[vrows].float(), HL) if latent.ENABLED else None)

    def ahead(layer: int) -> list[str]:
        """Every tensor of a layer, in read order, so the reader can share its reads across them."""

        out = []
        p = f"model.layers.{layer}.self_attn."
        for proj in glm53.ATTENTION:
            out += names(p + proj)
        out += [p + "q_a_layernorm.weight", p + "kv_a_layernorm.weight", p + "kv_b_proj.weight"]
        if sched.owns_indexer(layer):
            out += [p + "indexer.wk.weight", p + "indexer.k_norm.weight", p + "indexer.k_norm.bias",
                    p + "indexer.weights_proj.weight"] + names(p + "indexer.wq_b")
        out += [f"model.layers.{layer}.input_layernorm.weight", f"model.layers.{layer}.post_attention_layernorm.weight"]
        if layer == MTP_LAYER:
            out += [f"model.layers.{layer}.{n}" for n in
                    ("enorm.weight", "hnorm.weight", "shared_head.norm.weight")] + names(f"model.layers.{layer}.eh_proj")
        mlp_root = f"model.layers.{layer}.mlp."
        if layer != MTP_LAYER and sched.mlp[layer] == "dense":
            for proj in ("gate_proj", "up_proj", "down_proj"):
                out += names(mlp_root + proj)
            return out
        out += [mlp_root + "gate.weight", mlp_root + "gate.e_score_correction_bias"]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            out += names(mlp_root + "shared_experts." + proj)
        root = mlp_root + "experts."
        for proj, width in (("gate_proj", cfg.experts), ("up_proj", cfg.experts), ("down_proj", cfg.experts)):
            for e in range(width):
                out += names(f"{root}{e}.{proj}")
        return out

    built: list[LayerW] = []
    try:
        for i in list(range(cfg.layers)) + ([MTP_LAYER] if cfg.mtp_layers and mtp else []):
            rd.prefetch(ahead(i))                                    # one shared pass over the layer's tensors
            layer = LayerW(i, sched.kinds[i] if i < cfg.layers else "dsa", None, None,
                           t(f"model.layers.{i}.input_layernorm.weight"),
                           t(f"model.layers.{i}.post_attention_layernorm.weight"))
            layer.dsa = dsa(i)
            if i == MTP_LAYER:
                layer.moe = moe(i)
            elif sched.mlp[i] == "dense":
                layer.mlp = mlp(f"model.layers.{i}.mlp.")
            else:
                layer.moe = moe(i)
            built.append(layer)
            torch.cuda.empty_cache()
        dev_embed = t("model.embed_tokens.weight").to(torch.bfloat16)
        head = private_head(rd.get("lm_head.weight"), rank, world, cfg.vocab, dev)
        mtpw = None
        if cfg.mtp_layers and mtp:
            mtpw = MTPW(t(f"model.layers.{MTP_LAYER}.enorm.weight"), t(f"model.layers.{MTP_LAYER}.hnorm.weight"),
                        group(f"model.layers.{MTP_LAYER}.eh_proj"),
                        t(f"model.layers.{MTP_LAYER}.shared_head.norm.weight"), built.pop())
        w = Weights(cfg, dev_embed, built[:cfg.layers], t("model.norm.weight"), head, mtpw, rank, world, dev,
                    inv_freq=inv_freq(cfg.qk_rope, cfg.rope_theta, dev),
                    # the indexer rotates only its leading qk_rope columns, with the main attention's own table
                    index_inv_freq=inv_freq(cfg.qk_rope, cfg.rope_theta, dev))
        # EXL3 (and the BF16 head/norms) never read the 64-group row sums: the producers can skip them (K1-a)
        w.meta.update(valid_vocab=valid_vocab, norm_sums=False, layers=list(range(cfg.layers)), index_full=sched.full,
                      index_source=sched.source, kv_a_width=meta("model.layers.0.self_attn.kv_a_proj_with_mqa").n,
                      tuning=tuning, linear_mode=LINEAR_MODE)
    finally:
        rd.close()
    torch.cuda.empty_cache()
    return w


__all__ = ["Concat", "inv_freq", "load"]
