# GLM-5.3 (753B) EXL3 on TensorFold — TP=4 reference

This document describes the GLM-5.3 family port (`src/tensorfold/families/glm5_next/`)
and measured serving behavior on a tensor-parallel 4-way (TP=4) host of
4x NVIDIA RTX PRO 6000 Blackwell (SM120, PCIe).

All numbers here are from the **most recent** measurement round only. Earlier
figures from a synthetic fixed suite are **withdrawn** — see section 6.

## 1. What this port is

TensorFold's main line is the Zig engine and server. The Python tree under
`src/tensorfold/` is a maintained Python implementation; this document and the
`glm5_next` family belong to that Python line.

The work here does **not** train or re-quantize a model. It connects an
**existing** EXL3 checkpoint to the TensorFold runtime so that it can be served
at TP=4. Quantization parameters, calibration corpus and the conversion
pipeline are upstream artifacts and are not reproduced here.

The port adds a model family and its operators. It does not add a second
scheduler beside the engine: projections, MoE and collectives are shared across
rows, while MLA/DSA/MTP state is per request.

## 2. Checkpoint

| Property | Value |
|---|---|
| Architecture | `GlmMoeDsaForCausalLM` |
| Total parameters | 753B |
| Layers | 78 + 1 MTP layer |
| Experts | 256 routed (8 active) + 1 shared |
| Attention | MLA with DSA sparse indexer |
| Quantization | EXL3, 3.04 bpw average, `mul1` codebook, ExLlamaV3 `d3739fd` |
| Size on disk | 273 GiB |
| License | MIT |

### Lineage and modification (read this before comparing results)

The checkpoint is **not** stock GLM-5.3. Its lineage is:

```
zai-org/GLM-5.3
  -> dealignai/GLM-5.3-UNCENSORED-FP8   (weight edit, no fine-tune)
  -> GLM-5.3 EXL3 3.04 bpw              (EXL3 quantization)
```

The intermediate step is a **weight modification** documented in that
repository's `CRACK_SURGERY.json` (`mode: mpoa`, attention layer list). It is not
a fine-tune and is not affiliated with or endorsed by Z.ai.

Consequences for interpretation:

- Every number below reflects **engine + quantization + weight edit** together.
  It is not a measurement of stock GLM-5.3.
- The checkpoint's own documentation records a possible quality regression
  against stock GLM-5.3 (tau2-bench airline `-0.070`, ~1.1 SE; retail `-0.022`,
  not significant at that sample size).
- The same documentation records a known tool-calling defect: the `glm4_5`
  argument parser JSON-decodes every value without consulting the tool schema,
  so string parameters that look numeric arrive as integers.

Weights are not redistributed here. Obtain the checkpoint from its own
repository under its own license (MIT).

## 3. Host

- 4x NVIDIA RTX PRO 6000 Blackwell **Max-Q** Workstation Edition, 97,887 MiB each
- **PCIe, no NVLink** (SM120 has no NVLink)
- Driver 595.71.05, CUDA 13.0, NCCL 2.29.7
- Python 3.14.5, Torch 2.13.0+cu130, nvcc 13.0.88
- GPUs used: **0-3** for TP=4

## 4. Measured results (latest round)

Cache precision is **FP8 E4M3FN** (latent and rotary cache, per-row FP32 scale;
DSA index keys in FP16). MTP is enabled, max depth 7, policy `t7:0.5:0.7`.

### 4.1 Pure decode (prefill excluded)

All requests are prefilled first, all ranks and GPUs are synchronized, then
decode is timed by the engine's own decode walltime. MTP, sampling, NCCL and CPU
control are included; prefill and HTTP are not. The first token produced by
prefill is excluded from the numerator. Input 1K = 1024 tokens, up to 512
generated tokens, EOS respected. 4 inputs per condition, 1 sweep, **no CI**.

Decode rate, tok/s (prose / code):

| Input | C=1 | C=2 total | C=4 total |
|---|---:|---:|---:|
| 1K | **87.4 / 95.2** | 96.4 / 106.8 | 107.2 / 120.1 |
| 16K | 78.0 / 69.8 | 87.7 / 78.7 | 96.6 / 91.7 |
| 32K | 71.9 / 81.4 | 79.0 / 87.0 | 85.4 / 95.2 |
| 64K | 63.0 / 71.5 | 68.4 / 77.8 | *capacity* |
| 128K | 48.9 / 50.8 | *capacity* | *capacity* |

Per-request average (C=2), prose / code: 1K 51.3 / 56.4; 16K 44.8 / 43.8;
32K 40.0 / 49.8; 64K 35.4 / 40.4. Total rate divided by C does not equal the
per-request average.

### 4.2 Capacity

30 conditions were planned; 24 were measured and **6 were skipped by the native
MemoryGate capacity check**. 96 requests, 0 request failures. The largest
configurations actually admitted and verified:

| Input | C=1 | C=2 | C=4 |
|---|---|---|---|
| 1K | yes | yes | yes |
| 16K | yes | yes | yes |
| 32K | yes | yes | yes |
| 64K | yes | yes | no (needs 10.817 GB, 4.743 GB available) |
| 128K | yes | no (needs 7.183 GB, 1.041 GB available) | no |

A single 225,548-token request was an **estimate at startup** and was never
proven; the largest actually allocated and generated is 128K input + 512 output.

At 128K, rank0 startup allocated 82.195 GiB with 6.690 GiB private State per
rank; the context window is 131,584 slots.

### 4.3 Cold-arrival HTTP (for comparison, not steady-state)

Measured over the first-to-last delivery interval of simultaneously arriving
requests, so it **includes prefill interference from other requests** and is not
a steady-state decode rate.

| Input | C=1 | C=2 | C=4 |
|---|---:|---:|---:|
| 1K | 86.8 / 94.8 | 85.6 / 92.0 | 87.0 / 97.9 |
| 16K | 78.1 / 74.2 | 24.4 / 19.5 | 18.3 / 14.5 |
| 32K | 71.7 / 81.9 | 13.6 / 11.9 | 9.6 / 8.3 |
| 64K | 63.2 / 71.5 | 7.0 / 7.0 | *capacity* |
| 128K | 48.7 / 51.0 | *capacity* | *capacity* |

TTFT at C=1 (prose): 1.34 s at 1K, 28.32 s at 16K, 59.61 s at 32K, 128.84 s at
64K, 289.28 s at 128K.

At 16K C=4, 32K C=2/C=4 and 64K C=2, the generation intervals of the streams did
not overlap in common time, so these rows cannot be used to judge steady
concurrent scaling.

### 4.4 What these numbers are not

- **One sweep, 4 inputs per condition, no confidence intervals.** These are
  exploratory, not a contract-grade result.
- **FP8 vs BF16 have not been A/B'd on the same natural text input.** The
  contribution of the cache precision change to speed or quality is not
  separated. Do not attribute differences to FP8.
- Concurrency scaling is not established: `2x` and `4x` columns are totals for a
  finite batch, not sustained load.
- Short completions are kept, not excluded: a 16K code input produced 8 IDs and
  a 32K code input 195 IDs, both including EOS.

## 5. Correctness evidence

- **Original token IDs are preserved.** Drafted output equals `"draft": false`
  output token for token; all 96 requests matched the earlier HTTP run exactly.
- **Per-request state is isolated.** MLA/DSA/MTP state is per request while
  projections, MoE and collectives are shared across rows.
- **Boundary inputs are exercised.** Attention key scanning across the 32
  boundary and DSA capacity limits were checked with boundary inputs. An
  out-of-bounds read masked by the output mask was found and fixed; a clean
  output does not by itself imply memory safety.
- **An independent oracle is used.** Numerical behavior was compared against an
  independent reference, not only against the previous engine version.
- **Every run ended cleanly.** All ranks exited `[0,0,0,0]`, `source_unchanged`
  held, and only owned server processes were shut down.

Limit: matching token IDs demonstrates invariance across an optimization step.
It does not establish numerical identity with an independent implementation,
full model quality, or memory safety. A full sanitizer-clean run at TP=4 was not
obtained.

## 6. Withdrawn figures

Earlier rounds reported **180-198 tok/s** for this engine. Those came from a
frozen fixed suite whose prompts were padded with repeated instructions, and the
model then generated repetitions of those instructions instead of solving the
task. Repetitive output makes MTP accept far more drafts, inflating the rate.

Example, same checkpoint, 1K C=1:

| | synthetic fixed suite | natural input |
|---|---|---|
| prose | 221.35 tok/s (acceptance 100%) | 86.77 tok/s (acceptance 51.29%) |
| code | 196.45 tok/s (acceptance 97.05%) | 94.79 tok/s (acceptance 52.09%) |

The gap is already present at 1K C=1, so long context or concurrent prefill
cannot explain it. **Those earlier high figures are repetition-heavy synthetic
throughput and are withdrawn as a statement about practical inference speed.**
Exact output-ID invariance establishes consistency, not task completion or
representative throughput.

No practical single-stream 180-200 tok/s result was achieved.

## 7. Serving

```bash
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4 MAX_JOBS=3
export NCCL_P2P_LEVEL=SYS

# GLM-5.3 family paths exercised during development
export TF_GLM_MAIN_PAD=1export TF_GLM_CONTROL_FRAME=1
export TF_GLM_GRAPH_MISS_MODE=once
export TF_GLM_SAMPLING_GRAPH=1 TF_GLM_BATCH_SAMPLING=1
export TF_GLM_ROUTE_MODE=ballot
export TF_GLM_NUCLEUS_INITIAL=64 TF_GLM_NUCLEUS_FALLBACK=4096
export TF_GLM_GRAPH_CACHE=64 TF_GLM_SAMPLING_CACHE=32
export TF_GLM_ALL_GRAPH_ROWS=1
export TF_GLM_GROUPED_MODE=transpose
export TF_GLM_DOWN_COMBINE=1
export TF_GLM_LINEAR_MODE=transpose_narrow
export TF_GLM_INPLACE_GATHER=1
export TF_GLM_EXPERT_TILE=nt4_at8

export TF_NUCLEUS_AMAX=1 TF_NUCLEUS_UNSORTED=1 TF_NUCLEUS_SINGLETON=1

# KV cache: bf16 (default) or fp8. FP8 holds E4M3FN codes with one FP32 scale a row and
# roughly halves the cache, which is what lets a wide window fit on four cards.
# It changes the model's numerics; see section 8.
#   tensorfold serve MODEL --backend cuda --tp 4 ... --kv-dtype fp8
```

These are **experimental selectors**, left default-off or legacy-on in the code.
Enabling them is a deliberate choice, not a supported default.
`TF_GLM_EXPERT_TILE=nt4_at8` and `TF_GLM_INPLACE_GATHER=1` change kernel dispatch
and collective behavior; verify correctness on your checkpoint before relying on
them. Do not mix a native NCCL size tuner into a run unless it was rebuilt for
your exact NCCL version. `--kv-dtype` is accepted by full GLM-5.3 only; Flash
serves bf16.

## 8. Scope and non-claims

- Experimental switches were never adopted as service defaults.
- Robustness under mixed arrivals, prefix hit/miss, cancellation/reuse, grammar
  and tool constraints, and slow consumers was not fully exercised.
- Abnormal-failure injection for the NCCL path (rank delay, OOM rollback, fatal
  timeout) was not completed.
- Long-context quality was not evaluated; only capacity and rate were measured.
- The FP8 KV cache (`--kv-dtype fp8`) **changes the model's numerics**: the latent and
  rotary caches hold E4M3FN codes with one FP32 scale a row, so attention reads a quantized
  key and value, and exactness is then relative to the FP8 serial run rather than to bf16. It
  is opt-in and off by default. It has **not** been A/B'd against bf16 on the same natural
  text, so its effect on quality is not measured here and no quality-equivalence claim is
  made. Index keys stay FP16. `CONTRIBUTING.md` asks for an issue before a precision mode
  that a checkpoint format defines; this is such a mode.

If you need a performance receipt for your own hardware, measure against your
own checkpoint and publish the receipt with platform, library versions and
request parameters attached.

## 9. Licensing and notices

This tree follows TensorFold's Apache-2.0 license (`LICENSE`). See `NOTICE`,
`LICENSES/` and `THIRD_PARTY_NOTICES.md` for the retained notices.

Model weights carry their own license (MIT for the reference checkpoint above)
and are not included.
