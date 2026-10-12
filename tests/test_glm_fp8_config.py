"""CPU-only FP8 opt-in and memory geometry contracts."""
import json
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.glm5_next.cuda import cache_fp8
from tensorfold.cuda.geometry import mla_geometry


@pytest.mark.parametrize("width", [64, 512])
def test_packed_cache_geometry(width):
    for dtype, size, storage in [("bf16", 2 * width, torch.bfloat16), ("fp8", width + 4, torch.uint8)]:
        cache = cache_fp8.allocate(19, width, "cpu", dtype)
        assert cache.dtype == storage and cache.is_contiguous()
        assert cache.numel() * cache.element_size() == 19 * size
        assert cache_fp8.row_bytes(width, dtype) == size


@pytest.mark.parametrize("width,dtype", [(63, "fp8"), (0, "fp8"), (512, "int8"), (512, "garbage")])
def test_bad_storage_rejected(width, dtype):
    with pytest.raises(ValueError):
        cache_fp8.allocate(19, width, "cpu", dtype)


def config():
    return dict(model_type="glm_moe_dsa", num_hidden_layers=78, num_attention_heads=64,
                qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256, hidden_size=6144,
                num_experts_per_tok=8, moe_intermediate_size=2048, vocab_size=154880,
                kv_lora_rank=512, indexer_types=["full"] * 22 + ["shared"] * 56,
                num_nextn_predict_layers=1)


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("capacity", [2560, 16384, 131592])
def test_capacity_savings_match_storage_exactly(mtp, capacity):
    t = config()
    bf = mla_geometry(t, 4, 8, latent=True, mtp=mtp)
    fp = mla_geometry(t, 4, 8, latent=True, mtp=mtp, kv_dtype="fp8")
    assert bf.bytes_at(capacity) - fp.bytes_at(capacity) == (78 + int(mtp)) * capacity * (576 - 8)


def test_fp8_not_accepted_for_flash_or_expanded_geometry():
    for text, latent in [(dict(config(), model_type="glm5_next"), True), (config(), False)]:
        with pytest.raises(ValueError, match="full GLM"):
            mla_geometry(text, 4, 8, latent=latent, kv_dtype="fp8")


def test_family_passes_explicit_dtype_and_defaults_full_to_fp8(monkeypatch, tmp_path):
    from tensorfold.families import glm5_next as family
    from tensorfold.families.glm5_next.cuda import engine
    (tmp_path / "config.json").write_text(json.dumps(config()))
    monkeypatch.setattr(engine, "GlmEngine", lambda *a, **kw: kw)
    assert family.cuda_engine(tmp_path, tp=4, master="127.0.0.1", kv_dtype="fp8")["kv_dtype"] == "fp8"
    assert family.cuda_engine(tmp_path, tp=4, master="127.0.0.1", kv_dtype="bf16")["kv_dtype"] == "bf16"
    assert family.cuda_engine(tmp_path, tp=4, master="127.0.0.1")["kv_dtype"] == "fp8"   # full GLM defaults to fp8
    (tmp_path / "config.json").write_text(json.dumps(dict(config(), model_type="glm5_next")))
    with pytest.raises(ValueError, match="only by full"):
        family.cuda_engine(tmp_path, tp=4, master="127.0.0.1", kv_dtype="fp8")
