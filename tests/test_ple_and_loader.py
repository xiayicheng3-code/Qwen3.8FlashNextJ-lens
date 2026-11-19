"""Tests for MmapPleTable, streaming safetensors loader (split FP8 experts + PLE shards), Tier 2 NVMe slabs, and checkpoint resume/merge."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

from qwen38_jlens.engine import Qwen38JLensEngine
from qwen38_jlens.hardware import HardwareProfile, StorageTier, build_offload_plan
from qwen38_jlens.lens import Qwen38JacobianLens, render_slice_html
from qwen38_jlens.loader import (
    allocate_moe_payload,
    load_qwen38_streaming,
    read_moe_slab_into,
    write_moe_slab,
)
from qwen38_jlens.model_ops import MoeExpertPayload
from qwen38_jlens.ple_mmap import MmapPleTable, PleHashSpec, PreextractedPleProvider


def _make_synthetic_fp8_checkpoint(root: Path) -> tuple[dict, torch.Tensor, float]:
    """Create a realistic sharded multimodal Qwen3.8-Flash-Next-FP8 checkpoint on disk.

    Includes:
      - `model.language_model.*` multimodal prefix
      - `model.visual.*` and `mtp.*` decoy shards (must be skipped!)
      - Per-expert split FP8 weights + `weight_scale_inv` (`152,089`-tensor style!)
      - Sharded `ngram_embedding.shard_0..3.weight` (FP8 E4M3) + `weight_scale`
    """
    torch.manual_seed(7)
    D, hc, E, top_k, inter, V = 16, 4, 4, 2, 16, 64
    cfg = {
        "model_type": "qwen4_exp",
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [8, 8],
        },
        "text_config": {
            "vocab_size": V,
            "hidden_size": D,
            "num_hidden_layers": 2,
            "hc_count": hc,
            "hc_lowrank": 8,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "num_experts": E,
            "num_experts_per_tok": top_k,
            "moe_intermediate_size": inter,
            "shared_expert_intermediate_size": inter,
            "layer_types": ["linear_attention", "indexed_attention"],
            "ple_layer_ids": [1],  # Layer 0 has PLE (1-indexed=1) for compact 2-layer test
            "ple_embed_dim": D,
            "ple_conv_kernel_size": 4,
            "ngram_size": 3,
            "heads_per_ngram": 2,
            "ngram_vocab_size_base": 17,
            "make_ngram_vocab_size_divisible_by": 4,
            "split_ngram_parts": 4,
            "seed": 1234,
            "eos_token_id": V - 1,
            "bos_token_id": V - 1,
            "output_gate_type": "sigmoid",
            "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.25},
        },
    }
    (root / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    spec = PleHashSpec(
        vocab_size=V, ngram_size=3, heads_per_ngram=2,
        ngram_vocab_size_base=17, make_ngram_vocab_size_divisible_by=4,
        ple_embed_dim=D, seed=1234, eos_token_id=V - 1, ple_layer_index=0,
    )
    full_ple_fp8 = (torch.randn(spec.padded_vocab_size, spec.head_dim_per_ngram) * 2.0).to(torch.float8_e4m3fn)
    ple_scale = 0.03125

    weight_map: dict[str, str] = {}

    # Shard 1: Vision + MTP decoy (must be skipped!)
    decoy = {
        "model.visual.patch_embed.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        "mtp.layers.0.weight": torch.randn(4, 4, dtype=torch.bfloat16),
    }
    save_file(decoy, str(root / "model-00001-vision-mtp.safetensors"))
    for k in decoy:
        weight_map[k] = "model-00001-vision-mtp.safetensors"

    # Shards 2-3: 4 PLE n-gram shards across 2 safetensors files
    chunks = torch.chunk(full_ple_fp8, 4, dim=0)
    ple_f1 = {
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight": chunks[0].contiguous(),
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_1.weight": chunks[1].contiguous(),
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.weight_scale": torch.tensor([ple_scale], dtype=torch.bfloat16),
    }
    ple_f2 = {
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_2.weight": chunks[2].contiguous(),
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_3.weight": chunks[3].contiguous(),
    }
    save_file(ple_f1, str(root / "model-00002-ple.safetensors"))
    save_file(ple_f2, str(root / "model-00003-ple.safetensors"))
    for k in ple_f1:
        weight_map[k] = "model-00002-ple.safetensors"
    for k in ple_f2:
        weight_map[k] = "model-00003-ple.safetensors"

    # Shard 4: Dense backbone + Layer 0 & Layer 1 split FP8 experts
    main: dict[str, torch.Tensor] = {
        "model.language_model.embed_tokens.weight": torch.randn(V, D, dtype=torch.bfloat16) * 0.1,
        "lm_head.weight": torch.randn(V, D, dtype=torch.bfloat16) * 0.1,
        "model.language_model.hyper_connection_mixer.hc_norm.weight": torch.zeros(hc * D, dtype=torch.bfloat16),
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight": torch.randn(8, hc * D, dtype=torch.bfloat16) * 0.1,
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight": torch.randn(hc * D, 8, dtype=torch.bfloat16) * 0.1,
    }
    for l in range(2):
        p = f"model.language_model.layers.{l}"
        for hc_name in ("attn_hyper_connection", "mlp_hyper_connection"):
            main[f"{p}.{hc_name}.hc_norm.weight"] = torch.zeros(hc * D, dtype=torch.bfloat16)
            main[f"{p}.{hc_name}.input_mix_weight_down.weight"] = torch.randn(8, hc * D, dtype=torch.bfloat16) * 0.1
            main[f"{p}.{hc_name}.input_mix_weight_up.weight"] = torch.randn(hc * D, 8, dtype=torch.bfloat16) * 0.1
            main[f"{p}.{hc_name}.block_inject_weight.weight"] = torch.randn(hc, hc * D, dtype=torch.bfloat16) * 0.1

        main[f"{p}.mlp.gate.weight"] = torch.randn(E, D, dtype=torch.bfloat16) * 0.1
        main[f"{p}.mlp.shared_expert.gate_proj.weight"] = torch.randn(inter, D, dtype=torch.bfloat16) * 0.1
        main[f"{p}.mlp.shared_expert.up_proj.weight"] = torch.randn(inter, D, dtype=torch.bfloat16) * 0.1
        main[f"{p}.mlp.shared_expert.down_proj.weight"] = torch.randn(D, inter, dtype=torch.bfloat16) * 0.1
        main[f"{p}.mlp.shared_expert_gate.weight"] = torch.randn(1, D, dtype=torch.bfloat16) * 0.1

        for e in range(E):
            main[f"{p}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(inter, D).to(torch.float8_e4m3fn)
            main[f"{p}.mlp.experts.{e}.gate_proj.weight_scale_inv"] = torch.full((inter // 8, D // 8), 0.05, dtype=torch.float32)
            main[f"{p}.mlp.experts.{e}.up_proj.weight"] = torch.randn(inter, D).to(torch.float8_e4m3fn)
            main[f"{p}.mlp.experts.{e}.up_proj.weight_scale_inv"] = torch.full((inter // 8, D // 8), 0.05, dtype=torch.float32)
            main[f"{p}.mlp.experts.{e}.down_proj.weight"] = torch.randn(D, inter).to(torch.float8_e4m3fn)
            main[f"{p}.mlp.experts.{e}.down_proj.weight_scale_inv"] = torch.full((D // 8, inter // 8), 0.05, dtype=torch.float32)

        if l == 0:
            cd = 2 * 8 + 16
            main[f"{p}.linear_attn.in_proj_qkv.weight"] = torch.randn(cd, D, dtype=torch.bfloat16) * 0.1
            main[f"{p}.linear_attn.in_proj_z.weight"] = torch.randn(16, D, dtype=torch.bfloat16) * 0.1
            main[f"{p}.linear_attn.in_proj_b.weight"] = torch.randn(4, D, dtype=torch.bfloat16) * 0.1
            main[f"{p}.linear_attn.in_proj_a.weight"] = torch.randn(4, D, dtype=torch.bfloat16) * 0.1
            main[f"{p}.linear_attn.conv1d.weight"] = torch.randn(cd, 1, 4, dtype=torch.bfloat16) * 0.1
            main[f"{p}.linear_attn.dt_bias"] = torch.ones(4, dtype=torch.float32)
            main[f"{p}.linear_attn.A_log"] = torch.zeros(4, dtype=torch.float32)
            main[f"{p}.linear_attn.norm.weight"] = torch.ones(4, dtype=torch.bfloat16)
            main[f"{p}.linear_attn.out_proj.weight"] = torch.randn(D, 16, dtype=torch.bfloat16) * 0.1
            # PLE dense projections on layer 0
            main[f"{p}.ple.key_proj.weight"] = torch.randn(hc * D, D, dtype=torch.bfloat16) * 0.1
            main[f"{p}.ple.value_proj.weight"] = torch.randn(D, D, dtype=torch.bfloat16) * 0.1
            main[f"{p}.ple.norm_key.weight"] = torch.zeros(hc * D, dtype=torch.bfloat16)
            main[f"{p}.ple.norm_query.weight"] = torch.zeros(hc * D, dtype=torch.bfloat16)
            main[f"{p}.ple.norm_conv.weight"] = torch.zeros(hc * D, dtype=torch.bfloat16)
            main[f"{p}.ple.conv1d.weight"] = torch.randn(hc * D, 1, 4, dtype=torch.bfloat16) * 0.1
        else:
            main[f"{p}.self_attn.q_proj.weight"] = torch.randn(4 * 8 * 2, D, dtype=torch.bfloat16) * 0.1
            main[f"{p}.self_attn.k_proj.weight"] = torch.randn(2 * 8, D, dtype=torch.bfloat16) * 0.1
            main[f"{p}.self_attn.v_proj.weight"] = torch.randn(2 * 8, D, dtype=torch.bfloat16) * 0.1
            main[f"{p}.self_attn.o_proj.weight"] = torch.randn(D, 4 * 8, dtype=torch.bfloat16) * 0.1
            main[f"{p}.self_attn.q_norm.weight"] = torch.zeros(8, dtype=torch.bfloat16)
            main[f"{p}.self_attn.k_norm.weight"] = torch.zeros(8, dtype=torch.bfloat16)

    save_file(main, str(root / "model-00004-backbone.safetensors"))
    for k in main:
        weight_map[k] = "model-00004-backbone.safetensors"

    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )
    return cfg, full_ple_fp8, ple_scale


def test_ple_mmap_and_slab_roundtrip(tmp_path: Path):
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    _, full_ple_fp8, ple_scale = _make_synthetic_fp8_checkpoint(ckpt_dir)

    spec = PleHashSpec(
        vocab_size=64, ngram_size=3, heads_per_ngram=2,
        ngram_vocab_size_base=17, make_ngram_vocab_size_divisible_by=4,
        ple_embed_dim=16, seed=1234, eos_token_id=63, ple_layer_index=0,
    )
    mmap_tbl = MmapPleTable.from_safetensors_files(
        spec, [str(ckpt_dir / "model-00002-ple.safetensors"), str(ckpt_dir / "model-00003-ple.safetensors")]
    )
    ids = torch.randint(0, 62, (1, 32))
    out_mmap = mmap_tbl.lookup(ids, device="cpu")

    # Compare bit-for-bit against dense table lookup
    ngram_ids = spec.compute_ngram_ids(ids)
    out_ref = (full_ple_fp8[ngram_ids].to(torch.bfloat16) * ple_scale).flatten(-2)
    assert torch.equal(out_mmap, out_ref)

    # Verify PreextractedPleProvider matches bit-for-bit
    pre = PreextractedPleProvider.from_mmap_and_prompts(mmap_tbl, [ids])
    assert torch.equal(pre.lookup(ids, device="cpu"), out_ref)
    mmap_tbl.close()

    # Verify Tier 2 binary MoE slab write + readinto bit-for-bit
    p_src = MoeExpertPayload(
        gate_up_proj=torch.randn(4, 32, 16).to(torch.float8_e4m3fn),
        down_proj=torch.randn(4, 16, 16).to(torch.float8_e4m3fn),
        gate_up_scale=torch.randn(4, 4, 2, dtype=torch.float32),
        down_scale=torch.randn(4, 2, 2, dtype=torch.float32),
    )
    slab_path = str(tmp_path / "layer_00.q38slab")
    write_moe_slab(slab_path, p_src)
    p_dst = allocate_moe_payload(4, 16, 16, weight_dtype=torch.float8_e4m3fn, block_size=(8, 8))
    read_moe_slab_into(slab_path, p_dst)
    assert torch.equal(p_src.gate_up_proj.view(torch.uint8), p_dst.gate_up_proj.view(torch.uint8))
    assert torch.equal(p_src.down_proj.view(torch.uint8), p_dst.down_proj.view(torch.uint8))
    assert torch.equal(p_src.gate_up_scale, p_dst.gate_up_scale)
    assert torch.equal(p_src.down_scale, p_dst.down_scale)


def test_streaming_loader_with_nvme_slab_and_checkpoint_resume(tmp_path: Path):
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    _make_synthetic_fp8_checkpoint(ckpt_dir)

    prof = HardwareProfile("CPU", (0, 0), 0.0, 0.0, 32.0, 24.0, 200.0, 4, has_cuda=False)
    plan = build_offload_plan(
        prof, num_layers=2, d_model=16, hc_count=4, num_experts=4,
        moe_intermediate_size=16, max_seq_len=24, dense_backbone_gib=0.01, ple_table_gib=0.01,
    )
    # Explicitly test Tier 1 Host + Tier 2 NVMe slab simultaneously!
    plan.layer_tiers = {0: StorageTier.HOST_PINNED, 1: StorageTier.NVME_SLAB}

    loaded = load_qwen38_streaming(
        str(ckpt_dir), plan, slab_dir=str(tmp_path / "slabs"), compute_dtype=torch.bfloat16
    )
    assert isinstance(loaded.moe_tier_store[0], MoeExpertPayload)
    assert isinstance(loaded.moe_tier_store[1], str) and os.path.isfile(loaded.moe_tier_store[1])

    engine = Qwen38JLensEngine(loaded)
    prompts = [torch.randint(0, 62, (1, 20)) for _ in range(3)]
    ckpt_file = str(tmp_path / "fit_ckpt.pt")

    # Fit first 2 prompts -> save checkpoint -> resume on all 3!
    lens_part = engine.fit(prompts[:2], skip_first=2, checkpoint_path=ckpt_file, checkpoint_every=1, resume=False)
    assert lens_part.n_prompts == 2

    lens_full = engine.fit(prompts, skip_first=2, checkpoint_path=ckpt_file, checkpoint_every=1, resume=True)
    assert lens_full.n_prompts == 3

    # Verify direct fit without resume matches resumed fit!
    lens_direct = engine.fit(prompts, skip_first=2, resume=False)
    for l in (0, 1):
        assert torch.allclose(lens_full.jacobians[l], lens_direct.jacobians[l], atol=1e-5)

    # Save, reload, merge, and render HTML slice
    lens_path = str(tmp_path / "qwen38_lens.pt")
    lens_full.save(lens_path)
    reloaded = Qwen38JacobianLens.load(lens_path)
    merged = Qwen38JacobianLens.merge([lens_part, reloaded])
    assert merged.n_prompts == 5
    assert len(reloaded.branch_norms()[0]) == 4
    html_doc = render_slice_html(["The", "cat"], {0: [("dog", 0.5), ("sat", 0.8)]}, [("sat", 0.9), ("on", 0.7)])
    assert "Qwen3.8-Flash-Next" in html_doc

    engine.close()
    loaded.ple_providers[0].close()
