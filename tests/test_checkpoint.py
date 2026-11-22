"""Tests for crash-proof AsyncCheckpointManager, safetensors J_mean checkpoints, multi-source resume, and KeyboardInterrupt flush."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from qwen38_jlens.checkpoint import (
    AsyncCheckpointManager,
    CheckpointConfig,
    ConvergenceLogger,
    export_neuronpedia_bundle,
    load_jlens_safetensors,
    peek_checkpoint_metadata,
    save_jlens_safetensors,
)
from qwen38_jlens.engine import FitStats, Qwen38JLensEngine
from qwen38_jlens.hardware import HardwareProfile, StorageTier, build_offload_plan
from qwen38_jlens.lens import Qwen38JacobianLens
from qwen38_jlens.loader import load_qwen38_streaming
from tests.test_ple_and_loader import _make_synthetic_fp8_checkpoint


def test_safetensors_checkpoint_roundtrip_and_direct_lens_load(tmp_path: Path):
    torch.manual_seed(10)
    j_mean = {-1: torch.randn(16, 64), 0: torch.randn(16, 64), 1: torch.randn(16, 64)}
    ckpt = str(tmp_path / "ckpt.safetensors")

    save_jlens_safetensors(
        ckpt, j_mean, n_done=37, next_idx=40, skip_first=16, d_model=16, hc_count=4, dtype=torch.float16
    )

    from safetensors import safe_open
    with safe_open(ckpt, framework="pt", device="cpu") as sf:
        assert "J.embed" in sf.keys()
        assert "J.layer_00" in sf.keys()

    meta = peek_checkpoint_metadata(ckpt)
    assert meta is not None
    assert meta["n_done"] == 37
    assert meta["next_idx"] == 40
    assert meta["source_layers"] == [-1, 0, 1]
    assert meta["ckpt_dtype"] == "float16"

    # Any in-progress checkpoint is IMMEDIATELY loadable as a usable Qwen38JacobianLens!
    lens = Qwen38JacobianLens.load(ckpt)
    assert lens.n_prompts == 37
    assert lens.source_layers == [-1, 0, 1]
    assert lens.d_model == 16 and lens.hc_count == 4
    assert torch.allclose(lens.jacobians[-1], j_mean[-1], atol=2e-3)
    assert torch.allclose(lens.jacobians[0], j_mean[0], atol=2e-3)

    # Also test Qwen38JacobianLens.save() -> .safetensors with collapsed J
    out_st = str(tmp_path / "final.safetensors")
    lens.save(out_st, dtype=torch.float16, include_collapsed=True)
    reloaded = Qwen38JacobianLens.load(out_st)
    assert reloaded.source_layers == [-1, 0, 1]
    assert torch.allclose(reloaded.jacobians[1], lens.jacobians[1], atol=1e-4)


def test_async_manager_picks_freshest_across_local_and_drive(tmp_path: Path, caplog):
    local_p = str(tmp_path / "local" / "ckpt.safetensors")
    drive_p = str(tmp_path / "drive" / "ckpt.safetensors")
    os.makedirs(os.path.dirname(local_p), exist_ok=True)
    os.makedirs(os.path.dirname(drive_p), exist_ok=True)

    save_jlens_safetensors(
        local_p, {0: torch.ones(8, 32)}, n_done=10, next_idx=10, skip_first=2, d_model=8, hc_count=4, dtype=torch.float32
    )
    save_jlens_safetensors(
        drive_p, {0: torch.full((8, 32), 2.5)}, n_done=25, next_idx=25, skip_first=2, d_model=8, hc_count=4, dtype=torch.float32
    )

    mgr = AsyncCheckpointManager(
        CheckpointConfig(local_path=local_p, drive_path=drive_p, ckpt_dtype="float32", every_n_prompts=5)
    )
    resumed = mgr.load_best_resume(expected_source_layers=[0], expected_skip_first=2)
    assert resumed is not None
    j_sum, n_done, next_idx = resumed
    assert n_done == 25 and next_idx == 25
    assert torch.allclose(j_sum[0] / n_done, torch.full((8, 32), 2.5), atol=1e-6)
    mgr.close()

    # Enabling BOTH Drive and HF simultaneously emits a clear shared-bandwidth warning
    with caplog.at_level("WARNING"):
        mgr_dual = AsyncCheckpointManager(
            CheckpointConfig(local_path=local_p, drive_path=drive_p, hf_repo_id="user/dummy")
        )
        mgr_dual.close()
    assert any("share outbound" in rec.message for rec in caplog.records)


def test_engine_fit_async_checkpoint_resume_csv_and_keyboard_interrupt(tmp_path: Path):
    ckpt_dir = tmp_path / "model"
    ckpt_dir.mkdir()
    _make_synthetic_fp8_checkpoint(ckpt_dir)

    prof = HardwareProfile("CPU", (0, 0), 0.0, 0.0, 32.0, 24.0, 200.0, 4, has_cuda=False)
    plan = build_offload_plan(
        prof, num_layers=2, d_model=16, hc_count=4, num_experts=4,
        moe_intermediate_size=16, max_seq_len=24, dense_backbone_gib=0.01, ple_table_gib=0.01,
    )
    plan.layer_tiers = {0: StorageTier.HOST_PINNED, 1: StorageTier.NVME_SLAB}
    loaded = load_qwen38_streaming(str(ckpt_dir), plan, slab_dir=str(tmp_path / "slabs"), compute_dtype=torch.bfloat16)
    engine = Qwen38JLensEngine(loaded)

    prompts = [torch.randint(0, 62, (1, 20)) for _ in range(4)]
    local_ckpt = str(tmp_path / "local_ckpt.safetensors")
    drive_ckpt = str(tmp_path / "drive_ckpt.safetensors")
    local_csv = str(tmp_path / "local_conv.csv")
    drive_csv = str(tmp_path / "drive_conv.csv")

    mgr1 = AsyncCheckpointManager(
        CheckpointConfig(
            local_path=local_ckpt,
            drive_path=drive_ckpt,
            local_convergence_csv=local_csv,
            drive_convergence_csv=drive_csv,
            ckpt_dtype="float32",
            every_n_prompts=10,
            every_minutes=0.0,
        )
    )

    def _interrupt_after_two(stats):
        if stats.prompt_idx == 1:
            raise KeyboardInterrupt("Simulated Colab stop")

    with pytest.raises(KeyboardInterrupt):
        engine.fit(
            prompts, include_embedding_layer=True, skip_first=2,
            checkpoint_manager=mgr1, resume=False, on_prompt_done=_interrupt_after_two,
        )
    mgr1.close()

    # Verify BOTH Drive checkpoint AND Drive convergence CSV were safely flushed on KeyboardInterrupt!
    meta_drive = peek_checkpoint_metadata(drive_ckpt)
    assert meta_drive is not None and meta_drive["n_done"] == 2 and meta_drive["next_idx"] == 2
    assert meta_drive["source_layers"] == [-1, 0, 1]
    assert os.path.isfile(drive_csv)

    # Delete local NVMe checkpoint + CSV (simulating brand-new Colab VM!) -> auto-resume from Drive!
    os.unlink(local_ckpt)
    os.unlink(local_csv)
    mgr2 = AsyncCheckpointManager(
        CheckpointConfig(
            local_path=local_ckpt,
            drive_path=drive_ckpt,
            local_convergence_csv=local_csv,
            drive_convergence_csv=drive_csv,
            ckpt_dtype="float32",
            every_n_prompts=1,
            every_minutes=0.0,
        )
    )
    lens_resumed = engine.fit(prompts, include_embedding_layer=True, skip_first=2, checkpoint_manager=mgr2, resume=True)
    mgr2.close()
    assert lens_resumed.n_prompts == 4
    assert lens_resumed.source_layers == [-1, 0, 1]

    # Verify unbroken 4-row convergence CSV across the disconnect (header + 4 rows, no duplicates!)
    rows = Path(drive_csv).read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 5  # header + prompts 0, 1, 2, 3

    # Compare against uninterrupted reference fit
    lens_ref = engine.fit(prompts, include_embedding_layer=True, skip_first=2, resume=False)
    for l in (-1, 0, 1):
        assert torch.allclose(lens_resumed.jacobians[l], lens_ref.jacobians[l], atol=1e-5)

    # Test early stopping (`stop_at_delta`) with identical prompts so delta drops immediately below threshold
    same_prompts = [prompts[0].clone() for _ in range(6)]
    lens_early = engine.fit(same_prompts, skip_first=2, stop_at_delta=1.0, min_prompts=2, stop_window=2, resume=False)
    assert lens_early.n_prompts == 3  # prompt 0 (nan), prompt 1 (d1<1), prompt 2 (d2<1) -> stops at n=3 < 6!

    # Test Neuronpedia bundle export (drop-in .pt weights_only=True + full 4-branch .safetensors + CSV + config.yaml + CREDIT.md)
    bundle_dir = str(tmp_path / "bundle" / "qwen3.8-flash-next" / "jlens" / "Salesforce-wikitext")
    written = export_neuronpedia_bundle(
        lens_resumed,
        bundle_dir,
        author="Test Author",
        max_seq_len=20,
        skip_first=2,
        convergence_csv_path=drive_csv,
        gpu_name="NVIDIA A100-SXM4-80GB",
        gpu_vram_gib=79.2,
        gdn_kernel="fla",
    )
    assert set(written.keys()) == {"neuronpedia_pt", "full_4branch_safetensors", "convergence_csv", "config_yaml", "credit_md"}

    pt_data = torch.load(written["neuronpedia_pt"], map_location="cpu", weights_only=True)
    assert pt_data["n_prompts"] == 4
    assert pt_data["d_model"] == 16
    assert pt_data["source_layers"] == [-1, 0, 1]
    assert pt_data["J"][-1].shape == (16, 16)
    assert pt_data["J"][0].shape == (16, 16)
    assert pt_data["J"][-1].dtype == torch.float16
    assert pt_data["provenance"]["author"] == "Test Author"
    assert pt_data["provenance"]["include_embedding_layer"] is True
    assert pt_data["provenance"]["hc_count"] == 4

    reloaded_4b = Qwen38JacobianLens.load(written["full_4branch_safetensors"])
    assert reloaded_4b.source_layers == [-1, 0, 1]
    assert reloaded_4b.jacobians[-1].shape == (16, 64)
    assert torch.allclose(reloaded_4b.collapsed_jacobians()[-1], pt_data["J"][-1].float(), atol=1e-3)

    cfg_text = Path(written["config_yaml"]).read_text(encoding="utf-8")
    assert 'np_model_id: "qwen3.8-flash-next"' in cfg_text
    assert "include_embedding_layer: true" in cfg_text
    assert "prompts_fitted: 4" in cfg_text
    assert 'author: "Test Author"' in cfg_text

    engine.close()
    loaded.ple_providers[0].close()


def test_convergence_logger_truncates_crash_tail_on_resume(tmp_path: Path):
    csv_p = str(tmp_path / "conv.csv")
    lg1 = ConvergenceLogger(csv_p, resume_next_idx=0)
    for i in range(4):
        lg1.append(i + 1, FitStats(i, 128, 112, 10.0, 1.0, 9.0, 1.2, 0.01 / (i + 1)))

    # Simulate resuming from checkpoint at next_idx=2 (prompts 2 & 3 happened after last checkpoint before crash)
    lg2 = ConvergenceLogger(csv_p, resume_next_idx=2)
    rows = Path(csv_p).read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 3  # header + prompt 0 + prompt 1
    assert len(lg2.tail_deltas(10)) == 2


def test_bundle_fallback_resume_tiebreak_and_find_best_lens_path(tmp_path: Path):
    from qwen38_jlens import find_best_lens_path

    local_p = str(tmp_path / "local" / "ckpt.safetensors")
    drive_p = str(tmp_path / "drive" / "ckpt.safetensors")
    bundle_p = str(tmp_path / "bundle" / "release_4branch.safetensors")
    local_csv = str(tmp_path / "local" / "conv.csv")
    drive_csv = str(tmp_path / "drive" / "conv.csv")
    bundle_csv = str(tmp_path / "bundle" / "conv.csv")

    # Case 1: Drive checkpoint lagged at 985 while release bundle reached 1000
    save_jlens_safetensors(
        drive_p, {0: torch.ones(8, 32)}, n_done=985, next_idx=985, skip_first=2, d_model=8, hc_count=4, dtype=torch.float32
    )
    save_jlens_safetensors(
        bundle_p, {0: torch.full((8, 32), 3.0)}, n_done=1000, next_idx=1000, skip_first=2, d_model=8, hc_count=4, dtype=torch.float16
    )
    Path(drive_csv).write_text("prompt_idx,n_done\n0,1\n", encoding="utf-8")
    Path(bundle_csv).write_text("prompt_idx,n_done\n0,1\n999,1000\n", encoding="utf-8")

    mgr = AsyncCheckpointManager(
        CheckpointConfig(
            local_path=local_p,
            drive_path=drive_p,
            bundle_safetensors_path=bundle_p,
            local_convergence_csv=local_csv,
            drive_convergence_csv=drive_csv,
            bundle_convergence_csv=bundle_csv,
        )
    )
    resumed = mgr.load_best_resume(expected_source_layers=[0], expected_skip_first=2)
    assert resumed is not None
    j_sum, n_done, next_idx = resumed
    assert n_done == 1000 and next_idx == 1000
    assert torch.allclose(j_sum[0] / n_done, torch.full((8, 32), 3.0), atol=1e-3)
    assert "999,1000" in Path(local_csv).read_text(encoding="utf-8")
    mgr.close()

    best = find_best_lens_path([None, local_p, drive_p, bundle_p])
    assert best is not None and best[0] == bundle_p and best[1]["next_idx"] == 1000

    # Case 2: Equal next_idx=1000 -> prefer float32 Drive checkpoint over float16 release bundle
    save_jlens_safetensors(
        drive_p, {0: torch.full((8, 32), 4.0)}, n_done=1000, next_idx=1000, skip_first=2, d_model=8, hc_count=4, dtype=torch.float32
    )
    mgr2 = AsyncCheckpointManager(
        CheckpointConfig(local_path=local_p, drive_path=drive_p, bundle_safetensors_path=bundle_p)
    )
    resumed2 = mgr2.load_best_resume(expected_source_layers=[0], expected_skip_first=2)
    assert resumed2 is not None
    j_sum2, n_done2, next_idx2 = resumed2
    assert n_done2 == 1000 and next_idx2 == 1000
    assert torch.allclose(j_sum2[0] / n_done2, torch.full((8, 32), 4.0), atol=1e-6)
    mgr2.close()

    best2 = find_best_lens_path([bundle_p, drive_p])
    assert best2 is not None and best2[0] == drive_p and best2[1]["ckpt_dtype"] == "float32"


def test_drive_copy_readback_verification_detects_mismatch(tmp_path: Path):
    src_p = str(tmp_path / "src.safetensors")
    dst_p = str(tmp_path / "drive" / "dst.safetensors")
    save_jlens_safetensors(
        src_p, {0: torch.ones(8, 32)}, n_done=10, next_idx=10, skip_first=2, d_model=8, hc_count=4, dtype=torch.float32
    )
    mgr = AsyncCheckpointManager(CheckpointConfig(local_path=src_p, drive_path=dst_p, max_sync_retries=1))
    # Matching expected_next_idx succeeds
    mgr._copy_to_drive(src_p, dst_p, expected_next_idx=10)
    assert peek_checkpoint_metadata(dst_p)["next_idx"] == 10
    # Mismatched expected_next_idx raises RuntimeError so retry/error handling triggers
    with pytest.raises(RuntimeError, match="readback verification mismatch"):
        mgr._copy_to_drive(src_p, dst_p, expected_next_idx=15)
    mgr.close()


