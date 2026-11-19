"""Tests for crash-proof AsyncCheckpointManager, safetensors J_mean checkpoints, multi-source resume, and KeyboardInterrupt flush."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from qwen38_jlens.checkpoint import (
    AsyncCheckpointManager,
    CheckpointConfig,
    load_jlens_safetensors,
    peek_checkpoint_metadata,
    save_jlens_safetensors,
)
from qwen38_jlens.engine import Qwen38JLensEngine
from qwen38_jlens.hardware import HardwareProfile, StorageTier, build_offload_plan
from qwen38_jlens.lens import Qwen38JacobianLens
from qwen38_jlens.loader import load_qwen38_streaming
from tests.test_ple_and_loader import _make_synthetic_fp8_checkpoint


def test_safetensors_checkpoint_roundtrip_and_direct_lens_load(tmp_path: Path):
    torch.manual_seed(10)
    j_mean = {0: torch.randn(16, 64), 1: torch.randn(16, 64)}
    ckpt = str(tmp_path / "ckpt.safetensors")

    save_jlens_safetensors(
        ckpt, j_mean, n_done=37, next_idx=40, skip_first=16, d_model=16, hc_count=4, dtype=torch.float16
    )

    meta = peek_checkpoint_metadata(ckpt)
    assert meta is not None
    assert meta["n_done"] == 37
    assert meta["next_idx"] == 40
    assert meta["source_layers"] == [0, 1]
    assert meta["ckpt_dtype"] == "float16"

    # Any in-progress checkpoint is IMMEDIATELY loadable as a usable Qwen38JacobianLens!
    lens = Qwen38JacobianLens.load(ckpt)
    assert lens.n_prompts == 37
    assert lens.d_model == 16 and lens.hc_count == 4
    assert torch.allclose(lens.jacobians[0], j_mean[0], atol=2e-3)

    # Also test Qwen38JacobianLens.save() -> .safetensors with collapsed J
    out_st = str(tmp_path / "final.safetensors")
    lens.save(out_st, dtype=torch.float16, include_collapsed=True)
    reloaded = Qwen38JacobianLens.load(out_st)
    assert torch.allclose(reloaded.jacobians[1], lens.jacobians[1], atol=1e-4)


def test_async_manager_picks_freshest_across_local_and_drive(tmp_path: Path):
    local_p = str(tmp_path / "local" / "ckpt.safetensors")
    drive_p = str(tmp_path / "drive" / "ckpt.safetensors")
    os.makedirs(os.path.dirname(local_p), exist_ok=True)
    os.makedirs(os.path.dirname(drive_p), exist_ok=True)

    # Stale local (next_idx=10) vs fresher Drive mirror (next_idx=25)
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


def test_engine_fit_async_checkpoint_resume_and_keyboard_interrupt(tmp_path: Path):
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

    mgr1 = AsyncCheckpointManager(
        CheckpointConfig(
            local_path=local_ckpt,
            drive_path=drive_ckpt,
            ckpt_dtype="float32",
            every_n_prompts=10,  # Won't fire automatically before prompt 2; KeyboardInterrupt must flush!
            every_minutes=0.0,
        )
    )

    # Simulate user pressing Stop (KeyboardInterrupt) right after prompt 2 finishes!
    def _interrupt_after_two(stats):
        if stats.prompt_idx == 1:
            raise KeyboardInterrupt("Simulated Colab stop")

    with pytest.raises(KeyboardInterrupt):
        engine.fit(prompts, skip_first=2, checkpoint_manager=mgr1, resume=False, on_prompt_done=_interrupt_after_two)
    mgr1.close()

    # Verify BOTH local NVMe AND Drive mirror were safely flushed on KeyboardInterrupt!
    meta_drive = peek_checkpoint_metadata(drive_ckpt)
    assert meta_drive is not None and meta_drive["n_done"] == 2 and meta_drive["next_idx"] == 2

    # Delete local NVMe (simulating brand-new Colab VM!) -> must auto-resume from Drive mirror!
    os.unlink(local_ckpt)
    mgr2 = AsyncCheckpointManager(
        CheckpointConfig(
            local_path=local_ckpt,
            drive_path=drive_ckpt,
            ckpt_dtype="float32",
            every_n_prompts=1,
            every_minutes=0.0,
        )
    )
    lens_resumed = engine.fit(prompts, skip_first=2, checkpoint_manager=mgr2, resume=True)
    mgr2.close()
    assert lens_resumed.n_prompts == 4

    # Compare against uninterrupted reference fit
    lens_ref = engine.fit(prompts, skip_first=2, resume=False)
    for l in (0, 1):
        assert torch.allclose(lens_resumed.jacobians[l], lens_ref.jacobians[l], atol=1e-5)

    engine.close()
    loaded.ple_providers[0].close()
