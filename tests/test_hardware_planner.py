"""Tests for hardware auto-detection and concrete 3-tier interleaved offloading plans."""

from qwen38_jlens.hardware import HardwareProfile, StorageTier, build_offload_plan


def test_colab_g4_blackwell_plan():
    g4 = HardwareProfile(
        gpu_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        compute_capability=(12, 0),
        vram_total_gib=95.6,
        vram_free_gib=94.5,
        ram_total_gib=176.9,
        ram_avail_gib=170.0,
        disk_free_gib=195.0,
        cpu_count=24,
        has_cuda=True,
    )
    plan = build_offload_plan(g4, max_seq_len=128)
    counts = plan.tier_counts()
    print("\n" + plan.summary())

    assert plan.gdn_backend == "flash_qla"
    assert plan.moe_backend == "grouped_mm_bf16"
    assert plan.attn_backend == "l2_cached_exact"
    assert plan.k_pass == 2560  # Single weight sweep per prompt!
    assert plan.ple_mode == "mmap"
    assert counts[StorageTier.GPU_HBM] >= 24
    assert counts[StorageTier.NVME_SLAB] == 0
    assert counts[StorageTier.HOST_PAGEABLE] == 0
    assert sum(counts.values()) == 48
    assert plan.est_vram_peak_gib < g4.vram_free_gib
    assert plan.est_ram_peak_gib < g4.ram_avail_gib
    assert plan.est_disk_peak_gib < g4.disk_free_gib


def test_colab_a100_80g_plan():
    a100_80 = HardwareProfile(
        gpu_name="NVIDIA A100-SXM4-80GB",
        compute_capability=(8, 0),
        vram_total_gib=79.2,
        vram_free_gib=78.5,
        ram_total_gib=167.1,
        ram_avail_gib=160.0,
        disk_free_gib=195.0,
        cpu_count=12,
        has_cuda=True,
    )
    plan = build_offload_plan(a100_80, max_seq_len=128)
    counts = plan.tier_counts()
    print("\n" + plan.summary())

    assert plan.gdn_backend == "fla"  # SM80 cannot run FlashQLA
    assert plan.moe_backend == "grouped_mm_bf16"
    assert plan.k_pass == 2560
    assert plan.ple_mode == "mmap"
    assert counts[StorageTier.GPU_HBM] >= 18
    assert counts[StorageTier.NVME_SLAB] == 0
    assert sum(counts.values()) == 48
    assert plan.est_vram_peak_gib < a100_80.vram_free_gib
    assert plan.est_ram_peak_gib < a100_80.ram_avail_gib
    assert plan.est_disk_peak_gib < a100_80.disk_free_gib


def test_colab_a100_40g_plan_and_nvme_interleaving():
    a100_40 = HardwareProfile(
        gpu_name="NVIDIA A100-SXM4-40GB",
        compute_capability=(8, 0),
        vram_total_gib=39.6,
        vram_free_gib=39.0,
        ram_total_gib=83.5,
        ram_avail_gib=79.0,
        disk_free_gib=195.0,
        cpu_count=12,
        has_cuda=True,
    )
    plan = build_offload_plan(a100_40, max_seq_len=128)
    counts = plan.tier_counts()
    print("\n" + plan.summary())

    assert plan.gdn_backend == "fla"
    assert plan.k_pass == 2560  # Single weight sweep even on A100 40G!
    assert plan.expert_dequant_chunk == 256  # Chunked BF16 dequant caps peak scratch!
    assert counts[StorageTier.GPU_HBM] >= 3
    assert counts[StorageTier.NVME_SLAB] > 0  # Safely spills overflow to interleaved NVMe slabs (zero RAM OOM!)
    assert sum(counts.values()) == 48
    assert plan.est_vram_peak_gib < a100_40.vram_free_gib
    assert plan.est_ram_peak_gib < a100_40.ram_avail_gib
    assert plan.est_disk_peak_gib < a100_40.disk_free_gib

    # Verify Tier 2 NVMe layers are interleaved (never all bunched consecutively at one end!)
    t2_layers = sorted(l for l, t in plan.layer_tiers.items() if t == StorageTier.NVME_SLAB)
    assert t2_layers[0] <= 4
    assert t2_layers[-1] >= 43
    max_run = 1
    cur_run = 1
    for a, b in zip(t2_layers, t2_layers[1:]):
        if b == a + 1:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    assert max_run <= 2, f"Tier 2 layers should be interleaved, got run={max_run}: {t2_layers}"


def test_tight_disk_switches_to_ple_preextract():
    tight = HardwareProfile(
        gpu_name="NVIDIA A100-SXM4-40GB",
        compute_capability=(8, 0),
        vram_total_gib=39.6,
        vram_free_gib=39.0,
        ram_total_gib=83.5,
        ram_avail_gib=79.0,
        disk_free_gib=100.0,
        cpu_count=12,
        has_cuda=True,
    )
    plan = build_offload_plan(tight, max_seq_len=128)
    assert plan.ple_mode == "preextract"
    assert plan.evict_raw_shards is True
    assert plan.est_disk_peak_gib < tight.disk_free_gib


def _a100_80g_profile(vram_free_gib: float = 78.8) -> HardwareProfile:
    return HardwareProfile(
        gpu_name="NVIDIA A100-SXM4-80GB",
        compute_capability=(8, 0),
        vram_total_gib=79.3,
        vram_free_gib=vram_free_gib,
        ram_total_gib=167.1,
        ram_avail_gib=164.1,
        disk_free_gib=235.5,
        cpu_count=12,
        has_cuda=True,
    )


def test_a100_80g_stage_a_peak_leaves_headroom():
    """Replays the live Colab A100-80G OOM budget: dense + Tier-0 + staging + G + Stage-A scratch must fit."""
    plan = build_offload_plan(_a100_80g_profile(), max_seq_len=128)
    layer_gib = plan.bytes_per_moe_layer / (1024**3)
    n_t0 = plan.tier_counts()[StorageTier.GPU_HBM]
    assert plan.expert_dequant_chunk == 256  # BF16 dequant buffer capped at 2.35 GiB on 80G
    # Real Stage-A peak at T_v=112 (k=2560, D=2560, S=1120, K_blk=256, inter=640):
    g = 2560 * 112 * 10240 * 2 / 1024**3
    d_mlp = 2 * 2560 * 112 * 2560 * 2 / 1024**3
    w_bf16 = 256 * 3 * 640 * 2560 * 2 / 1024**3
    moe_chunk = 1120 * 256 * (2560 + 2 * 640) * 2 / 1024**3
    peak = plan.dense_backbone_gib + n_t0 * layer_gib + plan.gpu_staging_slots * layer_gib + g + d_mlp + w_bf16 + moe_chunk
    assert peak <= 79.3 - 5.0, f"Stage-A peak {peak:.1f} GiB leaves < 5 GiB headroom"
    assert plan.runtime_vram_reserve_gib >= g + d_mlp + w_bf16 + moe_chunk


def test_vram_self_heal_selects_spread_tier0_layers():
    from qwen38_jlens.engine import select_tier0_layers_to_demote

    plan = build_offload_plan(_a100_80g_profile(), max_seq_len=128)
    layer_gib = plan.bytes_per_moe_layer / (1024**3)
    need0 = plan.runtime_vram_reserve_gib + plan.gpu_staging_slots * layer_gib
    # Enough headroom -> nothing demoted
    assert select_tier0_layers_to_demote(plan, free_gib=need0 + 0.1, reserve_gib=plan.runtime_vram_reserve_gib)[0] == []
    # 5.5 GiB short (the live-run discrepancy) -> 3 evenly spread Tier-0 layers demoted
    demoted, need = select_tier0_layers_to_demote(plan, free_gib=need0 - 5.5, reserve_gib=plan.runtime_vram_reserve_gib)
    assert len(demoted) == 3
    assert all(plan.layer_tiers[l] == StorageTier.GPU_HBM for l in demoted)
    assert demoted[0] < 16 and demoted[-1] > 32  # spread across the stack, not bunched
    assert need == need0
