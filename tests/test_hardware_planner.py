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
