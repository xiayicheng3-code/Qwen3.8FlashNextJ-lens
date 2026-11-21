"""Hardware auto-detection and concrete 3-tier offloading planner for Qwen3.8-Flash-Next J-lens.

Targets randomly assigned Google Colab High-RAM runtimes:
  - Colab G4 High RAM: NVIDIA RTX PRO 6000 Blackwell (SM120, ~95.6 GiB VRAM, ~177 GiB RAM)
  - Colab A100 80G High RAM: NVIDIA A100-SXM4-80GB (SM80, ~79.2 GiB VRAM, ~167 GiB RAM)
  - Colab A100 40G High RAM: NVIDIA A100-SXM4-40GB (SM80, ~39.6 GiB VRAM, ~83.5 GiB RAM)
  - Also adapts cleanly to H100 80G (SM90), B200 (SM100), L4/V100, or synthetic CPU/GPU models.

Core memory insights for Qwen/Qwen3.8-Flash-Next-FP8 (48 layers, 512 experts, hc_count=4):
  - Dense BF16 backbone (excluding vision, MTP, lm_head): ~8.2 GiB -> ALWAYS Tier 0 (GPU HBM).
  - Routed MoE experts (FP8 E4M3 + 128x128 fp32 scales): 2.3443 GiB/layer -> 112.53 GiB across 48 layers.
  - 51B PLE n-gram embedding table (FP8 E4M3 + scalar scale, layer 2 only): 47.68 GiB -> NEVER in RAM/VRAM;
    served via MADV_RANDOM NVMe mmap (0 GiB RAM) or pre-extracted into a ~64 MB cache if disk is tight.
  - Tier placement interleaves slower tiers (Tier 2 NVMe, Tier 1 Host RAM) evenly between faster tiers
    so background NVMe reads and async H2D DMA transfers overlap 100% behind backward compute.
"""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import torch


class StorageTier(str, Enum):
    """Where a decoder layer's routed MoE FP8 expert weights reside."""

    GPU_HBM = "tier0_gpu_hbm"
    HOST_PINNED = "tier1_host_pinned"
    HOST_PAGEABLE = "tier1_host_pageable"
    NVME_SLAB = "tier2_nvme_slab"


@dataclass(frozen=True)
class HardwareProfile:
    """Detected hardware capabilities of the current runtime."""

    gpu_name: str
    compute_capability: tuple[int, int]
    vram_total_gib: float
    vram_free_gib: float
    ram_total_gib: float
    ram_avail_gib: float
    disk_free_gib: float
    cpu_count: int
    has_cuda: bool = True

    @property
    def arch_label(self) -> str:
        major, minor = self.compute_capability
        if major == 12:
            return f"Blackwell-Workstation (SM{major}{minor})"
        if major == 10:
            return f"Blackwell-Datacenter (SM{major}{minor})"
        if major == 9:
            return f"Hopper (SM{major}{minor})"
        if major == 8:
            return f"Ampere (SM{major}{minor})"
        return f"SM{major}{minor}"

    @property
    def supports_flash_qla(self) -> bool:
        """FlashQLA v0.1.3 supports SM90, SM100, SM103, SM120, SM121."""
        return self.has_cuda and self.compute_capability in (
            (9, 0),
            (10, 0),
            (10, 3),
            (12, 0),
            (12, 1),
        )

    @property
    def supports_native_fp8_tc(self) -> bool:
        """Hardware FP8 Tensor Cores exist on SM89+ (Ada/Hopper/Blackwell)."""
        return self.has_cuda and self.compute_capability >= (8, 9)

    @property
    def supports_grouped_mm(self) -> bool:
        """PyTorch CUTLASS/cuBLAS grouped_mm supports SM80+ on torch>=2.9."""
        return self.has_cuda and self.compute_capability >= (8, 0)


@dataclass
class OffloadPlan:
    """Concrete, deterministic offloading & kernel execution plan."""

    profile: HardwareProfile
    num_layers: int
    d_model: int
    hc_count: int
    max_seq_len: int
    bytes_per_moe_layer: int
    dense_backbone_gib: float

    # Tier assignment per layer index (0 .. num_layers - 1)
    layer_tiers: dict[int, StorageTier]

    # J-lens sweep & micro-chunk tuning parameters
    k_pass: int                  # Cotangents held in GPU G_curr per layer sweep (2560 => 1 single sweep!)
    attn_k_chunk: int            # Cotangents processed per GDN/QSA kernel call
    moe_k_chunk: int             # Cotangents processed per MoE grouped_mm VJP call (multiple of 16!)
    expert_dequant_chunk: int    # Max active experts dequantized simultaneously into BF16 (256 or 512)
    gpu_staging_slots: int       # Number of double-buffered FP8 GPU staging slots (0 or 2)
    host_pinned_staging_slots: int  # Pinned host staging slots for pageable/NVMe layers (0 or 2)

    # PLE & Disk strategy
    ple_mode: str                # "mmap" (keep 48 GiB PLE shards on disk) or "preextract" (~64 MB cache)
    evict_raw_shards: bool       # Delete raw HF safetensors shards after unpacking each layer group
    download_batch_gib: float    # Max raw shard download batch before packing + evicting

    # Preferred kernels
    gdn_backend: str             # "flash_qla", "fla", or "torch"
    attn_backend: str            # "l2_cached_exact" (T<=512) or "sdpa_flash"
    moe_backend: str             # "grouped_mm_bf16" (1x Triton FP8 dequant) or "loop_mm_bf16"
    use_triton_fp8_dequant: bool
    use_triton_swiglu_vjp: bool

    # Estimated memory usage summary (GiB)
    est_vram_peak_gib: float = 0.0
    est_ram_peak_gib: float = 0.0
    est_disk_peak_gib: float = 0.0
    notes: list[str] = field(default_factory=list)

    def tier_counts(self) -> dict[StorageTier, int]:
        counts = {t: 0 for t in StorageTier}
        for t in self.layer_tiers.values():
            counts[t] += 1
        return counts

    def summary(self) -> str:
        c = self.tier_counts()
        moe_gb = self.bytes_per_moe_layer / (1024**3)
        n_sweeps = math.ceil(self.d_model / self.k_pass)
        lines = [
            "========================================================================",
            f" Hardware Detected : {self.profile.gpu_name} [{self.profile.arch_label}]",
            f"   VRAM            : {self.profile.vram_free_gib:.1f} / {self.profile.vram_total_gib:.1f} GiB free",
            f"   System RAM      : {self.profile.ram_avail_gib:.1f} / {self.profile.ram_total_gib:.1f} GiB avail",
            f"   Disk Free       : {self.profile.disk_free_gib:.1f} GiB   |   CPUs: {self.profile.cpu_count}",
            "------------------------------------------------------------------------",
            " Concrete 3-Tier Offloading Plan (Qwen3.8-Flash-Next-FP8)",
            f"   Dense Backbone  : {self.dense_backbone_gib:.2f} GiB BF16 -> Tier 0 GPU HBM (permanent)",
            f"   Tier 0 (GPU HBM): {c[StorageTier.GPU_HBM]:2d}/{self.num_layers} MoE layers ({c[StorageTier.GPU_HBM]*moe_gb:5.1f} GiB FP8, 0 ms transfer)",
            f"   Tier 1 (Pinned) : {c[StorageTier.HOST_PINNED]:2d}/{self.num_layers} MoE layers ({c[StorageTier.HOST_PINNED]*moe_gb:5.1f} GiB FP8, async H2D DMA)",
            f"   Tier 1 (Pageabl): {c[StorageTier.HOST_PAGEABLE]:2d}/{self.num_layers} MoE layers ({c[StorageTier.HOST_PAGEABLE]*moe_gb:5.1f} GiB FP8, threaded pin->DMA)",
            f"   Tier 2 (NVMe)   : {c[StorageTier.NVME_SLAB]:2d}/{self.num_layers} MoE layers ({c[StorageTier.NVME_SLAB]*moe_gb:5.1f} GiB FP8, interleaved pread)",
            f"   51B PLE Table   : mode={self.ple_mode!r} (0 GiB VRAM, 0 GiB pinned RAM)",
            f"   Shard Eviction  : {self.evict_raw_shards} (download batches <= {self.download_batch_gib:.0f} GiB)",
            "------------------------------------------------------------------------",
            " Kernel & VJP Schedule",
            f"   GDN Kernel      : {self.gdn_backend}",
            f"   QSA Attention   : {self.attn_backend} (QSA indexer bypassed for T <= 2048)",
            f"   MoE VJP Kernel  : {self.moe_backend} | Triton FP8 dequant={self.use_triton_fp8_dequant} | Triton SwiGLU={self.use_triton_swiglu_vjp}",
            f"   Weight Sweeps   : {n_sweeps}x/prompt (k_pass={self.k_pass}, attn_k_chunk={self.attn_k_chunk}, moe_k_chunk={self.moe_k_chunk}, expert_dequant_chunk={self.expert_dequant_chunk})",
            f"   Est. Peak Usage : VRAM ~{self.est_vram_peak_gib:.1f} GiB | RAM ~{self.est_ram_peak_gib:.1f} GiB | Disk ~{self.est_disk_peak_gib:.1f} GiB",
            "========================================================================",
        ]
        if self.notes:
            lines.append(" Planner Notes:")
            for note in self.notes:
                lines.append(f"   - {note}")
        return "\n".join(lines)


def detect_hardware(disk_path: str = ".") -> HardwareProfile:
    """Inspect live GPU, Host RAM, Disk, and CPU resources."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_total_gib = vm.total / (1024**3)
        ram_avail_gib = vm.available / (1024**3)
    except ImportError:
        # Fallback to /proc/meminfo on Linux
        ram_total_gib, ram_avail_gib = 64.0, 48.0
        if os.path.exists("/proc/meminfo"):
            info = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        info[parts[0].rstrip(":")] = int(parts[1])
            ram_total_gib = info.get("MemTotal", 64 * 1024 * 1024) / (1024**2)
            ram_avail_gib = info.get("MemAvailable", 48 * 1024 * 1024) / (1024**2)

    check_path = disk_path if os.path.exists(disk_path) else "."
    disk_free_gib = shutil.disk_usage(check_path).free / (1024**3)
    cpu_count = os.cpu_count() or 4

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free_b, total_b = torch.cuda.mem_get_info(0)
        return HardwareProfile(
            gpu_name=props.name,
            compute_capability=(props.major, props.minor),
            vram_total_gib=total_b / (1024**3),
            vram_free_gib=free_b / (1024**3),
            ram_total_gib=ram_total_gib,
            ram_avail_gib=ram_avail_gib,
            disk_free_gib=disk_free_gib,
            cpu_count=cpu_count,
            has_cuda=True,
        )
    return HardwareProfile(
        gpu_name="CPU-only",
        compute_capability=(0, 0),
        vram_total_gib=0.0,
        vram_free_gib=0.0,
        ram_total_gib=ram_total_gib,
        ram_avail_gib=ram_avail_gib,
        disk_free_gib=disk_free_gib,
        cpu_count=cpu_count,
        has_cuda=False,
    )


def _evenly_spaced_indices(n_total: int, count: int, excluded: set[int] | None = None) -> list[int]:
    """Select `count` maximally spread-out indices from `[0, n_total) \\ excluded`.

    Used to interleave Tier 2 NVMe layers (and Tier 1 Host layers) evenly among
    faster tiers so background NVMe reads and async H2D DMAs hide 100% behind
    backward compute of adjacent layers.
    """
    available = [i for i in range(n_total) if excluded is None or i not in excluded]
    if count <= 0:
        return []
    if count >= len(available):
        return list(available)
    m = len(available)
    # Pick evenly spaced positions in [0, m - 1]
    chosen_pos: list[int] = []
    used: set[int] = set()
    for k in range(count):
        ideal = (k + 0.5) * m / count - 0.5
        pos = min(m - 1, max(0, int(round(ideal))))
        # Resolve rare rounding collision to nearest unused slot
        if pos in used:
            for delta in range(1, m):
                if pos + delta < m and (pos + delta) not in used:
                    pos = pos + delta
                    break
                if pos - delta >= 0 and (pos - delta) not in used:
                    pos = pos - delta
                    break
        used.add(pos)
        chosen_pos.append(pos)
    chosen_pos.sort()
    return [available[p] for p in chosen_pos]


def compute_moe_layer_bytes(
    num_experts: int = 512,
    hidden_size: int = 2560,
    moe_intermediate_size: int = 640,
    block_size: int = 128,
    weight_bytes: int = 1,
) -> int:
    """Exact byte footprint of one layer's routed MoE experts (weights + 128x128 fp32 scales)."""
    gate_up_w = num_experts * (2 * moe_intermediate_size) * hidden_size * weight_bytes
    down_w = num_experts * hidden_size * moe_intermediate_size * weight_bytes
    if weight_bytes == 1:
        gate_up_s = (
            num_experts
            * math.ceil((2 * moe_intermediate_size) / block_size)
            * math.ceil(hidden_size / block_size)
            * 4
        )
        down_s = (
            num_experts
            * math.ceil(hidden_size / block_size)
            * math.ceil(moe_intermediate_size / block_size)
            * 4
        )
    else:
        gate_up_s = 0
        down_s = 0
    return gate_up_w + down_w + gate_up_s + down_s


def build_offload_plan(
    profile: HardwareProfile,
    *,
    num_layers: int = 48,
    d_model: int = 2560,
    hc_count: int = 4,
    num_experts: int = 512,
    moe_intermediate_size: int = 640,
    max_seq_len: int = 128,
    num_source_layers: int | None = None,
    dense_backbone_gib: float = 8.2,
    ple_table_gib: float = 47.68,
    force_ple_mode: str | None = None,
    weight_bytes: int = 1,
) -> OffloadPlan:
    """Compute the optimal 3-tier placement and kernel configuration for `profile`."""
    notes: list[str] = []
    d_hc = d_model * hc_count
    n_src = num_layers if num_source_layers is None else num_source_layers
    bytes_per_layer = compute_moe_layer_bytes(
        num_experts=num_experts,
        hidden_size=d_model,
        moe_intermediate_size=moe_intermediate_size,
        weight_bytes=weight_bytes,
    )
    layer_gib = bytes_per_layer / (1024**3)
    vram_usable = min(profile.vram_free_gib, profile.vram_total_gib) if profile.has_cuda else 0.0

    # 1. Choose k_pass (cotangents held in GPU G_curr), attn_k_chunk, moe_k_chunk, expert_dequant_chunk
    #    For standard max_seq_len=128: full K=2560 G_curr is 6.25 GiB -> fits in 1 single sweep on A100-40G/80G/G4!
    full_g_curr_gib = (d_model * max_seq_len * d_hc * 2) / (1024**3)
    max_g_curr_gib = 6.5 if vram_usable < 60.0 else 13.0
    if full_g_curr_gib <= max_g_curr_gib or not profile.has_cuda:
        k_pass = d_model
    else:
        # Pick largest divisor of d_model (or multiple of 128) fitting max_g_curr_gib
        raw_k = max(128, int((max_g_curr_gib * (1024**3)) / (max_seq_len * d_hc * 2)))
        k_pass = max(128, (raw_k // 128) * 128)
        notes.append(f"max_seq_len={max_seq_len} exceeds 1-sweep G_curr budget; using k_pass={k_pass} ({math.ceil(d_model / k_pass)} sweeps/prompt).")

    g_curr_gib = (k_pass * max_seq_len * d_hc * 2) / (1024**3)

    if vram_usable >= 70.0:
        attn_k_chunk = 128
        moe_k_chunk = 256
        expert_dequant_chunk = num_experts  # Dequantize all active experts in 1 shot (~3.8-4.7 GiB max)
        vram_safety_gib = 6.0
    elif vram_usable >= 34.0:
        # Colab A100 40G profile
        attn_k_chunk = 64
        moe_k_chunk = 128
        expert_dequant_chunk = min(num_experts, 256)  # Chunked BF16 dequant caps peak scratch at 2.35 GiB
        vram_safety_gib = 4.0
    else:
        attn_k_chunk = 32
        moe_k_chunk = 64
        expert_dequant_chunk = min(num_experts, 128)
        vram_safety_gib = 2.0

    # Ensure chunks divide/bound cleanly by k_pass
    attn_k_chunk = min(attn_k_chunk, k_pass)
    moe_k_chunk = min(moe_k_chunk, k_pass)

    # Active BF16 expert dequant peak (2 bytes/param * expert_dequant_chunk / num_experts)
    dequant_bf16_gib = (
        (expert_dequant_chunk / max(1, num_experts))
        * (num_experts * 3 * moe_intermediate_size * d_model * 2)
        / (1024**3)
    )
    # Scratch for GDN/QSA + MoE VJP intermediates (d_y_mlp_TK + d_x_mlp_TK + grouped_mm chunk temps)
    s_pairs = max_seq_len * 10
    d_mlp_tk_gib = (2 * k_pass * max_seq_len * d_model * 2) / (1024**3)
    moe_chunk_gib = (s_pairs * moe_k_chunk * (d_model + 2 * moe_intermediate_size) * 2) / (1024**3)
    moe_scratch_gib = d_mlp_tk_gib + moe_chunk_gib
    attn_scratch_gib = max(0.8, (attn_k_chunk * max_seq_len * 48 * 128 * 10 * 2) / (1024**3))
    compute_scratch_gib = max(moe_scratch_gib, attn_scratch_gib)

    # Check if ALL layers fit directly in GPU HBM (no staging slots needed)
    all_gpu_vram = (
        dense_backbone_gib
        + g_curr_gib
        + dequant_bf16_gib
        + compute_scratch_gib
        + vram_safety_gib
        + num_layers * layer_gib
    )
    if profile.has_cuda and all_gpu_vram <= vram_usable:
        n_tier0 = num_layers
        gpu_staging_slots = 0
    elif profile.has_cuda:
        gpu_staging_slots = 2  # Double-buffered async H2D prefetch slots
        fixed_vram = (
            dense_backbone_gib
            + g_curr_gib
            + gpu_staging_slots * layer_gib
            + dequant_bf16_gib
            + compute_scratch_gib
            + vram_safety_gib
        )
        avail_for_tier0 = max(0.0, vram_usable - fixed_vram)
        n_tier0 = min(num_layers, int(avail_for_tier0 // layer_gib))
    else:
        n_tier0 = 0
        gpu_staging_slots = 0

    remaining_after_gpu = num_layers - n_tier0

    # 2. Host RAM budget (Tier 1 Pinned + Tier 1 Pageable) vs Tier 2 NVMe Slabs
    # Fixed Host RAM overhead:
    #   - Running fp32 Jacobian sums: n_src * d_model * d_hc * 4B (~4.69 GiB for all 48 layers)
    #   - lm_head on CPU (~1.18 GiB)
    #   - Saved B=1 layer inputs H_l across 48 layers (~0.12 GiB)
    #   - Python / PyTorch / OS page cache + streaming load temp buffer safety margin
    jac_sum_gib = (n_src * d_model * d_hc * 4) / (1024**3)
    lm_head_cpu_gib = 1.2 if d_model >= 2048 else 0.1
    ram_safety_gib = 14.0 if profile.ram_total_gib >= 120.0 else 13.5

    # Cap total pinned memory to <= 45% of total RAM (prevents Linux mlock/OOM-killer starvation)
    max_pinned_gib = min(
        0.45 * profile.ram_total_gib,
        max(0.0, profile.ram_avail_gib - ram_safety_gib - jac_sum_gib - lm_head_cpu_gib),
    )

    if remaining_after_gpu == 0:
        n_tier1_pinned = 0
        n_tier1_pageable = 0
        n_tier2_nvme = 0
        host_pinned_staging_slots = 0
    else:
        # First check if all remaining layers fit in directly pinned host RAM
        if remaining_after_gpu * layer_gib <= max_pinned_gib:
            n_tier1_pinned = remaining_after_gpu
            n_tier1_pageable = 0
            n_tier2_nvme = 0
            host_pinned_staging_slots = 0
        else:
            # Reserve 2 pinned host staging slots for pageable RAM / NVMe layers
            host_pinned_staging_slots = 2 if profile.has_cuda else 0
            staging_ram_gib = host_pinned_staging_slots * layer_gib
            avail_ram_for_layers = max(
                0.0,
                profile.ram_avail_gib - ram_safety_gib - jac_sum_gib - lm_head_cpu_gib - staging_ram_gib,
            )
            max_host_layers = min(remaining_after_gpu, int(avail_ram_for_layers // layer_gib))
            avail_pinned_for_layers = max(0.0, max_pinned_gib - staging_ram_gib)
            n_tier1_pinned = min(max_host_layers, int(avail_pinned_for_layers // layer_gib))
            n_tier1_pageable = max(0, max_host_layers - n_tier1_pinned)
            n_tier2_nvme = remaining_after_gpu - max_host_layers

    # 3. Interleaved layer-to-tier assignment!
    #    - Space Tier 2 NVMe layers evenly across [0..num_layers-1] so each NVMe read (~0.8s)
    #      hides behind 2-3 consecutive Tier 0/1 layers (~1.0-1.3s compute on A100).
    #    - Space Tier 0 GPU layers evenly among remaining layers so on G4/A100-80G every
    #      Tier 1 H2D DMA (~90 ms) hides behind Tier 0 + Tier 1 compute!
    layer_tiers: dict[int, StorageTier] = {}
    t2_indices = set(_evenly_spaced_indices(num_layers, n_tier2_nvme))
    for l in t2_indices:
        layer_tiers[l] = StorageTier.NVME_SLAB

    t0_indices = set(_evenly_spaced_indices(num_layers, n_tier0, excluded=t2_indices))
    for l in t0_indices:
        layer_tiers[l] = StorageTier.GPU_HBM

    assigned = t2_indices | t0_indices
    t1_pageable_indices = set(_evenly_spaced_indices(num_layers, n_tier1_pageable, excluded=assigned))
    for l in t1_pageable_indices:
        layer_tiers[l] = StorageTier.HOST_PAGEABLE

    for l in range(num_layers):
        if l not in layer_tiers:
            layer_tiers[l] = StorageTier.HOST_PINNED

    # 4. Disk & PLE strategy
    tier2_disk_gib = n_tier2_nvme * layer_gib
    download_batch_gib = 15.0 if profile.disk_free_gib >= 80.0 else 8.0
    if force_ple_mode is not None:
        ple_mode = force_ple_mode
    else:
        # Need room for: Tier 2 slabs + PLE shards (47.7 GiB) + download batch + 20 GiB OS/checkpoint buffer
        needed_for_mmap = tier2_disk_gib + ple_table_gib + download_batch_gib + 20.0
        if profile.disk_free_gib >= needed_for_mmap:
            ple_mode = "mmap"
        else:
            ple_mode = "preextract"
            notes.append(
                f"Disk free ({profile.disk_free_gib:.1f} GiB) < mmap threshold ({needed_for_mmap:.1f} GiB); "
                "auto-selecting two-stage PLE pre-extraction + shard cleanup!"
            )

    # Always evict raw MoE safetensors shards after unpacking if keeping full 164 GiB raw repo + slabs risks disk full
    evict_raw_shards = profile.disk_free_gib < (175.0 + tier2_disk_gib + 25.0)
    if evict_raw_shards:
        notes.append("Streaming shard-by-shard pack-and-evict enabled (keeps peak disk well below Colab 235 GiB limit).")

    # 5. Best kernel selection per GPU architecture
    if profile.supports_flash_qla:
        gdn_backend = "flash_qla"
    elif profile.has_cuda and profile.compute_capability >= (8, 0):
        gdn_backend = "fla"
    else:
        gdn_backend = "torch"

    attn_backend = "l2_cached_exact" if max_seq_len <= 512 else "sdpa_flash"
    moe_backend = "grouped_mm_bf16" if profile.supports_grouped_mm else "loop_mm_bf16"
    use_triton_fp8_dequant = profile.has_cuda and weight_bytes == 1
    use_triton_swiglu_vjp = profile.has_cuda

    est_vram_peak = (
        dense_backbone_gib
        + n_tier0 * layer_gib
        + gpu_staging_slots * layer_gib
        + g_curr_gib
        + dequant_bf16_gib
        + compute_scratch_gib
    ) if profile.has_cuda else 0.0

    est_ram_peak = (
        (n_tier1_pinned + n_tier1_pageable + host_pinned_staging_slots) * layer_gib
        + jac_sum_gib
        + lm_head_cpu_gib
        + 4.0  # Python/PyTorch base overhead
    )

    ple_disk = ple_table_gib if ple_mode == "mmap" else 0.5
    raw_disk = download_batch_gib if evict_raw_shards else (num_layers * layer_gib + dense_backbone_gib)
    est_disk_peak = tier2_disk_gib + ple_disk + raw_disk + 2.5  # +2.5 GiB saved lens/ckpt

    if n_tier2_nvme > 0:
        spacing = num_layers / n_tier2_nvme
        notes.append(
            f"Tier 2 NVMe layers ({n_tier2_nvme}) interleaved every ~{spacing:.1f} layers "
            "for zero-stall async pread overlap."
        )

    return OffloadPlan(
        profile=profile,
        num_layers=num_layers,
        d_model=d_model,
        hc_count=hc_count,
        max_seq_len=max_seq_len,
        bytes_per_moe_layer=bytes_per_layer,
        dense_backbone_gib=dense_backbone_gib,
        layer_tiers=layer_tiers,
        k_pass=k_pass,
        attn_k_chunk=attn_k_chunk,
        moe_k_chunk=moe_k_chunk,
        expert_dequant_chunk=expert_dequant_chunk,
        gpu_staging_slots=gpu_staging_slots,
        host_pinned_staging_slots=host_pinned_staging_slots,
        ple_mode=ple_mode,
        evict_raw_shards=evict_raw_shards,
        download_batch_gib=download_batch_gib,
        gdn_backend=gdn_backend,
        attn_backend=attn_backend,
        moe_backend=moe_backend,
        use_triton_fp8_dequant=use_triton_fp8_dequant,
        use_triton_swiglu_vjp=use_triton_swiglu_vjp,
        est_vram_peak_gib=est_vram_peak,
        est_ram_peak_gib=est_ram_peak,
        est_disk_peak_gib=est_disk_peak,
        notes=notes,
    )
