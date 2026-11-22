"""Async double-buffered 3-tier weight streamer and single-sweep layer-by-layer J-lens engine.

Architecture of one prompt pass (`jacobian_for_prompt_fast`):
  1. **Forward sweep (layers 0 -> L-1, B = 1)**:
     - Streams offloaded MoE layers asynchronously via `AsyncMoEStreamer`.
     - Saves ONLY each layer's input residual `H_{l-1}` (`[1, T, 10240]` = 2.5 MB/layer -> 120 MB total!).
     - Computes `h_final = hyper_connection_mixer(H_{L-1})`.
  2. **Backward sweep (layers L-1 -> min_source + 1, ALL K = k_pass cotangents at once!)**:
     - Initializes ONE in-place GPU cotangent buffer `G` (`[k_pass, T, 10240]`, 6.25 GiB for full K=2560!).
     - At each layer `l` (going downward):
       a) Async prefetch kicks off layer `l - 1` (H2D DMA) AND the next Tier-2 NVMe slab (background thread).
       b) Recomputes ONE tiny `B = 1` forward of layer `l` (~0.5 ms), dequantizing ONLY the hit active
          experts ONCE into BF16.
       c) Executes exact analytical VJP for ALL `k_pass` cotangents (`decoder_layer_vjp_inplace`) updating
          `G` in-place while layer `l` sits hot in GPU HBM!
       d) Immediately reduces `J_{l-1} = G[:, valid_positions, :].float().mean(dim=1)` onto CPU!
     - **Result**: Each offloaded layer's FP8 weights cross PCIe/NVMe AT MOST ONCE per sweep
       (vs 320x in Anthropic's reference `dim_batch=8` loop -> 40x-120x faster!).
"""

from __future__ import annotations

import logging
import math
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from qwen38_jlens.hardware import StorageTier
from qwen38_jlens.kernels import GatedDeltaRuleBackend
from qwen38_jlens.lens import Qwen38JacobianLens
from qwen38_jlens.loader import (
    LoadedQwen38Model,
    allocate_moe_payload,
    read_moe_slab_into,
)
from qwen38_jlens.model_ops import (
    MoeExpertPayload,
    build_rope_cos_sin,
    decoder_layer_attn_fwd,
    decoder_layer_fwd,
    decoder_layer_vjp_inplace,
    gated_residual_fwd,
    gated_residual_vjp,
    moe_block_fwd,
    slice_gated_residual_saved,
)

logger = logging.getLogger(__name__)

SKIP_FIRST_N_POSITIONS = 16


def valid_position_mask(seq_len: int, *, skip_first: int = SKIP_FIRST_N_POSITIONS) -> torch.Tensor:
    """Boolean mask `[seq_len]` matching `anthropics/jacobian-lens` (`[skip_first : seq_len - 1]`)."""
    if skip_first < 0:
        raise ValueError(f"skip_first must be >= 0, got {skip_first}")
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[skip_first : seq_len - 1] = True
    if int(mask.sum().item()) == 0:
        raise ValueError(f"prompt too short: seq_len={seq_len}, need > {skip_first + 1} tokens")
    return mask


def compute_aligned_bwd_start(seq_len: int, skip_first: int, *, align: int = 16) -> int:
    """Choose the largest suffix start `bwd_start <= skip_first` so `T_v = seq_len - bwd_start` is `align`-aligned.

    Exploits strict causal lower-triangularity of every layer: cotangents at masked prefix
    tokens `[0 : bwd_start]` never influence valid suffix tokens `[skip_first : seq_len - 1]`.
    Rounding `T_v` up to a multiple of `align = 16` (e.g. `T = 128, skip_first = 16 -> T_v = 112`)
    frees 12.5% of cotangent VRAM (`6.25 -> 5.47 GiB`) and skips prefix MoE/projection/QSA backward
    FLOPs while preserving 16-aligned Tensor Core / TMA shapes on QSA `bmm` and 128-aligned shapes
    on MoE `grouped_mm`, dense projections (`M = K_blk * T_v`), and zero-padded FlashQLA/FLA.
    Pass `align <= 0` to disable suffix pruning (`bwd_start = 0`).
    """
    if align <= 0 or skip_first <= 0:
        return 0
    if align == 1:
        return min(skip_first, seq_len)
    min_tv = max(1, seq_len - skip_first)
    aligned_tv = min(seq_len, ((min_tv + align - 1) // align) * align)
    return seq_len - aligned_tv


def _copy_payload_async(dst: MoeExpertPayload, src: MoeExpertPayload, non_blocking: bool) -> None:
    dst.gate_up_proj.copy_(src.gate_up_proj, non_blocking=non_blocking)
    dst.down_proj.copy_(src.down_proj, non_blocking=non_blocking)
    if dst.gate_up_scale is not None and src.gate_up_scale is not None:
        dst.gate_up_scale.copy_(src.gate_up_scale, non_blocking=non_blocking)
    if dst.down_scale is not None and src.down_scale is not None:
        dst.down_scale.copy_(src.down_scale, non_blocking=non_blocking)


class AsyncMoEStreamer:
    """Zero-stall double-buffered 3-tier MoE expert weight streamer.

    Manages:
      - Tier 0 (`GPU_HBM`): zero-copy direct return.
      - Tier 1 (`HOST_PINNED`): async H2D DMA on dedicated `h2d_stream` into 2-slot GPU staging ring.
      - Tier 1 (`HOST_PAGEABLE`) & Tier 2 (`NVME_SLAB`): background I/O thread populates 2-slot
        pinned host staging ring (overlapped across interleaved Tier 0/1 layers!), then `h2d_stream`
        DMAs into the 2-slot GPU staging ring!
    """

    def __init__(self, model: LoadedQwen38Model) -> None:
        self.model = model
        self.plan = model.plan
        self.cfg = model.config
        self.device = model.device
        self.has_cuda = self.device.type == "cuda"

        weight_dtype = torch.float8_e4m3fn if self.cfg.is_fp8_experts else model.compute_dtype

        self._gpu_slots: list[MoeExpertPayload] = [
            allocate_moe_payload(
                self.cfg.num_experts,
                self.cfg.hidden_size,
                self.cfg.moe_intermediate_size,
                device=self.device,
                pin_memory=False,
                weight_dtype=weight_dtype,
                block_size=self.cfg.fp8_block_size,
            )
            for _ in range(self.plan.gpu_staging_slots)
        ]
        self._host_slots: list[MoeExpertPayload] = [
            allocate_moe_payload(
                self.cfg.num_experts,
                self.cfg.hidden_size,
                self.cfg.moe_intermediate_size,
                device="cpu",
                pin_memory=self.has_cuda,
                weight_dtype=weight_dtype,
                block_size=self.cfg.fp8_block_size,
            )
            for _ in range(self.plan.host_pinned_staging_slots)
        ]

        if self.has_cuda:
            self._h2d_stream = torch.cuda.Stream(device=self.device)
            self._gpu_ready = [torch.cuda.Event() for _ in self._gpu_slots]
            self._gpu_free = [torch.cuda.Event() for _ in self._gpu_slots]
            self._host_free = [torch.cuda.Event() for _ in self._host_slots]
        else:
            self._h2d_stream = None
            self._gpu_ready = []
            self._gpu_free = []
            self._host_free = []

        self._io_pool = ThreadPoolExecutor(max_workers=1)
        # CPU-only scratch buffer for NVMe slabs
        self._cpu_nvme_buf: MoeExpertPayload | None = None
        if not self.has_cuda and any(
            t == StorageTier.NVME_SLAB for t in self.plan.layer_tiers.values()
        ):
            self._cpu_nvme_buf = allocate_moe_payload(
                self.cfg.num_experts,
                self.cfg.hidden_size,
                self.cfg.moe_intermediate_size,
                device="cpu",
                pin_memory=False,
                weight_dtype=weight_dtype,
                block_size=self.cfg.fp8_block_size,
            )

    def close(self) -> None:
        self._io_pool.shutdown(wait=True)

    def _stage_to_host_slot(self, layer_idx: int, host_slot_idx: int) -> int:
        """Runs on background I/O thread: waits for host slot DMA completion, then loads layer into pinned host slot."""
        if self.has_cuda:
            self._host_free[host_slot_idx].synchronize()
        dst = self._host_slots[host_slot_idx]
        item = self.model.moe_tier_store[layer_idx]
        if isinstance(item, str):
            read_moe_slab_into(item, dst)
        else:
            _copy_payload_async(dst, item, non_blocking=False)
        return host_slot_idx

    def iter_sweep(self, layer_order: Sequence[int]):
        """Context-style generator yielding `(layer_idx, gpu_moe_payload)` with full async double-buffering."""
        if not self.has_cuda or not self._gpu_slots:
            # Pure GPU-resident (all Tier 0) or CPU test path
            for l_idx in layer_order:
                item = self.model.moe_tier_store[l_idx]
                if isinstance(item, str):
                    assert self._cpu_nvme_buf is not None
                    yield l_idx, read_moe_slab_into(item, self._cpu_nvme_buf)
                else:
                    yield l_idx, item
            return

        order = list(layer_order)
        tiers = self.plan.layer_tiers

        # Pre-index all slow layers (Tier 1 pageable + Tier 2 NVMe) that require host staging
        slow_positions = [
            pos
            for pos, l_idx in enumerate(order)
            if tiers[l_idx] in (StorageTier.HOST_PAGEABLE, StorageTier.NVME_SLAB)
        ]
        # Track active host futures: layer_idx -> (Future, host_slot_idx)
        host_futures: dict[int, tuple[Future, int]] = {}
        next_slow_ptr = 0
        next_host_slot = 0
        n_host_slots = len(self._host_slots)

        def _kick_host_prefetch() -> None:
            nonlocal next_slow_ptr, next_host_slot
            if n_host_slots == 0:
                return
            while next_slow_ptr < len(slow_positions) and len(host_futures) < n_host_slots:
                l_slow = order[slow_positions[next_slow_ptr]]
                h_slot = next_host_slot % n_host_slots
                next_host_slot += 1
                fut = self._io_pool.submit(self._stage_to_host_slot, l_slow, h_slot)
                host_futures[l_slow] = (fut, h_slot)
                next_slow_ptr += 1

        # Active GPU H2D transfers: layer_idx -> gpu_slot_idx
        gpu_inflight: dict[int, int] = {}
        next_gpu_slot = 0
        n_gpu_slots = len(self._gpu_slots)

        def _kick_h2d(l_idx: int) -> None:
            nonlocal next_gpu_slot
            t = tiers[l_idx]
            if t == StorageTier.GPU_HBM or l_idx in gpu_inflight:
                return
            g_slot = next_gpu_slot % n_gpu_slots
            next_gpu_slot += 1

            if t == StorageTier.HOST_PINNED:
                src_payload = self.model.moe_tier_store[l_idx]
                assert not isinstance(src_payload, str)
                h_slot_used = None
            else:
                # Ensure background I/O finished staging into pinned host slot
                fut, h_slot_used = host_futures.pop(l_idx)
                fut.result()
                src_payload = self._host_slots[h_slot_used]

            with torch.cuda.stream(self._h2d_stream):
                self._h2d_stream.wait_event(self._gpu_free[g_slot])
                _copy_payload_async(self._gpu_slots[g_slot], src_payload, non_blocking=True)
                self._gpu_ready[g_slot].record(self._h2d_stream)
                if h_slot_used is not None:
                    self._host_free[h_slot_used].record(self._h2d_stream)
            gpu_inflight[l_idx] = g_slot
            # Immediately queue the next slow layer onto the freed host slot!
            if h_slot_used is not None:
                _kick_host_prefetch()

        # Prime pipeline: kick off up to 2 slow-tier host prefetches + first offloaded H2D
        _kick_host_prefetch()
        if order:
            _kick_h2d(order[0])

        compute_stream = torch.cuda.current_stream(self.device)
        for pos, l_idx in enumerate(order):
            # Look ahead to prefetch the NEXT offloaded layer in parallel with current layer's compute!
            for look_pos in range(pos + 1, min(len(order), pos + 3)):
                cand = order[look_pos]
                if tiers[cand] != StorageTier.GPU_HBM:
                    # Only launch H2D if we have a free GPU slot (at most 1 ahead of current)
                    if len(gpu_inflight) < n_gpu_slots:
                        _kick_h2d(cand)
                    break

            t = tiers[l_idx]
            if t == StorageTier.GPU_HBM:
                payload = self.model.moe_tier_store[l_idx]
                assert not isinstance(payload, str)
                yield l_idx, payload
            else:
                if l_idx not in gpu_inflight:
                    _kick_h2d(l_idx)
                g_slot = gpu_inflight.pop(l_idx)
                compute_stream.wait_event(self._gpu_ready[g_slot])
                try:
                    yield l_idx, self._gpu_slots[g_slot]
                finally:
                    self._gpu_free[g_slot].record(compute_stream)


@dataclass
class FitStats:
    prompt_idx: int
    seq_len: int
    n_valid: int
    elapsed_s: float
    fwd_s: float
    bwd_s: float
    max_norm: float
    mean_rel_change: float


@dataclass
class _CachedPromptForward:
    """Compact per-prompt forward cache (~120 MB in CPU RAM at T=128) for forward-only micro-batching."""

    seq_len: int
    n_valid: int
    bwd_start: int
    valid_pos_local: torch.Tensor
    layer_inputs: dict[int, torch.Tensor]
    mixer_saved_v: dict[str, Any]
    ple_embs: dict[int, torch.Tensor]


def _move_tensor_dict(d: dict[str, Any], device: torch.device) -> dict[str, Any]:
    non_blocking = device.type == "cuda"
    return {
        k: (
            v.to(device, non_blocking=non_blocking)
            if isinstance(v, torch.Tensor) and v.device != device
            else v
        )
        for k, v in d.items()
    }


def _reduce_valid_mean_cpu(
    G: torch.Tensor,
    valid_pos_local: torch.Tensor,
    *,
    k_chunk: int = 256,
) -> torch.Tensor:
    """Exact fp32 mean over valid positions without allocating a 16.26 GiB fancy-indexed fp32 copy of `G`!

    Previously `G[:, valid_pos_local, :].float().mean(dim=1)` allocated a 5.42 GiB BF16 fancy-index
    copy AND a 10.84 GiB FP32 cast on GPU simultaneously. Using a zero-copy slice + chunked
    `mean(dim=1, dtype=torch.float32)` keeps peak temporary GPU VRAM at ~10 MB (0.01 GiB)!
    """
    K_cur, _, D_hc = G.shape
    out = torch.empty((K_cur, D_hc), dtype=torch.float32)
    n_valid = int(valid_pos_local.numel())
    lo = int(valid_pos_local[0].item())
    hi = int(valid_pos_local[-1].item()) + 1
    is_contig = (hi - lo) == n_valid
    for k0 in range(0, K_cur, k_chunk):
        k1 = min(k0 + k_chunk, K_cur)
        sub = G[k0:k1, lo:hi, :] if is_contig else G[k0:k1, valid_pos_local, :]
        out[k0:k1] = sub.mean(dim=1, dtype=torch.float32).cpu()
    return out


def _gpu_tensor_bytes(obj: Any, seen: set[int] | None = None) -> int:
    """Recursively sum bytes of CUDA tensors inside dataclasses / dicts / lists (deduplicated by storage)."""
    if seen is None:
        seen = set()
    if isinstance(obj, torch.Tensor):
        if obj.device.type != "cuda":
            return 0
        key = obj.untyped_storage().data_ptr()
        if key in seen:
            return 0
        seen.add(key)
        return obj.untyped_storage().nbytes()
    if isinstance(obj, dict):
        return sum(_gpu_tensor_bytes(v, seen) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_gpu_tensor_bytes(v, seen) for v in obj)
    if hasattr(obj, "__dataclass_fields__"):
        return sum(_gpu_tensor_bytes(getattr(obj, f), seen) for f in obj.__dataclass_fields__)
    return 0


def model_vram_breakdown(model: LoadedQwen38Model, streamer: "AsyncMoEStreamer | None" = None) -> dict[str, float]:
    """Live per-component GPU memory breakdown (GiB) — makes any budget discrepancy visible immediately."""
    gib = 1024**3
    out = {
        "dense_gib": (
            _gpu_tensor_bytes(model.layers)
            + _gpu_tensor_bytes(model.embed_tokens)
            + _gpu_tensor_bytes(model.hyper_connection_mixer)
        ) / gib,
        "tier0_moe_gib": _gpu_tensor_bytes(
            [v for v in model.moe_tier_store.values() if not isinstance(v, str)]
        ) / gib,
        "staging_gib": (_gpu_tensor_bytes(streamer._gpu_slots) / gib) if streamer is not None else 0.0,
        "allocated_gib": 0.0,
        "reserved_gib": 0.0,
        "free_gib": 0.0,
        "total_gib": 0.0,
    }
    if model.device.type == "cuda" and torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info(model.device)
        out["allocated_gib"] = torch.cuda.memory_allocated(model.device) / gib
        out["reserved_gib"] = torch.cuda.memory_reserved(model.device) / gib
        out["free_gib"] = free_b / gib
        out["total_gib"] = total_b / gib
    return out


def ensure_vram_headroom(
    model: LoadedQwen38Model,
    *,
    reserve_gib: float | None = None,
    free_gib: float | None = None,
) -> list[int]:
    """Self-healing OOM guard: demote Tier-0 MoE layers to pinned host RAM until live free VRAM covers
    `plan.runtime_vram_reserve_gib` (+ any GPU staging slots the streamer still has to allocate).

    Backward is compute-bound and Tier-1 H2D DMA hides behind adjacent layers, so demoting a few Tier-0
    layers costs ~0% speed while turning a guaranteed mid-run OOM into a clean run. Returns demoted layers.
    """
    plan = model.plan
    if model.device.type != "cuda":
        return []
    reserve = plan.runtime_vram_reserve_gib if reserve_gib is None else float(reserve_gib)
    if reserve <= 0.0:
        return []
    if free_gib is None:
        torch.cuda.empty_cache()
        free_gib = torch.cuda.mem_get_info(model.device)[0] / (1024**3)

    demoted, need = select_tier0_layers_to_demote(plan, free_gib=free_gib, reserve_gib=reserve)
    if not demoted:
        return []
    weight_dtype = torch.float8_e4m3fn if model.config.is_fp8_experts else model.compute_dtype
    for l_idx in demoted:
        gpu_payload = model.moe_tier_store[l_idx]
        assert not isinstance(gpu_payload, str)
        alloc_kwargs = dict(
            device="cpu",
            weight_dtype=weight_dtype,
            block_size=model.config.fp8_block_size,
        )
        dims = (model.config.num_experts, model.config.hidden_size, model.config.moe_intermediate_size)
        try:
            host = allocate_moe_payload(*dims, pin_memory=True, **alloc_kwargs)
            new_tier = StorageTier.HOST_PINNED
        except RuntimeError:
            # Pinning can fail on tight-RAM hosts: keep it pageable and stream via the pinned host ring.
            host = allocate_moe_payload(*dims, pin_memory=False, **alloc_kwargs)
            new_tier = StorageTier.HOST_PAGEABLE
            plan.host_pinned_staging_slots = max(2, plan.host_pinned_staging_slots)
        _copy_payload_async(host, gpu_payload, non_blocking=False)
        model.moe_tier_store[l_idx] = host
        plan.layer_tiers[l_idx] = new_tier
        del gpu_payload
    if plan.gpu_staging_slots == 0:
        plan.gpu_staging_slots = 2
    torch.cuda.synchronize(model.device)
    torch.cuda.empty_cache()
    logger.warning(
        "VRAM self-heal: free %.1f GiB < required %.1f GiB -> demoted %d Tier-0 MoE layers to pinned RAM: %s",
        free_gib, need, len(demoted), demoted,
    )
    return demoted


def select_tier0_layers_to_demote(
    plan: Any,
    *,
    free_gib: float,
    reserve_gib: float,
) -> tuple[list[int], float]:
    """Pure planner: which Tier-0 layers to demote so `free_gib >= reserve + staging slots`.

    Returns `(layers, required_gib)`. Picks maximally spread-out Tier-0 layers so the remaining GPU layers
    stay interleaved with the Tier-1 layers (every H2D DMA keeps a Tier-0 layer to hide behind).
    """
    layer_gib = plan.bytes_per_moe_layer / (1024**3)
    tier0 = sorted(l for l, t in plan.layer_tiers.items() if t == StorageTier.GPU_HBM)
    need = reserve_gib + plan.gpu_staging_slots * layer_gib
    if free_gib >= need or not tier0:
        return [], need
    # Demoting means streaming from host -> the 2 double-buffered GPU staging slots must exist too.
    need = reserve_gib + max(2, plan.gpu_staging_slots) * layer_gib
    n_demote = min(len(tier0), math.ceil((need - free_gib) / layer_gib))
    return [tier0[p] for p in _evenly_spaced_positions(len(tier0), n_demote)], need


def _evenly_spaced_positions(n: int, count: int) -> list[int]:
    if count <= 0:
        return []
    if count >= n:
        return list(range(n))
    return sorted({min(n - 1, int((k + 0.5) * n / count)) for k in range(count)})


class Qwen38JLensEngine:
    """High-performance single-sweep Jacobian Lens fitter and inference applier for Qwen3.8-Flash-Next."""

    def __init__(self, model: LoadedQwen38Model, *, auto_vram_headroom: bool = True) -> None:
        self.model = model
        self.cfg = model.config
        self.plan = model.plan
        self.device = model.device
        self.dtype = model.compute_dtype
        self.gdn_backend = GatedDeltaRuleBackend(
            preferred=self.plan.gdn_backend,
            max_seq_len=self.plan.max_seq_len,
        )
        # Verify REAL post-load headroom before allocating staging slots (self-healing OOM guard).
        self.demoted_layers = ensure_vram_headroom(model) if auto_vram_headroom else []
        self.streamer = AsyncMoEStreamer(model)

    def vram_breakdown(self) -> dict[str, float]:
        return model_vram_breakdown(self.model, self.streamer)

    def close(self) -> None:
        self.streamer.close()

    def _get_ple_embeddings(
        self,
        input_ids: torch.Tensor,
        *,
        device: torch.device | None = None,
    ) -> dict[int, torch.Tensor]:
        """Gather `B=1` PLE n-gram embeddings for all PLE layers (takes <2 ms via mmap)."""
        target_dev = self.device if device is None else device
        out: dict[int, torch.Tensor] = {}
        for l_idx, provider in self.model.ple_providers.items():
            out[l_idx] = provider.lookup(input_ids, device=target_dev).to(self.dtype)
        return out

    @torch.no_grad()
    def forward_capture(
        self,
        input_ids: torch.Tensor,  # [1, T]
        *,
        capture_layers: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Run ONE fast `B=1` forward pass streaming layers 0..L-1; returns `(h_final, captured_H)`.

        Layer `-1` denotes the post-embedding 4-branch state `H_{-1} = emb.repeat(1, 1, hc_count)` before Layer 0.
        """
        input_ids = input_ids.to(self.device)
        T = input_ids.shape[1]
        capture_set = (
            {-1, *range(self.cfg.num_hidden_layers)}
            if capture_layers is None
            else set(capture_layers)
        )

        cos, sin = build_rope_cos_sin(
            seq_len=T,
            head_dim=self.cfg.head_dim,
            partial_rotary_factor=self.cfg.partial_rotary_factor,
            rope_theta=self.cfg.rope_theta,
            device=self.device,
            dtype=self.dtype,
        )
        ple_embs = self._get_ple_embeddings(input_ids)

        emb = F.embedding(input_ids, self.model.embed_tokens)  # [1, T, D]
        H = emb.repeat(1, 1, self.cfg.hc_count).contiguous()   # [1, T, hc_count * D]
        captured: dict[int, torch.Tensor] = {}
        if -1 in capture_set:
            captured[-1] = H.detach()

        for l_idx, experts in self.streamer.iter_sweep(range(self.cfg.num_hidden_layers)):
            H, _ = decoder_layer_fwd(
                H,
                self.model.layers[l_idx],
                experts,
                cos,
                sin,
                self.gdn_backend,
                ple_embeddings=ple_embs.get(l_idx),
                dequant_chunk_size=self.plan.expert_dequant_chunk,
                use_triton_dequant=self.plan.use_triton_fp8_dequant,
                keep_experts_dequantized=False,
            )
            if l_idx in capture_set:
                captured[l_idx] = H.detach()

        h_final, _, _ = gated_residual_fwd(H, self.model.hyper_connection_mixer)
        return h_final, captured

    @torch.no_grad()
    def _forward_microbatch(
        self,
        batch_input_ids: Sequence[torch.Tensor],
        min_src: int,
        *,
        skip_first: int = SKIP_FIRST_N_POSITIONS,
        bwd_align: int = 16,
        cache_on_cpu: bool = False,
    ) -> tuple[list[_CachedPromptForward], float]:
        """Run ONE forward weight sweep (0 -> L-1) amortized across `len(batch_input_ids)` prompts.

        Supports variable-length prompts without padding. When `cache_on_cpu=True` (used for
        `fwd_batch_size > 1` on CUDA), stores each prompt's `[1, T, D_hc]` layer inputs (~120 MB/prompt)
        in CPU RAM so peak GPU VRAM stays identical to `fwd_batch_size=1` (0 GiB VRAM overhead!).
        """
        t0 = time.perf_counter()
        B = len(batch_input_ids)
        hc = self.cfg.hc_count
        n_layers = self.cfg.num_hidden_layers
        offload_to_cpu = cache_on_cpu and self.device.type != "cpu"
        ple_dev = torch.device("cpu") if offload_to_cpu else self.device

        seq_lens: list[int] = []
        n_valids: list[int] = []
        bwd_starts: list[int] = []
        valid_pos_locals: list[torch.Tensor] = []
        rope_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        ple_embs_list: list[dict[int, torch.Tensor]] = []
        H_cur: list[torch.Tensor] = []

        for ids in batch_input_ids:
            T = int(ids.shape[1])
            pos_mask = valid_position_mask(T, skip_first=skip_first)
            valid_pos = pos_mask.nonzero(as_tuple=True)[0]
            bwd_start = compute_aligned_bwd_start(T, skip_first, align=bwd_align)
            seq_lens.append(T)
            n_valids.append(int(valid_pos.numel()))
            bwd_starts.append(bwd_start)
            valid_pos_locals.append(valid_pos - bwd_start)

            if T not in rope_cache:
                rope_cache[T] = build_rope_cos_sin(
                    seq_len=T,
                    head_dim=self.cfg.head_dim,
                    partial_rotary_factor=self.cfg.partial_rotary_factor,
                    rope_theta=self.cfg.rope_theta,
                    device=self.device,
                    dtype=self.dtype,
                )
            ids_dev = ids.to(self.device)
            ple_embs_list.append(self._get_ple_embeddings(ids_dev, device=ple_dev))
            emb = F.embedding(ids_dev, self.model.embed_tokens)
            H_cur.append(emb.repeat(1, 1, hc).contiguous())

        saved_inputs: list[dict[int, torch.Tensor]] = [{} for _ in range(B)]

        for l_idx, experts in self.streamer.iter_sweep(range(n_layers)):
            layer_w = self.model.layers[l_idx]
            if B == 1:
                H_b = H_cur[0]
                if l_idx > min_src:
                    saved_inputs[0][l_idx] = H_b.to("cpu") if offload_to_cpu else H_b
                cos_b, sin_b = rope_cache[seq_lens[0]]
                ple_b = ple_embs_list[0].get(l_idx)
                if ple_b is not None and ple_b.device != self.device:
                    ple_b = ple_b.to(self.device)
                H_next, _ = decoder_layer_fwd(
                    H_b,
                    layer_w,
                    experts,
                    cos_b,
                    sin_b,
                    self.gdn_backend,
                    ple_embeddings=ple_b,
                    dequant_chunk_size=self.plan.expert_dequant_chunk,
                    use_triton_dequant=self.plan.use_triton_fp8_dequant,
                    keep_experts_dequantized=False,
                )
                H_cur[0] = H_next
            else:
                # Amortize FP8 expert dequantization & grouped_mm launches across ALL B prompts
                # via a single concatenated token-wise moe_block_fwd (supports variable seq_lens!).
                H_mid_list: list[torch.Tensor] = []
                x_mlp_list: list[torch.Tensor] = []
                w_mlp_list: list[torch.Tensor] = []
                for b in range(B):
                    H_b = H_cur[b]
                    if l_idx > min_src:
                        saved_inputs[b][l_idx] = H_b.to("cpu") if offload_to_cpu else H_b
                    cos_b, sin_b = rope_cache[seq_lens[b]]
                    ple_b = ple_embs_list[b].get(l_idx)
                    if ple_b is not None and ple_b.device != self.device:
                        ple_b = ple_b.to(self.device)
                    H_mid_b = decoder_layer_attn_fwd(
                        H_b,
                        layer_w,
                        cos_b,
                        sin_b,
                        self.gdn_backend,
                        ple_embeddings=ple_b,
                    )
                    x_mlp_b, w_mlp_b, _ = gated_residual_fwd(H_mid_b, layer_w.mlp_hc)
                    assert w_mlp_b is not None
                    H_mid_list.append(H_mid_b)
                    x_mlp_list.append(x_mlp_b)
                    w_mlp_list.append(w_mlp_b)
                    H_cur[b] = torch.empty(0)

                x_mlp_cat = torch.cat(x_mlp_list, dim=1)  # [1, sum(T_b), D]
                del x_mlp_list
                y_mlp_cat, _ = moe_block_fwd(
                    x_mlp_cat,
                    layer_w.moe_dense,
                    experts,
                    dequant_chunk_size=self.plan.expert_dequant_chunk,
                    use_triton_dequant=self.plan.use_triton_fp8_dequant,
                    keep_dequantized_for_bwd=False,
                )
                del x_mlp_cat
                y_mlp_splits = y_mlp_cat.split(seq_lens, dim=1)
                for b in range(B):
                    H_cur[b] = (
                        H_mid_list[b]
                        + (y_mlp_splits[b].unsqueeze(-2) * w_mlp_list[b].unsqueeze(-1)).flatten(-2)
                    )
                del H_mid_list, w_mlp_list, y_mlp_cat, y_mlp_splits

        cached_list: list[_CachedPromptForward] = []
        cpu_dev = torch.device("cpu")
        for b in range(B):
            _, _, mixer_saved = gated_residual_fwd(H_cur[b], self.model.hyper_connection_mixer)
            H_cur[b] = torch.empty(0)
            mixer_saved_v = slice_gated_residual_saved(mixer_saved, bwd_starts[b])
            del mixer_saved
            if offload_to_cpu:
                mixer_saved_v = _move_tensor_dict(mixer_saved_v, cpu_dev)
            cached_list.append(
                _CachedPromptForward(
                    seq_len=seq_lens[b],
                    n_valid=n_valids[b],
                    bwd_start=bwd_starts[b],
                    valid_pos_local=valid_pos_locals[b],
                    layer_inputs=saved_inputs[b],
                    mixer_saved_v=mixer_saved_v,
                    ple_embs=ple_embs_list[b],
                )
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_fwd_total = time.perf_counter() - t0
        return cached_list, t_fwd_total

    @torch.no_grad()
    def _backward_from_cached_forward(
        self,
        cached: _CachedPromptForward,
        src_list: list[int],
    ) -> tuple[dict[int, torch.Tensor], float]:
        """Run exact single-sweep backward propagation (L-1 -> min_src + 1) from a cached forward pass."""
        t_bwd_start = time.perf_counter()
        T = cached.seq_len
        bwd_start = cached.bwd_start
        T_v = T - bwd_start
        valid_pos_local = cached.valid_pos_local.to(self.device)

        D = self.cfg.hidden_size
        D_hc = D * self.cfg.hc_count
        n_layers = self.cfg.num_hidden_layers
        src_set = set(src_list)
        min_src = src_list[0]

        cos, sin = build_rope_cos_sin(
            seq_len=T,
            head_dim=self.cfg.head_dim,
            partial_rotary_factor=self.cfg.partial_rotary_factor,
            rope_theta=self.cfg.rope_theta,
            device=self.device,
            dtype=self.dtype,
        )
        mixer_saved_v = _move_tensor_dict(cached.mixer_saved_v, self.device)
        ple_embs = {
            l_idx: (t.to(self.device) if t.device != self.device else t)
            for l_idx, t in cached.ple_embs.items()
        }

        jacobians = {l: torch.zeros((D, D_hc), dtype=torch.float32) for l in src_list}
        k_pass = min(self.plan.k_pass, D)
        bwd_layers = list(range(n_layers - 1, min_src, -1))
        non_blocking = self.device.type == "cuda"

        for k_start in range(0, D, k_pass):
            k_end = min(k_start + k_pass, D)
            K_cur = k_end - k_start

            # Build initial one-hot cotangent at target h_final for valid positions on [bwd_start:],
            # then immediately backprop through hyper_connection_mixer -> G_{L-1} [K_cur, T_v, D_hc]!
            G: torch.Tensor | None = torch.zeros((K_cur, T_v, D_hc), device=self.device, dtype=self.dtype)
            try:
                hc_k_chunk = min(64, self.plan.moe_k_chunk, K_cur)
                d_target_chunk = torch.zeros(
                    (hc_k_chunk, T_v, D), device=self.device, dtype=self.dtype
                )
                k_sub_idx = torch.arange(hc_k_chunk, device=self.device)
                for c0 in range(0, K_cur, hc_k_chunk):
                    c1 = min(c0 + hc_k_chunk, K_cur)
                    nc = c1 - c0
                    d_target_chunk[:nc].zero_()
                    d_target_chunk[
                        k_sub_idx[:nc, None],
                        valid_pos_local[None, :],
                        (k_start + c0) + k_sub_idx[:nc, None],
                    ] = 1.0
                    G[c0:c1] = gated_residual_vjp(
                        d_target_chunk[:nc], None, self.model.hyper_connection_mixer, mixer_saved_v
                    )
                del d_target_chunk

                # Layer L-1 output H_{L-1} Jacobian (chunked fp32 reduction -> 0.01 GiB peak vs 16.26 GiB!)
                if (n_layers - 1) in src_set:
                    jacobians[n_layers - 1][k_start:k_end, :] = _reduce_valid_mean_cpu(
                        G, valid_pos_local, k_chunk=self.plan.moe_k_chunk
                    )

                # Reverse layer sweep: L-1 -> min_src + 1
                for l_idx, experts in self.streamer.iter_sweep(bwd_layers):
                    H_in_l = cached.layer_inputs[l_idx]
                    if H_in_l.device != self.device:
                        H_in_l = H_in_l.to(self.device, non_blocking=non_blocking)

                    # Recompute ONE tiny B=1 forward of layer l_idx (~0.5 ms), keeping suffix-hit active
                    # experts dequantized in BF16 when E_hit <= expert_dequant_chunk!
                    _, saved_l = decoder_layer_fwd(
                        H_in_l,
                        self.model.layers[l_idx],
                        experts,
                        cos,
                        sin,
                        self.gdn_backend,
                        ple_embeddings=ple_embs.get(l_idx),
                        dequant_chunk_size=self.plan.expert_dequant_chunk,
                        use_triton_dequant=self.plan.use_triton_fp8_dequant,
                        keep_experts_dequantized=True,
                        bwd_start=bwd_start,
                    )
                    del H_in_l

                    # In-place VJP across ALL K_cur cotangents on [bwd_start:]!
                    decoder_layer_vjp_inplace(
                        G,
                        self.model.layers[l_idx],
                        experts,
                        saved_l,
                        self.gdn_backend,
                        attn_k_chunk=self.plan.attn_k_chunk,
                        moe_k_chunk=self.plan.moe_k_chunk,
                        dequant_chunk_size=self.plan.expert_dequant_chunk,
                        use_triton_dequant=self.plan.use_triton_fp8_dequant,
                        use_triton_swiglu=self.plan.use_triton_swiglu_vjp,
                    )
                    del saved_l

                    # After layer l_idx backward, G is now d h_final / d H_{l_idx - 1} on [bwd_start:]!
                    prev_layer = l_idx - 1
                    if prev_layer in src_set:
                        jacobians[prev_layer][k_start:k_end, :] = _reduce_valid_mean_cpu(
                            G, valid_pos_local, k_chunk=self.plan.moe_k_chunk
                        )
            finally:
                # Explicitly drop 5.47 GiB cotangent buffer so Jupyter tracebacks never trap it on crash/interrupt!
                G = None

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_bwd = time.perf_counter() - t_bwd_start
        return jacobians, t_bwd

    @torch.no_grad()
    def jacobian_for_tokens(
        self,
        input_ids: torch.Tensor,  # [1, T]
        source_layers: Sequence[int],
        *,
        skip_first: int = SKIP_FIRST_N_POSITIONS,
        bwd_align: int = 16,
    ) -> tuple[dict[int, torch.Tensor], int, int, float, float]:
        """Compute exact per-prompt `J_l = E[d h_final / d H_l]` (`[D, hc_count * D]` fp32 CPU) for all `source_layers`.

        Layer `-1` (optional) represents the embedding output `H_{-1} = emb.repeat(1, 1, hc_count)` before Layer 0,
        whose 4-branch sum `J_collapsed[-1] = sum_b J_{-1}[:, b, :]` equals the exact token-embedding Jacobian
        `E[d h_final / d emb]`.

        Uses single-sweep layer-by-layer reverse propagation with 16-aligned exact causal suffix pruning:
          - Forward sweep (0 -> L-1): runs on full `[0:T]`, saving ONLY `[1, T, 10240]` layer inputs (~120 MB total!).
          - Backward sweep (L-1 -> min_source + 1): drops masked prefix tokens `[0 : bwd_start]` while keeping
            `T_v = T - bwd_start` aligned to `bwd_align` (e.g. `112` at `T=128, skip_first=16`), holds ALL `k_pass`
            cotangents in ONE in-place GPU buffer `G [k_pass, T_v, 10240]` (5.47 GiB vs 6.25 GiB), dequantizes
            only suffix-hit experts ONCE, and overlaps next-layer DMA/NVMe prefetch 100% behind backward compute!
        """
        n_layers = self.cfg.num_hidden_layers
        src_list = sorted(set(int(l) for l in source_layers))
        if not src_list or src_list[0] < -1 or src_list[-1] >= n_layers:
            raise ValueError(f"source_layers {src_list} out of valid range [-1, {n_layers - 1}]")

        cached_list, t_fwd = self._forward_microbatch(
            [input_ids],
            min_src=src_list[0],
            skip_first=skip_first,
            bwd_align=bwd_align,
            cache_on_cpu=False,
        )
        cached = cached_list[0]
        jacobians, t_bwd = self._backward_from_cached_forward(cached, src_list)
        return jacobians, cached.seq_len, cached.n_valid, t_fwd, t_bwd

    def fit(
        self,
        tokenized_prompts: Sequence[torch.Tensor],
        *,
        source_layers: Sequence[int] | None = None,
        include_embedding_layer: bool = False,
        skip_first: int = SKIP_FIRST_N_POSITIONS,
        bwd_align: int = 16,
        fwd_batch_size: int = 8,
        checkpoint_path: str | None = None,
        checkpoint_every: int = 10,
        checkpoint_manager: Any | None = None,
        convergence_csv_path: str | None = None,
        stop_at_delta: float | None = None,
        min_prompts: int = 100,
        stop_window: int = 10,
        resume: bool = True,
        on_prompt_done: Any = None,
    ) -> Qwen38JacobianLens:
        """Fit `J_l` over tokenized prompts `[1, T]` with zero-GPU-stall async checkpointing & resume.

        Supports:
          - `include_embedding_layer=True` (or `-1` in `source_layers`) to include layer `-1`
            (`embed_tokens -> H_{-1}` before Layer 0, where `J_collapsed[-1] = d h_final / d emb`).
          - `bwd_align=16` (default): exact 16-aligned causal suffix-only backward pruning of masked
            prefix tokens `[0 : bwd_start]` (`bwd_align=0` disables pruning).
          - `fwd_batch_size=8` (default): zero-VRAM forward-only micro-batching! Amortizes the
            weight-transfer-bound forward sweep across `fwd_batch_size` prompts (caching ~120 MB/prompt
            in CPU RAM -> 0 GiB extra VRAM, 8x fewer forward weight sweeps!) while running exact
            per-prompt `B=1` backward sweeps, checkpointing, and early stopping.
          - `checkpoint_manager=AsyncCheckpointManager(...)` (recommended: Local NVMe + async Google Drive
            during fitting, plus one-shot `publish_bundle_to_hf(...)` once complete!), OR
          - `checkpoint_path=".../ckpt.safetensors"` (or legacy `.pt`) for simple local checkpointing.
          - Optional Neuronpedia-style convergence logging (`convergence_csv_path`) and early stopping
            (`stop_at_delta=0.002`, `min_prompts=100`, `stop_window=10`).
        """
        from qwen38_jlens.checkpoint import AsyncCheckpointManager, CheckpointConfig, ConvergenceLogger

        n_layers = self.cfg.num_hidden_layers
        D = self.cfg.hidden_size
        D_hc = D * self.cfg.hc_count
        if source_layers is None:
            src_layers = ([-1] if include_embedding_layer else []) + list(range(n_layers))
        else:
            src_layers = sorted(set(int(l) for l in source_layers))
            if include_embedding_layer and -1 not in src_layers:
                src_layers = [-1] + src_layers
        if not src_layers or src_layers[0] < -1 or src_layers[-1] >= n_layers:
            raise ValueError(f"source_layers {src_layers} out of valid range [-1, {n_layers - 1}]")

        owned_mgr = False
        mgr: AsyncCheckpointManager | None = checkpoint_manager
        if mgr is None and checkpoint_path is not None:
            if checkpoint_path.endswith(".safetensors"):
                mgr = AsyncCheckpointManager(
                    CheckpointConfig(
                        local_path=checkpoint_path,
                        every_n_prompts=max(1, checkpoint_every),
                        every_minutes=0.0,
                        ckpt_dtype="float32",
                    )
                )
                owned_mgr = True

        # ---- Resume from freshest checkpoint (Local / Drive or legacy .pt) ----
        jacobian_sum: dict[int, torch.Tensor]
        n_done: int
        next_idx: int
        resumed = None
        if resume and mgr is not None:
            resumed = mgr.load_best_resume(
                expected_source_layers=src_layers, expected_skip_first=skip_first
            )
        elif resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if state.get("source_layers") != src_layers or state.get("skip_first") != skip_first:
                raise ValueError(
                    f"Checkpoint at {checkpoint_path} has incompatible source_layers/skip_first; pass resume=False."
                )
            resumed = (state["jacobian_sum"], int(state["n_done"]), int(state["next_idx"]))

        if resumed is not None:
            jacobian_sum, n_done, next_idx = resumed
            logger.info("Resuming J-lens fit: %d/%d prompts already processed (n_done=%d).", next_idx, len(tokenized_prompts), n_done)
        else:
            jacobian_sum = {l: torch.zeros((D, D_hc), dtype=torch.float32) for l in src_layers}
            n_done = 0
            next_idx = 0
        resumed_next_idx = next_idx
        resumed_n_done = n_done

        csv_path = convergence_csv_path or (mgr.cfg.local_convergence_csv if mgr is not None else None)
        conv_logger = ConvergenceLogger(csv_path, resume_next_idx=next_idx) if csv_path else None
        recent_deltas: list[float] = conv_logger.tail_deltas(stop_window) if conv_logger is not None else []

        def _save_legacy_pt() -> None:
            if checkpoint_path is None:
                return
            parent = os.path.dirname(os.path.abspath(checkpoint_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp = f"{checkpoint_path}.tmp.{os.getpid()}"
            torch.save(
                {
                    "jacobian_sum": jacobian_sum,
                    "n_done": n_done,
                    "next_idx": next_idx,
                    "source_layers": src_layers,
                    "skip_first": skip_first,
                    "d_model": D,
                    "hc_count": self.cfg.hc_count,
                },
                tmp,
            )
            os.replace(tmp, checkpoint_path)

        fwd_bs = max(1, int(fwd_batch_size))
        total_prompts = len(tokenized_prompts)
        sqrt_d = math.sqrt(D)
        prompts_since_ckpt = 0
        early_stopped = False
        try:
            batch_start = next_idx
            while batch_start < total_prompts and not early_stopped:
                batch_end = min(batch_start + fwd_bs, total_prompts)

                # Separate valid prompts (amortized in 1 forward sweep) from too-short prompts
                valid_entries: list[tuple[int, torch.Tensor]] = []
                skipped_errors: dict[int, ValueError] = {}
                for p_idx in range(batch_start, batch_end):
                    ids = tokenized_prompts[p_idx]
                    try:
                        valid_position_mask(int(ids.shape[1]), skip_first=skip_first)
                        valid_entries.append((p_idx, ids))
                    except ValueError as exc:
                        skipped_errors[p_idx] = exc

                cached_by_pidx: dict[int, _CachedPromptForward] = {}
                t_fwd_amort = 0.0
                if valid_entries:
                    cache_on_cpu = len(valid_entries) > 1 and self.device.type == "cuda"
                    cached_list, t_fwd_batch = self._forward_microbatch(
                        [ids for _, ids in valid_entries],
                        min_src=src_layers[0],
                        skip_first=skip_first,
                        bwd_align=bwd_align,
                        cache_on_cpu=cache_on_cpu,
                    )
                    t_fwd_amort = t_fwd_batch / len(valid_entries)
                    for (p_idx, _), c_item in zip(valid_entries, cached_list, strict=True):
                        cached_by_pidx[p_idx] = c_item
                    del cached_list

                for p_idx in range(batch_start, batch_end):
                    if p_idx in skipped_errors:
                        logger.warning("Skipping prompt %d: %s", p_idx, skipped_errors[p_idx])
                        next_idx = p_idx + 1
                        continue

                    cached = cached_by_pidx.pop(p_idx)
                    seq_len = cached.seq_len
                    n_valid = cached.n_valid
                    per_J, t_bwd = self._backward_from_cached_forward(cached, src_layers)
                    del cached
                    if self.device.type == "cuda" and n_done == resumed_n_done:
                        peak = torch.cuda.max_memory_allocated(self.device) / (1024**3)
                        total = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
                        logger.info(
                            "First-prompt peak VRAM: %.1f / %.1f GiB allocated (%.1f GiB headroom).",
                            peak, total, total - peak,
                        )

                    max_norm = max(per_J[l].norm().item() for l in src_layers) / sqrt_d
                    if n_done > 0:
                        rel_change = max(
                            (
                                (per_J[l] - jacobian_sum[l] / n_done).norm()
                                / ((n_done + 1) * max(1e-8, (jacobian_sum[l] / n_done).norm().item()))
                            ).item()
                            for l in src_layers
                        )
                    else:
                        rel_change = float("nan")

                    for l in src_layers:
                        jacobian_sum[l].add_(per_J[l])
                    n_done += 1
                    next_idx = p_idx + 1
                    prompts_since_ckpt += 1
                    elapsed = t_fwd_amort + t_bwd

                    stats = FitStats(
                        prompt_idx=p_idx,
                        seq_len=seq_len,
                        n_valid=n_valid,
                        elapsed_s=elapsed,
                        fwd_s=t_fwd_amort,
                        bwd_s=t_bwd,
                        max_norm=max_norm,
                        mean_rel_change=rel_change,
                    )
                    if conv_logger is not None:
                        conv_logger.append(n_done, stats)
                    if math.isfinite(rel_change):
                        recent_deltas.append(rel_change)

                    logger.info(
                        "Prompt %d/%d  T=%d (valid=%d)  %.2fs (fwd=%.2fs bwd=%.2fs)  max||J||/sqrt(d)=%.3f  d_mean=%.2e",
                        p_idx + 1, total_prompts, seq_len, n_valid,
                        elapsed, t_fwd_amort, t_bwd, max_norm, rel_change,
                    )
                    if on_prompt_done is not None:
                        on_prompt_done(stats)

                    if mgr is not None and mgr.should_checkpoint(prompts_since_ckpt):
                        mgr.save_async(
                            jacobian_sum,
                            n_done=n_done,
                            next_idx=next_idx,
                            skip_first=skip_first,
                            d_model=D,
                            hc_count=self.cfg.hc_count,
                            wait_remote=False,
                        )
                        prompts_since_ckpt = 0
                    elif mgr is None and checkpoint_path is not None and checkpoint_every > 0 and prompts_since_ckpt >= checkpoint_every:
                        _save_legacy_pt()
                        prompts_since_ckpt = 0

                    if (
                        stop_at_delta is not None
                        and stop_at_delta > 0.0
                        and n_done >= min_prompts
                        and len(recent_deltas) >= stop_window
                        and all(d < stop_at_delta for d in recent_deltas[-stop_window:])
                    ):
                        logger.info(
                            "Converged early at n_done=%d (last %d mean_rel_change < %.4g); stopping cleanly.",
                            n_done,
                            stop_window,
                            stop_at_delta,
                        )
                        early_stopped = True
                        break

                cached_by_pidx.clear()
                batch_start = batch_end

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt caught at next_idx=%d (n_done=%d); flushing checkpoint safely before exit!", next_idx, n_done)
            if mgr is not None and n_done > 0 and next_idx > resumed_next_idx:
                mgr.save_async(
                    jacobian_sum,
                    n_done=n_done,
                    next_idx=next_idx,
                    skip_first=skip_first,
                    d_model=D,
                    hc_count=self.cfg.hc_count,
                    wait_remote=True,
                )
            elif checkpoint_path is not None and n_done > 0 and next_idx > resumed_next_idx:
                _save_legacy_pt()
            if owned_mgr and mgr is not None:
                mgr.close()
            raise

        # Final checkpoint flush (wait for Drive / HF Hub upload to finish!)
        if mgr is not None and n_done > 0:
            mgr.flush()
            if next_idx > resumed_next_idx:
                needs_final = (
                    prompts_since_ckpt > 0
                    or mgr._last_local_next_idx < next_idx
                    or (mgr.has_remote and mgr._last_synced_next_idx < next_idx)
                )
                if needs_final:
                    mgr.save_async(
                        jacobian_sum,
                        n_done=n_done,
                        next_idx=next_idx,
                        skip_first=skip_first,
                        d_model=D,
                        hc_count=self.cfg.hc_count,
                        wait_remote=True,
                    )
            if owned_mgr:
                mgr.close()
        elif checkpoint_path is not None and n_done > 0 and next_idx > resumed_next_idx:
            _save_legacy_pt()

        if n_done == 0:
            raise ValueError("No prompts were long enough to fit the Jacobian lens.")

        # Divide in-place to avoid allocating a duplicate 4.69 GiB fp32 dict on tight-RAM hosts!
        for l in src_layers:
            jacobian_sum[l].div_(float(n_done))
        return Qwen38JacobianLens(
            jacobians=jacobian_sum,
            n_prompts=n_done,
            d_model=D,
            hc_count=self.cfg.hc_count,
        )

    @torch.no_grad()
    def apply_lens(
        self,
        lens: Qwen38JacobianLens,
        input_ids: torch.Tensor,  # [1, T]
        *,
        layers: Sequence[int] | None = None,
        positions: Sequence[int] | None = None,
        use_jacobian: bool = True,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Apply fitted J-lens (or logit-lens baseline if `use_jacobian=False`) on `input_ids`.

        Returns `(lens_logits_by_layer, model_final_logits)`, each of shape `[n_positions, vocab_size]` on CPU.
        """
        req_layers = lens.source_layers if layers is None else sorted(set(layers))
        h_final, captured = self.forward_capture(input_ids, capture_layers=req_layers)

        def _sel(x: torch.Tensor) -> torch.Tensor:
            t0 = x[0]  # [T, ...]
            return t0 if positions is None else t0[list(positions)]

        # Unembed via CPU or GPU lm_head
        lm_w = self.model.lm_head.float()  # [vocab, D] on CPU (avoids extra 1.2 GiB VRAM peak)
        model_logits = F.linear(_sel(h_final).float().cpu(), lm_w)

        lens_logits: dict[int, torch.Tensor] = {}
        for l in req_layers:
            H_sel = _sel(captured[l])  # [P, hc_count * D]
            if use_jacobian:
                h_proj = lens.transport(H_sel, l).cpu()
            else:
                # Logit-lens baseline for 4-branch GR: pass H_l through final hyper_connection_mixer!
                h_proj, _, _ = gated_residual_fwd(H_sel.unsqueeze(0), self.model.hyper_connection_mixer)
                h_proj = h_proj[0].float().cpu()
            lens_logits[l] = F.linear(h_proj, lm_w)

        return lens_logits, model_logits
