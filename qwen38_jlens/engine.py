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
    decoder_layer_fwd,
    decoder_layer_vjp_inplace,
    gated_residual_fwd,
    gated_residual_vjp,
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
                assert isinstance(src_payload, MoeExpertPayload)
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
                assert isinstance(payload, MoeExpertPayload)
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


class Qwen38JLensEngine:
    """High-performance single-sweep Jacobian Lens fitter and inference applier for Qwen3.8-Flash-Next."""

    def __init__(self, model: LoadedQwen38Model) -> None:
        self.model = model
        self.cfg = model.config
        self.plan = model.plan
        self.device = model.device
        self.dtype = model.compute_dtype
        self.gdn_backend = GatedDeltaRuleBackend(preferred=self.plan.gdn_backend)
        self.streamer = AsyncMoEStreamer(model)

    def close(self) -> None:
        self.streamer.close()

    def _get_ple_embeddings(self, input_ids: torch.Tensor) -> dict[int, torch.Tensor]:
        """Gather `B=1` PLE n-gram embeddings on GPU for all PLE layers (takes <2 ms via mmap)."""
        out: dict[int, torch.Tensor] = {}
        for l_idx, provider in self.model.ple_providers.items():
            out[l_idx] = provider.lookup(input_ids, device=self.device).to(self.dtype)
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
        t0 = time.perf_counter()
        input_ids = input_ids.to(self.device)
        T = int(input_ids.shape[1])
        pos_mask = valid_position_mask(T, skip_first=skip_first)
        valid_pos = pos_mask.nonzero(as_tuple=True)[0].to(self.device)
        n_valid = int(valid_pos.numel())
        bwd_start = compute_aligned_bwd_start(T, skip_first, align=bwd_align)
        T_v = T - bwd_start
        valid_pos_local = valid_pos - bwd_start

        D = self.cfg.hidden_size
        hc = self.cfg.hc_count
        D_hc = D * hc
        n_layers = self.cfg.num_hidden_layers
        src_list = sorted(set(int(l) for l in source_layers))
        if not src_list or src_list[0] < -1 or src_list[-1] >= n_layers:
            raise ValueError(f"source_layers {src_list} out of valid range [-1, {n_layers - 1}]")
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
        ple_embs = self._get_ple_embeddings(input_ids)

        # ---- 1. Forward Sweep (0 -> L-1, B = 1) ----
        emb = F.embedding(input_ids, self.model.embed_tokens)
        H = emb.repeat(1, 1, hc).contiguous()

        # Save each layer's input H_{l-1} for layers >= min_src + 1 (only ~2.5 MB/layer!)
        layer_inputs: dict[int, torch.Tensor] = {}
        for l_idx, experts in self.streamer.iter_sweep(range(n_layers)):
            if l_idx > min_src:
                layer_inputs[l_idx] = H
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

        # Final hyper_connection_mixer forward (H_{L-1} -> h_final [1, T, D])
        _, _, mixer_saved = gated_residual_fwd(H, self.model.hyper_connection_mixer)
        mixer_saved_v = slice_gated_residual_saved(mixer_saved, bwd_start)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_fwd = time.perf_counter() - t0

        # ---- 2. Backward Sweep (L-1 -> min_src + 1, k_pass cotangents on [bwd_start:]!) ----
        t_bwd_start = time.perf_counter()
        jacobians = {l: torch.zeros((D, D_hc), dtype=torch.float32) for l in src_list}
        k_pass = min(self.plan.k_pass, D)
        bwd_layers = list(range(n_layers - 1, min_src, -1))

        for k_start in range(0, D, k_pass):
            k_end = min(k_start + k_pass, D)
            K_cur = k_end - k_start

            # Build initial one-hot cotangent at target h_final for valid positions on [bwd_start:],
            # then immediately backprop through hyper_connection_mixer -> G_{L-1} [K_cur, T_v, D_hc]!
            G = torch.zeros((K_cur, T_v, D_hc), device=self.device, dtype=self.dtype)
            d_target_chunk = torch.zeros((min(self.plan.moe_k_chunk, K_cur), T_v, D), device=self.device, dtype=self.dtype)
            k_sub_idx = torch.arange(d_target_chunk.shape[0], device=self.device)
            for c0 in range(0, K_cur, d_target_chunk.shape[0]):
                c1 = min(c0 + d_target_chunk.shape[0], K_cur)
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

            # Layer L-1 output H_{L-1} Jacobian!
            if (n_layers - 1) in src_set:
                rows = G[:, valid_pos_local, :].float().mean(dim=1).cpu()
                jacobians[n_layers - 1][k_start:k_end, :] = rows

            # Reverse layer sweep: L-1 -> min_src + 1
            for l_idx, experts in self.streamer.iter_sweep(bwd_layers):
                # Recompute ONE tiny B=1 forward of layer l_idx (~0.5 ms), keeping suffix-hit active
                # experts dequantized in BF16 when E_hit <= expert_dequant_chunk!
                _, saved_l = decoder_layer_fwd(
                    layer_inputs[l_idx],
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
                    rows = G[:, valid_pos_local, :].float().mean(dim=1).cpu()
                    jacobians[prev_layer][k_start:k_end, :] = rows

            del G

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_bwd = time.perf_counter() - t_bwd_start
        return jacobians, T, n_valid, t_fwd, t_bwd

    def fit(
        self,
        tokenized_prompts: Sequence[torch.Tensor],
        *,
        source_layers: Sequence[int] | None = None,
        include_embedding_layer: bool = False,
        skip_first: int = SKIP_FIRST_N_POSITIONS,
        bwd_align: int = 16,
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

        sqrt_d = math.sqrt(D)
        prompts_since_ckpt = 0
        try:
            for p_idx, ids in enumerate(tokenized_prompts):
                if p_idx < next_idx:
                    continue
                t0 = time.perf_counter()
                try:
                    per_J, seq_len, n_valid, t_fwd, t_bwd = self.jacobian_for_tokens(
                        ids, src_layers, skip_first=skip_first, bwd_align=bwd_align
                    )
                except ValueError as exc:
                    logger.warning("Skipping prompt %d: %s", p_idx, exc)
                    next_idx = p_idx + 1
                    continue

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
                elapsed = time.perf_counter() - t0

                stats = FitStats(
                    prompt_idx=p_idx,
                    seq_len=seq_len,
                    n_valid=n_valid,
                    elapsed_s=elapsed,
                    fwd_s=t_fwd,
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
                    p_idx + 1, len(tokenized_prompts), seq_len, n_valid,
                    elapsed, t_fwd, t_bwd, max_norm, rel_change,
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
                    break

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
