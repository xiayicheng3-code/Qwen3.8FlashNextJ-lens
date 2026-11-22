"""Best-in-class GPU kernels with hardware auto-dispatch and verified fallbacks.

Kernel stack:
  1. Gated DeltaNet (36/48 layers):
       - `T <= 256` (default J-lens `T = 128`): L2-Cached Exact WY (`l2_cached_wy`) — precomputes
         tiny L2-resident `[H_v, T, T]` WY matrices ONCE in `B = 1` forward and runs all `K`
         cotangents on suffix `[bwd_start:]` via 6 pure cuBLAS `bmm`s (~8x faster than FLA on SM80, 0s compile!)
       - `T > 256`: FlashQLA v0.1.3 (SM90/100/120) > FLA Triton (SM80) > exact chunked PyTorch
  2. FP8 128x128 Block-Wise Weight Dequantization (48 MoE layers):
       - Custom fused Triton 2D-block kernel (native FP8 PTX on SM89+; uint8+L1 LUT on SM80)
  3. Fused Triton VJP Kernels (zero FP32 HBM temps, `tl.int64` indexing for 2.94B-elem `G`):
       - `_rmsnorm_vjp_kernel` (with fused in-place `out_add += dx` directly into `G`)
       - `_rmsnorm_gated_vjp_kernel` (single-pass GDN `D_v = 128`)
       - `_hc_bwd_prep_kernel` & `_hc_inject_vjp_hc4_kernel` (single-read 4-branch Hyper-Connection VJP)
       - `_moe_gather_scale_kernel`, `_swiglu_vjp_pack_kernel` (2D-tiled), `_moe_scatter_add_tk_kernel` (zero-atomic)
       - `_l2norm_bwd_kernel` & `_causal_conv1d_silu_bwd_slice_kernel` (4-tap causal conv1d+SiLU VJP in `[K, T, C]`)
  4. Grouped MoE GEMM (48 MoE layers):
       - `torch.nn.functional.grouped_mm` / `torch._grouped_mm` (SM80+ BF16) + per-expert fallback
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Optional Triton kernels
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


# --------------------------------------------------------------------------- #
# 1. Custom Fused Triton Kernel: 128x128 Block-Wise FP8 -> BF16 Dequantization
# --------------------------------------------------------------------------- #
_FP8_E4M3_LUT_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_fp8_e4m3_lut(device: torch.device) -> torch.Tensor:
    """Cached 256-entry fp32 LUT (1 KB, L1-resident) for exact FP8 E4M3FN decode on SM80."""
    lut = _FP8_E4M3_LUT_CACHE.get(device)
    if lut is None:
        lut = (
            torch.arange(256, dtype=torch.uint8)
            .view(torch.float8_e4m3fn)
            .to(torch.float32)
            .to(device)
            .contiguous()
        )
        _FP8_E4M3_LUT_CACHE[device] = lut
    return lut


if _HAS_TRITON:

    @triton.jit
    def _dequant_fp8_block_kernel(
        w_ptr,
        s_ptr,
        out_ptr,
        stride_we,
        stride_wm,
        stride_wn,
        stride_se,
        stride_sm,
        stride_sn,
        stride_oe,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SCALE_M_RATIO: tl.constexpr,
    ):
        """SM89+ native hardware FP8 E4M3FN -> BF16 block-128 dequant kernel."""
        pid_e = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)

        s_m = pid_m // SCALE_M_RATIO
        scale = tl.load(s_ptr + pid_e * stride_se + s_m * stride_sm + pid_n * stride_sn).to(tl.float32)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        w_ptrs = w_ptr + pid_e * stride_we + offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn
        o_ptrs = out_ptr + pid_e * stride_oe + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

        w_fp8 = tl.load(w_ptrs)
        w_bf16 = (w_fp8.to(tl.float32) * scale).to(tl.bfloat16)
        tl.store(o_ptrs, w_bf16)

    @triton.jit
    def _dequant_fp8_u8_lut_block_kernel(
        w_u8_ptr,
        lut_ptr,
        s_ptr,
        out_ptr,
        stride_we,
        stride_wm,
        stride_wn,
        stride_se,
        stride_sm,
        stride_sn,
        stride_oe,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SCALE_M_RATIO: tl.constexpr,
    ):
        """SM80 (Ampere A100) exact FP8 E4M3FN -> BF16 block-128 dequant via raw uint8 + 1 KB L1 LUT.

        Triton rejects `fp8e4nv` pointer signatures on SM80 (no hardware FP8 PTX), so passing
        `w.view(torch.uint8)` + a 256-entry fp32 LUT achieves exact bit-for-bit decoding in a
        single HBM pass with zero temp allocations!
        """
        pid_e = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)

        s_m = pid_m // SCALE_M_RATIO
        scale = tl.load(s_ptr + pid_e * stride_se + s_m * stride_sm + pid_n * stride_sn).to(tl.float32)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        w_ptrs = w_u8_ptr + pid_e * stride_we + offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn
        o_ptrs = out_ptr + pid_e * stride_oe + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

        w_u8 = tl.load(w_ptrs).to(tl.int32)
        w_f32 = tl.load(lut_ptr + w_u8)
        w_bf16 = (w_f32 * scale).to(tl.bfloat16)
        tl.store(o_ptrs, w_bf16)

    @triton.jit
    def _swiglu_vjp_pack_kernel(
        da_ptr,      # [S, K, H]
        da_dg_ptr,   # [S, H]
        da_du_ptr,   # [S, H]
        out_ptr,     # [S, K, 2*H]
        S,
        K,
        H,
        stride_da_s,
        stride_da_k,
        stride_da_h,
        stride_g_s,
        stride_g_h,
        stride_o_s,
        stride_o_k,
        stride_o_h,
        BLOCK_K: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        """2D-tiled fused SwiGLU VJP + pack `[dG | dU]`: loads `da_dg/da_du` once per BLOCK_K cotangents."""
        pid_s = tl.program_id(0).to(tl.int64)
        pid_k = tl.program_id(1).to(tl.int64)
        pid_h = tl.program_id(2).to(tl.int64)

        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_k = offs_k < K
        mask_h = offs_h < H
        mask_kh = mask_k[:, None] & mask_h[None, :]

        g_base = pid_s * stride_g_s + offs_h.to(tl.int64) * stride_g_h
        dg_coeff = tl.load(da_dg_ptr + g_base, mask=mask_h, other=0.0).to(tl.float32)
        du_coeff = tl.load(da_du_ptr + g_base, mask=mask_h, other=0.0).to(tl.float32)

        da_ptrs = (
            da_ptr
            + pid_s * stride_da_s
            + offs_k[:, None].to(tl.int64) * stride_da_k
            + offs_h[None, :].to(tl.int64) * stride_da_h
        )
        da = tl.load(da_ptrs, mask=mask_kh, other=0.0).to(tl.float32)

        out_base = out_ptr + pid_s * stride_o_s + offs_k[:, None].to(tl.int64) * stride_o_k
        tl.store(out_base + offs_h[None, :].to(tl.int64) * stride_o_h, (da * dg_coeff[None, :]).to(tl.bfloat16), mask=mask_kh)
        tl.store(
            out_base + (H + offs_h[None, :]).to(tl.int64) * stride_o_h,
            (da * du_coeff[None, :]).to(tl.bfloat16),
            mask=mask_kh,
        )

    @triton.jit
    def _rmsnorm_vjp_kernel(
        dy_ptr,       # [K, R, D] bf16
        xh_ptr,       # [R, D] fp32
        rstd_ptr,     # [R] fp32
        w_ptr,        # [W_groups, D] bf16
        out_ptr,      # [K, R, D] bf16
        D,
        G_GROUPS,
        stride_dy_k,
        stride_dy_r,
        stride_xh_r,
        stride_w_g,
        stride_out_k,
        stride_out_r,
        HAS_GROUP_WEIGHT: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SINGLE_PASS: tl.constexpr,
        ADD_INPLACE: tl.constexpr,
    ):
        """Fused RMSNorm VJP (`1 + w` zero-centered) with optional in-place `out += dx` (0 FP32 temps)."""
        pid_k = tl.program_id(0).to(tl.int64)
        pid_r = tl.program_id(1).to(tl.int64)

        rstd = tl.load(rstd_ptr + pid_r).to(tl.float32)

        dy_base = dy_ptr + pid_k * stride_dy_k + pid_r * stride_dy_r
        xh_base = xh_ptr + pid_r * stride_xh_r
        if HAS_GROUP_WEIGHT:
            w_base = w_ptr + (pid_r % G_GROUPS) * stride_w_g
        else:
            w_base = w_ptr
        out_base = out_ptr + pid_k * stride_out_k + pid_r * stride_out_r

        if SINGLE_PASS:
            offs_d = tl.arange(0, BLOCK_D)
            mask = offs_d < D
            dy = tl.load(dy_base + offs_d, mask=mask, other=0.0).to(tl.float32)
            wf = 1.0 + tl.load(w_base + offs_d, mask=mask, other=0.0).to(tl.float32)
            xh = tl.load(xh_base + offs_d, mask=mask, other=0.0).to(tl.float32)
            g = dy * wf
            dot = tl.sum(g * xh, axis=0) / D
            dx = rstd * (g - xh * dot)
            if ADD_INPLACE:
                prev = tl.load(out_base + offs_d, mask=mask, other=0.0).to(tl.float32)
                tl.store(out_base + offs_d, (prev + dx).to(tl.bfloat16), mask=mask)
            else:
                tl.store(out_base + offs_d, dx.to(tl.bfloat16), mask=mask)
        else:
            dot_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
            for d0 in range(0, D, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D)
                mask = offs_d < D
                dy = tl.load(dy_base + offs_d, mask=mask, other=0.0).to(tl.float32)
                wf = 1.0 + tl.load(w_base + offs_d, mask=mask, other=0.0).to(tl.float32)
                xh = tl.load(xh_base + offs_d, mask=mask, other=0.0).to(tl.float32)
                dot_acc += dy * wf * xh
            dot = tl.sum(dot_acc, axis=0) / D
            for d0 in range(0, D, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D)
                mask = offs_d < D
                dy = tl.load(dy_base + offs_d, mask=mask, other=0.0).to(tl.float32)
                wf = 1.0 + tl.load(w_base + offs_d, mask=mask, other=0.0).to(tl.float32)
                xh = tl.load(xh_base + offs_d, mask=mask, other=0.0).to(tl.float32)
                dx = rstd * (dy * wf - xh * dot)
                if ADD_INPLACE:
                    prev = tl.load(out_base + offs_d, mask=mask, other=0.0).to(tl.float32)
                    tl.store(out_base + offs_d, (prev + dx).to(tl.bfloat16), mask=mask)
                else:
                    tl.store(out_base + offs_d, dx.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _rmsnorm_gated_vjp_kernel(
        dy_ptr,       # [K, R, D] bf16
        normed_ptr,   # [R, D] fp32
        dact_ptr,     # [R, D] fp32
        act_ptr,      # [R, D] fp32
        xh_ptr,       # [R, D] fp32
        rstd_ptr,     # [R] fp32
        w_ptr,        # [D] fp32
        dx_ptr,       # [K, R, D] bf16
        dgate_ptr,    # [K, R, D] bf16
        D,
        stride_dy_k,
        stride_dy_r,
        stride_r,
        BLOCK_D: tl.constexpr,
    ):
        """Single-pass register-resident VJP for GDN `RMSNormGated` (`D=128`)."""
        pid_k = tl.program_id(0).to(tl.int64)
        pid_r = tl.program_id(1).to(tl.int64)

        offs_d = tl.arange(0, BLOCK_D)
        mask = offs_d < D

        kr_off = pid_k * stride_dy_k + pid_r * stride_dy_r + offs_d
        r_off = pid_r * stride_r + offs_d

        dy = tl.load(dy_ptr + kr_off, mask=mask, other=0.0).to(tl.float32)
        normed = tl.load(normed_ptr + r_off, mask=mask, other=0.0).to(tl.float32)
        dact = tl.load(dact_ptr + r_off, mask=mask, other=0.0).to(tl.float32)
        act = tl.load(act_ptr + r_off, mask=mask, other=0.0).to(tl.float32)
        xh = tl.load(xh_ptr + r_off, mask=mask, other=0.0).to(tl.float32)
        wf = tl.load(w_ptr + offs_d, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + pid_r).to(tl.float32)

        tl.store(dgate_ptr + kr_off, (dy * normed * dact).to(tl.bfloat16), mask=mask)
        g = dy * act * wf
        dot = tl.sum(g * xh, axis=0) / D
        dx = rstd * (g - xh * dot)
        tl.store(dx_ptr + kr_off, dx.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _hc_bwd_prep_kernel(
        dm_ptr,       # [K, T, D] bf16 (arbitrary strides!)
        m4_ptr,       # [T, HC, D] bf16
        hn4_ptr,      # [T, HC, D] bf16
        dmu_ptr,      # [T, HC, D] bf16
        dhn_ptr,      # [K, T, HC, D] bf16
        du2_ptr,      # [K, T, HC, D] bf16
        D,
        INV_HC,
        stride_dm_k,
        stride_dm_t,
        stride_dm_d,
        stride_s_t,
        stride_s_b,
        stride_o_k,
        stride_o_t,
        stride_o_b,
        BLOCK_D: tl.constexpr,
    ):
        """Fused Hyper-Connection VJP prep: computes `d_h_norm` & `d_u2` in 1 pass directly from `d_mixed`."""
        pid_k = tl.program_id(0).to(tl.int64)
        pid_t = tl.program_id(1).to(tl.int64)
        pid_b = tl.program_id(2).to(tl.int64)

        dm_base = dm_ptr + pid_k * stride_dm_k + pid_t * stride_dm_t
        s_base = pid_t * stride_s_t + pid_b * stride_s_b
        o_base = pid_k * stride_o_k + pid_t * stride_o_t + pid_b * stride_o_b

        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask = offs_d < D
            dm = tl.load(dm_base + offs_d.to(tl.int64) * stride_dm_d, mask=mask, other=0.0).to(tl.float32) * INV_HC
            m = tl.load(m4_ptr + s_base + offs_d, mask=mask, other=0.0).to(tl.float32)
            hn = tl.load(hn4_ptr + s_base + offs_d, mask=mask, other=0.0).to(tl.float32)
            dmu = tl.load(dmu_ptr + s_base + offs_d, mask=mask, other=0.0).to(tl.float32)
            tl.store(dhn_ptr + o_base + offs_d, (dm * m).to(tl.bfloat16), mask=mask)
            tl.store(du2_ptr + o_base + offs_d, (dm * hn * dmu).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _hc_inject_vjp_hc4_kernel(
        g_ptr,        # [K, T, 4, D] bf16
        y_ptr,        # [T, D] bf16
        w_ptr,        # [T, 4] bf16
        dy_ptr,       # [K, T, D] or [T, K, D] bf16
        dw_ptr,       # [K, T, 4] bf16
        D,
        stride_gk,
        stride_gt,
        stride_gb,
        stride_yt,
        stride_wt,
        stride_dyk,
        stride_dyt,
        stride_dwk,
        stride_dwt,
        BLOCK_D: tl.constexpr,
    ):
        """Fused 4-branch injection VJP: 1 HBM read of `G` produces BOTH `d_y` and FP32-reduced `d_w`."""
        pid_k = tl.program_id(0).to(tl.int64)
        pid_t = tl.program_id(1).to(tl.int64)

        w_base = w_ptr + pid_t * stride_wt
        w0 = tl.load(w_base + 0).to(tl.float32)
        w1 = tl.load(w_base + 1).to(tl.float32)
        w2 = tl.load(w_base + 2).to(tl.float32)
        w3 = tl.load(w_base + 3).to(tl.float32)

        g_base = g_ptr + pid_k * stride_gk + pid_t * stride_gt
        y_base = y_ptr + pid_t * stride_yt
        dy_base = dy_ptr + pid_k * stride_dyk + pid_t * stride_dyt

        dw0_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        dw1_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        dw2_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        dw3_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask = offs_d < D
            y_t = tl.load(y_base + offs_d, mask=mask, other=0.0).to(tl.float32)
            g0 = tl.load(g_base + 0 * stride_gb + offs_d, mask=mask, other=0.0).to(tl.float32)
            g1 = tl.load(g_base + 1 * stride_gb + offs_d, mask=mask, other=0.0).to(tl.float32)
            g2 = tl.load(g_base + 2 * stride_gb + offs_d, mask=mask, other=0.0).to(tl.float32)
            g3 = tl.load(g_base + 3 * stride_gb + offs_d, mask=mask, other=0.0).to(tl.float32)

            dy = g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3
            tl.store(dy_base + offs_d, dy.to(tl.bfloat16), mask=mask)

            dw0_acc += g0 * y_t
            dw1_acc += g1 * y_t
            dw2_acc += g2 * y_t
            dw3_acc += g3 * y_t

        dw_base = dw_ptr + pid_k * stride_dwk + pid_t * stride_dwt
        tl.store(dw_base + 0, tl.sum(dw0_acc, axis=0).to(tl.bfloat16))
        tl.store(dw_base + 1, tl.sum(dw1_acc, axis=0).to(tl.bfloat16))
        tl.store(dw_base + 2, tl.sum(dw2_acc, axis=0).to(tl.bfloat16))
        tl.store(dw_base + 3, tl.sum(dw3_acc, axis=0).to(tl.bfloat16))

    @triton.jit
    def _moe_gather_scale_kernel(
        dy_tk_ptr,    # [T, K_total, D] bf16
        sub_tok_ptr,  # [n_rows] int64
        sub_w_ptr,    # [n_rows] bf16
        out_ptr,      # [n_rows, K_blk, D] bf16
        K_blk,
        D,
        k0,
        stride_tk_t,
        stride_tk_k,
        stride_o_n,
        stride_o_k,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused token-gather + routing-weight scale into `[n_rows, K_blk, D]` in 1 HBM pass."""
        pid_n = tl.program_id(0).to(tl.int64)
        pid_k = tl.program_id(1).to(tl.int64)
        pid_d = tl.program_id(2).to(tl.int64)

        tok = tl.load(sub_tok_ptr + pid_n).to(tl.int64)
        w = tl.load(sub_w_ptr + pid_n).to(tl.float32)

        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = (offs_k[:, None] < K_blk) & (offs_d[None, :] < D)

        src_ptrs = (
            dy_tk_ptr
            + tok * stride_tk_t
            + (k0 + offs_k[:, None].to(tl.int64)) * stride_tk_k
            + offs_d[None, :].to(tl.int64)
        )
        dst_ptrs = (
            out_ptr
            + pid_n * stride_o_n
            + offs_k[:, None].to(tl.int64) * stride_o_k
            + offs_d[None, :].to(tl.int64)
        )
        val = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32) * w
        tl.store(dst_ptrs, val.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _moe_scatter_add_tk_kernel(
        dx_tk_ptr,    # [T, K_total, D] bf16 (updated in-place!)
        dx_s_ptr,     # [n_rows, K_blk, D] bf16
        inv_perm_ptr, # [T, TOP_K] int64
        K_blk,
        D,
        k0,
        row_lo,
        n_rows,
        stride_tk_t,
        stride_tk_k,
        stride_s_n,
        stride_s_k,
        stride_inv_t,
        TOP_K: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Zero-atomic deterministic FP32-accum gather-add replacing BF16 `index_add_` (7.1s -> ~0.9s)."""
        pid_t = tl.program_id(0).to(tl.int64)
        pid_k = tl.program_id(1).to(tl.int64)
        pid_d = tl.program_id(2).to(tl.int64)

        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = (offs_k[:, None] < K_blk) & (offs_d[None, :] < D)

        dst_ptrs = (
            dx_tk_ptr
            + pid_t * stride_tk_t
            + (k0 + offs_k[:, None].to(tl.int64)) * stride_tk_k
            + offs_d[None, :].to(tl.int64)
        )
        acc = tl.load(dst_ptrs, mask=mask, other=0.0).to(tl.float32)

        inv_base = inv_perm_ptr + pid_t * stride_inv_t
        k_off_s = offs_k[:, None].to(tl.int64) * stride_s_k
        d_off_s = offs_d[None, :].to(tl.int64)

        for r in tl.static_range(TOP_K):
            slot = tl.load(inv_base + r).to(tl.int64) - row_lo
            valid = (slot >= 0) & (slot < n_rows)
            if valid:
                src_ptrs = dx_s_ptr + slot * stride_s_n + k_off_s + d_off_s
                acc = acc + tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)

        tl.store(dst_ptrs, acc.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _l2norm_bwd_kernel(
        dy_ptr,       # [K, R, D] bf16
        xn_ptr,       # [R, D] bf16
        rstd_ptr,     # [R] fp32
        dx_ptr,       # [K, R, D] bf16
        D,
        stride_k,
        stride_r,
        stride_xn_r,
        BLOCK_D: tl.constexpr,
    ):
        """Single-pass register-resident L2Norm VJP (`D=128`)."""
        pid_k = tl.program_id(0).to(tl.int64)
        pid_r = tl.program_id(1).to(tl.int64)
        offs_d = tl.arange(0, BLOCK_D)
        mask = offs_d < D

        kr_off = pid_k * stride_k + pid_r * stride_r + offs_d
        r_off = pid_r * stride_xn_r + offs_d

        dyf = tl.load(dy_ptr + kr_off, mask=mask, other=0.0).to(tl.float32)
        yf = tl.load(xn_ptr + r_off, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + pid_r).to(tl.float32)
        dot = tl.sum(dyf * yf, axis=0)
        tl.store(dx_ptr + kr_off, (rstd * (dyf - yf * dot)).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _causal_conv1d_silu_bwd_slice_kernel(
        in_ptr,       # [K, T, C_slice] bf16
        pre_ptr,      # [C_slice, T] bf16/fp32 (sliced view of pre_silu[0, c_lo:c_hi, :])
        w_ptr,        # [C_slice, 4] bf16 (sliced view of weight[c_lo:c_hi, :])
        out_ptr,      # [K, T, C_total] bf16 (offset to c_lo)
        T,
        C_slice,
        stride_ik,
        stride_it,
        stride_pc,
        stride_pt,
        stride_wc,
        stride_wk,
        stride_ok,
        stride_ot,
        BLOCK_C: tl.constexpr,
    ):
        """Fused 4-tap causal depthwise conv1d + SiLU VJP directly in `[K, T, C]` (zero transposes/conv_transpose1d)."""
        pid_k = tl.program_id(0).to(tl.int64)
        pid_t = tl.program_id(1).to(tl.int64)
        pid_c = tl.program_id(2).to(tl.int64)

        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C_slice
        offs_c64 = offs_c.to(tl.int64)

        acc = tl.zeros([BLOCK_C], dtype=tl.float32)
        in_k_base = in_ptr + pid_k * stride_ik
        for tap in tl.static_range(4):
            t_src = pid_t + tap
            if t_src < T:
                w_tap = tl.load(w_ptr + offs_c64 * stride_wc + (3 - tap) * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
                u = tl.load(pre_ptr + offs_c64 * stride_pc + t_src * stride_pt, mask=mask_c, other=0.0).to(tl.float32)
                sig = 1.0 / (1.0 + tl.exp(-u))
                d_silu = sig * (1.0 + u * (1.0 - sig))
                d_in = tl.load(in_k_base + t_src * stride_it + offs_c64, mask=mask_c, other=0.0).to(tl.float32)
                acc = acc + w_tap * d_in * d_silu

        tl.store(out_ptr + pid_k * stride_ok + pid_t * stride_ot + offs_c64, acc.to(tl.bfloat16), mask=mask_c)


def dequantize_fp8_blockwise(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
    out_dtype: torch.dtype = torch.bfloat16,
    use_triton: bool = True,
) -> torch.Tensor:
    """Dequantize `[E, M, N]` FP8 E4M3 weights with `[E, M//bm, N//bn]` block scales.

    Uses a single-pass Triton 2D-block kernel on CUDA (native FP8 PTX on SM89+ Hopper/Blackwell;
    raw uint8 + 1 KB L1 LUT on SM80 Ampere A100 where Triton does not support `fp8e4nv`).
    """
    if weight.element_size() > 1:
        return weight.to(out_dtype)

    bm, bn = block_size
    if weight.ndim == 2:
        return dequantize_fp8_blockwise(
            weight.unsqueeze(0), scale_inv.unsqueeze(0), block_size=block_size, out_dtype=out_dtype, use_triton=use_triton
        ).squeeze(0)

    E, M, N = weight.shape
    if (
        use_triton
        and _HAS_TRITON
        and weight.is_cuda
        and weight.dtype == torch.float8_e4m3fn
        and out_dtype == torch.bfloat16
        and bm == 128
        and bn == 128
        and M % 128 == 0
        and N % 128 == 0
    ):
        w_c = weight.contiguous()
        s_c = scale_inv.contiguous()
        out = torch.empty((E, M, N), device=weight.device, dtype=torch.bfloat16)
        block_m, block_n = 32, 128
        scale_m_ratio = bm // block_m  # 4
        grid = (E, M // block_m, N // block_n)
        cc = torch.cuda.get_device_capability(weight.device)
        if cc >= (8, 9):
            _dequant_fp8_block_kernel[grid](
                w_c,
                s_c,
                out,
                w_c.stride(0),
                w_c.stride(1),
                w_c.stride(2),
                s_c.stride(0),
                s_c.stride(1),
                s_c.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                SCALE_M_RATIO=scale_m_ratio,
                num_warps=4,
            )
        else:
            w_u8 = w_c.view(torch.uint8)
            lut = _get_fp8_e4m3_lut(weight.device)
            _dequant_fp8_u8_lut_block_kernel[grid](
                w_u8,
                lut,
                s_c,
                out,
                w_u8.stride(0),
                w_u8.stride(1),
                w_u8.stride(2),
                s_c.stride(0),
                s_c.stride(1),
                s_c.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                SCALE_M_RATIO=scale_m_ratio,
                num_warps=4,
            )
        return out

    # Universal PyTorch exact fallback (handles arbitrary shapes & CPU testing)
    sm, sn = scale_inv.shape[-2], scale_inv.shape[-1]
    if M % sm == 0 and N % sn == 0:
        b_m = M // sm
        b_n = N // sn
        w_f = weight.to(torch.float32).reshape(E, sm, b_m, sn, b_n)
        s_f = scale_inv.to(torch.float32).reshape(E, sm, 1, sn, 1)
        return (w_f * s_f).reshape(E, M, N).to(out_dtype)

    # Non-divisible edge case
    s_rep = scale_inv.to(torch.float32).repeat_interleave(bm, dim=-2).repeat_interleave(bn, dim=-1)[..., :M, :N]
    return (weight.to(torch.float32) * s_rep).to(out_dtype)


def swiglu_vjp_pack(
    dA: torch.Tensor,      # [S, K, H]
    da_dg: torch.Tensor,   # [S, H]
    da_du: torch.Tensor,   # [S, H]
    use_triton: bool = True,
) -> torch.Tensor:
    """Fused compute & pack `dGU = cat([dA * da_dg, dA * da_du], dim=-1)` -> `[S, K, 2*H]`."""
    S, K, H = dA.shape
    if use_triton and _HAS_TRITON and dA.is_cuda and dA.dtype == torch.bfloat16:
        dA_c = dA.contiguous()
        dg_c = da_dg.contiguous()
        du_c = da_du.contiguous()
        out = torch.empty((S, K, 2 * H), device=dA.device, dtype=torch.bfloat16)
        BLOCK_K = 16 if K >= 16 else 8
        BLOCK_H = 128 if H >= 128 else 64
        grid = (S, triton.cdiv(K, BLOCK_K), triton.cdiv(H, BLOCK_H))
        _swiglu_vjp_pack_kernel[grid](
            dA_c,
            dg_c,
            du_c,
            out,
            S,
            K,
            H,
            dA_c.stride(0),
            dA_c.stride(1),
            dA_c.stride(2),
            dg_c.stride(0),
            dg_c.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BLOCK_K=BLOCK_K,
            BLOCK_H=BLOCK_H,
            num_warps=4,
        )
        return out

    dG = dA * da_dg.unsqueeze(1)
    dU = dA * da_du.unsqueeze(1)
    return torch.cat([dG, dU], dim=-1)


def rmsnorm_vjp_fused(
    dy: torch.Tensor,            # [K, *orig_shape_no_batch] or [K, T, D]
    x_hat: torch.Tensor,         # [1, ..., (G,) gs] float32
    rstd: torch.Tensor,          # [1, ..., (G,) 1]  float32
    weight: torch.Tensor,        # [D] or [G * gs]
    group_size: int | None = None,
    *,
    out_add: torch.Tensor | None = None,
    use_triton: bool = True,
) -> torch.Tensor:
    """Exact batched-cotangent VJP of Qwen4ExpTextRMSNorm with fused Triton path & optional `out_add += dx`."""
    orig_dy_shape = dy.shape
    if (
        use_triton
        and _HAS_TRITON
        and dy.is_cuda
        and dy.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and (out_add is None or (out_add.is_cuda and out_add.dtype == torch.bfloat16 and out_add.is_contiguous()))
    ):
        K = orig_dy_shape[0]
        if group_size is not None:
            D = int(group_size)
            G_groups = orig_dy_shape[-1] // D
            dy_3d = dy.contiguous().reshape(K, -1, D)
            xh_2d = x_hat.contiguous().reshape(-1, D)
            rstd_1d = rstd.contiguous().reshape(-1)
            w_2d = weight.contiguous().reshape(G_groups, D)
            has_group_weight = G_groups > 1
        else:
            D = orig_dy_shape[-1]
            G_groups = orig_dy_shape[-2] if dy.ndim >= 3 else 1
            dy_3d = dy.contiguous().reshape(K, -1, D)
            xh_2d = x_hat.contiguous().reshape(-1, D)
            rstd_1d = rstd.contiguous().reshape(-1)
            w_2d = weight.contiguous().reshape(1, D)
            has_group_weight = False
        R = dy_3d.shape[1]
        if out_add is not None:
            out_3d = out_add.reshape(K, R, D)
            add_inplace = True
        else:
            out_3d = torch.empty_like(dy_3d)
            add_inplace = False
        block_d = min(512, triton.next_power_of_2(D))
        single_pass = D <= block_d
        _rmsnorm_vjp_kernel[(K, R)](
            dy_3d,
            xh_2d,
            rstd_1d,
            w_2d,
            out_3d,
            D,
            G_groups,
            dy_3d.stride(0),
            dy_3d.stride(1),
            xh_2d.stride(0),
            w_2d.stride(0),
            out_3d.stride(0),
            out_3d.stride(1),
            HAS_GROUP_WEIGHT=has_group_weight,
            BLOCK_D=block_d,
            SINGLE_PASS=single_pass,
            ADD_INPLACE=add_inplace,
            num_warps=4,
        )
        return out_add if out_add is not None else out_3d.reshape(orig_dy_shape)

    # Low-VRAM in-place PyTorch fallback (CPU unit tests & fp32 verification)
    g = dy.to(torch.float32, copy=True)
    wf = 1.0 + weight.float()
    if group_size is not None:
        g = g.reshape(*orig_dy_shape[:-1], -1, group_size)
        wf = wf.reshape(-1, group_size)
    g.mul_(wf)
    dot = (g * x_hat).mean(dim=-1, keepdim=True)
    g.addcmul_(x_hat, dot, value=-1.0)
    del dot
    g.mul_(rstd)
    dx = g.to(dy.dtype)
    del g
    if group_size is not None:
        dx = dx.reshape(orig_dy_shape)
    if out_add is not None:
        out_add.add_(dx)
        return out_add
    return dx


def rmsnorm_gated_vjp_fused(
    dy: torch.Tensor,  # [K, T, H_v, D_v]
    saved: dict[str, torch.Tensor],
    *,
    use_triton: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass fused Triton VJP for GDN `RMSNormGated` -> `(dx, dgate)`."""
    K, T, H_v, D_v = dy.shape
    if use_triton and _HAS_TRITON and dy.is_cuda and dy.dtype == torch.bfloat16 and D_v <= 512:
        R = T * H_v
        dy_3d = dy.contiguous().reshape(K, R, D_v)
        normed_2d = saved["normed"].contiguous().reshape(R, D_v)
        dact_2d = saved["d_act_g"].contiguous().reshape(R, D_v)
        act_2d = saved["act_g"].contiguous().reshape(R, D_v)
        xh_2d = saved["x_hat_f"].contiguous().reshape(R, D_v)
        rstd_1d = saved["rstd"].contiguous().reshape(R)
        wf_1d = saved["weight_f"].contiguous().reshape(D_v)
        dx_3d = torch.empty_like(dy_3d)
        dgate_3d = torch.empty_like(dy_3d)
        block_d = triton.next_power_of_2(D_v)
        _rmsnorm_gated_vjp_kernel[(K, R)](
            dy_3d,
            normed_2d,
            dact_2d,
            act_2d,
            xh_2d,
            rstd_1d,
            wf_1d,
            dx_3d,
            dgate_3d,
            D_v,
            dy_3d.stride(0),
            dy_3d.stride(1),
            normed_2d.stride(0),
            BLOCK_D=block_d,
            num_warps=4,
        )
        return dx_3d.reshape(K, T, H_v, D_v), dgate_3d.reshape(K, T, H_v, D_v)

    dyf = dy.to(torch.float32, copy=True)
    d_gate = (dyf * saved["normed"] * saved["d_act_g"]).to(dy.dtype)
    dyf.mul_(saved["act_g"]).mul_(saved["weight_f"])
    x_hat_f = saved["x_hat_f"]
    dot = (dyf * x_hat_f).mean(dim=-1, keepdim=True)
    dyf.addcmul_(x_hat_f, dot, value=-1.0)
    del dot
    dyf.mul_(saved["rstd"])
    dx = dyf.to(dy.dtype)
    del dyf
    return dx, d_gate


def hc_bwd_prep_fused(
    d_mixed: torch.Tensor,    # [K, T, D] (may be non-contiguous!)
    m_4: torch.Tensor,        # [1, T, hc, D]
    h_norm_4: torch.Tensor,   # [1, T, hc, D]
    dm_du2: torch.Tensor,     # [1, T, hc, D]
    inv_hc: float,
    *,
    use_triton: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused prep of `(d_h_norm, d_u2)` both `[K, T, hc * D]` in 1 pass directly from `d_mixed`."""
    K, T, D = d_mixed.shape
    hc = m_4.shape[2]
    if (
        use_triton
        and _HAS_TRITON
        and d_mixed.is_cuda
        and d_mixed.dtype == torch.bfloat16
        and m_4.dtype == torch.bfloat16
    ):
        m4_c = m_4[0].contiguous()
        hn4_c = h_norm_4[0].contiguous()
        dmu_c = dm_du2[0].contiguous()
        d_h_norm = torch.empty((K, T, hc, D), device=d_mixed.device, dtype=torch.bfloat16)
        d_u2 = torch.empty((K, T, hc, D), device=d_mixed.device, dtype=torch.bfloat16)
        block_d = min(512, triton.next_power_of_2(D))
        _hc_bwd_prep_kernel[(K, T, hc)](
            d_mixed,
            m4_c,
            hn4_c,
            dmu_c,
            d_h_norm,
            d_u2,
            D,
            float(inv_hc),
            d_mixed.stride(0),
            d_mixed.stride(1),
            d_mixed.stride(2),
            m4_c.stride(0),
            m4_c.stride(1),
            d_h_norm.stride(0),
            d_h_norm.stride(1),
            d_h_norm.stride(2),
            BLOCK_D=block_d,
            num_warps=4,
        )
        return d_h_norm.view(K, T, hc * D), d_u2.view(K, T, hc * D)

    d_prod = (d_mixed * inv_hc).unsqueeze(-2)
    d_h_norm = (d_prod * m_4).flatten(-2).contiguous()
    d_u2 = (d_prod * h_norm_4 * dm_du2).flatten(-2).contiguous()
    return d_h_norm, d_u2


def hc_inject_vjp_fused(
    G_sub: torch.Tensor,      # [K_sub, T_v, hc * D]
    y: torch.Tensor,          # [T_v, D]
    w: torch.Tensor,          # [T_v, hc]
    out_dy: torch.Tensor,     # [T_v, K_sub, D] if tk_layout else [K_sub, T_v, D]
    out_dw: torch.Tensor,     # [K_sub, T_v, hc]
    *,
    tk_layout: bool = False,
    use_triton: bool = True,
) -> None:
    """Single-read fused injection VJP computing BOTH `out_dy` and FP32-reduced `out_dw`."""
    K_sub, T_v, D_hc = G_sub.shape
    hc = w.shape[1]
    D = y.shape[1]
    if (
        use_triton
        and _HAS_TRITON
        and hc == 4
        and G_sub.is_cuda
        and G_sub.dtype == torch.bfloat16
        and G_sub.is_contiguous()
        and out_dy.stride(-1) == 1
        and out_dw.stride(-1) == 1
    ):
        g4 = G_sub.view(K_sub, T_v, 4, D)
        y_c = y.contiguous()
        w_c = w.contiguous()
        stride_dyk = out_dy.stride(1) if tk_layout else out_dy.stride(0)
        stride_dyt = out_dy.stride(0) if tk_layout else out_dy.stride(1)
        block_d = min(512, triton.next_power_of_2(D))
        _hc_inject_vjp_hc4_kernel[(K_sub, T_v)](
            g4,
            y_c,
            w_c,
            out_dy,
            out_dw,
            D,
            g4.stride(0),
            g4.stride(1),
            g4.stride(2),
            y_c.stride(0),
            w_c.stride(0),
            stride_dyk,
            stride_dyt,
            out_dw.stride(0),
            out_dw.stride(1),
            BLOCK_D=block_d,
            num_warps=4,
        )
        return

    g4 = G_sub.view(K_sub, T_v, hc, D)
    dy = (g4 * w[None, :, :, None]).sum(dim=2)
    if tk_layout:
        out_dy.copy_(dy.transpose(0, 1))
    else:
        out_dy.copy_(dy)
    out_dw.copy_((g4.float() * y.float()[None, :, None, :]).sum(dim=-1).to(G_sub.dtype))


def moe_gather_scale_fused(
    dY_TK: torch.Tensor,      # [T, K_total, D]
    sub_tok: torch.Tensor,    # [n_rows] int64
    sub_w: torch.Tensor,      # [n_rows] dtype
    k0: int,
    k1: int,
    *,
    use_triton: bool = True,
) -> torch.Tensor:
    """Fused token-gather + routing-weight scale -> `[n_rows, k1 - k0, D]` in 1 HBM pass."""
    n_rows = int(sub_tok.numel())
    K_blk = k1 - k0
    D = dY_TK.shape[2]
    if (
        use_triton
        and _HAS_TRITON
        and dY_TK.is_cuda
        and dY_TK.dtype == torch.bfloat16
        and dY_TK.stride(2) == 1
    ):
        sub_tok_c = sub_tok.contiguous()
        sub_w_c = sub_w.contiguous()
        out = torch.empty((n_rows, K_blk, D), device=dY_TK.device, dtype=torch.bfloat16)
        block_k = 16 if K_blk >= 16 else 8
        block_d = 128 if D >= 128 else 64
        grid = (n_rows, triton.cdiv(K_blk, block_k), triton.cdiv(D, block_d))
        _moe_gather_scale_kernel[grid](
            dY_TK,
            sub_tok_c,
            sub_w_c,
            out,
            K_blk,
            D,
            k0,
            dY_TK.stride(0),
            dY_TK.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            num_warps=4,
        )
        return out

    dy_s = dY_TK[sub_tok, k0:k1, :]
    dy_s.mul_(sub_w[:, None, None])
    return dy_s


def moe_scatter_add_tk_fused(
    dX_TK: torch.Tensor,        # [T, K_total, D] (updated in-place!)
    dX_s: torch.Tensor,         # [n_rows, K_blk, D]
    inv_perm_2d: torch.Tensor,  # [T, top_k] int64
    row_lo: int,
    k0: int,
    k1: int,
    *,
    sub_tok: torch.Tensor | None = None,
    use_triton: bool = True,
) -> None:
    """Zero-atomic deterministic FP32-accum scatter-add into `dX_TK[:, k0:k1, :]`."""
    T, _, D = dX_TK.shape
    n_rows, K_blk, _ = dX_s.shape
    top_k = inv_perm_2d.shape[1]
    if (
        use_triton
        and _HAS_TRITON
        and dX_TK.is_cuda
        and dX_TK.dtype == torch.bfloat16
        and dX_s.dtype == torch.bfloat16
        and dX_TK.stride(2) == 1
        and dX_s.stride(2) == 1
    ):
        inv_c = inv_perm_2d.contiguous()
        block_k = 16 if K_blk >= 16 else 8
        block_d = 128 if D >= 128 else 64
        grid = (T, triton.cdiv(K_blk, block_k), triton.cdiv(D, block_d))
        _moe_scatter_add_tk_kernel[grid](
            dX_TK,
            dX_s,
            inv_c,
            K_blk,
            D,
            k0,
            row_lo,
            n_rows,
            dX_TK.stride(0),
            dX_TK.stride(1),
            dX_s.stride(0),
            dX_s.stride(1),
            inv_c.stride(0),
            TOP_K=top_k,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            num_warps=4,
        )
        return

    if row_lo == 0 and n_rows == T * top_k:
        for r in range(top_k):
            dX_TK[:, k0:k1, :].add_(dX_s[inv_perm_2d[:, r]])
    else:
        assert sub_tok is not None
        dX_TK[:, k0:k1, :].index_add_(0, sub_tok, dX_s)


# --------------------------------------------------------------------------- #
# 2. Grouped MoE GEMM Dispatcher (CUTLASS/cuBLAS grouped_mm + safe fallback)
# --------------------------------------------------------------------------- #
def _has_native_cuda_grouped_mm(device: torch.device, dtype: torch.dtype) -> bool:
    if device.type != "cuda" or dtype != torch.bfloat16:
        return False
    cc = torch.cuda.get_device_capability(device)
    if hasattr(torch.nn.functional, "grouped_mm"):
        return cc >= (8, 0)
    if hasattr(torch, "_grouped_mm"):
        # torch>=2.9 supports SM80+; torch 2.8 supports SM90
        parts = torch.__version__.split("+")[0].split(".")
        ver = (int(parts[0]), int(parts[1]))
        return cc >= (8, 0) if ver >= (2, 9) else cc[0] >= 9
    return False


def grouped_mm(
    mat_a: torch.Tensor,    # [M_total, K_in]
    mat_b: torch.Tensor,    # [E, K_in, N_out]
    offs: torch.Tensor,     # [E] int32 cumulative end offsets
    offs_list: list[int] | None = None,
) -> torch.Tensor:
    """Fast grouped GEMM: `out[start_e:end_e] = mat_a[start_e:end_e] @ mat_b[e]`.

    Dispatches to CUTLASS/cuBLAS `torch.nn.functional.grouped_mm` / `torch._grouped_mm`
    on CUDA SM80+ (A100, H100, Blackwell G4) when inputs are BF16, with automatic safe
    fallback if a driver/torch build rejects unaligned `B=1` forward offsets.
    """
    if _has_native_cuda_grouped_mm(mat_a.device, mat_a.dtype) and mat_b.dtype == mat_a.dtype:
        offs_i32 = offs.to(dtype=torch.int32, device=mat_a.device)
        try:
            if hasattr(torch.nn.functional, "grouped_mm"):
                return torch.nn.functional.grouped_mm(mat_a, mat_b, offs=offs_i32)
            return torch._grouped_mm(mat_a, mat_b, offs=offs_i32)
        except RuntimeError as exc:
            logger.debug("Native grouped_mm declined shape %s (%s); using per-expert mm fallback.", tuple(mat_a.shape), exc)

    # Clean fallback (CPU unit tests / fp32 verification / unaligned edge cases)
    ends = offs.tolist() if offs_list is None else offs_list
    out = torch.empty((mat_a.size(0), mat_b.size(2)), device=mat_a.device, dtype=mat_a.dtype)
    start = 0
    for e, end in enumerate(ends):
        if end > start:
            torch.mm(mat_a[start:end], mat_b[e], out=out[start:end])
        start = end
    return out


# --------------------------------------------------------------------------- #
# 3. Gated DeltaNet (GDN) Kernel Dispatcher: FlashQLA > FLA Triton > PyTorch
# --------------------------------------------------------------------------- #
def l2norm_torch(x: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """L2 normalization matching FLA/Qwen4Exp: returns `(x_norm, rstd)` in fp32."""
    xf = x.float()
    rstd = torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)
    return (xf * rstd).to(x.dtype), rstd


def l2norm_bwd_torch(x_norm: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """Exact VJP of `y = x * rstd` given saved `x_norm = x * rstd` and `rstd`."""
    yf = x_norm.float()
    dyf = dy.float()
    dot = (dyf * yf).sum(dim=-1, keepdim=True)
    return (rstd * (dyf - yf * dot)).to(dy.dtype)


def l2norm_bwd_fused(
    x_norm: torch.Tensor,  # [1, T, H_k, D]
    rstd: torch.Tensor,    # [1, T, H_k, 1] fp32
    dy: torch.Tensor,      # [K, T, H_k, D]
    *,
    use_triton: bool = True,
) -> torch.Tensor:
    """Single-pass register-resident Triton VJP of L2 normalization (`D=128`)."""
    K, T, H_k, D = dy.shape
    if (
        use_triton
        and _HAS_TRITON
        and dy.is_cuda
        and dy.dtype == torch.bfloat16
        and x_norm.dtype == torch.bfloat16
        and D <= 512
    ):
        R = T * H_k
        dy_3d = dy.contiguous().reshape(K, R, D)
        xn_2d = x_norm.contiguous().reshape(R, D)
        rstd_1d = rstd.contiguous().reshape(R)
        dx_3d = torch.empty_like(dy_3d)
        block_d = triton.next_power_of_2(D)
        _l2norm_bwd_kernel[(K, R)](
            dy_3d,
            xn_2d,
            rstd_1d,
            dx_3d,
            D,
            dy_3d.stride(0),
            dy_3d.stride(1),
            xn_2d.stride(0),
            BLOCK_D=block_d,
            num_warps=4,
        )
        return dx_3d.reshape(K, T, H_k, D)
    return l2norm_bwd_torch(x_norm, rstd, dy)


def wy_gdn_fwd_single(
    q: torch.Tensor,     # [1, T, H_k, D]
    k: torch.Tensor,     # [1, T, H_k, D]
    v: torch.Tensor,     # [1, T, H_v, D]
    g: torch.Tensor,     # [1, T, H_v] float32 log-decay <= 0
    beta: torch.Tensor,  # [1, T, H_v]
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Exact `B=1` WY representation of Gated Delta Rule (`O = A @ L^{-1} @ (beta * v)`).

    For `T <= 256`, precomputes the tiny L2-resident `[H_v, T, T]` matrices `A`, `L_strict`,
    `W_eff = A @ L^{-1}`, and `U = L^{-1} @ v_beta` ONCE in FP32 (`< 0.3 ms`), turning all
    `K = 2560` cotangent backward passes into 6 pure Tensor-Core-saturated cuBLAS `torch.bmm`s
    with ZERO recurrent state recomputation and ZERO FLA Triton autotuning!
    """
    _, T, H_k, D = q.shape
    H_v = v.shape[2]
    rep = H_v // H_k
    scale = D ** -0.5
    dtype = q.dtype

    q_n, q_rstd = l2norm_torch(q)
    k_n, k_rstd = l2norm_torch(k)

    qf = q_n[0].transpose(0, 1).float().repeat_interleave(rep, dim=0) * scale  # [H_v, T, D]
    kf = k_n[0].transpose(0, 1).float().repeat_interleave(rep, dim=0)          # [H_v, T, D]
    vf = v[0].transpose(0, 1).float()                                          # [H_v, T, D]
    bf = beta[0].transpose(0, 1).float()                                       # [H_v, T]
    gf = g[0].transpose(0, 1).float()                                          # [H_v, T]

    cum_g = gf.cumsum(dim=-1)  # [H_v, T]
    triu1 = torch.ones((T, T), dtype=torch.bool, device=q.device).triu(1)
    triu0 = torch.ones((T, T), dtype=torch.bool, device=q.device).triu(0)
    diff_g = cum_g.unsqueeze(2) - cum_g.unsqueeze(1)  # [H_v, T, T]
    pair_lo = diff_g.masked_fill(triu1, float("-inf")).exp()
    pair_st = diff_g.masked_fill(triu0, float("-inf")).exp()

    k_beta = kf * bf.unsqueeze(-1)
    v_beta = vf * bf.unsqueeze(-1)

    A = torch.bmm(qf, kf.transpose(1, 2)) * pair_lo
    L_strict = torch.bmm(k_beta, kf.transpose(1, 2)) * pair_st
    eye = torch.eye(T, dtype=torch.float32, device=q.device).unsqueeze(0).expand(H_v, -1, -1)
    L = eye + L_strict
    U = torch.linalg.solve_triangular(L, v_beta, upper=False, unitriangular=True)
    L_inv = torch.linalg.solve_triangular(L, eye, upper=False, unitriangular=True)
    W_eff = torch.bmm(A, L_inv)
    O_out = torch.bmm(A, U).transpose(0, 1).unsqueeze(0).to(dtype)

    return O_out, {
        "mode": "l2_cached_wy",
        "H_k": H_k,
        "H_v": H_v,
        "T": T,
        "D": D,
        "scale": scale,
        "q_n": q_n,
        "q_rstd": q_rstd,
        "k_n": k_n,
        "k_rstd": k_rstd,
        "qf_b": qf.to(dtype).contiguous(),
        "kf_b": kf.to(dtype).contiguous(),
        "vf_b": vf.to(dtype).contiguous(),
        "bf_b": bf.to(dtype).contiguous(),
        "kb_b": k_beta.to(dtype).contiguous(),
        "U_b": U.to(dtype).contiguous(),
        "W_eff_b": W_eff.to(dtype).contiguous(),
        "pair_lo_b": pair_lo.to(dtype).contiguous(),
        "pair_st_b": pair_st.to(dtype).contiguous(),
        "A_f": A,
        "L_strict_f": L_strict,
    }


def wy_gdn_bwd_batched(
    saved: dict[str, Any],
    do: torch.Tensor,  # [K_sub, T_v, H_v, D]
    *,
    bwd_start: int = 0,
    use_triton: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact L2-cached WY GDN VJP on suffix `[bwd_start:]` via 6 pure cuBLAS `torch.bmm`s."""
    K_sub, Tv, H_v, D = do.shape
    H_k = saved["H_k"]
    T = saved["T"]
    rep = H_v // H_k
    s = int(bwd_start)
    assert Tv == T - s, f"Expected T_v={T - s}, got {Tv}"
    dtype = do.dtype

    dO_h = do.permute(2, 0, 1, 3).contiguous()  # [H_v, K_sub, Tv, D]
    W_vv_b = saved["W_eff_b"][:, s:, s:] if s > 0 else saved["W_eff_b"]
    U_b = saved["U_b"]
    kf_b = saved["kf_b"]
    qf_v_b = saved["qf_b"][:, s:, :] if s > 0 else saved["qf_b"]
    kb_v_b = saved["kb_b"][:, s:, :] if s > 0 else saved["kb_b"]
    vf_v_b = saved["vf_b"][:, s:, :] if s > 0 else saved["vf_b"]
    kf_v_b = kf_b[:, s:, :] if s > 0 else kf_b
    p_lo_b = (saved["pair_lo_b"][:, s:, :] if s > 0 else saved["pair_lo_b"]).unsqueeze(1)
    p_st_b = (saved["pair_st_b"][:, s:, :] if s > 0 else saved["pair_st_b"]).unsqueeze(1)
    A_va_f = (saved["A_f"][:, s:, :] if s > 0 else saved["A_f"]).unsqueeze(1)
    L_va_f = (saved["L_strict_f"][:, s:, :] if s > 0 else saved["L_strict_f"]).unsqueeze(1)

    # 1. d_v_beta[s:] = W_eff[s:, s:]^T @ dO[s:]
    dO_for_v = dO_h.permute(0, 2, 1, 3).reshape(H_v, Tv, K_sub * D)
    d_vb_v = torch.bmm(W_vv_b.transpose(1, 2), dO_for_v).view(H_v, Tv, K_sub, D).permute(0, 2, 1, 3).contiguous()
    del dO_for_v

    # 2. Branch A: dA = dO @ U^T -> dQ, dK_A, and A-contribution to d_cum_g
    dA_va = torch.bmm(dO_h.view(H_v, K_sub * Tv, D), U_b.transpose(1, 2)).view(H_v, K_sub, Tv, T)
    del dO_h
    d_lp_a = dA_va.float() * A_va_f
    d_cum_v = d_lp_a.sum(dim=-1) - d_lp_a[:, :, :, s:].sum(dim=-2)  # [H_v, K_sub, Tv] fp32
    del d_lp_a
    dS_A = dA_va * p_lo_b
    del dA_va
    dQ_v = torch.bmm(dS_A.view(H_v, K_sub * Tv, T), kf_b).view(H_v, K_sub, Tv, D)
    dK_v = torch.bmm(
        dS_A[:, :, :, s:].transpose(-1, -2).reshape(H_v, K_sub * Tv, Tv), qf_v_b
    ).view(H_v, K_sub, Tv, D)
    del dS_A

    # 3. Branch L: dL = -(d_v_beta @ U^T) -> d_k_beta, dK_L, and L-contribution to d_cum_g
    dL_va = torch.bmm(d_vb_v.view(H_v, K_sub * Tv, D), U_b.transpose(1, 2)).view(H_v, K_sub, Tv, T).neg_()
    d_lp_l = dL_va.float() * L_va_f
    d_cum_v.add_(d_lp_l.sum(dim=-1)).sub_(d_lp_l[:, :, :, s:].sum(dim=-2))
    del d_lp_l
    dS_L = dL_va * p_st_b
    del dL_va
    d_kb_v = torch.bmm(dS_L.view(H_v, K_sub * Tv, T), kf_b).view(H_v, K_sub, Tv, D)
    dK_v.add_(
        torch.bmm(
            dS_L[:, :, :, s:].transpose(-1, -2).reshape(H_v, K_sub * Tv, Tv), kb_v_b
        ).view(H_v, K_sub, Tv, D)
    )
    del dS_L

    # 4. dV, d_beta, dK += d_k_beta * beta, dQ *= scale, d_g = reverse_cumsum(d_cum_v)
    b_v = (saved["bf_b"][:, s:] if s > 0 else saved["bf_b"]).unsqueeze(1).unsqueeze(-1)
    dV_v = d_vb_v * b_v
    d_beta_v = (d_vb_v * vf_v_b.unsqueeze(1)).sum(dim=-1, dtype=torch.float32)
    del d_vb_v
    d_beta_v.add_((d_kb_v * kf_v_b.unsqueeze(1)).sum(dim=-1, dtype=torch.float32))
    dK_v.add_(d_kb_v * b_v)
    del d_kb_v
    dQ_v.mul_(saved["scale"])
    d_g_v = d_cum_v.flip(-1).cumsum(dim=-1).flip(-1)
    del d_cum_v

    # 5. Reduce GVA heads H_v -> H_k and apply fused L2Norm VJP
    dQ_k = dQ_v.view(H_k, rep, K_sub, Tv, D).sum(dim=1).permute(1, 2, 0, 3)
    del dQ_v
    dK_k = dK_v.view(H_k, rep, K_sub, Tv, D).sum(dim=1).permute(1, 2, 0, 3)
    del dK_v
    q_n_s = saved["q_n"][:, s:] if s > 0 else saved["q_n"]
    q_rstd_s = saved["q_rstd"][:, s:] if s > 0 else saved["q_rstd"]
    k_n_s = saved["k_n"][:, s:] if s > 0 else saved["k_n"]
    k_rstd_s = saved["k_rstd"][:, s:] if s > 0 else saved["k_rstd"]

    dQ_raw = l2norm_bwd_fused(q_n_s, q_rstd_s, dQ_k, use_triton=use_triton)
    del dQ_k
    dK_raw = l2norm_bwd_fused(k_n_s, k_rstd_s, dK_k, use_triton=use_triton)
    del dK_k
    return (
        dQ_raw,
        dK_raw,
        dV_v.permute(1, 2, 0, 3).contiguous(),
        d_g_v.permute(1, 2, 0).contiguous(),
        d_beta_v.permute(1, 2, 0).to(dtype).contiguous(),
    )


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    """Exact reference chunked Gated Delta Rule matching `modeling_qwen4_exp.py`."""
    initial_dtype = query.dtype
    batch_size, sequence_length, num_k_heads, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]

    if num_k_heads != num_v_heads:
        rep = num_v_heads // num_k_heads
        query = query.repeat_interleave(rep, dim=2)
        key = key.repeat_interleave(rep, dim=2)

    q, k, v, b, decay = [
        x.transpose(1, 2).to(torch.float32).contiguous() for x in (query, key, value, beta, g)
    ]
    if use_qk_l2norm_in_kernel:
        q = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)
    q = q * (k_head_dim ** -0.5)

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    if pad_size:
        q = F.pad(q, (0, 0, 0, pad_size))
        k = F.pad(k, (0, 0, 0, pad_size))
        v = F.pad(v, (0, 0, 0, pad_size))
        b = F.pad(b, (0, pad_size))
        decay = F.pad(decay, (0, pad_size))

    num_chunks = (sequence_length + pad_size) // chunk_size
    v_beta = v * b.unsqueeze(-1)
    k_beta = k * b.unsqueeze(-1)

    q, k, k_beta, v_beta = [
        x.reshape(batch_size, num_v_heads, num_chunks, chunk_size, x.shape[-1])
        for x in (q, k, k_beta, v_beta)
    ]
    decay = decay.reshape(batch_size, num_v_heads, num_chunks, chunk_size)

    strictly_upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device).triu(1)
    cum_decay = decay.cumsum(dim=3)
    pairwise = (cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)).masked_fill(strictly_upper, float("-inf")).exp()

    ut_system = (k_beta @ k.transpose(-1, -2)) * pairwise
    intra_attn = (q @ k.transpose(-1, -2)) * pairwise
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_k_beta, upper=False, unitriangular=True)

    state = torch.zeros((batch_size, num_v_heads, k_head_dim, v_head_dim), dtype=torch.float32, device=q.device)
    q_dec = q * cum_decay.exp().unsqueeze(-1)
    k_dec = k * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    out_chunks = []
    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
        inter = q_dec[:, :, i] @ state
        out_chunks.append(inter + intra_attn[:, :, i] @ v_new)
        state = state * chunk_decay[:, :, i] + k_dec[:, :, i].transpose(-1, -2) @ v_new

    core_out = torch.cat(out_chunks, dim=2)[:, :, :sequence_length]
    return core_out.transpose(1, 2).contiguous().to(initial_dtype)


class GatedDeltaRuleBackend:
    """Unified GDN forward + batched-cotangent backward backend with startup parity verification.

    Automatically dispatches:
      - `T <= wy_max_seq_len` (default 256): **L2-Cached Exact WY (`l2_cached_wy`)** — precomputes
        the tiny `[H_v, T, T]` WY matrices ONCE in `B=1` forward and runs all `K` cotangents via
        6 pure cuBLAS `torch.bmm`s (~10x faster than FLA on SM80, zero autotune compile!).
      - `T > wy_max_seq_len`: FlashQLA (SM90/100/120) > FLA Triton (SM80) > PyTorch chunked.
    """

    def __init__(
        self,
        preferred: str = "auto",
        verify_on_init: bool = True,
        *,
        wy_max_seq_len: int = 256,
        max_seq_len: int | None = None,
    ) -> None:
        self.name = "torch"
        self.wy_max_seq_len = int(wy_max_seq_len)
        self._flash_qla: dict[str, Any] | None = None
        self._fla: dict[str, Any] | None = None
        self._use_direct = True
        self._init_backend(preferred, verify_on_init=verify_on_init, max_seq_len=max_seq_len)

    def _init_backend(
        self,
        preferred: str,
        verify_on_init: bool = True,
        max_seq_len: int | None = None,
    ) -> None:
        if preferred == "torch":
            self.wy_max_seq_len = 0
            self.name = "torch"
            return

        if preferred == "l2_cached_wy" or (
            max_seq_len is not None and self.wy_max_seq_len > 0 and max_seq_len <= self.wy_max_seq_len
        ):
            self.name = "l2_cached_wy"
            self._use_direct = True
            if verify_on_init and torch.cuda.is_available():
                assert self._self_test_gva(force_wy=True), "L2-cached WY GDN self-test failed!"
                logger.info("GDN backend verified: L2-cached exact WY cuBLAS bmm (T <= %d, 0s compile)", self.wy_max_seq_len)
            return

        if not torch.cuda.is_available():
            self.name = "l2_cached_wy" if self.wy_max_seq_len > 0 else "torch"
            return

        cc = torch.cuda.get_device_capability(0)
        order = (
            ["flash_qla", "fla", "torch"]
            if preferred in ("auto", "flash_qla")
            else ["fla", "flash_qla", "torch"]
        )

        for cand in order:
            if cand == "flash_qla" and cc in ((9, 0), (10, 0), (10, 3), (12, 0), (12, 1)):
                try:
                    import flash_qla
                    from flash_qla.ops.gated_delta_rule.chunk import (
                        chunk_gated_delta_rule_fwd,
                        chunk_gated_delta_rule_bwd,
                    )
                    from flash_qla.utils import l2norm_fwd, l2norm_bwd

                    self._flash_qla = {
                        "high": flash_qla.chunk_gated_delta_rule,
                        "fwd": chunk_gated_delta_rule_fwd,
                        "bwd": chunk_gated_delta_rule_bwd,
                        "l2_fwd": l2norm_fwd,
                        "l2_bwd": l2norm_bwd,
                    }
                    self.name = "flash_qla"
                    self._use_direct = True
                    if not verify_on_init or self._self_test_gva(force_wy=False):
                        logger.info("GDN backend verified: FlashQLA direct (SM%d%d TileLang)", cc[0], cc[1])
                        return
                    self._use_direct = False
                    if self._self_test_gva(force_wy=False):
                        logger.info("GDN backend verified: FlashQLA high-level (SM%d%d TileLang)", cc[0], cc[1])
                        return
                except Exception as exc:
                    logger.warning("FlashQLA unavailable (%s); trying next GDN backend.", exc)

            if cand == "fla" and cc >= (8, 0):
                try:
                    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
                    from fla.ops.gated_delta_rule.chunk import (
                        chunk_gated_delta_rule_fwd,
                        chunk_gated_delta_rule_bwd,
                    )
                    from fla.modules.norm.l2norm import l2norm_fwd, l2norm_bwd

                    self._fla = {
                        "high": chunk_gated_delta_rule,
                        "fwd": chunk_gated_delta_rule_fwd,
                        "bwd": chunk_gated_delta_rule_bwd,
                        "l2_fwd": l2norm_fwd,
                        "l2_bwd": l2norm_bwd,
                    }
                    self.name = "fla"
                    self._use_direct = True
                    if not verify_on_init or self._self_test_gva(force_wy=False):
                        logger.info("GDN backend verified: FLA Triton direct (SM%d%d)", cc[0], cc[1])
                        return
                    self._use_direct = False
                    if self._self_test_gva(force_wy=False):
                        logger.info("GDN backend verified: FLA Triton high-level (SM%d%d)", cc[0], cc[1])
                        return
                except Exception as exc:
                    logger.warning("FLA unavailable (%s); falling back to PyTorch chunked GDN.", exc)

        self.name = "torch"

    def _self_test_gva(self, *, force_wy: bool = False) -> bool:
        """1-ms numerical parity smoke test on Qwen3.8's exact 3:1 GVA head ratio & D=128."""
        prev_wy = self.wy_max_seq_len
        if not force_wy:
            self.wy_max_seq_len = 0
        try:
            dev = torch.device("cuda:0")
            q = torch.randn(1, 64, 2, 128, device=dev, dtype=torch.bfloat16) * 0.2
            k = torch.randn(1, 64, 2, 128, device=dev, dtype=torch.bfloat16) * 0.2
            v = torch.randn(1, 64, 6, 128, device=dev, dtype=torch.bfloat16) * 0.2
            g = -F.softplus(torch.randn(1, 64, 6, device=dev, dtype=torch.float32))
            beta = torch.sigmoid(torch.randn(1, 64, 6, device=dev, dtype=torch.bfloat16))
            do = torch.randn(2, 64, 6, 128, device=dev, dtype=torch.bfloat16) * 0.2

            o_fast, saved = self.forward_single(q, k, v, g, beta)
            dq, dk, dv, dg, db = self.backward_batched(saved, do)

            with torch.enable_grad():
                q_r = q.expand(2, -1, -1, -1).clone().requires_grad_(True)
                k_r = k.expand(2, -1, -1, -1).clone().requires_grad_(True)
                v_r = v.expand(2, -1, -1, -1).clone().requires_grad_(True)
                g_r = g.expand(2, -1, -1).clone().requires_grad_(True)
                b_r = beta.expand(2, -1, -1).clone().requires_grad_(True)
                o_r = torch_chunk_gated_delta_rule(q_r, k_r, v_r, g_r, b_r, use_qk_l2norm_in_kernel=True)
                dq_r, dk_r, dv_r, dg_r, db_r = torch.autograd.grad(o_r, (q_r, k_r, v_r, g_r, b_r), grad_outputs=do)

            ok = (
                (o_fast.float() - o_r[:1].float()).abs().max().item() < 5e-2
                and (dq.float() - dq_r.float()).abs().max().item() < 1e-1
                and (dv.float() - dv_r.float()).abs().max().item() < 1e-1
                and torch.isfinite(dg).all().item()
                and torch.isfinite(db).all().item()
            )
            return bool(ok)
        except Exception as exc:
            logger.warning("GDN kernel self-test failed (%s direct=%s): %s", self.name, self._use_direct, exc)
            return False
        finally:
            self.wy_max_seq_len = prev_wy

    def forward_single(
        self,
        q: torch.Tensor,     # [1, T, H_k, D]
        k: torch.Tensor,     # [1, T, H_k, D]
        v: torch.Tensor,     # [1, T, H_v, D]
        g: torch.Tensor,     # [1, T, H_v] float32 log-decay <= 0
        beta: torch.Tensor,  # [1, T, H_v]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run ONE `B=1` forward pass and return `(core_attn_out, saved_ctx)` for fast multi-cotangent bwd."""
        T = q.shape[1]
        if self.wy_max_seq_len > 0 and T <= self.wy_max_seq_len:
            return wy_gdn_fwd_single(q, k, v, g, beta)

        scale = k.shape[-1] ** -0.5

        if self.name == "flash_qla" and q.is_cuda and q.dtype in (torch.bfloat16, torch.float16):
            api = self._flash_qla
            assert api is not None
            if self._use_direct:
                q_n, q_rstd = api["l2_fwd"](q.contiguous())
                k_n, k_rstd = api["l2_fwd"](k.contiguous())
                v_c = v.contiguous()
                g_c = g.float().contiguous()
                b_c = beta.to(q.dtype).contiguous()
                g_cum, A, o, *_ = api["fwd"](
                    q=q_n,
                    k=k_n,
                    v=v_c,
                    g=g_c,
                    beta=b_c,
                    scale=scale,
                    initial_state=None,
                    output_final_state=False,
                    auto_cp=False,
                    enable_fwd_cp_cache=False,
                    is_train=True,
                )
                return o.to(q.dtype), {
                    "mode": "flash_qla_direct",
                    "q_n": q_n, "q_rstd": q_rstd,
                    "k_n": k_n, "k_rstd": k_rstd,
                    "v": v_c, "g_cum": g_cum, "beta": b_c, "A": A, "scale": scale,
                }
            o, _ = api["high"](
                q.contiguous(), k.contiguous(), v.contiguous(),
                g=g.float().contiguous(), beta=beta.to(q.dtype).contiguous(),
                scale=scale, output_final_state=False, use_qk_l2norm_in_kernel=True, auto_cp=False,
            )
            return o.to(q.dtype), {"mode": "flash_qla_high", "q": q, "k": k, "v": v, "g": g, "beta": beta, "scale": scale}

        if self.name == "fla" and q.is_cuda and q.dtype in (torch.bfloat16, torch.float16):
            api = self._fla
            assert api is not None
            if self._use_direct:
                q_n, q_rstd = api["l2_fwd"](q.contiguous())
                k_n, k_rstd = api["l2_fwd"](k.contiguous())
                v_c = v.contiguous()
                g_c = g.float().contiguous()
                b_c = beta.to(q.dtype).contiguous()
                g_cum, o, A, *_ = api["fwd"](
                    q=q_n,
                    k=k_n,
                    v=v_c,
                    g=g_c,
                    beta=b_c,
                    scale=scale,
                    initial_state=None,
                    output_final_state=False,
                )
                return o.to(q.dtype), {
                    "mode": "fla_direct",
                    "q_n": q_n, "q_rstd": q_rstd,
                    "k_n": k_n, "k_rstd": k_rstd,
                    "v": v_c, "g_cum": g_cum, "beta": b_c, "A": A, "scale": scale,
                }
            o, _ = api["high"](
                q.contiguous(), k.contiguous(), v.contiguous(),
                g=g.float().contiguous(), beta=beta.to(q.dtype).contiguous(),
                scale=scale, output_final_state=False, use_qk_l2norm_in_kernel=True,
            )
            return o.to(q.dtype), {"mode": "fla_high", "q": q, "k": k, "v": v, "g": g, "beta": beta, "scale": scale}

        # Universal PyTorch exact path
        o = torch_chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
        return o, {"mode": "torch", "q": q, "k": k, "v": v, "g": g, "beta": beta}

    def backward_batched(
        self,
        saved: dict[str, Any],
        do: torch.Tensor,  # [K_sub, T_v, H_v, D]
        *,
        bwd_start: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute `(dq, dk, dv, dg, dbeta)` on suffix `[bwd_start:]` for `K_sub` cotangents at once."""
        K_sub = do.shape[0]
        mode = saved["mode"]
        s = int(bwd_start)

        if mode == "l2_cached_wy":
            return wy_gdn_bwd_batched(saved, do, bwd_start=s)

        do_full = F.pad(do, (0, 0, 0, 0, s, 0)) if s > 0 else do

        def _trim(res_tuple: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            dq_, dk_, dv_, dg_, db_ = res_tuple
            if s > 0:
                return dq_[:, s:], dk_[:, s:], dv_[:, s:], dg_[:, s:], db_[:, s:]
            return dq_, dk_, dv_, dg_, db_

        if mode == "flash_qla_direct":
            api = self._flash_qla
            assert api is not None
            q_e = saved["q_n"].expand(K_sub, -1, -1, -1).contiguous()
            k_e = saved["k_n"].expand(K_sub, -1, -1, -1).contiguous()
            v_e = saved["v"].expand(K_sub, -1, -1, -1).contiguous()
            g_e = saved["g_cum"].expand(K_sub, -1, -1).contiguous()
            b_e = saved["beta"].expand(K_sub, -1, -1).contiguous()
            A_e = saved["A"].expand(K_sub, -1, -1, -1).contiguous()
            res = api["bwd"](
                q=q_e,
                k=k_e,
                v=v_e,
                g=g_e,
                beta=b_e,
                A=A_e,
                do=do_full.to(q_e.dtype).contiguous(),
                dht=None,
                scale=saved["scale"],
                initial_state=None,
                auto_cp=False,
            )
            dq, dk, dv, db, dg = res[0], res[1], res[2], res[3], res[4]
            q_rstd_e = saved["q_rstd"].expand(K_sub, -1, -1, -1).contiguous()
            k_rstd_e = saved["k_rstd"].expand(K_sub, -1, -1, -1).contiguous()
            dq = api["l2_bwd"](q_e, q_rstd_e, dq)
            dk = api["l2_bwd"](k_e, k_rstd_e, dk)
            return _trim((dq.to(do.dtype), dk.to(do.dtype), dv.to(do.dtype), dg.float(), db.to(do.dtype)))

        if mode == "fla_direct":
            api = self._fla
            assert api is not None
            q_e = saved["q_n"].expand(K_sub, -1, -1, -1).contiguous()
            k_e = saved["k_n"].expand(K_sub, -1, -1, -1).contiguous()
            v_e = saved["v"].expand(K_sub, -1, -1, -1).contiguous()
            g_e = saved["g_cum"].expand(K_sub, -1, -1).contiguous()
            b_e = saved["beta"].expand(K_sub, -1, -1).contiguous()
            A_e = saved["A"].expand(K_sub, -1, -1, -1).contiguous()
            res = api["bwd"](
                q=q_e,
                k=k_e,
                v=v_e,
                g=g_e,
                beta=b_e,
                A=A_e,
                scale=saved["scale"],
                initial_state=None,
                do=do_full.to(q_e.dtype).contiguous(),
                dht=None,
            )
            dq, dk, dv, db, dg = res[0], res[1], res[2], res[3], res[4]
            q_rstd_e = saved["q_rstd"].expand(K_sub, -1, -1, -1).contiguous()
            k_rstd_e = saved["k_rstd"].expand(K_sub, -1, -1, -1).contiguous()
            dq = api["l2_bwd"](q_e, q_rstd_e, dq)
            dk = api["l2_bwd"](k_e, k_rstd_e, dk)
            return _trim((dq.to(do.dtype), dk.to(do.dtype), dv.to(do.dtype), dg.float(), db.to(do.dtype)))

        if mode in ("flash_qla_high", "fla_high"):
            api = self._flash_qla if mode == "flash_qla_high" else self._fla
            assert api is not None
            extra = {"auto_cp": False} if mode == "flash_qla_high" else {}
            with torch.enable_grad():
                q_e = saved["q"].detach().expand(K_sub, -1, -1, -1).contiguous().requires_grad_(True)
                k_e = saved["k"].detach().expand(K_sub, -1, -1, -1).contiguous().requires_grad_(True)
                v_e = saved["v"].detach().expand(K_sub, -1, -1, -1).contiguous().requires_grad_(True)
                g_e = saved["g"].detach().float().expand(K_sub, -1, -1).contiguous().requires_grad_(True)
                b_e = saved["beta"].detach().to(q_e.dtype).expand(K_sub, -1, -1).contiguous().requires_grad_(True)
                o_e, _ = api["high"](
                    q_e, k_e, v_e, g=g_e, beta=b_e, scale=saved["scale"],
                    output_final_state=False, use_qk_l2norm_in_kernel=True, **extra,
                )
                dq, dk, dv, dg, db = torch.autograd.grad(
                    outputs=o_e, inputs=(q_e, k_e, v_e, g_e, b_e), grad_outputs=do_full.to(o_e.dtype),
                )
            return _trim((dq.to(do.dtype), dk.to(do.dtype), dv.to(do.dtype), dg.float(), db.to(do.dtype)))

        # Pure PyTorch exact chunked backward
        with torch.enable_grad():
            q_e = saved["q"].detach().expand(K_sub, -1, -1, -1).clone().requires_grad_(True)
            k_e = saved["k"].detach().expand(K_sub, -1, -1, -1).clone().requires_grad_(True)
            v_e = saved["v"].detach().expand(K_sub, -1, -1, -1).clone().requires_grad_(True)
            g_e = saved["g"].detach().expand(K_sub, -1, -1).clone().requires_grad_(True)
            b_e = saved["beta"].detach().expand(K_sub, -1, -1).clone().requires_grad_(True)
            o_e = torch_chunk_gated_delta_rule(q_e, k_e, v_e, g_e, b_e, use_qk_l2norm_in_kernel=True)
            dq, dk, dv, dg, db = torch.autograd.grad(
                outputs=o_e,
                inputs=(q_e, k_e, v_e, g_e, b_e),
                grad_outputs=do_full.to(o_e.dtype),
            )
        return _trim((dq, dk, dv, dg.float(), db))


# --------------------------------------------------------------------------- #
# 4. Causal Conv1D Forward + Exact Backward
# --------------------------------------------------------------------------- #
try:
    from causal_conv1d import causal_conv1d_fn as _cuda_causal_conv1d_fn
except Exception:
    _cuda_causal_conv1d_fn = None


def causal_conv1d_silu_fwd(
    x: torch.Tensor,        # [B, C, T]
    weight: torch.Tensor,   # [C, K_conv]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact depthwise causal conv1d + SiLU; returns `(out_silu, pre_silu)`."""
    C, K_conv = weight.shape
    T = x.shape[-1]
    pre = F.conv1d(x.to(weight.dtype), weight.unsqueeze(1), bias=None, padding=K_conv - 1, groups=C)[:, :, :T]
    out = F.silu(pre).to(x.dtype)
    return out, pre


def causal_conv1d_silu_bwd(
    d_out: torch.Tensor,     # [K_sub, C, T]
    pre_silu: torch.Tensor,  # [1, C, T]
    weight: torch.Tensor,    # [C, K_conv]
) -> torch.Tensor:
    """Exact VJP of depthwise causal conv1d + SiLU w.r.t. input `x` -> `[K_sub, C, T]`."""
    C, K_conv = weight.shape
    T = d_out.shape[-1]
    u = pre_silu.float()
    sig = torch.sigmoid(u)
    d_silu = (sig * (1.0 + u * (1.0 - sig))).to(d_out.dtype)
    d_pre = (d_out * d_silu).to(weight.dtype)
    pad_l = K_conv - 1
    d_x = F.conv_transpose1d(d_pre, weight.unsqueeze(1), bias=None, padding=0, groups=C)
    return d_x[:, :, pad_l : pad_l + T].to(d_out.dtype)


def causal_conv1d_silu_bwd_qkv_ktc(
    dq: torch.Tensor,        # [K_sub, T_v, H_k, D] or [K_sub, T_v, key_dim]
    dk: torch.Tensor,        # [K_sub, T_v, H_k, D] or [K_sub, T_v, key_dim]
    dv: torch.Tensor,        # [K_sub, T_v, H_v, D] or [K_sub, T_v, val_dim]
    pre_silu: torch.Tensor,  # [1, C, T_v]
    weight: torch.Tensor,    # [C, K_conv]
    *,
    use_triton: bool = True,
) -> torch.Tensor:
    """Fused `[dq, dk, dv]` + SiLU + 4-tap causal depthwise conv1d VJP directly in `[K_sub, T_v, C]`."""
    K_sub, T_v = dq.shape[0], dq.shape[1]
    dq_f = dq.reshape(K_sub, T_v, -1)
    dk_f = dk.reshape(K_sub, T_v, -1)
    dv_f = dv.reshape(K_sub, T_v, -1)
    key_dim = dq_f.shape[2]
    val_dim = dv_f.shape[2]
    C, K_conv = weight.shape
    assert C == 2 * key_dim + val_dim

    if (
        use_triton
        and _HAS_TRITON
        and dq_f.is_cuda
        and dq_f.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and K_conv == 4
    ):
        out = torch.empty((K_sub, T_v, C), device=dq_f.device, dtype=torch.bfloat16)
        pre_2d = pre_silu[0]  # [C, T_v]
        block_c = 128
        for src, c_lo, c_len in (
            (dq_f.contiguous(), 0, key_dim),
            (dk_f.contiguous(), key_dim, key_dim),
            (dv_f.contiguous(), 2 * key_dim, val_dim),
        ):
            pre_s = pre_2d[c_lo : c_lo + c_len]
            w_s = weight[c_lo : c_lo + c_len]
            out_s = out[:, :, c_lo : c_lo + c_len]
            grid = (K_sub, T_v, triton.cdiv(c_len, block_c))
            _causal_conv1d_silu_bwd_slice_kernel[grid](
                src,
                pre_s,
                w_s,
                out_s,
                T_v,
                c_len,
                src.stride(0),
                src.stride(1),
                pre_s.stride(0),
                pre_s.stride(1),
                w_s.stride(0),
                w_s.stride(1),
                out_s.stride(0),
                out_s.stride(1),
                BLOCK_C=block_c,
                num_warps=4,
            )
        return out

    d_mixed_t = torch.cat([dq_f, dk_f, dv_f], dim=-1)
    return causal_conv1d_silu_bwd(d_mixed_t.transpose(1, 2), pre_silu, weight).transpose(1, 2)


@torch.no_grad()
def verify_cuda_triton_kernels(
    device: torch.device | str | None = None,
    *,
    verbose: bool = True,
) -> dict[str, float]:
    """~0.5s live parity & compile verification for ALL fused Triton VJP kernels + L2-cached WY GDN.

    Compares every custom Triton kernel in BF16 on CUDA against its exact `use_triton=False` PyTorch
    reference (and verifies `wy_gdn_fwd_single` + `wy_gdn_bwd_batched` at `T=128, bwd_start=16`
    against `torch_chunk_gated_delta_rule` + `torch.autograd.grad`). Returns `{name: max_abs_diff}`.
    """
    dev = torch.device(device) if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
    use_tri = dev.type == "cuda" and _HAS_TRITON
    atol_def = 2e-2 if dt == torch.bfloat16 else 1e-4
    rtol_def = 2e-2 if dt == torch.bfloat16 else 1e-4
    errs: dict[str, float] = {}

    def _chk(
        name: str,
        got: torch.Tensor,
        ref: torch.Tensor,
        *,
        atol: float = atol_def,
        rtol: float = rtol_def,
    ) -> None:
        assert torch.isfinite(got).all(), f"{name}: non-finite output!"
        gf, rf = got.float(), ref.float()
        diff = (gf - rf).abs()
        err = float(diff.max().item())
        errs[name] = err
        ok = torch.allclose(gf, rf, rtol=rtol, atol=atol)
        if not ok:
            idx = int(diff.argmax().item())
            gv, rv = float(gf.reshape(-1)[idx].item()), float(rf.reshape(-1)[idx].item())
            raise AssertionError(
                f"{name} parity failed: max_abs_diff={err:.4e} (got={gv:.4f}, ref={rv:.4f}, atol={atol:.2e}, rtol={rtol:.2e})"
            )

    # 1. FP8 blockwise dequant (128x128) — bit-exact on BF16!
    w_fp8 = torch.randn(2, 256, 256, device=dev).to(torch.float8_e4m3fn)
    s_fp32 = torch.rand(2, 2, 2, device=dev, dtype=torch.float32) * 0.1 + 0.01
    _chk(
        "fp8_dequant",
        dequantize_fp8_blockwise(w_fp8, s_fp32, out_dtype=dt, use_triton=use_tri),
        dequantize_fp8_blockwise(w_fp8, s_fp32, out_dtype=dt, use_triton=False),
        atol=0.0 if dt == torch.bfloat16 else 1e-6,
        rtol=0.0 if dt == torch.bfloat16 else 1e-6,
    )

    # 2. 2D-tiled SwiGLU VJP pack
    dA = torch.randn(16, 32, 640, device=dev, dtype=dt) * 0.3
    dg = torch.randn(16, 640, device=dev, dtype=dt) * 0.3
    du = torch.randn(16, 640, device=dev, dtype=dt) * 0.3
    _chk("swiglu_vjp_pack", swiglu_vjp_pack(dA, dg, du, use_triton=use_tri), swiglu_vjp_pack(dA, dg, du, use_triton=False))

    # 3. RMSNorm VJP: grouped (hc=4, D=2560, 2-pass) + out_add, AND ungrouped (D=256, single-pass)
    K_t, T_t, D_t, hc = 8, 16, 2560, 4
    x_g = torch.randn(1, T_t, hc * D_t, device=dev, dtype=dt) * 0.3
    w_g = torch.randn(hc * D_t, device=dev, dtype=dt) * 0.1
    xf = x_g.float().reshape(1, T_t, hc, D_t)
    rstd_g = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    xhat_g = xf * rstd_g
    dy_g = torch.randn(K_t, T_t, hc * D_t, device=dev, dtype=dt) * 0.3
    _chk(
        "rmsnorm_vjp_grouped",
        rmsnorm_vjp_fused(dy_g, xhat_g, rstd_g, w_g, group_size=D_t, use_triton=use_tri),
        rmsnorm_vjp_fused(dy_g, xhat_g, rstd_g, w_g, group_size=D_t, use_triton=False),
    )
    buf_tri = torch.randn(K_t, T_t, hc * D_t, device=dev, dtype=dt) * 0.3
    buf_ref = buf_tri.clone()
    rmsnorm_vjp_fused(dy_g, xhat_g, rstd_g, w_g, group_size=D_t, out_add=buf_tri, use_triton=use_tri)
    rmsnorm_vjp_fused(dy_g, xhat_g, rstd_g, w_g, group_size=D_t, out_add=buf_ref, use_triton=False)
    _chk("rmsnorm_vjp_out_add", buf_tri, buf_ref)

    x_u = torch.randn(1, T_t, 6, 256, device=dev, dtype=dt) * 0.3
    w_u = torch.randn(256, device=dev, dtype=dt) * 0.1
    rstd_u = torch.rsqrt(x_u.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    xhat_u = x_u.float() * rstd_u
    dy_u = torch.randn(K_t, T_t, 6, 256, device=dev, dtype=dt) * 0.3
    _chk(
        "rmsnorm_vjp_ungrouped",
        rmsnorm_vjp_fused(dy_u, xhat_u, rstd_u, w_u, group_size=None, use_triton=use_tri),
        rmsnorm_vjp_fused(dy_u, xhat_u, rstd_u, w_u, group_size=None, use_triton=False),
    )

    # 4. RMSNormGated VJP (D_v=128 single-pass)
    xg = torch.randn(1, T_t, 12, 128, device=dev, dtype=dt) * 0.3
    gg = torch.randn(1, T_t, 12, 128, device=dev, dtype=dt) * 0.3
    wg = torch.randn(128, device=dev, dtype=dt) + 1.0
    rstd_rg = torch.rsqrt(xg.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    xhat_rg = xg.float() * rstd_rg
    act_g = torch.sigmoid(gg.float())
    saved_rg = {
        "x_hat_f": xhat_rg,
        "rstd": rstd_rg,
        "normed": (wg * xhat_rg.to(dt)).float(),
        "act_g": act_g,
        "d_act_g": act_g * (1.0 - act_g),
        "weight_f": wg.float(),
    }
    dy_rg = torch.randn(K_t, T_t, 12, 128, device=dev, dtype=dt) * 0.3
    dx_tri, dg_tri = rmsnorm_gated_vjp_fused(dy_rg, saved_rg, use_triton=use_tri)
    dx_ref, dg_ref = rmsnorm_gated_vjp_fused(dy_rg, saved_rg, use_triton=False)
    _chk("rmsnorm_gated_dx", dx_tri, dx_ref)
    _chk("rmsnorm_gated_dgate", dg_tri, dg_ref)

    # 5. HC bwd prep with non-contiguous transposed d_mixed
    dm_parent = torch.randn(T_t, 64, D_t, device=dev, dtype=dt) * 0.3
    dm_nc = dm_parent[:, 8:24, :].transpose(0, 1)  # [16, T_t, D_t] non-contiguous
    m4 = torch.rand(1, T_t, hc, D_t, device=dev, dtype=dt)
    hn4 = torch.randn(1, T_t, hc, D_t, device=dev, dtype=dt) * 0.3
    dmu = torch.rand(1, T_t, hc, D_t, device=dev, dtype=dt) * 0.25
    dhn_t, du2_t = hc_bwd_prep_fused(dm_nc, m4, hn4, dmu, 0.25, use_triton=use_tri)
    dhn_r, du2_r = hc_bwd_prep_fused(dm_nc, m4, hn4, dmu, 0.25, use_triton=False)
    assert dhn_t.is_contiguous() and du2_t.is_contiguous()
    _chk("hc_bwd_prep_dhn", dhn_t, dhn_r)
    _chk("hc_bwd_prep_du2", du2_t, du2_r)

    # 6. HC inject VJP (both tk_layout=True and tk_layout=False)
    G_s = torch.randn(16, T_t, hc * D_t, device=dev, dtype=dt) * 0.2
    y_s = torch.randn(T_t, D_t, device=dev, dtype=dt) * 0.2
    w_s = torch.rand(T_t, hc, device=dev, dtype=dt) * 2.0
    for tk_flag in (True, False):
        dy_shape = (T_t, 16, D_t) if tk_flag else (16, T_t, D_t)
        dy_t = torch.empty(dy_shape, device=dev, dtype=dt)
        dw_t = torch.empty((16, T_t, hc), device=dev, dtype=dt)
        dy_r = torch.empty_like(dy_t)
        dw_r = torch.empty_like(dw_t)
        hc_inject_vjp_fused(G_s, y_s, w_s, dy_t, dw_t, tk_layout=tk_flag, use_triton=use_tri)
        hc_inject_vjp_fused(G_s, y_s, w_s, dy_r, dw_r, tk_layout=tk_flag, use_triton=False)
        tag = "tk" if tk_flag else "kt"
        _chk(f"hc_inject_dy_{tag}", dy_t, dy_r)
        _chk(f"hc_inject_dw_{tag}", dw_t, dw_r, atol=3e-2 if dt == torch.bfloat16 else 1e-4)

    # 7 & 8. MoE gather+scale and zero-atomic scatter-add (unchunked AND 2-chunked!)
    top_k = 8
    S_m = T_t * top_k
    perm = torch.randperm(S_m, device=dev)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(S_m, device=dev)
    inv_2d = inv_perm.view(T_t, top_k)
    tok_sorted = perm // top_k
    w_sorted = torch.rand(S_m, device=dev, dtype=dt) * 0.25
    dY_tk = torch.randn(T_t, 64, D_t, device=dev, dtype=dt) * 0.3
    _chk(
        "moe_gather_scale",
        moe_gather_scale_fused(dY_tk, tok_sorted, w_sorted, 16, 48, use_triton=use_tri),
        moe_gather_scale_fused(dY_tk, tok_sorted, w_sorted, 16, 48, use_triton=False),
    )
    dX_s = torch.randn(S_m, 32, D_t, device=dev, dtype=dt) * 0.2
    for mode_lbl, splits in (("full", [(0, S_m)]), ("chunked", [(0, S_m // 2), (S_m // 2, S_m)])):
        dst_t = torch.randn(T_t, 64, D_t, device=dev, dtype=dt) * 0.1
        dst_r = dst_t.clone()
        for r0, r1 in splits:
            moe_scatter_add_tk_fused(
                dst_t, dX_s[r0:r1], inv_2d, r0, 16, 48, sub_tok=tok_sorted[r0:r1], use_triton=use_tri
            )
            moe_scatter_add_tk_fused(
                dst_r, dX_s[r0:r1], inv_2d, r0, 16, 48, sub_tok=tok_sorted[r0:r1], use_triton=False
            )
        _chk(f"moe_scatter_add_{mode_lbl}", dst_t, dst_r, atol=3e-2 if dt == torch.bfloat16 else 1e-4)

    # 9. L2Norm VJP (D=128)
    x_l2 = torch.randn(1, T_t, 16, 128, device=dev, dtype=dt) * 0.3
    xn_l2, rstd_l2 = l2norm_torch(x_l2)
    dy_l2 = torch.randn(K_t, T_t, 16, 128, device=dev, dtype=dt) * 0.3
    _chk(
        "l2norm_bwd",
        l2norm_bwd_fused(xn_l2, rstd_l2, dy_l2, use_triton=use_tri),
        l2norm_bwd_fused(xn_l2, rstd_l2, dy_l2, use_triton=False),
    )

    # 10. Fused [dq, dk, dv] + SiLU + 4-tap causal conv1d VJP (on sliced suffix T_v=112 from T_full=128!)
    T_full, s_conv = 128, 16
    T_v = T_full - s_conv
    Hk_c, Hv_c, D_c = 4, 12, 128
    C_c = (2 * Hk_c + Hv_c) * D_c
    pre_full = torch.randn(1, C_c, T_full, device=dev, dtype=dt) * 0.5
    w_conv = torch.randn(C_c, 4, device=dev, dtype=dt) * 0.2
    dq_c = torch.randn(8, T_v, Hk_c, D_c, device=dev, dtype=dt) * 0.3
    dk_c = torch.randn(8, T_v, Hk_c, D_c, device=dev, dtype=dt) * 0.3
    dv_c = torch.randn(8, T_v, Hv_c, D_c, device=dev, dtype=dt) * 0.3
    _chk(
        "causal_conv1d_silu_bwd",
        causal_conv1d_silu_bwd_qkv_ktc(dq_c, dk_c, dv_c, pre_full[:, :, s_conv:], w_conv, use_triton=use_tri),
        causal_conv1d_silu_bwd_qkv_ktc(dq_c, dk_c, dv_c, pre_full[:, :, s_conv:], w_conv, use_triton=False),
        atol=3e-2 if dt == torch.bfloat16 else 1e-4,
    )

    # 11. L2-Cached Exact WY GDN (T=128, bwd_start=16) vs chunked PyTorch autograd
    q_w = torch.randn(1, T_full, 2, 128, device=dev, dtype=dt) * 0.2
    k_w = torch.randn(1, T_full, 2, 128, device=dev, dtype=dt) * 0.2
    v_w = torch.randn(1, T_full, 6, 128, device=dev, dtype=dt) * 0.2
    g_w = -F.softplus(torch.randn(1, T_full, 6, device=dev, dtype=torch.float32))
    b_w = torch.sigmoid(torch.randn(1, T_full, 6, device=dev, dtype=dt))
    do_full = torch.zeros(4, T_full, 6, 128, device=dev, dtype=dt)
    do_full[:, s_conv:] = torch.randn(4, T_v, 6, 128, device=dev, dtype=dt) * 0.2
    o_wy, saved_wy = wy_gdn_fwd_single(q_w, k_w, v_w, g_w, b_w)
    dq_wy, dk_wy, dv_wy, dg_wy, db_wy = wy_gdn_bwd_batched(saved_wy, do_full[:, s_conv:], bwd_start=s_conv, use_triton=use_tri)
    with torch.enable_grad():
        q_r = q_w.expand(4, -1, -1, -1).clone().requires_grad_(True)
        k_r = k_w.expand(4, -1, -1, -1).clone().requires_grad_(True)
        v_r = v_w.expand(4, -1, -1, -1).clone().requires_grad_(True)
        g_r = g_w.expand(4, -1, -1).clone().requires_grad_(True)
        b_r = b_w.expand(4, -1, -1).clone().requires_grad_(True)
        o_r = torch_chunk_gated_delta_rule(q_r, k_r, v_r, g_r, b_r, use_qk_l2norm_in_kernel=True)
        dq_r, dk_r, dv_r, dg_r, db_r = torch.autograd.grad(o_r, (q_r, k_r, v_r, g_r, b_r), grad_outputs=do_full)
    wy_atol = 5e-2 if dt == torch.bfloat16 else 2e-4
    wy_rtol = 4e-2 if dt == torch.bfloat16 else 1e-3
    _chk("wy_gdn_fwd", o_wy, o_r[:1], atol=3e-2 if dt == torch.bfloat16 else 1e-4, rtol=wy_rtol)
    _chk("wy_gdn_dq", dq_wy, dq_r[:, s_conv:], atol=wy_atol, rtol=wy_rtol)
    _chk("wy_gdn_dk", dk_wy, dk_r[:, s_conv:], atol=wy_atol, rtol=wy_rtol)
    _chk("wy_gdn_dv", dv_wy, dv_r[:, s_conv:], atol=wy_atol, rtol=wy_rtol)
    _chk("wy_gdn_dg", dg_wy, dg_r[:, s_conv:], atol=wy_atol, rtol=wy_rtol)
    _chk("wy_gdn_db", db_wy, db_r[:, s_conv:], atol=wy_atol, rtol=wy_rtol)

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    if verbose:
        max_k = max(errs, key=errs.get)  # type: ignore[arg-type]
        print(
            f"  [OK] Verified {len(errs)} fused {'CUDA/Triton BF16' if use_tri else 'PyTorch'} kernels + "
            f"WY GDN (T=128, s=16) | worst max_diff={errs[max_k]:.2e} ({max_k})"
        )
    return errs

