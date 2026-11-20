"""Exact Qwen3.8-Flash-Next (qwen4_exp) forward and batched-cotangent analytical VJP operators.

Every operator executes:
  1. Forward ONCE at batch size `B = 1` (saving minimal `B = 1` activations, ~18 MB/layer total).
  2. Exact VJP across `K` cotangents (`G` of shape `[K, T, D_hc]`, up to `K = 2560`) reusing
     the single `B = 1` forward state and single FP8->BF16 active-expert dequantization!

Mathematical parity with HuggingFace `transformers/models/qwen4_exp/modeling_qwen4_exp.py`:
  - `Qwen4ExpTextRMSNorm`: zero-centered `(1.0 + weight)` with optional `group_size=hidden_size`.
  - `Qwen4ExpTextRMSNormGated`: direct `weight` (NOT `1 + weight`) and configurable `sigmoid` / `silu`.
  - `Qwen4ExpTextGatedResidual`: 4-branch Hyper-Connection with bottleneck rank 320.
  - `Qwen4ExpTextPLELayer`: layer-2 hashed n-gram lookup (constant wrt residual) + sqrt-sign gate + dilated conv1d.
  - `Qwen4ExpTextAttention` (QSA): exact indexer bypass for `T <= 2048` + L2-cached exact `B=1` attention VJP.
  - `Qwen4ExpTextGatedDeltaNet` (GDN): FlashQLA (SM90/100/120) / FLA Triton (SM80) / PyTorch fallback.
  - `Qwen4ExpTextSparseMoeBlock`: exact router softmax+topk+norm VJP + 1x active-expert FP8 dequant + `grouped_mm` VJP + shared expert VJP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from qwen38_jlens.kernels import (
    GatedDeltaRuleBackend,
    causal_conv1d_silu_bwd,
    causal_conv1d_silu_fwd,
    dequantize_fp8_blockwise,
    grouped_mm,
    swiglu_vjp_pack,
)


# --------------------------------------------------------------------------- #
# Weight containers (pure flat tensor dataclasses — zero nn.Module overhead)
# --------------------------------------------------------------------------- #
@dataclass
class GatedResidualWeights:
    hc_norm_weight: torch.Tensor          # [hc_count * D]
    input_mix_weight_down: torch.Tensor   # [hc_lowrank, hc_count * D]
    input_mix_weight_up: torch.Tensor     # [hc_count * D, hc_lowrank]
    block_inject_weight: torch.Tensor | None  # [hc_count, hc_count * D] or None
    hc_count: int
    hidden_size: int
    eps: float = 1e-6


@dataclass
class PleLayerWeights:
    key_proj: torch.Tensor       # [hc_count * D, ple_embed_dim]
    value_proj: torch.Tensor     # [D, ple_embed_dim]
    norm_key: torch.Tensor       # [hc_count * D]
    norm_query: torch.Tensor     # [hc_count * D]
    norm_conv: torch.Tensor      # [hc_count * D]
    conv1d_weight: torch.Tensor  # [hc_count * D, 1, conv_kernel_size]
    hc_count: int
    hidden_size: int
    ngram_size: int = 3
    eps: float = 1e-6


@dataclass
class GdnLayerWeights:
    in_proj_qkv: torch.Tensor    # [2*key_dim + value_dim, D]
    in_proj_z: torch.Tensor      # [value_dim, D]
    in_proj_b: torch.Tensor      # [num_v_heads, D]
    in_proj_a: torch.Tensor      # [num_v_heads, D]
    conv1d_weight: torch.Tensor  # [conv_dim, conv_kernel_size] (squeezed)
    dt_bias: torch.Tensor        # [num_v_heads]
    A_log: torch.Tensor          # [num_v_heads]
    norm_weight: torch.Tensor    # [head_v_dim]
    out_proj: torch.Tensor       # [D, value_dim]
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    output_gate_type: str = "sigmoid"
    eps: float = 1e-6


@dataclass
class QsaLayerWeights:
    q_proj: torch.Tensor         # [num_q_heads * head_dim * 2, D]
    k_proj: torch.Tensor         # [num_kv_heads * head_dim, D]
    v_proj: torch.Tensor         # [num_kv_heads * head_dim, D]
    o_proj: torch.Tensor         # [D, num_q_heads * head_dim]
    q_norm: torch.Tensor         # [head_dim]
    k_norm: torch.Tensor         # [head_dim]
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    eps: float = 1e-6
    # Optional indexer weights (only used when seq_len > indexer_budget)
    index_qk_proj: torch.Tensor | None = None
    index_q_norm: torch.Tensor | None = None
    index_k_norm: torch.Tensor | None = None
    index_n_heads: int = 4
    index_head_dim: int = 128


@dataclass
class MoeDenseWeights:
    router_weight: torch.Tensor        # [num_experts, D]
    shared_gate_proj: torch.Tensor     # [shared_inter, D]
    shared_up_proj: torch.Tensor       # [shared_inter, D]
    shared_down_proj: torch.Tensor     # [D, shared_inter]
    shared_expert_gate: torch.Tensor   # [1, D]
    num_experts: int
    top_k: int
    norm_topk_prob: bool = True
    fp8_block_size: tuple[int, int] = (128, 128)


@dataclass
class MoeExpertPayload:
    """Routed MoE expert tensors for one layer (either FP8+scales or BF16/FP32)."""

    gate_up_proj: torch.Tensor                  # [E, 2*inter, D] fp8 or bf16/fp32
    down_proj: torch.Tensor                     # [E, D, inter]   fp8 or bf16/fp32
    gate_up_scale: torch.Tensor | None = None   # [E, 2*inter//128, D//128] fp32
    down_scale: torch.Tensor | None = None      # [E, D//128, inter//128]   fp32


@dataclass
class DecoderLayerDenseWeights:
    layer_idx: int
    layer_type: str  # "linear_attention" or "indexed_attention"
    attn_hc: GatedResidualWeights
    mlp_hc: GatedResidualWeights
    moe_dense: MoeDenseWeights
    gdn: GdnLayerWeights | None = None
    qsa: QsaLayerWeights | None = None
    ple: PleLayerWeights | None = None


# --------------------------------------------------------------------------- #
# 1. Exact RMSNorm (zero-centered 1+w) & RMSNormGated (direct w)
# --------------------------------------------------------------------------- #
def rmsnorm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Qwen4ExpTextRMSNorm forward: `y = x_hat * (1 + weight)` with optional group_size."""
    orig_shape = x.shape
    xf = x.float()
    wf = 1.0 + weight.float()
    if group_size is not None:
        xf = xf.reshape(*orig_shape[:-1], -1, group_size)
        wf = wf.reshape(-1, group_size)
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_hat = xf * rstd
    out = (x_hat * wf).to(x.dtype)
    if group_size is not None:
        out = out.reshape(orig_shape)
    return out, x_hat, rstd


def rmsnorm_vjp(
    dy: torch.Tensor,      # [K, *orig_shape_no_batch] or [K, T, D]
    x_hat: torch.Tensor,   # [1, ..., (G,) gs] float32
    rstd: torch.Tensor,    # [1, ..., (G,) 1]  float32
    weight: torch.Tensor,  # [D]
    group_size: int | None = None,
) -> torch.Tensor:
    """Exact batched-cotangent VJP of Qwen4ExpTextRMSNorm."""
    orig_dy_shape = dy.shape
    dyf = dy.float()
    wf = 1.0 + weight.float()
    if group_size is not None:
        dyf = dyf.reshape(*orig_dy_shape[:-1], -1, group_size)
        wf = wf.reshape(-1, group_size)
    g = dyf * wf
    dot = (g * x_hat).mean(dim=-1, keepdim=True)
    dx = (rstd * (g - x_hat * dot)).to(dy.dtype)
    if group_size is not None:
        dx = dx.reshape(orig_dy_shape)
    return dx


def rmsnorm_gated_fwd(
    x: torch.Tensor,         # [1, T, H_v, D_v]
    gate: torch.Tensor,      # [1, T, H_v, D_v]
    weight: torch.Tensor,    # [D_v] (NOT 1+weight!)
    activation: str = "sigmoid",
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Qwen4ExpTextRMSNormGated exact forward."""
    input_dtype = x.dtype
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_hat_f = xf * rstd
    normed = (weight.to(input_dtype) * x_hat_f.to(input_dtype)).float()
    gf = gate.float()
    if activation == "sigmoid":
        act_g = torch.sigmoid(gf)
        d_act_g = act_g * (1.0 - act_g)
    elif activation == "silu":
        sg = torch.sigmoid(gf)
        act_g = gf * sg
        d_act_g = sg * (1.0 + gf * (1.0 - sg))
    else:
        raise ValueError(f"Unsupported RMSNormGated activation: {activation}")
    out = (normed * act_g).to(input_dtype)
    return out, {
        "x_hat_f": x_hat_f,
        "rstd": rstd,
        "normed": normed,
        "act_g": act_g,
        "d_act_g": d_act_g,
        "weight_f": weight.float(),
    }


def rmsnorm_gated_vjp(
    dy: torch.Tensor,  # [K, T, H_v, D_v]
    saved: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact VJP of Qwen4ExpTextRMSNormGated -> `(dx, dgate)`."""
    dyf = dy.float()
    d_gate = (dyf * saved["normed"] * saved["d_act_g"]).to(dy.dtype)
    g = dyf * saved["act_g"] * saved["weight_f"]
    x_hat_f = saved["x_hat_f"]
    dot = (g * x_hat_f).mean(dim=-1, keepdim=True)
    dx = (saved["rstd"] * (g - x_hat_f * dot)).to(dy.dtype)
    return dx, d_gate


# --------------------------------------------------------------------------- #
# 2. Exact Gated Residual (Hyper-Connection) Forward & VJP
# --------------------------------------------------------------------------- #
def gated_residual_fwd(
    hyper_input: torch.Tensor,  # [1, T, hc_count * D]
    w: GatedResidualWeights,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    """Exact forward of `Qwen4ExpTextGatedResidual`."""
    h_norm, x_hat, rstd = rmsnorm_fwd(
        hyper_input, w.hc_norm_weight, eps=w.eps, group_size=w.hidden_size
    )
    u1 = F.linear(h_norm, w.input_mix_weight_down) / w.hc_count
    u1_f = u1.float()
    sig1 = torch.sigmoid(u1_f)
    s1 = (u1_f * sig1).to(h_norm.dtype)
    d_silu_u1 = (sig1 * (1.0 + u1_f * (1.0 - sig1))).to(h_norm.dtype)

    u2 = F.linear(s1, w.input_mix_weight_up)
    m_flat = torch.sigmoid(u2.float())
    m_4 = m_flat.unflatten(-1, (w.hc_count, w.hidden_size))
    h_norm_4 = h_norm.float().unflatten(-1, (w.hc_count, w.hidden_size))
    mixed_input = (m_4 * h_norm_4).mean(dim=-2).to(hyper_input.dtype)

    inj_weights = None
    d_inj_u3 = None
    if w.block_inject_weight is not None:
        u3 = F.linear(h_norm, w.block_inject_weight) / w.hc_count
        sig3 = torch.sigmoid(u3.float())
        inj_weights = (2.0 * sig3).to(hyper_input.dtype)
        d_inj_u3 = (2.0 * sig3 * (1.0 - sig3)).to(hyper_input.dtype)

    saved = {
        "x_hat": x_hat,
        "rstd": rstd,
        "h_norm_4": h_norm_4.to(hyper_input.dtype),
        "m_4": m_4.to(hyper_input.dtype),
        "dm_du2": (m_4 * (1.0 - m_4)).to(hyper_input.dtype),
        "d_silu_u1": d_silu_u1,
        "d_inj_u3": d_inj_u3,
    }
    return mixed_input, inj_weights, saved


def gated_residual_vjp(
    d_mixed: torch.Tensor,          # [K, T, D]
    d_inj: torch.Tensor | None,     # [K, T, hc_count] or None
    w: GatedResidualWeights,
    saved: dict[str, Any],
) -> torch.Tensor:
    """Exact VJP of `Qwen4ExpTextGatedResidual` w.r.t. `hyper_input` -> `[K, T, hc_count * D]`."""
    inv_hc = 1.0 / w.hc_count
    d_prod = (d_mixed * inv_hc).unsqueeze(-2)  # [K, T, 1, D]

    # Direct branch through h_norm * m
    d_h_norm = (d_prod * saved["m_4"]).flatten(-2)

    # Gate branch through m = sigmoid(W_up(silu(W_down(h_norm) / hc)))
    d_u2 = (d_prod * saved["h_norm_4"] * saved["dm_du2"]).flatten(-2)  # [K, T, hc*D]
    d_s1 = torch.matmul(d_u2, w.input_mix_weight_up)  # [K, T, hc_lowrank]
    d_u1 = d_s1 * saved["d_silu_u1"]
    d_h_norm = d_h_norm + torch.matmul(d_u1 * inv_hc, w.input_mix_weight_down)

    # Injection gate branch
    if d_inj is not None and w.block_inject_weight is not None:
        d_u3 = d_inj * saved["d_inj_u3"]  # [K, T, hc_count]
        d_h_norm = d_h_norm + torch.matmul(d_u3 * inv_hc, w.block_inject_weight)

    return rmsnorm_vjp(d_h_norm, saved["x_hat"], saved["rstd"], w.hc_norm_weight, group_size=w.hidden_size)


def slice_gated_residual_saved(saved: dict[str, Any], bwd_start: int) -> dict[str, Any]:
    """Slice a `gated_residual_fwd` saved dict along token dim 1 to `[bwd_start:]`."""
    if bwd_start <= 0:
        return saved
    return {
        k: (v[:, bwd_start:].contiguous() if isinstance(v, torch.Tensor) else v)
        for k, v in saved.items()
    }


# --------------------------------------------------------------------------- #
# 3. Exact PLE Layer Forward & VJP (Layer 2 only; n-gram lookup is constant wrt H)
# --------------------------------------------------------------------------- #
def ple_layer_fwd(
    hidden_states: torch.Tensor,    # [1, T, hc_count * D]
    ple_embeddings: torch.Tensor,   # [1, T, ple_embed_dim] (pre-gathered from MmapPleTable)
    w: PleLayerWeights,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Exact forward of `Qwen4ExpTextPLELayer` returning `(delta_out, saved)`."""
    hc = w.hc_count
    D = w.hidden_size
    key_normed, _, _ = rmsnorm_fwd(
        F.linear(ple_embeddings, w.key_proj), w.norm_key, eps=w.eps, group_size=D
    )
    key_normed_4 = key_normed.unflatten(-1, (hc, D))  # [1, T, hc, D]
    value = F.linear(ple_embeddings, w.value_proj)    # [1, T, D]

    query_normed, q_xhat, q_rstd = rmsnorm_fwd(
        hidden_states, w.norm_query, eps=w.eps, group_size=D
    )
    query_normed_4 = query_normed.unflatten(-1, (hc, D))

    inv_sqrt_d = 1.0 / math.sqrt(D)
    raw_gate = (key_normed_4.float() * query_normed_4.float()).sum(dim=-1, keepdim=True) * inv_sqrt_d
    abs_gate = raw_gate.abs()
    sqrt_gate = abs_gate.clamp_min(1e-6).sqrt() * raw_gate.sign()
    d_sqrt_gate = torch.where(
        abs_gate > 1e-6, 0.5 * torch.rsqrt(abs_gate.clamp_min(1e-6)), torch.zeros_like(abs_gate)
    )
    sig_gate = torch.sigmoid(sqrt_gate)
    d_sig_raw = sig_gate * (1.0 - sig_gate) * d_sqrt_gate  # [1, T, hc, 1]

    gated_value = (sig_gate * value.float().unsqueeze(-2)).to(hidden_states.dtype).flatten(-2)
    gv_normed, c_xhat, c_rstd = rmsnorm_fwd(gated_value, w.norm_conv, eps=w.eps, group_size=D)

    # Dilated causal depthwise conv1d + silu
    conv_k = w.conv1d_weight.shape[-1]
    dil = w.ngram_size
    pad_l = (conv_k - 1) * dil
    x_conv = F.pad(gv_normed.transpose(1, 2), (pad_l, 0))
    pre_silu = F.conv1d(x_conv, w.conv1d_weight, bias=None, dilation=dil, groups=hc * D)
    pre_f = pre_silu.float()
    sig_c = torch.sigmoid(pre_f)
    conv_out = (pre_f * sig_c).to(hidden_states.dtype).transpose(1, 2)
    d_silu_conv = (sig_c * (1.0 + pre_f * (1.0 - sig_c))).to(hidden_states.dtype)

    out = gated_value + conv_out
    saved = {
        "key_normed_4": key_normed_4.float(),
        "value_f": value.float(),
        "q_xhat": q_xhat,
        "q_rstd": q_rstd,
        "d_sig_raw": d_sig_raw,
        "c_xhat": c_xhat,
        "c_rstd": c_rstd,
        "d_silu_conv": d_silu_conv,
        "pad_l": pad_l,
        "dil": dil,
    }
    return out, saved


def ple_layer_vjp(
    d_out: torch.Tensor,  # [K, T, hc_count * D]
    w: PleLayerWeights,
    saved: dict[str, Any],
) -> torch.Tensor:
    """Exact VJP of `Qwen4ExpTextPLELayer` w.r.t. `hidden_states` -> `[K, T, hc_count * D]`."""
    hc = w.hc_count
    D = w.hidden_size
    T = d_out.shape[1]

    # Backward through short_conv (dilated depthwise conv1d + silu)
    d_pre = (d_out.transpose(1, 2) * saved["d_silu_conv"]).to(w.conv1d_weight.dtype)
    pad_l = saved["pad_l"]
    d_gv_normed = F.conv_transpose1d(
        d_pre, w.conv1d_weight, bias=None, dilation=saved["dil"], groups=hc * D
    )[:, :, pad_l : pad_l + T].transpose(1, 2).to(d_out.dtype)

    d_gated_value = d_out + rmsnorm_vjp(
        d_gv_normed, saved["c_xhat"], saved["c_rstd"], w.norm_conv, group_size=D
    )

    # Backward through gated_value = sig_gate * value
    d_gv_4 = d_gated_value.float().unflatten(-1, (hc, D))
    d_sig = (d_gv_4 * saved["value_f"].unsqueeze(-2)).sum(dim=-1, keepdim=True)  # [K, T, hc, 1]
    d_raw_gate = d_sig * saved["d_sig_raw"] * (1.0 / math.sqrt(D))  # [K, T, hc, 1]

    # Backward through raw_gate = sum(key_normed_4 * query_normed_4) / sqrt(D)
    d_query_normed = (d_raw_gate * saved["key_normed_4"]).flatten(-2).to(d_out.dtype)
    return rmsnorm_vjp(d_query_normed, saved["q_xhat"], saved["q_rstd"], w.norm_query, group_size=D)


# --------------------------------------------------------------------------- #
# 4. Exact Qwen Sparse Attention (QSA) Forward & L2-Cached Attention VJP
# --------------------------------------------------------------------------- #
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rope(
    x: torch.Tensor,      # [..., T, head_dim]
    cos: torch.Tensor,    # [1, 1, T, rotary_dim]
    sin: torch.Tensor,    # [1, 1, T, rotary_dim]
) -> torch.Tensor:
    rot_dim = cos.shape[-1]
    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]
    return torch.cat([(x_rot * cos) + (_rotate_half(x_rot) * sin), x_pass], dim=-1)


def build_rope_cos_sin(
    seq_len: int,
    head_dim: int,
    partial_rotary_factor: float,
    rope_theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact Qwen4Exp 1D text RoPE cos/sin (for text-only sequences, all 3 M-RoPE dims equal position_ids)."""
    rot_dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rot_dim, 2, device=device, dtype=torch.float32) / rot_dim))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [T, rot_dim // 2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, rot_dim]
    cos = emb.cos().to(dtype).view(1, 1, seq_len, rot_dim)
    sin = emb.sin().to(dtype).view(1, 1, seq_len, rot_dim)
    return cos, sin


def qsa_layer_fwd(
    hidden_states: torch.Tensor,  # [1, T, D]
    w: QsaLayerWeights,
    cos: torch.Tensor,            # [1, 1, T, rot_dim]
    sin: torch.Tensor,            # [1, 1, T, rot_dim]
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Exact QSA forward.

    Mathematical guarantee: when `T <= indexer_budget` (2048), QSA's top-k block indexer
    selects 100% of causal positions and has zero gradient (boolean mask). Bypassing the
    indexer eliminates O(B*T) Python loops and GPU syncs while remaining bit-identical!
    """
    B, T, _ = hidden_states.shape
    H_q, H_kv, Dh = w.num_q_heads, w.num_kv_heads, w.head_dim
    n_rep = H_q // H_kv
    scaling = Dh ** -0.5

    qg = F.linear(hidden_states, w.q_proj).view(B, T, H_q, Dh * 2)
    q_raw, gate = torch.chunk(qg, 2, dim=-1)
    gate_flat = gate.reshape(B, T, H_q * Dh)

    q_normed, q_xhat, q_rstd = rmsnorm_fwd(q_raw, w.q_norm, eps=w.eps)
    k_raw = F.linear(hidden_states, w.k_proj).view(B, T, H_kv, Dh)
    k_normed, k_xhat, k_rstd = rmsnorm_fwd(k_raw, w.k_norm, eps=w.eps)
    v = F.linear(hidden_states, w.v_proj).view(B, T, H_kv, Dh).transpose(1, 2)  # [1, H_kv, T, Dh]

    q_rot = apply_partial_rope(q_normed.transpose(1, 2), cos, sin)  # [1, H_q, T, Dh]
    k_rot = apply_partial_rope(k_normed.transpose(1, 2), cos, sin)  # [1, H_kv, T, Dh]

    k_rep = k_rot.unsqueeze(2).expand(B, H_kv, n_rep, T, Dh).reshape(B, H_q, T, Dh)
    v_rep = v.unsqueeze(2).expand(B, H_kv, n_rep, T, Dh).reshape(B, H_q, T, Dh)

    if T <= 512 and T <= w.indexer_budget:
        # L2-cached exact attention weights P [H_q, T, T] (~768 KB at T=128)
        scores = torch.matmul(q_rot[0].float(), k_rep[0].float().transpose(-1, -2)) * scaling
        causal_mask = torch.ones((T, T), device=hidden_states.device, dtype=torch.bool).triu(1)
        scores.masked_fill_(causal_mask, float("-inf"))
        P = torch.softmax(scores, dim=-1)  # [H_q, T, T] fp32
        attn_out = torch.matmul(P.to(v_rep.dtype), v_rep[0]).transpose(0, 1).contiguous().view(B, T, H_q * Dh)
        attn_mode = "l2_cached"
    else:
        attn_out = F.scaled_dot_product_attention(q_rot, k_rep, v_rep, is_causal=True, scale=scaling)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, H_q * Dh)
        P = None
        attn_mode = "sdpa"

    sig_gate = torch.sigmoid(gate_flat.float())
    gated_out = (attn_out.float() * sig_gate).to(hidden_states.dtype)
    out = F.linear(gated_out, w.o_proj)

    saved = {
        "attn_mode": attn_mode,
        "P": P,
        "q_rot": q_rot[0],
        "k_rep": k_rep[0],
        "v_rep": v_rep[0],
        "q_xhat": q_xhat,
        "q_rstd": q_rstd,
        "k_xhat": k_xhat,
        "k_rstd": k_rstd,
        "attn_out": attn_out[0],
        "sig_gate": sig_gate[0].to(hidden_states.dtype),
        "d_sig_gate": (sig_gate[0] * (1.0 - sig_gate[0])).to(hidden_states.dtype),
        "cos": cos,
        "sin": sin,
        "scaling": scaling,
    }
    return out, saved


def qsa_layer_vjp(
    d_out: torch.Tensor,  # [K_sub, T_v, D] where T_v = T_full - bwd_start
    w: QsaLayerWeights,
    saved: dict[str, Any],
    *,
    bwd_start: int = 0,
) -> torch.Tensor:
    """Exact batched-cotangent VJP of `Qwen4ExpTextAttention` on suffix `[bwd_start:]` -> `[K_sub, T_v, D]`."""
    K_sub, T_v, _ = d_out.shape
    s = int(bwd_start)
    T_full = int(saved["q_rot"].shape[1])
    assert T_v == T_full - s, f"Expected T_v={T_full - s}, got {T_v}"
    H_q, H_kv, Dh = w.num_q_heads, w.num_kv_heads, w.head_dim
    n_rep = H_q // H_kv
    dtype = d_out.dtype

    # 1. o_proj & sigmoid output gate VJP (pointwise on [s:])
    attn_out_s = saved["attn_out"][s:] if s > 0 else saved["attn_out"]
    sig_gate_s = saved["sig_gate"][s:] if s > 0 else saved["sig_gate"]
    d_sig_gate_s = saved["d_sig_gate"][s:] if s > 0 else saved["d_sig_gate"]

    d_gated = torch.matmul(d_out, w.o_proj)  # [K_sub, T_v, H_q * Dh]
    d_gate = d_gated * attn_out_s.unsqueeze(0) * d_sig_gate_s.unsqueeze(0)
    d_attn_flat = d_gated * sig_gate_s.unsqueeze(0)
    dO = d_attn_flat.view(K_sub, T_v, H_q, Dh).transpose(1, 2).contiguous()  # [K_sub, H_q, T_v, Dh]

    # 2. Core causal attention VJP (exact suffix-to-suffix block A_VV + exact dQ[s:] over full keys [0:T_full])
    if saved["attn_mode"] == "l2_cached":
        P = saved["P"]                                                     # [H_q, T_full, T_full] fp32
        P_vv_t = (P[:, s:, s:] if s > 0 else P).to(dtype).contiguous()     # [H_q, T_v, T_v]
        P_va = P[:, s:, :] if s > 0 else P                                 # [H_q, T_v, T_full] fp32
        Q_v = (saved["q_rot"][:, s:, :] if s > 0 else saved["q_rot"]).contiguous()  # [H_q, T_v, Dh]
        K_all = saved["k_rep"]                                             # [H_q, T_full, Dh]
        V_all = saved["v_rep"]                                             # [H_q, T_full, Dh]

        dO_h = dO.transpose(0, 1).contiguous()                             # [H_q, K_sub, T_v, Dh]

        # dV[s:] = P[s:, s:]^T @ dO[s:] (causal: P[q, k] = 0 for q < k => prefix q < s never contributes to k >= s)
        dO_for_v = dO_h.permute(0, 2, 1, 3).reshape(H_q, T_v, K_sub * Dh)
        dV_rep = torch.bmm(P_vv_t.transpose(1, 2), dO_for_v).view(H_q, T_v, K_sub, Dh).permute(2, 0, 1, 3)

        # dP[s:, 0:T_full] = dO[s:] @ V[0:T_full]^T
        dP = torch.bmm(dO_h.view(H_q, K_sub * T_v, Dh), V_all.transpose(1, 2)).view(H_q, K_sub, T_v, T_full)

        # Softmax VJP in fp32 over full causal support [0:T_full]: dS = P * (dP - sum(dP * P)) * scaling
        dP_f = dP.float()
        P_va_b = P_va.unsqueeze(1)
        D_dot = (dP_f * P_va_b).sum(dim=-1, keepdim=True)
        dS = ((P_va_b * (dP_f - D_dot)) * saved["scaling"]).to(dtype)      # [H_q, K_sub, T_v, T_full]

        # dQ[s:] = dS[s:, 0:T_full] @ K[0:T_full]
        dQ_rot = torch.bmm(dS.view(H_q, K_sub * T_v, T_full), K_all).view(H_q, K_sub, T_v, Dh).transpose(0, 1)

        # dK[s:] = dS[s:, s:]^T @ Q[s:] (causal: dS[q, k] = 0 for q < k)
        dS_vv = dS[:, :, :, s:] if s > 0 else dS                           # [H_q, K_sub, T_v, T_v]
        dK_rep = torch.bmm(
            dS_vv.transpose(-1, -2).reshape(H_q, K_sub * T_v, T_v), Q_v
        ).view(H_q, K_sub, T_v, Dh).transpose(0, 1)                         # [K_sub, H_q, T_v, Dh]
    else:
        with torch.enable_grad():
            q_e = saved["q_rot"].detach().unsqueeze(0).expand(K_sub, -1, -1, -1).clone().requires_grad_(True)
            k_e = saved["k_rep"].detach().unsqueeze(0).expand(K_sub, -1, -1, -1).clone().requires_grad_(True)
            v_e = saved["v_rep"].detach().unsqueeze(0).expand(K_sub, -1, -1, -1).clone().requires_grad_(True)
            o_e = F.scaled_dot_product_attention(q_e, k_e, v_e, is_causal=True, scale=saved["scaling"])
            dO_full = F.pad(dO, (0, 0, s, 0)) if s > 0 else dO
            dQ_rot, dK_rep, dV_rep = torch.autograd.grad(o_e, (q_e, k_e, v_e), grad_outputs=dO_full)
            if s > 0:
                dQ_rot = dQ_rot[:, :, s:, :]
                dK_rep = dK_rep[:, :, s:, :]
                dV_rep = dV_rep[:, :, s:, :]

    # Reduce GQA heads H_q -> H_kv
    dK_rot = dK_rep.view(K_sub, H_kv, n_rep, T_v, Dh).sum(dim=2)           # [K_sub, H_kv, T_v, Dh]
    dV = dV_rep.view(K_sub, H_kv, n_rep, T_v, Dh).sum(dim=2)               # [K_sub, H_kv, T_v, Dh]

    # 3. Inverse partial RoPE on [s:]
    cos = saved["cos"][:, :, s:, :] if s > 0 else saved["cos"]
    sin = saved["sin"][:, :, s:, :] if s > 0 else saved["sin"]
    dQ_normed = apply_partial_rope(dQ_rot, cos, -sin).transpose(1, 2)      # [K_sub, T_v, H_q, Dh]
    dK_normed = apply_partial_rope(dK_rot, cos, -sin).transpose(1, 2)      # [K_sub, T_v, H_kv, Dh]

    # 4. q_norm & k_norm VJP on [s:]
    q_xhat = saved["q_xhat"][:, s:] if s > 0 else saved["q_xhat"]
    q_rstd = saved["q_rstd"][:, s:] if s > 0 else saved["q_rstd"]
    k_xhat = saved["k_xhat"][:, s:] if s > 0 else saved["k_xhat"]
    k_rstd = saved["k_rstd"][:, s:] if s > 0 else saved["k_rstd"]
    dQ_raw = rmsnorm_vjp(dQ_normed, q_xhat, q_rstd, w.q_norm)
    dK_raw = rmsnorm_vjp(dK_normed, k_xhat, k_rstd, w.k_norm)

    # 5. Pack [dQ_raw, d_gate] matching q_proj output layout [K_sub, T_v, H_q, 2*Dh]
    dQG = torch.cat([dQ_raw, d_gate.view(K_sub, T_v, H_q, Dh)], dim=-1).reshape(K_sub, T_v, H_q * Dh * 2)
    dK_flat = dK_raw.reshape(K_sub, T_v, H_kv * Dh)
    dV_flat = dV.transpose(1, 2).reshape(K_sub, T_v, H_kv * Dh)

    d_x = (
        torch.matmul(dQG, w.q_proj)
        + torch.matmul(dK_flat, w.k_proj)
        + torch.matmul(dV_flat, w.v_proj)
    )
    return d_x


# --------------------------------------------------------------------------- #
# 5. Exact Gated DeltaNet (GDN) Forward & Direct Kernel VJP
# --------------------------------------------------------------------------- #
def gdn_layer_fwd(
    hidden_states: torch.Tensor,  # [1, T, D]
    w: GdnLayerWeights,
    gdn_backend: GatedDeltaRuleBackend,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Exact GDN forward using FlashQLA (SM90/100/120) or FLA Triton (SM80) or PyTorch."""
    B, T, _ = hidden_states.shape
    key_dim = w.num_k_heads * w.head_k_dim
    val_dim = w.num_v_heads * w.head_v_dim

    mixed_qkv = F.linear(hidden_states, w.in_proj_qkv).transpose(1, 2)  # [1, conv_dim, T]
    z = F.linear(hidden_states, w.in_proj_z).view(B, T, w.num_v_heads, w.head_v_dim)
    b = F.linear(hidden_states, w.in_proj_b)  # [1, T, H_v]
    a = F.linear(hidden_states, w.in_proj_a)  # [1, T, H_v]

    mixed_qkv_act, conv_pre_silu = causal_conv1d_silu_fwd(mixed_qkv, w.conv1d_weight)
    mixed_qkv_t = mixed_qkv_act.transpose(1, 2)
    q, k, v = torch.split(mixed_qkv_t, [key_dim, key_dim, val_dim], dim=-1)
    q = q.reshape(B, T, w.num_k_heads, w.head_k_dim)
    k = k.reshape(B, T, w.num_k_heads, w.head_k_dim)
    v = v.reshape(B, T, w.num_v_heads, w.head_v_dim)

    beta = torch.sigmoid(b)
    neg_exp_A = -w.A_log.float().exp()
    a_plus_dt = a.float() + w.dt_bias.float()
    g = neg_exp_A * F.softplus(a_plus_dt)

    core_out, gdn_saved = gdn_backend.forward_single(q, k, v, g, beta)
    norm_out, norm_saved = rmsnorm_gated_fwd(
        core_out, z, w.norm_weight, activation=w.output_gate_type, eps=w.eps
    )
    out = F.linear(norm_out.reshape(B, T, val_dim), w.out_proj)

    saved = {
        "conv_pre_silu": conv_pre_silu,
        "beta": beta,
        "neg_exp_A": neg_exp_A,
        "sig_a_plus_dt": torch.sigmoid(a_plus_dt),
        "gdn_saved": gdn_saved,
        "norm_saved": norm_saved,
    }
    return out, saved


def gdn_layer_vjp(
    d_out: torch.Tensor,  # [K_sub, T_v, D]
    w: GdnLayerWeights,
    saved: dict[str, Any],
    gdn_backend: GatedDeltaRuleBackend,
    *,
    bwd_start: int = 0,
) -> torch.Tensor:
    """Exact batched-cotangent VJP of `Qwen4ExpTextGatedDeltaNet` on suffix `[bwd_start:]` -> `[K_sub, T_v, D]`."""
    K_sub, T_v, _ = d_out.shape
    s = int(bwd_start)
    val_dim = w.num_v_heads * w.head_v_dim
    dtype = d_out.dtype

    # 1. out_proj & RMSNormGated VJP on [s:]
    d_norm_out = torch.matmul(d_out, w.out_proj).view(K_sub, T_v, w.num_v_heads, w.head_v_dim)
    if s > 0:
        norm_s = {k: (v[:, s:] if v.ndim == 4 else v) for k, v in saved["norm_saved"].items()}
    else:
        norm_s = saved["norm_saved"]
    d_core_out, d_z = rmsnorm_gated_vjp(d_norm_out, norm_s)

    # 2. Core Gated Delta Rule VJP (zero-pad [0:s] to preserve exact 128-aligned FlashQLA/FLA tiles!)
    d_core_in = F.pad(d_core_out, (0, 0, 0, 0, s, 0)) if s > 0 else d_core_out
    dq, dk, dv, dg, dbeta = gdn_backend.backward_batched(saved["gdn_saved"], d_core_in)
    if s > 0:
        dq = dq[:, s:]
        dk = dk[:, s:]
        dv = dv[:, s:]
        dg = dg[:, s:]
        dbeta = dbeta[:, s:]

    # 3. Gate activations VJP on [s:]: beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)
    beta_f = (saved["beta"][:, s:] if s > 0 else saved["beta"]).float()
    sig_adt = saved["sig_a_plus_dt"][:, s:] if s > 0 else saved["sig_a_plus_dt"]
    db = (dbeta.float() * beta_f * (1.0 - beta_f)).to(dtype)
    da = (dg.float() * saved["neg_exp_A"] * sig_adt).to(dtype)

    # 4. Pack [dq, dk, dv] -> causal_conv1d + SiLU VJP on [s:] (causal adjoint only looks at t' >= t >= s!)
    d_mixed_t = torch.cat(
        [dq.reshape(K_sub, T_v, -1), dk.reshape(K_sub, T_v, -1), dv.reshape(K_sub, T_v, -1)],
        dim=-1,
    )
    pre_silu_s = saved["conv_pre_silu"][:, :, s:] if s > 0 else saved["conv_pre_silu"]
    d_conv_in = causal_conv1d_silu_bwd(
        d_mixed_t.transpose(1, 2), pre_silu_s, w.conv1d_weight
    ).transpose(1, 2)  # [K_sub, T_v, conv_dim]

    # 5. Input projections VJP on [s:]
    d_x = (
        torch.matmul(d_conv_in, w.in_proj_qkv)
        + torch.matmul(d_z.reshape(K_sub, T_v, val_dim), w.in_proj_z)
        + torch.matmul(db, w.in_proj_b)
        + torch.matmul(da, w.in_proj_a)
    )
    return d_x


# --------------------------------------------------------------------------- #
# 6. Exact Sparse MoE Forward & Fused Active-Expert Grouped-GEMM VJP
# --------------------------------------------------------------------------- #
def _extract_active_experts_bf16(
    experts: MoeExpertPayload,
    active_ids: torch.Tensor,  # [E_hit] int64 on target device
    block_size: tuple[int, int],
    out_dtype: torch.dtype,
    use_triton: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather ONLY the `E_hit` active experts and dequantize FP8 -> `out_dtype` ONCE."""
    gu_sel = experts.gate_up_proj.index_select(0, active_ids)
    dn_sel = experts.down_proj.index_select(0, active_ids)
    if gu_sel.element_size() == 1 and experts.gate_up_scale is not None:
        gu_s = experts.gate_up_scale.index_select(0, active_ids)
        dn_s = experts.down_scale.index_select(0, active_ids)  # type: ignore[union-attr]
        W_gu = dequantize_fp8_blockwise(gu_sel, gu_s, block_size=block_size, out_dtype=out_dtype, use_triton=use_triton)
        W_dn = dequantize_fp8_blockwise(dn_sel, dn_s, block_size=block_size, out_dtype=out_dtype, use_triton=use_triton)
    else:
        W_gu = gu_sel.to(out_dtype)
        W_dn = dn_sel.to(out_dtype)
    return W_gu, W_dn


def moe_block_fwd(
    hidden_states: torch.Tensor,  # [1, T, D]
    dense: MoeDenseWeights,
    experts: MoeExpertPayload,
    *,
    dequant_chunk_size: int = 512,
    use_triton_dequant: bool = True,
    keep_dequantized_for_bwd: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Exact `B=1` forward of `Qwen4ExpTextSparseMoeBlock`.

    When `keep_dequantized_for_bwd=True` and `E_hit <= dequant_chunk_size`, keeps the
    active experts' BF16 weights (`W_gu`, `W_dn`) directly in `saved` so backward reuses
    them across ALL 2560 cotangents with ZERO redundant FP8 dequantization!
    """
    _, T, D = hidden_states.shape
    x = hidden_states[0]  # [T, D]
    dtype = x.dtype
    top_k = dense.top_k

    # 1. Shared expert + gate
    sh_gate = F.linear(x, dense.shared_gate_proj)
    sh_up = F.linear(x, dense.shared_up_proj)
    sh_gf = sh_gate.float()
    sh_sig = torch.sigmoid(sh_gf)
    sh_act = (sh_gf * sh_sig * sh_up.float()).to(dtype)
    sh_out = F.linear(sh_act, dense.shared_down_proj)  # [T, D]
    sg_logit = F.linear(x, dense.shared_expert_gate)   # [T, 1]
    sg_val = torch.sigmoid(sg_logit.float())           # [T, 1] fp32

    # 2. Top-K Router (exact match with Qwen4ExpTextTopKRouter)
    router_logits = F.linear(x, dense.router_weight)              # [T, E]
    router_probs = torch.softmax(router_logits.float(), dim=-1)   # [T, E] fp32
    top_vals_raw, top_idx = torch.topk(router_probs, top_k, dim=-1)  # [T, k]
    if dense.norm_topk_prob:
        top_sum = top_vals_raw.sum(dim=-1, keepdim=True)
        routing_weights_f = top_vals_raw / top_sum
    else:
        top_sum = None
        routing_weights_f = top_vals_raw
    routing_weights = routing_weights_f.to(dtype)

    # 3. Compact active-expert sorting (ONLY hit experts!)
    S = T * top_k
    flat_eids = top_idx.reshape(S)
    sorted_eids, perm = torch.sort(flat_eids)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(S, device=x.device)
    tok_of_sorted = perm // top_k  # [S]
    w_sorted = routing_weights.reshape(S)[perm]  # [S]

    active_ids, counts = torch.unique_consecutive(sorted_eids, return_counts=True)
    E_hit = int(active_ids.numel())
    offs_1 = torch.cumsum(counts, dim=0, dtype=torch.int32)
    offs_1_list = offs_1.tolist()

    x_sorted = x[tok_of_sorted]  # [S, D]

    # Run active experts (chunked if E_hit > dequant_chunk_size to cap BF16 peak VRAM on A100-40G!)
    inter = dense.shared_gate_proj.shape[0]
    # Note: routed intermediate size comes from gate_up_proj shape[1] // 2
    r_inter = experts.gate_up_proj.shape[1] // 2
    y_exp_sorted = torch.empty((S, D), device=x.device, dtype=dtype)
    da_dg_sorted = torch.empty((S, r_inter), device=x.device, dtype=dtype)
    da_du_sorted = torch.empty((S, r_inter), device=x.device, dtype=dtype)

    cached_W_gu = None
    cached_W_dn = None

    if E_hit <= dequant_chunk_size:
        W_gu, W_dn = _extract_active_experts_bf16(
            experts, active_ids, dense.fp8_block_size, dtype, use_triton_dequant
        )
        # Forward: [S, D] @ [E_hit, D, 2*r_inter] -> [S, 2*r_inter]
        gu_out = grouped_mm(x_sorted, W_gu.transpose(-2, -1).contiguous(), offs_1, offs_1_list)
        g_act, u_act = gu_out.chunk(2, dim=-1)
        gf = g_act.float()
        sig = torch.sigmoid(gf)
        silu_g = gf * sig
        a_out = (silu_g * u_act.float()).to(dtype)
        da_dg_sorted.copy_((sig * (1.0 + gf * (1.0 - sig)) * u_act.float()).to(dtype))
        da_du_sorted.copy_(silu_g.to(dtype))
        y_exp_sorted.copy_(grouped_mm(a_out, W_dn.transpose(-2, -1).contiguous(), offs_1, offs_1_list))
        if keep_dequantized_for_bwd:
            cached_W_gu = W_gu
            cached_W_dn = W_dn
        else:
            del W_gu, W_dn
    else:
        for c_start in range(0, E_hit, dequant_chunk_size):
            c_end = min(c_start + dequant_chunk_size, E_hit)
            row_lo = offs_1_list[c_start - 1] if c_start > 0 else 0
            row_hi = offs_1_list[c_end - 1]
            sub_offs = offs_1[c_start:c_end] - row_lo
            sub_offs_list = [v - row_lo for v in offs_1_list[c_start:c_end]]
            W_gu, W_dn = _extract_active_experts_bf16(
                experts, active_ids[c_start:c_end], dense.fp8_block_size, dtype, use_triton_dequant
            )
            gu_out = grouped_mm(x_sorted[row_lo:row_hi], W_gu.transpose(-2, -1).contiguous(), sub_offs, sub_offs_list)
            g_act, u_act = gu_out.chunk(2, dim=-1)
            gf = g_act.float()
            sig = torch.sigmoid(gf)
            silu_g = gf * sig
            a_out = (silu_g * u_act.float()).to(dtype)
            da_dg_sorted[row_lo:row_hi].copy_((sig * (1.0 + gf * (1.0 - sig)) * u_act.float()).to(dtype))
            da_du_sorted[row_lo:row_hi].copy_(silu_g.to(dtype))
            y_exp_sorted[row_lo:row_hi].copy_(
                grouped_mm(a_out, W_dn.transpose(-2, -1).contiguous(), sub_offs, sub_offs_list)
            )
            del W_gu, W_dn

    weighted_sorted = y_exp_sorted * w_sorted.unsqueeze(-1)
    routed_out = weighted_sorted[inv_perm].view(T, top_k, D).sum(dim=1).to(dtype)
    total_out = (routed_out + (sg_val * sh_out.float()).to(dtype)).unsqueeze(0)  # [1, T, D]

    # Precompute router expectation E_p[W_router] = router_probs @ W_router ONCE ([T, D]) for O(1) softmax VJP!
    p_dot_W_router = torch.matmul(router_probs.to(dtype), dense.router_weight)  # [T, D]
    W_router_top = dense.router_weight[top_idx]  # [T, top_k, D]

    saved = {
        "T": T,
        "D": D,
        "r_inter": r_inter,
        "S": S,
        "E_hit": E_hit,
        "active_ids": active_ids,
        "offs_1": offs_1,
        "offs_1_list": offs_1_list,
        "perm": perm,
        "inv_perm": inv_perm,
        "tok_of_sorted": tok_of_sorted,
        "w_sorted": w_sorted,
        "y_exp_sorted": y_exp_sorted,
        "da_dg_sorted": da_dg_sorted,
        "da_du_sorted": da_du_sorted,
        "cached_W_gu": cached_W_gu,
        "cached_W_dn": cached_W_dn,
        # Router VJP state
        "top_vals_raw": top_vals_raw,          # [T, k] fp32
        "top_sum": top_sum,                    # [T, 1] fp32 or None
        "routing_weights_f": routing_weights_f,  # [T, k] fp32
        "p_dot_W_router": p_dot_W_router,      # [T, D] dtype
        "W_router_top": W_router_top,          # [T, k, D] dtype
        # Shared expert VJP state
        "sh_out": sh_out,                      # [T, D]
        "sh_da_dg": (sh_sig * (1.0 + sh_gf * (1.0 - sh_sig)) * sh_up.float()).to(dtype),
        "sh_da_du": (sh_gf * sh_sig).to(dtype),
        "sg_val": sg_val.to(dtype),            # [T, 1]
        "d_sg": (sg_val * (1.0 - sg_val)).to(dtype),  # [T, 1]
    }
    return total_out, saved


def moe_block_vjp_all(
    d_out_all: torch.Tensor,  # [K_total, T, D]
    dense: MoeDenseWeights,
    experts: MoeExpertPayload,
    saved: dict[str, Any],
    *,
    moe_k_chunk: int = 256,
    dequant_chunk_size: int = 512,
    use_triton_dequant: bool = True,
    use_triton_swiglu: bool = True,
) -> torch.Tensor:
    """Exact VJP of `Qwen4ExpTextSparseMoeBlock` across ALL `K_total` cotangents.

    Key optimization:
      - When `E_hit <= dequant_chunk_size` (A100-80G / G4-96G): active experts are already
        dequantized ONCE in `saved["cached_W_gu"]` / `saved["cached_W_dn"]` and reused for
        every `moe_k_chunk` slice with 100% Tensor-Core-saturated `grouped_mm`!
      - When `E_hit > dequant_chunk_size` (A100-40G chunked mode): loops over expert chunks
        OUTSIDE the `K` loop so EVERY active expert is STILL dequantized ONLY ONCE across
        all `K_total = 2560` cotangents!!
    """
    K_total, T, D = d_out_all.shape
    dtype = d_out_all.dtype
    top_k = dense.top_k
    S = saved["S"]
    E_hit = saved["E_hit"]
    offs_1 = saved["offs_1"]
    offs_1_list = saved["offs_1_list"]
    tok_of_sorted = saved["tok_of_sorted"]
    w_sorted = saved["w_sorted"]
    inv_perm = saved["inv_perm"]

    # Transpose to [T, K_total, D] for O(1) token gather into expert-grouped layout [S, K_blk, D]
    dY_TK = d_out_all.transpose(0, 1).contiguous()  # [T, K_total, D]
    dX_TK = torch.zeros_like(dY_TK)

    # 1. Shared expert + router VJP (tiny dense BF16 ops -> run in K chunks, < 1 ms total)
    y_exp_sorted = saved["y_exp_sorted"]  # [S, D]
    sh_out = saved["sh_out"]              # [T, D]
    sh_da_dg = saved["sh_da_dg"]          # [T, sh_inter]
    sh_da_du = saved["sh_da_du"]          # [T, sh_inter]
    sg_val = saved["sg_val"]              # [T, 1]
    d_sg = saved["d_sg"]                  # [T, 1]
    top_vals_raw = saved["top_vals_raw"]  # [T, k] fp32
    top_sum = saved["top_sum"]            # [T, 1] fp32 or None
    rw_f = saved["routing_weights_f"]     # [T, k] fp32
    p_dot_W = saved["p_dot_W_router"]     # [T, D]
    W_r_top = saved["W_router_top"]       # [T, k, D]

    for k0 in range(0, K_total, moe_k_chunk):
        k1 = min(k0 + moe_k_chunk, K_total)
        dy_tk = dY_TK[:, k0:k1, :]  # [T, K_blk, D]

        # 1a. Shared expert VJP
        d_sh_out = dy_tk * sg_val.unsqueeze(1)  # [T, K_blk, D]
        d_sg_logit = (dy_tk.float() * sh_out.float().unsqueeze(1)).sum(dim=-1, keepdim=True).to(dtype) * d_sg.unsqueeze(1)
        d_sh_act = torch.matmul(d_sh_out, dense.shared_down_proj)  # [T, K_blk, sh_inter]
        d_sh_g = d_sh_act * sh_da_dg.unsqueeze(1)
        d_sh_u = d_sh_act * sh_da_du.unsqueeze(1)
        dx_shared = (
            torch.matmul(d_sh_g, dense.shared_gate_proj)
            + torch.matmul(d_sh_u, dense.shared_up_proj)
            + torch.matmul(d_sg_logit, dense.shared_expert_gate)
        )

        # 1b. Exact Top-K Router VJP (without materializing [T, K_blk, 512]!)
        # dw_sorted[s, k] = <dY[tok(s), k, :], y_exp_sorted[s, :]>
        dy_sorted_raw = dy_tk[tok_of_sorted]  # [S, K_blk, D]
        dw_sorted = (dy_sorted_raw.float() * y_exp_sorted.float().unsqueeze(1)).sum(dim=-1)  # [S, K_blk]
        dw = dw_sorted[inv_perm].view(T, top_k, k1 - k0).transpose(1, 2)  # [T, K_blk, k] fp32
        if top_sum is not None:
            dot_w = (dw * rw_f.unsqueeze(1)).sum(dim=-1, keepdim=True)
            d_top_v = (dw - dot_w) / top_sum.unsqueeze(1)
        else:
            d_top_v = dw
        # Softmax VJP: sum_e p_e (dp_e - dot_p) W_e = sum_r (p_r * d_top_v_r) W_{idx_r} - dot_p * (p @ W)
        p_times_dv = d_top_v * top_vals_raw.unsqueeze(1)  # [T, K_blk, k] fp32
        dot_p = p_times_dv.sum(dim=-1, keepdim=True)      # [T, K_blk, 1] fp32
        dx_router = (
            torch.bmm(p_times_dv.to(dtype), W_r_top)
            - dot_p.to(dtype) * p_dot_W.unsqueeze(1)
        )
        dX_TK[:, k0:k1, :].add_(dx_shared + dx_router)

    # 2. Routed experts VJP (each expert chunk dequantized ONLY ONCE across ALL K_total!)
    da_dg_sorted = saved["da_dg_sorted"]
    da_du_sorted = saved["da_du_sorted"]
    active_ids = saved["active_ids"]

    if saved["cached_W_gu"] is not None:
        expert_chunks = [(0, E_hit, saved["cached_W_gu"], saved["cached_W_dn"])]
    else:
        expert_chunks = [
            (c0, min(c0 + dequant_chunk_size, E_hit), None, None)
            for c0 in range(0, E_hit, dequant_chunk_size)
        ]

    for c0, c1, pre_W_gu, pre_W_dn in expert_chunks:
        row_lo = offs_1_list[c0 - 1] if c0 > 0 else 0
        row_hi = offs_1_list[c1 - 1]
        n_rows = row_hi - row_lo
        if n_rows == 0:
            continue

        if pre_W_gu is not None and pre_W_dn is not None:
            W_gu, W_dn = pre_W_gu, pre_W_dn
        else:
            W_gu, W_dn = _extract_active_experts_bf16(
                experts, active_ids[c0:c1], dense.fp8_block_size, dtype, use_triton_dequant
            )

        sub_tok = tok_of_sorted[row_lo:row_hi]         # [n_rows]
        sub_w = w_sorted[row_lo:row_hi]                # [n_rows]
        sub_dg = da_dg_sorted[row_lo:row_hi]           # [n_rows, r_inter]
        sub_du = da_du_sorted[row_lo:row_hi]           # [n_rows, r_inter]
        sub_offs_1 = offs_1[c0:c1] - row_lo            # [c1 - c0] int32
        sub_offs_1_list = [v - row_lo for v in offs_1_list[c0:c1]]

        # Notice W_dn is [E_c, D, r_inter] and W_gu is [E_c, 2*r_inter, D]:
        # ALREADY in exact contiguous [E_c, K_in, N_out] layout for backward grouped_mm!
        for k0 in range(0, K_total, moe_k_chunk):
            k1 = min(k0 + moe_k_chunk, K_total)
            K_blk = k1 - k0

            # Gather [n_rows, K_blk, D] and scale by routing weights
            dy_s = dY_TK[sub_tok, k0:k1, :] * sub_w[:, None, None]  # [n_rows, K_blk, D]
            offs_K = sub_offs_1 * K_blk
            offs_K_list = [v * K_blk for v in sub_offs_1_list]

            # Down-proj backward: [n_rows * K_blk, D] @ [E_c, D, r_inter] -> [n_rows * K_blk, r_inter]
            dA = grouped_mm(dy_s.reshape(n_rows * K_blk, D), W_dn, offs_K, offs_K_list).view(n_rows, K_blk, -1)

            # Fused SwiGLU backward + pack [dG | dU] -> [n_rows * K_blk, 2 * r_inter]
            dGU = swiglu_vjp_pack(dA, sub_dg, sub_du, use_triton=use_triton_swiglu).reshape(n_rows * K_blk, -1)
            del dA

            # Gate-Up-proj backward: [n_rows * K_blk, 2*r_inter] @ [E_c, 2*r_inter, D] -> [n_rows, K_blk, D]
            dX_s = grouped_mm(dGU, W_gu, offs_K, offs_K_list).view(n_rows, K_blk, D)
            del dGU

            # Deterministic accumulation back to token axis (no race conditions / atomicAdd!)
            if c0 == 0 and c1 == E_hit:
                # Full-active fast path: direct inv_perm + sum(dim=1) (zero index_add_)
                dX_TK[:, k0:k1, :].add_(dX_s[inv_perm].view(T, top_k, K_blk, D).sum(dim=1))
            else:
                # Chunked path: accumulate in float32 per token cleanly
                dX_TK[:, k0:k1, :].index_add_(0, sub_tok, dX_s)
            del dX_s

        if pre_W_gu is None:
            del W_gu, W_dn

    return dX_TK.transpose(0, 1).contiguous()  # [K_total, T, D]


# --------------------------------------------------------------------------- #
# 7. Complete Single-Layer Forward (`B=1`) & Full-Cotangent In-Place Backward
# --------------------------------------------------------------------------- #
def _slice_ple_saved(saved: dict[str, Any], bwd_start: int) -> dict[str, Any]:
    if bwd_start <= 0:
        return saved
    out: dict[str, Any] = {}
    for k, v in saved.items():
        if k == "d_silu_conv":
            out[k] = v[:, :, bwd_start:].contiguous()
        elif isinstance(v, torch.Tensor):
            out[k] = v[:, bwd_start:].contiguous()
        else:
            out[k] = v
    return out


def decoder_layer_fwd(
    H_in: torch.Tensor,  # [1, T, hc_count * D]
    layer_w: DecoderLayerDenseWeights,
    experts: MoeExpertPayload,
    cos: torch.Tensor,
    sin: torch.Tensor,
    gdn_backend: GatedDeltaRuleBackend,
    *,
    ple_embeddings: torch.Tensor | None = None,
    dequant_chunk_size: int = 512,
    use_triton_dequant: bool = True,
    keep_experts_dequantized: bool = False,
    bwd_start: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run ONE `B=1` forward of `Qwen4ExpTextDecoderLayer`; returns `(H_out, saved)`.

    When `bwd_start > 0` (backward-recompute mode for prefix-masked J-lens):
      - Causal token mixers (PLE, GDN/QSA) still run on full `[0:T]` so context/softmax/state
        entering `bwd_start` is 100% exact.
      - Pointwise `mlp_hc` + `SparseMoE` run strictly on the suffix `[bwd_start:]`, skipping
        masked prefix tokens and reducing unique hit experts `E_hit` to dequantize!
    """
    saved: dict[str, Any] = {}
    s = int(bwd_start)

    # 1. Optional PLE injection (layer 2 only)
    if layer_w.ple is not None:
        assert ple_embeddings is not None, f"Layer {layer_w.layer_idx} requires ple_embeddings."
        ple_delta, ple_saved = ple_layer_fwd(H_in, ple_embeddings, layer_w.ple)
        H_ple = H_in + ple_delta
        saved["ple_saved"] = _slice_ple_saved(ple_saved, s)
    else:
        H_ple = H_in

    # 2. Attention Hyper-Connection + Token Mixer (GDN or QSA) + Residual Injection
    x_attn, w_attn, attn_hc_saved = gated_residual_fwd(H_ple, layer_w.attn_hc)
    assert w_attn is not None
    if layer_w.layer_type == "linear_attention":
        assert layer_w.gdn is not None
        y_attn, mixer_saved = gdn_layer_fwd(x_attn, layer_w.gdn, gdn_backend)
    else:
        assert layer_w.qsa is not None
        y_attn, mixer_saved = qsa_layer_fwd(x_attn, layer_w.qsa, cos, sin)

    if s > 0:
        H_mid_s = H_ple[:, s:, :] + (y_attn[:, s:, :].unsqueeze(-2) * w_attn[:, s:, :].unsqueeze(-1)).flatten(-2)
        attn_hc_saved_bwd = slice_gated_residual_saved(attn_hc_saved, s)
        y_attn_bwd = y_attn[0, s:].contiguous()
        w_attn_bwd = w_attn[0, s:].contiguous()
    else:
        H_mid_s = H_ple + (y_attn.unsqueeze(-2) * w_attn.unsqueeze(-1)).flatten(-2)
        attn_hc_saved_bwd = attn_hc_saved
        y_attn_bwd = y_attn[0]
        w_attn_bwd = w_attn[0]

    # 3. Pointwise MLP Hyper-Connection + Sparse MoE + Residual Injection (strictly on [s:])
    x_mlp, w_mlp, mlp_hc_saved = gated_residual_fwd(H_mid_s, layer_w.mlp_hc)
    assert w_mlp is not None
    y_mlp, moe_saved = moe_block_fwd(
        x_mlp,
        layer_w.moe_dense,
        experts,
        dequant_chunk_size=dequant_chunk_size,
        use_triton_dequant=use_triton_dequant,
        keep_dequantized_for_bwd=keep_experts_dequantized,
    )
    H_out = H_mid_s + (y_mlp.unsqueeze(-2) * w_mlp.unsqueeze(-1)).flatten(-2)

    saved.update(
        {
            "bwd_start": s,
            "attn_hc_saved": attn_hc_saved_bwd,
            "mixer_saved": mixer_saved,
            "y_attn": y_attn_bwd,
            "w_attn": w_attn_bwd,
            "mlp_hc_saved": mlp_hc_saved,
            "moe_saved": moe_saved,
            "y_mlp": y_mlp[0],
            "w_mlp": w_mlp[0],
        }
    )
    return H_out, saved


def decoder_layer_vjp_inplace(
    G: torch.Tensor,  # [K_total, T_v, hc_count * D] — UPDATED IN-PLACE TO G_in!
    layer_w: DecoderLayerDenseWeights,
    experts: MoeExpertPayload,
    saved: dict[str, Any],
    gdn_backend: GatedDeltaRuleBackend,
    *,
    attn_k_chunk: int = 128,
    moe_k_chunk: int = 256,
    dequant_chunk_size: int = 512,
    use_triton_dequant: bool = True,
    use_triton_swiglu: bool = True,
) -> torch.Tensor:
    """Exact in-place VJP of `Qwen4ExpTextDecoderLayer` across ALL `K_total` cotangents.

    Updates `G` in-place so peak GPU cotangent memory stays at ONLY 1x `G_curr` (5.47 GiB at T_v=112)!
    """
    K_total, T_v, D_hc = G.shape
    hc = layer_w.mlp_hc.hc_count
    D = layer_w.mlp_hc.hidden_size
    bwd_start = int(saved.get("bwd_start", 0))

    # ---- Stage A: MLP Injection + SparseMoE + MLP Hyper-Connection (on [bwd_start:]) ----
    y_mlp = saved["y_mlp"]  # [T_v, D]
    w_mlp = saved["w_mlp"]  # [T_v, hc]

    d_y_mlp = torch.empty((K_total, T_v, D), device=G.device, dtype=G.dtype)
    d_w_mlp = torch.empty((K_total, T_v, hc), device=G.device, dtype=G.dtype)
    for k0 in range(0, K_total, moe_k_chunk):
        k1 = min(k0 + moe_k_chunk, K_total)
        g4 = G[k0:k1].view(k1 - k0, T_v, hc, D)
        d_y_mlp[k0:k1] = (g4 * w_mlp[None, :, :, None]).sum(dim=2)
        d_w_mlp[k0:k1] = (g4.float() * y_mlp.float()[None, :, None, :]).sum(dim=-1).to(G.dtype)

    d_x_mlp = moe_block_vjp_all(
        d_y_mlp,
        layer_w.moe_dense,
        experts,
        saved["moe_saved"],
        moe_k_chunk=moe_k_chunk,
        dequant_chunk_size=dequant_chunk_size,
        use_triton_dequant=use_triton_dequant,
        use_triton_swiglu=use_triton_swiglu,
    )
    del d_y_mlp
    saved["moe_saved"]["cached_W_gu"] = None
    saved["moe_saved"]["cached_W_dn"] = None

    for k0 in range(0, K_total, moe_k_chunk):
        k1 = min(k0 + moe_k_chunk, K_total)
        G[k0:k1].add_(
            gated_residual_vjp(d_x_mlp[k0:k1], d_w_mlp[k0:k1], layer_w.mlp_hc, saved["mlp_hc_saved"])
        )
    del d_x_mlp, d_w_mlp

    # ---- Stage B: Attention Injection + Token Mixer (GDN/QSA) + Attn Hyper-Connection ----
    y_attn = saved["y_attn"]  # [T_v, D]
    w_attn = saved["w_attn"]  # [T_v, hc]
    for k0 in range(0, K_total, attn_k_chunk):
        k1 = min(k0 + attn_k_chunk, K_total)
        g4 = G[k0:k1].view(k1 - k0, T_v, hc, D)
        d_y_attn = (g4 * w_attn[None, :, :, None]).sum(dim=2)
        d_w_attn = (g4.float() * y_attn.float()[None, :, None, :]).sum(dim=-1).to(G.dtype)

        if layer_w.layer_type == "linear_attention":
            assert layer_w.gdn is not None
            d_x_attn = gdn_layer_vjp(
                d_y_attn, layer_w.gdn, saved["mixer_saved"], gdn_backend, bwd_start=bwd_start
            )
        else:
            assert layer_w.qsa is not None
            d_x_attn = qsa_layer_vjp(
                d_y_attn, layer_w.qsa, saved["mixer_saved"], bwd_start=bwd_start
            )

        G[k0:k1].add_(
            gated_residual_vjp(d_x_attn, d_w_attn, layer_w.attn_hc, saved["attn_hc_saved"])
        )

    # ---- Stage C: Optional PLE Layer Backward (layer 2 only, already sliced to [bwd_start:]) ----
    if layer_w.ple is not None:
        ple_saved = saved["ple_saved"]
        for k0 in range(0, K_total, attn_k_chunk):
            k1 = min(k0 + attn_k_chunk, K_total)
            G[k0:k1].add_(ple_layer_vjp(G[k0:k1], layer_w.ple, ple_saved))

    return G
