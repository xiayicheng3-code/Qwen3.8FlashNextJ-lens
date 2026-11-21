"""Rigorous numerical parity tests: EVERY analytical VJP vs PyTorch `torch.autograd.grad`.

Compares our custom batched-cotangent VJP operators against the exact HuggingFace
`modeling_qwen4_exp.py` forward formulas differentiated via `torch.autograd.grad`,
including:
  1. Grouped & ungrouped `Qwen4ExpTextRMSNorm` (1+w) and `Qwen4ExpTextRMSNormGated` (sigmoid/silu)
  2. `Qwen4ExpTextGatedResidual` (both `use_combine=True` and `use_combine=False`)
  3. `Qwen4ExpTextPLELayer` (sqrt-sign gate + dilated depthwise conv1d)
  4. `Qwen4ExpTextAttention` (QSA with partial RoPE, GQA, sigmoid output gate, L2-cached VJP)
  5. `Qwen4ExpTextGatedDeltaNet` (causal_conv1d + SiLU + chunked gated delta rule + RMSNormGated)
  6. `Qwen4ExpTextSparseMoeBlock` (FP8 E4M3 block-quantized experts, chunked & full active dequant, exact router grad!)
  7. End-to-end 4-layer `Qwen4Exp` Jacobian Lens fit (`engine.jacobian_for_tokens` vs Anthropic `jlens` estimator!)
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from qwen38_jlens.engine import Qwen38JLensEngine, compute_aligned_bwd_start, valid_position_mask
from qwen38_jlens.hardware import HardwareProfile, StorageTier, build_offload_plan
from qwen38_jlens.kernels import GatedDeltaRuleBackend, dequantize_fp8_blockwise, torch_chunk_gated_delta_rule
from qwen38_jlens.loader import LoadedQwen38Model, Qwen38TextConfig
from qwen38_jlens.model_ops import (
    DecoderLayerDenseWeights,
    GatedResidualWeights,
    GdnLayerWeights,
    MoeDenseWeights,
    MoeExpertPayload,
    PleLayerWeights,
    QsaLayerWeights,
    _slice_ple_saved,
    apply_partial_rope,
    build_rope_cos_sin,
    decoder_layer_fwd,
    decoder_layer_vjp_inplace,
    gated_residual_fwd,
    gated_residual_vjp,
    gdn_layer_fwd,
    gdn_layer_vjp,
    moe_block_fwd,
    moe_block_vjp_all,
    ple_layer_fwd,
    ple_layer_vjp,
    qsa_layer_fwd,
    qsa_layer_vjp,
    rmsnorm_fwd,
    rmsnorm_gated_fwd,
    rmsnorm_gated_vjp,
    rmsnorm_vjp,
)
from qwen38_jlens.ple_mmap import InMemoryPleProvider, PleHashSpec


# --------------------------------------------------------------------------- #
# Reference HuggingFace Qwen4Exp forward functions (for autograd ground truth)
# --------------------------------------------------------------------------- #
def _ref_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6, group_size: int | None = None) -> torch.Tensor:
    xf = x.float()
    if group_size is not None:
        xf = xf.reshape(*x.shape[:-1], -1, group_size)
    out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    if group_size is not None:
        out = out.flatten(-2)
    return (out * (1.0 + w.float())).to(x.dtype)


def _ref_rmsnorm_gated(x: torch.Tensor, gate: torch.Tensor, w: torch.Tensor, act: str, eps: float = 1e-6) -> torch.Tensor:
    dt = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    h = xf * torch.rsqrt(var + eps)
    h = w * h.to(dt)
    g = torch.sigmoid(gate.float()) if act == "sigmoid" else F.silu(gate.float())
    return (h * g).to(dt)


def _ref_gated_residual(h: torch.Tensor, w: GatedResidualWeights):
    h_norm = _ref_rmsnorm(h, w.hc_norm_weight, eps=w.eps, group_size=w.hidden_size)
    mix = F.silu(F.linear(h_norm, w.input_mix_weight_down) / w.hc_count)
    mix = torch.sigmoid(F.linear(mix, w.input_mix_weight_up)).unflatten(-1, (w.hc_count, w.hidden_size))
    mixed = (mix * h_norm.unflatten(-1, (w.hc_count, w.hidden_size))).mean(dim=-2)
    if w.block_inject_weight is None:
        return mixed, None
    inj = 2.0 * torch.sigmoid(F.linear(h_norm, w.block_inject_weight) / w.hc_count)
    return mixed, inj


def _ref_ple(h: torch.Tensor, emb: torch.Tensor, w: PleLayerWeights) -> torch.Tensor:
    key_n = _ref_rmsnorm(F.linear(emb, w.key_proj), w.norm_key, eps=w.eps, group_size=w.hidden_size).unflatten(
        -1, (w.hc_count, w.hidden_size)
    )
    val = F.linear(emb, w.value_proj)
    q_n = _ref_rmsnorm(h, w.norm_query, eps=w.eps, group_size=w.hidden_size).unflatten(
        -1, (w.hc_count, w.hidden_size)
    )
    gate = (key_n * q_n).sum(dim=-1, keepdim=True) / math.sqrt(w.hidden_size)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gv = (torch.sigmoid(gate) * val.unsqueeze(-2)).flatten(-2)
    gv_n = _ref_rmsnorm(gv, w.norm_conv, eps=w.eps, group_size=w.hidden_size)
    pad_l = (w.conv1d_weight.shape[-1] - 1) * w.ngram_size
    c_in = F.pad(gv_n.transpose(1, 2), (pad_l, 0))
    c_out = F.silu(F.conv1d(c_in, w.conv1d_weight, dilation=w.ngram_size, groups=w.hc_count * w.hidden_size)).transpose(1, 2)
    return gv + c_out


def _ref_qsa(x: torch.Tensor, w: QsaLayerWeights, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, T, _ = x.shape
    Hq, Hkv, Dh = w.num_q_heads, w.num_kv_heads, w.head_dim
    q, gate = F.linear(x, w.q_proj).view(B, T, Hq, Dh * 2).chunk(2, dim=-1)
    gate = gate.reshape(B, T, Hq * Dh)
    q = apply_partial_rope(_ref_rmsnorm(q, w.q_norm, w.eps).transpose(1, 2), cos, sin)
    k = apply_partial_rope(_ref_rmsnorm(F.linear(x, w.k_proj).view(B, T, Hkv, Dh), w.k_norm, w.eps).transpose(1, 2), cos, sin)
    v = F.linear(x, w.v_proj).view(B, T, Hkv, Dh).transpose(1, 2)
    n_rep = Hq // Hkv
    k = k.unsqueeze(2).expand(B, Hkv, n_rep, T, Dh).reshape(B, Hq, T, Dh)
    v = v.unsqueeze(2).expand(B, Hkv, n_rep, T, Dh).reshape(B, Hq, T, Dh)
    scores = torch.matmul(q, k.transpose(-1, -2)) * (Dh ** -0.5)
    mask = torch.ones((T, T), dtype=torch.bool, device=x.device).triu(1)
    scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.matmul(F.softmax(scores.float(), dim=-1).to(x.dtype), v).transpose(1, 2).reshape(B, T, Hq * Dh)
    return F.linear(attn * torch.sigmoid(gate), w.o_proj)


def _ref_gdn(x: torch.Tensor, w: GdnLayerWeights) -> torch.Tensor:
    B, T, _ = x.shape
    kd = w.num_k_heads * w.head_k_dim
    vd = w.num_v_heads * w.head_v_dim
    mqkv = F.linear(x, w.in_proj_qkv).transpose(1, 2)
    Kc = w.conv1d_weight.shape[-1]
    mqkv = F.silu(F.conv1d(mqkv, w.conv1d_weight.unsqueeze(1), padding=Kc - 1, groups=mqkv.shape[1])[:, :, :T]).transpose(1, 2)
    q, k, v = torch.split(mqkv, [kd, kd, vd], dim=-1)
    q = q.reshape(B, T, w.num_k_heads, w.head_k_dim)
    k = k.reshape(B, T, w.num_k_heads, w.head_k_dim)
    v = v.reshape(B, T, w.num_v_heads, w.head_v_dim)
    z = F.linear(x, w.in_proj_z).reshape(B, T, w.num_v_heads, w.head_v_dim)
    beta = torch.sigmoid(F.linear(x, w.in_proj_b))
    g = -w.A_log.float().exp() * F.softplus(F.linear(x, w.in_proj_a).float() + w.dt_bias.float())
    core = torch_chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    normed = _ref_rmsnorm_gated(core, z, w.norm_weight, w.output_gate_type, w.eps).reshape(B, T, vd)
    return F.linear(normed, w.out_proj)


def _ref_moe(x: torch.Tensor, dense: MoeDenseWeights, experts: MoeExpertPayload) -> torch.Tensor:
    B, T, D = x.shape
    xf = x.reshape(-1, D)
    sh = F.linear(F.silu(F.linear(xf, dense.shared_gate_proj)) * F.linear(xf, dense.shared_up_proj), dense.shared_down_proj)
    sh = torch.sigmoid(F.linear(xf, dense.shared_expert_gate)) * sh

    logits = F.linear(xf, dense.router_weight)
    probs = F.softmax(logits.float(), dim=-1)
    top_v, top_i = torch.topk(probs, dense.top_k, dim=-1)
    if dense.norm_topk_prob:
        top_v = top_v / top_v.sum(dim=-1, keepdim=True)
    top_v = top_v.to(x.dtype)

    W_gu = (
        dequantize_fp8_blockwise(experts.gate_up_proj, experts.gate_up_scale, dense.fp8_block_size, x.dtype, use_triton=False)
        if experts.gate_up_proj.element_size() == 1 and experts.gate_up_scale is not None
        else experts.gate_up_proj.to(x.dtype)
    )
    W_dn = (
        dequantize_fp8_blockwise(experts.down_proj, experts.down_scale, dense.fp8_block_size, x.dtype, use_triton=False)
        if experts.down_proj.element_size() == 1 and experts.down_scale is not None
        else experts.down_proj.to(x.dtype)
    )
    out = torch.zeros_like(xf)
    for t in range(xf.shape[0]):
        for r in range(dense.top_k):
            e = int(top_i[t, r].item())
            gu = F.linear(xf[t : t + 1], W_gu[e])
            g, u = gu.chunk(2, dim=-1)
            y = F.linear(F.silu(g) * u, W_dn[e])
            out[t : t + 1] = out[t : t + 1] + top_v[t, r] * y
    return (out + sh).reshape(B, T, D)


def _quantize_fp8_block(w: torch.Tensor, bm: int, bn: int) -> tuple[torch.Tensor, torch.Tensor]:
    E, M, N = w.shape
    rm, rn = M // bm, N // bn
    wf = w.float().reshape(E, rm, bm, rn, bn)
    amax = wf.abs().amax(dim=(-3, -1)).clamp_min(1e-6)
    fmax = torch.finfo(torch.float8_e4m3fn).max
    scale_inv = amax / fmax
    q = (wf / scale_inv[:, :, None, :, None]).clamp(-fmax, fmax).to(torch.float8_e4m3fn).reshape(E, M, N)
    return q, scale_inv.float()


# --------------------------------------------------------------------------- #
# Unit Tests
# --------------------------------------------------------------------------- #
def test_rmsnorm_and_gated_vjp_parity():
    torch.manual_seed(0)
    T, D, hc, K = 12, 32, 4, 7
    x = torch.randn(1, T, hc * D)
    w = torch.randn(hc * D) * 0.2
    dy = torch.randn(K, T, hc * D)

    y_fast, xhat, rstd = rmsnorm_fwd(x, w, group_size=D)
    dx_fast = rmsnorm_vjp(dy, xhat, rstd, w, group_size=D)

    x_ref = x.expand(K, -1, -1).clone().requires_grad_(True)
    y_ref = _ref_rmsnorm(x_ref, w, group_size=D)
    (dx_ref,) = torch.autograd.grad(y_ref, x_ref, grad_outputs=dy)

    assert torch.allclose(y_fast, y_ref[:1], atol=1e-6)
    assert torch.allclose(dx_fast, dx_ref, atol=1e-5)

    for act in ("sigmoid", "silu"):
        xg = torch.randn(1, T, 6, 16)
        gg = torch.randn(1, T, 6, 16)
        wg = torch.randn(16) + 1.0
        dyg = torch.randn(K, T, 6, 16)

        yg_fast, sg = rmsnorm_gated_fwd(xg, gg, wg, activation=act)
        dxg_fast, dgg_fast = rmsnorm_gated_vjp(dyg, sg)

        xg_r = xg.expand(K, -1, -1, -1).clone().requires_grad_(True)
        gg_r = gg.expand(K, -1, -1, -1).clone().requires_grad_(True)
        yg_r = _ref_rmsnorm_gated(xg_r, gg_r, wg, act)
        dxg_r, dgg_r = torch.autograd.grad(yg_r, (xg_r, gg_r), grad_outputs=dyg)

        assert torch.allclose(yg_fast, yg_r[:1], atol=1e-6)
        assert torch.allclose(dxg_fast, dxg_r, atol=1e-5)
        assert torch.allclose(dgg_fast, dgg_r, atol=1e-5)


@pytest.mark.parametrize("use_combine", [True, False])
def test_gated_residual_vjp_parity(use_combine: bool):
    torch.manual_seed(1)
    T, D, hc, lowrank, K = 16, 32, 4, 12, 9
    w = GatedResidualWeights(
        hc_norm_weight=torch.randn(hc * D) * 0.1,
        input_mix_weight_down=torch.randn(lowrank, hc * D) * 0.1,
        input_mix_weight_up=torch.randn(hc * D, lowrank) * 0.1,
        block_inject_weight=(torch.randn(hc, hc * D) * 0.1) if use_combine else None,
        hc_count=hc,
        hidden_size=D,
    )
    h = torch.randn(1, T, hc * D)
    d_mixed = torch.randn(K, T, D)
    d_inj = torch.randn(K, T, hc) if use_combine else None

    mixed_f, inj_f, saved = gated_residual_fwd(h, w)
    dh_fast = gated_residual_vjp(d_mixed, d_inj, w, saved)

    h_r = h.expand(K, -1, -1).clone().requires_grad_(True)
    mixed_r, inj_r = _ref_gated_residual(h_r, w)
    loss = (mixed_r * d_mixed).sum()
    if use_combine:
        loss = loss + (inj_r * d_inj).sum()
    (dh_ref,) = torch.autograd.grad(loss, h_r)

    assert torch.allclose(mixed_f, mixed_r[:1], atol=1e-6)
    if use_combine:
        assert torch.allclose(inj_f, inj_r[:1], atol=1e-6)
    assert torch.allclose(dh_fast, dh_ref, atol=1e-5)


def test_ple_layer_vjp_parity():
    torch.manual_seed(2)
    T, D, hc, K = 20, 16, 4, 8
    w = PleLayerWeights(
        key_proj=torch.randn(hc * D, D) * 0.1,
        value_proj=torch.randn(D, D) * 0.1,
        norm_key=torch.randn(hc * D) * 0.1,
        norm_query=torch.randn(hc * D) * 0.1,
        norm_conv=torch.randn(hc * D) * 0.1,
        conv1d_weight=torch.randn(hc * D, 1, 4) * 0.2,
        hc_count=hc,
        hidden_size=D,
        ngram_size=3,
    )
    h = torch.randn(1, T, hc * D)
    ple_emb = torch.randn(1, T, D)
    dy = torch.randn(K, T, hc * D)

    y_f, saved = ple_layer_fwd(h, ple_emb, w)
    dh_fast = ple_layer_vjp(dy, w, saved)

    h_r = h.expand(K, -1, -1).clone().requires_grad_(True)
    y_r = _ref_ple(h_r, ple_emb.expand(K, -1, -1), w)
    (dh_ref,) = torch.autograd.grad(y_r, h_r, grad_outputs=dy)

    assert torch.allclose(y_f, y_r[:1], atol=1e-5)
    assert torch.allclose(dh_fast, dh_ref, atol=1e-4)


def test_qsa_layer_vjp_parity():
    torch.manual_seed(3)
    T, D, Hq, Hkv, Dh, K = 24, 32, 6, 2, 16, 11
    w = QsaLayerWeights(
        q_proj=torch.randn(Hq * Dh * 2, D) * 0.1,
        k_proj=torch.randn(Hkv * Dh, D) * 0.1,
        v_proj=torch.randn(Hkv * Dh, D) * 0.1,
        o_proj=torch.randn(D, Hq * Dh) * 0.1,
        q_norm=torch.randn(Dh) * 0.1,
        k_norm=torch.randn(Dh) * 0.1,
        num_q_heads=Hq,
        num_kv_heads=Hkv,
        head_dim=Dh,
    )
    cos, sin = build_rope_cos_sin(T, Dh, 0.25, 10_000.0, torch.device("cpu"), torch.float32)
    x = torch.randn(1, T, D)
    dy = torch.randn(K, T, D)

    y_f, saved = qsa_layer_fwd(x, w, cos, sin)
    dx_fast = qsa_layer_vjp(dy, w, saved)

    x_r = x.expand(K, -1, -1).clone().requires_grad_(True)
    y_r = _ref_qsa(x_r, w, cos, sin)
    (dx_ref,) = torch.autograd.grad(y_r, x_r, grad_outputs=dy)

    assert torch.allclose(y_f, y_r[:1], atol=1e-5)
    assert torch.allclose(dx_fast, dx_ref, atol=1e-5)


def test_gdn_layer_vjp_parity():
    torch.manual_seed(4)
    T, D, Hk, Hv, Dk, Dv, K = 20, 32, 2, 6, 8, 8, 9
    kd, vd = Hk * Dk, Hv * Dv
    cd = 2 * kd + vd
    w = GdnLayerWeights(
        in_proj_qkv=torch.randn(cd, D) * 0.1,
        in_proj_z=torch.randn(vd, D) * 0.1,
        in_proj_b=torch.randn(Hv, D) * 0.1,
        in_proj_a=torch.randn(Hv, D) * 0.1,
        conv1d_weight=torch.randn(cd, 4) * 0.2,
        dt_bias=torch.ones(Hv) * 0.5,
        A_log=torch.randn(Hv) * 0.3,
        norm_weight=torch.ones(Dv) + torch.randn(Dv) * 0.1,
        out_proj=torch.randn(D, vd) * 0.1,
        num_k_heads=Hk,
        num_v_heads=Hv,
        head_k_dim=Dk,
        head_v_dim=Dv,
        output_gate_type="sigmoid",
    )
    backend = GatedDeltaRuleBackend(preferred="torch")
    x = torch.randn(1, T, D)
    dy = torch.randn(K, T, D)

    y_f, saved = gdn_layer_fwd(x, w, backend)
    dx_fast = gdn_layer_vjp(dy, w, saved, backend)

    x_r = x.expand(K, -1, -1).clone().requires_grad_(True)
    y_r = _ref_gdn(x_r, w)
    (dx_ref,) = torch.autograd.grad(y_r, x_r, grad_outputs=dy)

    assert torch.allclose(y_f, y_r[:1], atol=1e-5)
    assert torch.allclose(dx_fast, dx_ref, atol=1e-4)


@pytest.mark.parametrize("dequant_chunk_size", [16, 4])  # Tests BOTH unchunked AND chunked expert paths!
def test_sparse_moe_fp8_vjp_parity(dequant_chunk_size: int):
    torch.manual_seed(5)
    T, D, E, top_k, inter, K = 18, 32, 16, 4, 16, 12
    dense = MoeDenseWeights(
        router_weight=torch.randn(E, D) * 0.2,
        shared_gate_proj=torch.randn(inter, D) * 0.1,
        shared_up_proj=torch.randn(inter, D) * 0.1,
        shared_down_proj=torch.randn(D, inter) * 0.1,
        shared_expert_gate=torch.randn(1, D) * 0.1,
        num_experts=E,
        top_k=top_k,
        norm_topk_prob=True,
        fp8_block_size=(16, 16),
    )
    gu_q, gu_s = _quantize_fp8_block(torch.randn(E, 2 * inter, D) * 0.15, 16, 16)
    dn_q, dn_s = _quantize_fp8_block(torch.randn(E, D, inter) * 0.15, 16, 16)
    experts = MoeExpertPayload(gate_up_proj=gu_q, down_proj=dn_q, gate_up_scale=gu_s, down_scale=dn_s)

    x = torch.randn(1, T, D)
    dy = torch.randn(K, T, D)

    y_f, saved = moe_block_fwd(
        x, dense, experts, dequant_chunk_size=dequant_chunk_size, use_triton_dequant=False, keep_dequantized_for_bwd=True
    )
    dx_fast = moe_block_vjp_all(
        dy, dense, experts, saved,
        moe_k_chunk=5, dequant_chunk_size=dequant_chunk_size, use_triton_dequant=False, use_triton_swiglu=False,
    )

    x_r = x.expand(K, -1, -1).clone().requires_grad_(True)
    y_r = _ref_moe(x_r, dense, experts)
    (dx_ref,) = torch.autograd.grad(y_r, x_r, grad_outputs=dy)

    assert torch.allclose(y_f, y_r[:1], atol=1e-5)
    assert torch.allclose(dx_fast, dx_ref, atol=1e-4)


@pytest.mark.parametrize("norm_topk_prob", [False, True])
def test_sparse_moe_vjp_tk_layout_multichunk_router_parity(norm_topk_prob: bool):
    """K=150 > 64 exercises the 64-wide bmm shared/router VJP chunks (64, 64, 22) + tk_layout I/O."""
    torch.manual_seed(11)
    T, D, E, top_k, inter, K = 10, 32, 16, 4, 16, 150
    dense = MoeDenseWeights(
        router_weight=torch.randn(E, D) * 0.2,
        shared_gate_proj=torch.randn(inter, D) * 0.1,
        shared_up_proj=torch.randn(inter, D) * 0.1,
        shared_down_proj=torch.randn(D, inter) * 0.1,
        shared_expert_gate=torch.randn(1, D) * 0.1,
        num_experts=E,
        top_k=top_k,
        norm_topk_prob=norm_topk_prob,
        fp8_block_size=(16, 16),
    )
    gu_q, gu_s = _quantize_fp8_block(torch.randn(E, 2 * inter, D) * 0.15, 16, 16)
    dn_q, dn_s = _quantize_fp8_block(torch.randn(E, D, inter) * 0.15, 16, 16)
    experts = MoeExpertPayload(gate_up_proj=gu_q, down_proj=dn_q, gate_up_scale=gu_s, down_scale=dn_s)

    x = torch.randn(1, T, D)
    dy = torch.randn(K, T, D)
    _, saved = moe_block_fwd(x, dense, experts, dequant_chunk_size=16, use_triton_dequant=False, keep_dequantized_for_bwd=True)
    dx_tk = moe_block_vjp_all(
        dy.transpose(0, 1).contiguous(), dense, experts, saved,
        moe_k_chunk=128, dequant_chunk_size=16, use_triton_dequant=False, use_triton_swiglu=False, tk_layout=True,
    )
    assert dx_tk.shape == (T, K, D)

    x_r = x.expand(K, -1, -1).clone().requires_grad_(True)
    y_r = _ref_moe(x_r, dense, experts)
    (dx_ref,) = torch.autograd.grad(y_r, x_r, grad_outputs=dy)
    assert torch.allclose(dx_tk.transpose(0, 1), dx_ref, atol=1e-4)


def test_fp8_e4m3_lut_exact_all_256_bytes():
    """Verify the SM80 uint8->fp32 LUT matches PyTorch float8_e4m3fn bit-for-bit across all 256 byte values."""
    from qwen38_jlens.kernels import _get_fp8_e4m3_lut

    lut = _get_fp8_e4m3_lut(torch.device("cpu"))
    assert lut.shape == (256,) and lut.dtype == torch.float32
    ref = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
    assert torch.equal(torch.isnan(lut), torch.isnan(ref))
    finite = ~torch.isnan(ref)
    assert torch.equal(lut[finite], ref[finite])
    assert lut[0x7E].item() == 448.0 and lut[0xFE].item() == -448.0


def test_compute_aligned_bwd_start():
    assert compute_aligned_bwd_start(128, 16, align=16) == 16  # T_v = 112 (16-aligned!)
    assert compute_aligned_bwd_start(128, 32, align=16) == 32  # T_v = 96
    assert compute_aligned_bwd_start(128, 20, align=16) == 16  # T_v = 112 (rounds up so <= skip_first)
    assert compute_aligned_bwd_start(128, 10, align=16) == 0   # T_v = 128
    assert compute_aligned_bwd_start(100, 16, align=16) == 4   # T_v = 96 (16-aligned, <= skip_first)
    assert compute_aligned_bwd_start(48, 16, align=16) == 16   # T_v = 32
    assert compute_aligned_bwd_start(128, 0, align=16) == 0
    assert compute_aligned_bwd_start(128, 16, align=0) == 0    # explicit disable
    assert compute_aligned_bwd_start(128, 20, align=1) == 20   # unaligned exact skip_first


def test_suffix_vjp_exact_parity():
    """Verify causal suffix-only VJP (`bwd_start > 0`) matches full autograd for QSA (L2 & SDPA), GDN, and PLE."""
    torch.manual_seed(7)
    T, s = 32, 16  # T_v = 16
    D, Hq, Hkv, Dh, K = 32, 6, 2, 16, 9

    # 1. QSA suffix VJP (both l2_cached AND sdpa fallback paths!)
    w_qsa = QsaLayerWeights(
        q_proj=torch.randn(Hq * Dh * 2, D) * 0.1,
        k_proj=torch.randn(Hkv * Dh, D) * 0.1,
        v_proj=torch.randn(Hkv * Dh, D) * 0.1,
        o_proj=torch.randn(D, Hq * Dh) * 0.1,
        q_norm=torch.randn(Dh) * 0.1,
        k_norm=torch.randn(Dh) * 0.1,
        num_q_heads=Hq,
        num_kv_heads=Hkv,
        head_dim=Dh,
    )
    cos, sin = build_rope_cos_sin(T, Dh, 0.25, 10_000.0, torch.device("cpu"), torch.float32)
    x_qsa = torch.randn(1, T, D)
    dy_full = torch.zeros(K, T, D)
    dy_full[:, s:, :] = torch.randn(K, T - s, D)

    _, saved_qsa = qsa_layer_fwd(x_qsa, w_qsa, cos, sin)
    dx_qsa_l2 = qsa_layer_vjp(dy_full[:, s:, :], w_qsa, saved_qsa, bwd_start=s)
    saved_qsa_sdpa = dict(saved_qsa, attn_mode="sdpa")
    dx_qsa_sdpa = qsa_layer_vjp(dy_full[:, s:, :], w_qsa, saved_qsa_sdpa, bwd_start=s)

    x_r = x_qsa.expand(K, -1, -1).clone().requires_grad_(True)
    (dx_qsa_ref,) = torch.autograd.grad(_ref_qsa(x_r, w_qsa, cos, sin), x_r, grad_outputs=dy_full)
    assert torch.allclose(dx_qsa_l2, dx_qsa_ref[:, s:, :], atol=1e-5)
    assert torch.allclose(dx_qsa_sdpa, dx_qsa_ref[:, s:, :], atol=1e-5)

    # 2. GDN suffix VJP
    Hk, Hv, Dk, Dv = 2, 6, 8, 8
    kd, vd = Hk * Dk, Hv * Dv
    cd = 2 * kd + vd
    w_gdn = GdnLayerWeights(
        in_proj_qkv=torch.randn(cd, D) * 0.1,
        in_proj_z=torch.randn(vd, D) * 0.1,
        in_proj_b=torch.randn(Hv, D) * 0.1,
        in_proj_a=torch.randn(Hv, D) * 0.1,
        conv1d_weight=torch.randn(cd, 4) * 0.2,
        dt_bias=torch.ones(Hv) * 0.5,
        A_log=torch.randn(Hv) * 0.3,
        norm_weight=torch.ones(Dv) + torch.randn(Dv) * 0.1,
        out_proj=torch.randn(D, vd) * 0.1,
        num_k_heads=Hk,
        num_v_heads=Hv,
        head_k_dim=Dk,
        head_v_dim=Dv,
        output_gate_type="sigmoid",
    )
    backend = GatedDeltaRuleBackend(preferred="torch")
    x_gdn = torch.randn(1, T, D)
    _, saved_gdn = gdn_layer_fwd(x_gdn, w_gdn, backend)
    dx_gdn_s = gdn_layer_vjp(dy_full[:, s:, :], w_gdn, saved_gdn, backend, bwd_start=s)

    xg_r = x_gdn.expand(K, -1, -1).clone().requires_grad_(True)
    (dx_gdn_ref,) = torch.autograd.grad(_ref_gdn(xg_r, w_gdn), xg_r, grad_outputs=dy_full)
    assert torch.allclose(dx_gdn_s, dx_gdn_ref[:, s:, :], atol=1e-4)

    # 3. PLE suffix VJP (dilated causal conv1d crossing bwd_start boundary!)
    hc, D_ple = 4, 16
    w_ple = PleLayerWeights(
        key_proj=torch.randn(hc * D_ple, D_ple) * 0.1,
        value_proj=torch.randn(D_ple, D_ple) * 0.1,
        norm_key=torch.randn(hc * D_ple) * 0.1,
        norm_query=torch.randn(hc * D_ple) * 0.1,
        norm_conv=torch.randn(hc * D_ple) * 0.1,
        conv1d_weight=torch.randn(hc * D_ple, 1, 4) * 0.2,
        hc_count=hc,
        hidden_size=D_ple,
        ngram_size=3,
    )
    h_ple = torch.randn(1, T, hc * D_ple)
    emb_ple = torch.randn(1, T, D_ple)
    dy_ple = torch.zeros(K, T, hc * D_ple)
    dy_ple[:, s:, :] = torch.randn(K, T - s, hc * D_ple)

    _, saved_ple = ple_layer_fwd(h_ple, emb_ple, w_ple)
    dh_ple_s = ple_layer_vjp(dy_ple[:, s:, :], w_ple, _slice_ple_saved(saved_ple, s))

    hp_r = h_ple.expand(K, -1, -1).clone().requires_grad_(True)
    (dh_ple_ref,) = torch.autograd.grad(_ref_ple(hp_r, emb_ple.expand(K, -1, -1), w_ple), hp_r, grad_outputs=dy_ple)
    assert torch.allclose(dh_ple_s, dh_ple_ref[:, s:, :], atol=1e-4)


def test_full_4layer_qwen4exp_jlens_fit_exact_parity():
    """End-to-end test: 4 layers [GDN, GDN+PLE, GDN, QSA] + hyper_connection_mixer + J-lens fit.

    Verifies our single-sweep layer-by-layer engine with 16-aligned causal suffix pruning
    (`T=48, skip_first=16 -> bwd_start=16, T_v=32`) produces the EXACT same `[D, 4*D]`
    Jacobians as BOTH `bwd_align=0` (full `[0:T]`) AND Anthropic's `jlens` estimator via autograd!
    """
    torch.manual_seed(42)
    D, hc, E, top_k, inter, T, V = 16, 4, 8, 2, 16, 48, 128
    skip_first = 16
    assert compute_aligned_bwd_start(T, skip_first, align=16) == 16
    cfg = Qwen38TextConfig(
        vocab_size=V,
        hidden_size=D,
        num_hidden_layers=4,
        hc_count=hc,
        hc_lowrank=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        num_experts=E,
        num_experts_per_tok=top_k,
        moe_intermediate_size=inter,
        shared_expert_intermediate_size=inter,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "indexed_attention"],
        ple_layer_ids=[2],
        ple_embed_dim=D,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        eos_token_id=V - 1,
        fp8_block_size=(8, 8),
        is_fp8_experts=True,
    )

    spec = PleHashSpec(
        vocab_size=V, ngram_size=3, heads_per_ngram=2,
        ngram_vocab_size_base=17, make_ngram_vocab_size_divisible_by=4,
        ple_embed_dim=D, eos_token_id=V - 1, ple_layer_index=0,
    )
    ple_tbl = torch.randn(spec.padded_vocab_size, spec.head_dim_per_ngram) * 0.1
    ple_provider = InMemoryPleProvider(spec, ple_tbl)

    embed_tokens = torch.randn(V, D) * 0.2
    lm_head = torch.randn(V, D) * 0.1
    hc_mixer = GatedResidualWeights(
        hc_norm_weight=torch.randn(hc * D) * 0.1,
        input_mix_weight_down=torch.randn(cfg.hc_lowrank, hc * D) * 0.1,
        input_mix_weight_up=torch.randn(hc * D, cfg.hc_lowrank) * 0.1,
        block_inject_weight=None,
        hc_count=hc,
        hidden_size=D,
    )

    layers = []
    moe_store = {}
    for l in range(4):
        attn_hc = GatedResidualWeights(
            torch.randn(hc * D) * 0.1, torch.randn(cfg.hc_lowrank, hc * D) * 0.1,
            torch.randn(hc * D, cfg.hc_lowrank) * 0.1, torch.randn(hc, hc * D) * 0.1, hc, D,
        )
        mlp_hc = GatedResidualWeights(
            torch.randn(hc * D) * 0.1, torch.randn(cfg.hc_lowrank, hc * D) * 0.1,
            torch.randn(hc * D, cfg.hc_lowrank) * 0.1, torch.randn(hc, hc * D) * 0.1, hc, D,
        )
        moe_d = MoeDenseWeights(
            torch.randn(E, D) * 0.15, torch.randn(inter, D) * 0.1,
            torch.randn(inter, D) * 0.1, torch.randn(D, inter) * 0.1,
            torch.randn(1, D) * 0.1, E, top_k, True, (8, 8),
        )
        gu_q, gu_s = _quantize_fp8_block(torch.randn(E, 2 * inter, D) * 0.1, 8, 8)
        dn_q, dn_s = _quantize_fp8_block(torch.randn(E, D, inter) * 0.1, 8, 8)
        moe_store[l] = MoeExpertPayload(gu_q, dn_q, gu_s, dn_s)

        gdn = qsa = ple = None
        if cfg.layer_types[l] == "linear_attention":
            kd, vd = 8, 16
            cd = 2 * kd + vd
            gdn = GdnLayerWeights(
                torch.randn(cd, D) * 0.1, torch.randn(vd, D) * 0.1, torch.randn(4, D) * 0.1,
                torch.randn(4, D) * 0.1, torch.randn(cd, 4) * 0.1, torch.ones(4) * 0.2,
                torch.randn(4) * 0.2, torch.ones(4) + torch.randn(4) * 0.1, torch.randn(D, vd) * 0.1,
                2, 4, 4, 4, "sigmoid",
            )
        else:
            qsa = QsaLayerWeights(
                torch.randn(4 * 8 * 2, D) * 0.1, torch.randn(2 * 8, D) * 0.1,
                torch.randn(2 * 8, D) * 0.1, torch.randn(D, 4 * 8) * 0.1,
                torch.randn(8) * 0.1, torch.randn(8) * 0.1, 4, 2, 8,
            )
        if l == 1:
            ple = PleLayerWeights(
                torch.randn(hc * D, D) * 0.1, torch.randn(D, D) * 0.1,
                torch.randn(hc * D) * 0.1, torch.randn(hc * D) * 0.1,
                torch.randn(hc * D) * 0.1, torch.randn(hc * D, 1, 4) * 0.1, hc, D, 3,
            )
        layers.append(DecoderLayerDenseWeights(l, cfg.layer_types[l], attn_hc, mlp_hc, moe_d, gdn, qsa, ple))

    prof = HardwareProfile("CPU-test", (0, 0), 0.0, 0.0, 32.0, 24.0, 100.0, 4, has_cuda=False)
    plan = build_offload_plan(
        prof, num_layers=4, d_model=D, hc_count=hc, num_experts=E,
        moe_intermediate_size=inter, max_seq_len=T, dense_backbone_gib=0.01, ple_table_gib=0.01,
    )
    # Test multi-sweep k_pass < D too!
    plan.k_pass = 8
    plan.attn_k_chunk = 4
    plan.moe_k_chunk = 4
    plan.expert_dequant_chunk = 4
    plan.layer_tiers = {l: StorageTier.HOST_PINNED for l in range(4)}

    loaded = LoadedQwen38Model(
        config=cfg, plan=plan, device=torch.device("cpu"), compute_dtype=torch.float32,
        embed_tokens=embed_tokens, hyper_connection_mixer=hc_mixer, lm_head=lm_head,
        layers=layers, moe_tier_store=moe_store, ple_providers={1: ple_provider},
    )
    engine = Qwen38JLensEngine(loaded)
    input_ids = torch.randint(0, V - 2, (1, T))

    # Fast single-sweep engine Jacobian with 16-aligned suffix pruning (bwd_start=16, T_v=32) AND unpruned (bwd_align=0)
    src_layers = [-1, 0, 1, 2, 3]
    J_fast, seq_len, n_valid, _, _ = engine.jacobian_for_tokens(
        input_ids, source_layers=src_layers, skip_first=skip_first, bwd_align=16
    )
    J_full_bwd, _, _, _, _ = engine.jacobian_for_tokens(
        input_ids, source_layers=src_layers, skip_first=skip_first, bwd_align=0
    )
    assert seq_len == T and n_valid == T - skip_first - 1

    # Reference Anthropic jlens estimator via full PyTorch autograd starting from embed_tokens!
    cos, sin = build_rope_cos_sin(T, 8, 0.25, cfg.rope_theta, torch.device("cpu"), torch.float32)
    ple_emb = ple_provider.lookup(input_ids, device="cpu").float()
    valid_pos = valid_position_mask(T, skip_first=skip_first).nonzero(as_tuple=True)[0]

    emb_rep = F.embedding(input_ids, embed_tokens).expand(D, -1, -1).clone().requires_grad_(True)
    H_emb_rep = emb_rep.repeat(1, 1, hc)
    H_emb_rep.retain_grad()
    acts = [H_emb_rep]
    cur = H_emb_rep
    for l in range(4):
        if layers[l].ple is not None:
            cur = cur + _ref_ple(cur, ple_emb.expand(D, -1, -1), layers[l].ple)
        xm, wm = _ref_gated_residual(cur, layers[l].attn_hc)
        ym = _ref_gdn(xm, layers[l].gdn) if layers[l].layer_type == "linear_attention" else _ref_qsa(xm, layers[l].qsa, cos, sin)
        cur = cur + (ym.unsqueeze(-2) * wm.unsqueeze(-1)).flatten(-2)
        xm, wm = _ref_gated_residual(cur, layers[l].mlp_hc)
        cur = cur + (_ref_moe(xm, layers[l].moe_dense, moe_store[l]).unsqueeze(-2) * wm.unsqueeze(-1)).flatten(-2)
        acts.append(cur)
    h_final, _ = _ref_gated_residual(cur, hc_mixer)

    cot = torch.zeros_like(h_final)
    idx = torch.arange(D)
    cot[idx[:, None], valid_pos[None, :], idx[:, None]] = 1.0
    grads = torch.autograd.grad(h_final, acts, grad_outputs=cot, retain_graph=True)
    (g_emb_ref,) = torch.autograd.grad(h_final, emb_rep, grad_outputs=cot)

    for l, g_ref in zip(src_layers, grads, strict=True):
        J_ref_l = g_ref[:, valid_pos, :].mean(dim=1)
        max_diff = (J_fast[l] - J_ref_l).abs().max().item()
        assert torch.allclose(J_fast[l], J_ref_l, atol=1e-4), f"Layer {l} mismatch vs autograd: max_diff={max_diff}"
        pruned_vs_full = (J_fast[l] - J_full_bwd[l]).abs().max().item()
        assert torch.allclose(J_fast[l], J_full_bwd[l], atol=1e-5), (
            f"Layer {l} mismatch between suffix-pruned and full backward: max_diff={pruned_vs_full}"
        )

    # Collapsed embedding Jacobian sum_{b=0..3} J_{-1}[:, b, :] must match exact d h_final / d emb!
    J_emb_collapsed = J_fast[-1].view(D, hc, D).sum(dim=1)
    J_emb_ref = g_emb_ref[:, valid_pos, :].mean(dim=1)
    assert torch.allclose(J_emb_collapsed, J_emb_ref, atol=1e-4)

    # Verify fit(include_embedding_layer=True) & apply_lens work seamlessly for layer -1..3
    lens = engine.fit([input_ids, input_ids], include_embedding_layer=True, skip_first=skip_first)
    assert lens.source_layers == [-1, 0, 1, 2, 3]
    jl_logits, m_logits = engine.apply_lens(lens, input_ids, positions=[-2], use_jacobian=True)
    ll_logits, _ = engine.apply_lens(lens, input_ids, positions=[-2], use_jacobian=False)
    assert m_logits.shape == (1, V)
    assert set(jl_logits.keys()) == {-1, 0, 1, 2, 3}
    assert set(ll_logits.keys()) == {-1, 0, 1, 2, 3}

    # Verify 4-branch transport on H_{-1} equals single-stream collapsed readout on emb!
    emb_pos = F.embedding(input_ids[:, -2:-1], embed_tokens)[0]  # [1, D]
    expected_emb_logits = F.linear(F.linear(emb_pos.float(), lens.collapsed_jacobians()[-1]), lm_head.float())
    assert torch.allclose(jl_logits[-1], expected_emb_logits, atol=1e-4)

    # Verify zero-VRAM forward-only micro-batching (`fwd_batch_size > 1`) across variable-length
    # prompts + a too-short skipped prompt produces bit-exact Jacobians AND amortizes forward sweeps!
    var_prompts = [
        torch.randint(0, V - 2, (1, 48)),
        torch.randint(0, V - 2, (1, 10)),  # too short for skip_first=16 -> skipped cleanly
        torch.randint(0, V - 2, (1, 32)),
        torch.randint(0, V - 2, (1, 48)),
        torch.randint(0, V - 2, (1, 36)),
        torch.randint(0, V - 2, (1, 40)),
    ]
    orig_iter_sweep = engine.streamer.iter_sweep
    sweep_counts = {"fwd": 0, "bwd": 0}

    def _counting_sweep(layer_order):
        order = list(layer_order)
        if len(order) >= 2 and order[0] < order[1]:
            sweep_counts["fwd"] += 1
        elif order:
            sweep_counts["bwd"] += 1
        return orig_iter_sweep(order)

    engine.streamer.iter_sweep = _counting_sweep  # type: ignore[method-assign]
    try:
        # 1. Batched forward (fwd_batch_size=3): chunks [0..2] (2 valid) and [3..5] (3 valid) -> ONLY 2 fwd sweeps!
        lens_batched = engine.fit(
            var_prompts, include_embedding_layer=True, skip_first=skip_first, fwd_batch_size=3, resume=False
        )
        batched_fwd_sweeps = sweep_counts["fwd"]
        batched_bwd_sweeps = sweep_counts["bwd"]

        # 2. Single-prompt forward (fwd_batch_size=1): 5 valid prompts -> 5 fwd sweeps!
        sweep_counts["fwd"] = 0
        sweep_counts["bwd"] = 0
        lens_single = engine.fit(
            var_prompts, include_embedding_layer=True, skip_first=skip_first, fwd_batch_size=1, resume=False
        )
        single_fwd_sweeps = sweep_counts["fwd"]
        single_bwd_sweeps = sweep_counts["bwd"]
    finally:
        engine.streamer.iter_sweep = orig_iter_sweep  # type: ignore[method-assign]

    assert lens_batched.n_prompts == 5 and lens_single.n_prompts == 5
    assert batched_fwd_sweeps == 2 and single_fwd_sweeps == 5
    assert batched_bwd_sweeps == single_bwd_sweeps
    for l in src_layers:
        assert torch.equal(lens_batched.jacobians[l], lens_single.jacobians[l]), (
            f"Layer {l} mismatch between fwd_batch_size=3 and fwd_batch_size=1"
        )
    engine.close()
