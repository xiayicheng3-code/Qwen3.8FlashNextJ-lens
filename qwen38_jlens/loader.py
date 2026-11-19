"""Streaming shard-by-shard safetensors loader & 3-tier MoE storage manager for Qwen3.8-Flash-Next.

Why a custom loader instead of `AutoModelForCausalLM.from_pretrained(..., device_map="auto")`?
  1. Avoids HF `transformers#48349` (multimodal `modules_to_not_convert` prefix mismatch silently
     corrupting BF16 PLE/GDN/GatedResidual weights into unscaled FP8!).
  2. Avoids HF `transformers#48350` & massive temporary CPU buffers during 51B PLE concatenation.
  3. Skips downloading/loading Vision encoder (`model.visual.*`) and MTP head (`mtp.*`) shards.
  4. Streams raw shards in small batches (~15 GiB), packs routed MoE experts directly into
     Tier 0 (GPU HBM), Tier 1 (Pinned/Pageable Host RAM), or Tier 2 (contiguous 2.34 GiB
     binary NVMe slabs for max sequential pread), and immediately deletes consumed raw HF blobs!
     Peak disk usage drops from >210 GiB to ~65-95 GiB (100% safe on Google Colab's 235 GiB disk!).
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from safetensors import safe_open

from qwen38_jlens.hardware import HardwareProfile, OffloadPlan, StorageTier, build_offload_plan
from qwen38_jlens.model_ops import (
    DecoderLayerDenseWeights,
    GatedResidualWeights,
    GdnLayerWeights,
    MoeDenseWeights,
    MoeExpertPayload,
    PleLayerWeights,
    QsaLayerWeights,
)
from qwen38_jlens.ple_mmap import MmapPleTable, PleHashSpec, PreextractedPleProvider

logger = logging.getLogger(__name__)

_EXPERT_SPLIT_RE = re.compile(
    r"^(?:model\.language_model\.|model\.)?layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv|weight_scale)$"
)
_EXPERT_FUSED_RE = re.compile(
    r"^(?:model\.language_model\.|model\.)?layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj|gate_up_proj_scale_inv|down_proj_scale_inv)$"
)
_LAYER_RE = re.compile(r"^(?:model\.language_model\.|model\.)?layers\.(\d+)\.(.+)$")
_PLE_SHARD_RE = re.compile(r"ngram_embedding\.shard_\d+\.weight$")

# Binary slab header for Tier 2 contiguous NVMe MoE storage
_SLAB_MAGIC = b"Q38M"
_SLAB_DTYPE_CODE = {
    torch.float8_e4m3fn: 1,
    torch.bfloat16: 2,
    torch.float32: 3,
}
_CODE_TO_DTYPE = {v: k for k, v in _SLAB_DTYPE_CODE.items()}


def write_moe_slab(path: str, payload: MoeExpertPayload) -> None:
    """Write one layer's packed MoE experts as a single contiguous binary file for max NVMe pread speed."""
    tensors = [payload.gate_up_proj, payload.down_proj]
    has_scales = payload.gate_up_scale is not None and payload.down_scale is not None
    if has_scales:
        assert payload.gate_up_scale is not None and payload.down_scale is not None
        tensors.extend([payload.gate_up_scale, payload.down_scale])

    meta = {
        "has_scales": has_scales,
        "tensors": [
            {
                "shape": list(t.shape),
                "dtype": _SLAB_DTYPE_CODE[t.dtype],
                "nbytes": t.numel() * t.element_size(),
            }
            for t in tensors
        ],
    }
    header_bytes = json.dumps(meta).encode("utf-8")
    # 4096-byte align data start for optimal NVMe page/DMA alignment
    raw_prefix_len = 8 + len(header_bytes)
    pad_len = (4096 - (raw_prefix_len % 4096)) % 4096

    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        f.write(_SLAB_MAGIC)
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        if pad_len:
            f.write(b"\x00" * pad_len)
        for t in tensors:
            t_cpu = t.detach().cpu().contiguous()
            f.write(t_cpu.view(torch.uint8).numpy().tobytes())
    os.replace(tmp_path, path)


def read_moe_slab_into(
    path: str,
    dst: MoeExpertPayload,
) -> MoeExpertPayload:
    """Sequentially read a contiguous Tier 2 NVMe MoE slab directly into pre-allocated pinned host buffers."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != _SLAB_MAGIC:
            raise ValueError(f"Invalid MoE slab magic in {path}: {magic!r}")
        (header_len,) = struct.unpack("<I", f.read(4))
        meta = json.loads(f.read(header_len).decode("utf-8"))
        raw_prefix_len = 8 + header_len
        pad_len = (4096 - (raw_prefix_len % 4096)) % 4096
        if pad_len:
            f.seek(pad_len, os.SEEK_CUR)

        targets = [dst.gate_up_proj, dst.down_proj]
        if meta["has_scales"]:
            assert dst.gate_up_scale is not None and dst.down_scale is not None
            targets.extend([dst.gate_up_scale, dst.down_scale])

        for t_info, dst_t in zip(meta["tensors"], targets, strict=True):
            nbytes = int(t_info["nbytes"])
            # Direct zero-copy readinto pinned buffer via memoryview (1 memcpy total!)
            mv = memoryview(dst_t.view(torch.uint8).numpy().reshape(-1))
            assert len(mv) == nbytes, f"Slab size mismatch: {len(mv)} vs {nbytes}"
            read_so_far = 0
            while read_so_far < nbytes:
                n = f.readinto(mv[read_so_far:])
                if not n:
                    raise EOFError(f"Unexpected EOF reading {path} at {read_so_far}/{nbytes}")
                read_so_far += n
    return dst


def allocate_moe_payload(
    num_experts: int,
    hidden_size: int,
    moe_intermediate_size: int,
    *,
    device: torch.device | str = "cpu",
    pin_memory: bool = False,
    weight_dtype: torch.dtype = torch.float8_e4m3fn,
    block_size: tuple[int, int] = (128, 128),
) -> MoeExpertPayload:
    """Pre-allocate one layer's packed MoE expert tensors (on GPU, pinned host, or pageable host)."""
    dev = torch.device(device)
    use_pin = pin_memory and dev.type == "cpu"
    gu = torch.empty((num_experts, 2 * moe_intermediate_size, hidden_size), dtype=weight_dtype, device=dev, pin_memory=use_pin)
    dn = torch.empty((num_experts, hidden_size, moe_intermediate_size), dtype=weight_dtype, device=dev, pin_memory=use_pin)

    if weight_dtype.itemsize == 1:
        bm, bn = block_size
        gu_s = torch.empty(
            (num_experts, (2 * moe_intermediate_size + bm - 1) // bm, (hidden_size + bn - 1) // bn),
            dtype=torch.float32,
            device=dev,
            pin_memory=use_pin,
        )
        dn_s = torch.empty(
            (num_experts, (hidden_size + bm - 1) // bm, (moe_intermediate_size + bn - 1) // bn),
            dtype=torch.float32,
            device=dev,
            pin_memory=use_pin,
        )
    else:
        gu_s, dn_s = None, None

    return MoeExpertPayload(gate_up_proj=gu, down_proj=dn, gate_up_scale=gu_s, down_scale=dn_s)


@dataclass
class Qwen38TextConfig:
    """Minimal normalized text configuration extracted from `config.json`."""

    vocab_size: int = 248320
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    hc_count: int = 4
    hc_lowrank: int = 320
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = True
    rms_norm_eps: float = 1e-6
    output_gate_type: str = "sigmoid"
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    layer_types: list[str] = field(default_factory=list)
    ple_layer_ids: list[int] = field(default_factory=lambda: [2])
    ple_embed_dim: int = 2560
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    seed: int = 1234
    eos_token_id: int = 248044
    bos_token_id: int = 248044
    fp8_block_size: tuple[int, int] = (128, 128)
    is_fp8_experts: bool = True

    @classmethod
    def from_config_dict(cls, raw: dict[str, Any]) -> "Qwen38TextConfig":
        tc = raw.get("text_config", raw)
        qcfg = raw.get("quantization_config") or tc.get("quantization_config") or {}
        rope = tc.get("rope_parameters") or {}
        num_layers = int(tc.get("num_hidden_layers", 48))
        raw_lt = tc.get("layer_types")
        if not raw_lt:
            interval = int(tc.get("full_attention_interval", 4))
            layer_types = [
                "linear_attention" if (i + 1) % interval else "indexed_attention"
                for i in range(num_layers)
            ]
        else:
            layer_types = [
                "indexed_attention" if t == "full_attention" else t for t in raw_lt
            ]
        eos = tc.get("eos_token_id", raw.get("eos_token_id", 248044))
        if isinstance(eos, list):
            eos = eos[0]
        bos = tc.get("bos_token_id", raw.get("bos_token_id", eos))
        if isinstance(bos, list):
            bos = bos[0]
        bs = qcfg.get("weight_block_size", [128, 128])
        is_fp8 = str(qcfg.get("quant_method", "")).lower() == "fp8"
        hidden = int(tc.get("hidden_size", 2560))
        return cls(
            vocab_size=int(tc.get("vocab_size", 248320)),
            hidden_size=hidden,
            num_hidden_layers=num_layers,
            hc_count=int(tc.get("hc_count", 4)),
            hc_lowrank=int(tc.get("hc_lowrank", 320)),
            num_attention_heads=int(tc.get("num_attention_heads", 24)),
            num_key_value_heads=int(tc.get("num_key_value_heads", 2)),
            head_dim=int(tc.get("head_dim", 256)),
            linear_num_key_heads=int(tc.get("linear_num_key_heads", 16)),
            linear_num_value_heads=int(tc.get("linear_num_value_heads", 48)),
            linear_key_head_dim=int(tc.get("linear_key_head_dim", 128)),
            linear_value_head_dim=int(tc.get("linear_value_head_dim", 128)),
            linear_conv_kernel_dim=int(tc.get("linear_conv_kernel_dim", 4)),
            num_experts=int(tc.get("num_experts", 512)),
            num_experts_per_tok=int(tc.get("num_experts_per_tok", 10)),
            moe_intermediate_size=int(tc.get("moe_intermediate_size", 640)),
            shared_expert_intermediate_size=int(tc.get("shared_expert_intermediate_size", 640)),
            norm_topk_prob=bool(tc.get("norm_topk_prob", True)),
            rms_norm_eps=float(tc.get("rms_norm_eps", 1e-6)),
            output_gate_type=str(tc.get("output_gate_type") or tc.get("hidden_act") or "sigmoid"),
            rope_theta=float(rope.get("rope_theta", tc.get("rope_theta", 10_000_000.0))),
            partial_rotary_factor=float(rope.get("partial_rotary_factor", tc.get("partial_rotary_factor", 0.25))),
            indexer_budget=int(tc.get("indexer_budget", 2048)),
            indexer_compress_ratio=int(tc.get("indexer_compress_ratio", 4)),
            layer_types=layer_types,
            ple_layer_ids=[int(x) for x in (tc.get("ple_layer_ids") or [])],
            ple_embed_dim=int(tc.get("ple_embed_dim") or hidden),
            ple_conv_kernel_size=int(tc.get("ple_conv_kernel_size", 4)),
            ngram_size=int(tc.get("ngram_size", 3)),
            heads_per_ngram=int(tc.get("heads_per_ngram", 8)),
            ngram_vocab_size_base=int(tc.get("ngram_vocab_size_base", 20_000_000)),
            make_ngram_vocab_size_divisible_by=int(tc.get("make_ngram_vocab_size_divisible_by", 128)),
            seed=int(tc.get("seed", 1234)),
            eos_token_id=int(eos),
            bos_token_id=int(bos),
            fp8_block_size=(int(bs[0]), int(bs[1])),
            is_fp8_experts=is_fp8,
        )


@dataclass
class LoadedQwen38Model:
    """All dense GPU-resident weights + 3-tier MoE store + PLE provider ready for J-lens fitting."""

    config: Qwen38TextConfig
    plan: OffloadPlan
    device: torch.device
    compute_dtype: torch.dtype

    embed_tokens: torch.Tensor                        # [vocab, D] on GPU
    hyper_connection_mixer: GatedResidualWeights      # final collapse [10240 -> 2560] on GPU
    lm_head: torch.Tensor                             # [vocab, D] on CPU (moved to GPU for apply())
    layers: list[DecoderLayerDenseWeights]            # 48 dense layer weight structs on GPU

    # 3-Tier MoE expert storage:
    #   - Tier 0 (GPU_HBM), Tier 1 (HOST_PINNED / HOST_PAGEABLE): MoeExpertPayload
    #   - Tier 2 (NVME_SLAB): path string to contiguous binary slab file
    moe_tier_store: dict[int, MoeExpertPayload | str]

    # PLE provider (layer_idx -> MmapPleTable | PreextractedPleProvider | InMemoryPleProvider)
    ple_providers: dict[int, Any]


def _normalize_text_key(key: str) -> str | None:
    """Strip multimodal `model.language_model.` / `model.` prefix; return None for vision/MTP keys."""
    if key.startswith("mtp.") or key.startswith("model.visual.") or key.startswith("visual."):
        return None
    if key == "lm_head.weight":
        return "lm_head.weight"
    if key.startswith("model.language_model."):
        return key[len("model.language_model.") :]
    if key.startswith("model."):
        return key[len("model.") :]
    return key


def _safe_delete_hf_file(path: str) -> None:
    """Delete both the symlink and its underlying HF Hub blob so disk space is actually reclaimed."""
    try:
        real = os.path.realpath(path)
        if os.path.islink(path):
            os.unlink(path)
        if os.path.isfile(real):
            os.unlink(real)
        elif os.path.isfile(path):
            os.unlink(path)
    except OSError as exc:
        logger.debug("Could not delete cached shard %s: %s", path, exc)


def _assemble_layer_dense(
    layer_idx: int,
    raw_dense: dict[str, torch.Tensor],
    cfg: Qwen38TextConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> DecoderLayerDenseWeights:
    def get(name: str, cast: torch.dtype | None = dtype) -> torch.Tensor:
        if name not in raw_dense:
            raise KeyError(f"Missing dense tensor for layer {layer_idx}: {name}")
        t = raw_dense.pop(name)
        return t.to(device=device, dtype=cast if cast is not None else t.dtype, non_blocking=True)

    attn_hc = GatedResidualWeights(
        hc_norm_weight=get("attn_hyper_connection.hc_norm.weight"),
        input_mix_weight_down=get("attn_hyper_connection.input_mix_weight_down.weight"),
        input_mix_weight_up=get("attn_hyper_connection.input_mix_weight_up.weight"),
        block_inject_weight=get("attn_hyper_connection.block_inject_weight.weight"),
        hc_count=cfg.hc_count,
        hidden_size=cfg.hidden_size,
        eps=cfg.rms_norm_eps,
    )
    mlp_hc = GatedResidualWeights(
        hc_norm_weight=get("mlp_hyper_connection.hc_norm.weight"),
        input_mix_weight_down=get("mlp_hyper_connection.input_mix_weight_down.weight"),
        input_mix_weight_up=get("mlp_hyper_connection.input_mix_weight_up.weight"),
        block_inject_weight=get("mlp_hyper_connection.block_inject_weight.weight"),
        hc_count=cfg.hc_count,
        hidden_size=cfg.hidden_size,
        eps=cfg.rms_norm_eps,
    )
    moe_dense = MoeDenseWeights(
        router_weight=get("mlp.gate.weight"),
        shared_gate_proj=get("mlp.shared_expert.gate_proj.weight"),
        shared_up_proj=get("mlp.shared_expert.up_proj.weight"),
        shared_down_proj=get("mlp.shared_expert.down_proj.weight"),
        shared_expert_gate=get("mlp.shared_expert_gate.weight"),
        num_experts=cfg.num_experts,
        top_k=cfg.num_experts_per_tok,
        norm_topk_prob=cfg.norm_topk_prob,
        fp8_block_size=cfg.fp8_block_size,
    )

    layer_type = cfg.layer_types[layer_idx]
    gdn = None
    qsa = None
    if layer_type == "linear_attention":
        conv_w = get("linear_attn.conv1d.weight")
        if conv_w.ndim == 3:
            conv_w = conv_w.squeeze(1)
        gdn = GdnLayerWeights(
            in_proj_qkv=get("linear_attn.in_proj_qkv.weight"),
            in_proj_z=get("linear_attn.in_proj_z.weight"),
            in_proj_b=get("linear_attn.in_proj_b.weight"),
            in_proj_a=get("linear_attn.in_proj_a.weight"),
            conv1d_weight=conv_w.contiguous(),
            dt_bias=get("linear_attn.dt_bias", cast=torch.float32),
            A_log=get("linear_attn.A_log", cast=torch.float32),
            norm_weight=get("linear_attn.norm.weight"),
            out_proj=get("linear_attn.out_proj.weight"),
            num_k_heads=cfg.linear_num_key_heads,
            num_v_heads=cfg.linear_num_value_heads,
            head_k_dim=cfg.linear_key_head_dim,
            head_v_dim=cfg.linear_value_head_dim,
            output_gate_type=cfg.output_gate_type,
            eps=cfg.rms_norm_eps,
        )
    else:
        qsa = QsaLayerWeights(
            q_proj=get("self_attn.q_proj.weight"),
            k_proj=get("self_attn.k_proj.weight"),
            v_proj=get("self_attn.v_proj.weight"),
            o_proj=get("self_attn.o_proj.weight"),
            q_norm=get("self_attn.q_norm.weight"),
            k_norm=get("self_attn.k_norm.weight"),
            num_q_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            indexer_budget=cfg.indexer_budget,
            indexer_compress_ratio=cfg.indexer_compress_ratio,
            eps=cfg.rms_norm_eps,
        )
        # Discard unused QSA indexer weights for T <= 2048 (or pop silently)
        for k in list(raw_dense.keys()):
            if k.startswith("self_attn.indexer."):
                raw_dense.pop(k, None)

    ple = None
    if (layer_idx + 1) in cfg.ple_layer_ids:
        ple = PleLayerWeights(
            key_proj=get("ple.key_proj.weight"),
            value_proj=get("ple.value_proj.weight"),
            norm_key=get("ple.norm_key.weight"),
            norm_query=get("ple.norm_query.weight"),
            norm_conv=get("ple.norm_conv.weight"),
            conv1d_weight=get("ple.conv1d.weight"),
            hc_count=cfg.hc_count,
            hidden_size=cfg.hidden_size,
            ngram_size=cfg.ngram_size,
            eps=cfg.rms_norm_eps,
        )

    return DecoderLayerDenseWeights(
        layer_idx=layer_idx,
        layer_type=layer_type,
        attn_hc=attn_hc,
        mlp_hc=mlp_hc,
        moe_dense=moe_dense,
        gdn=gdn,
        qsa=qsa,
        ple=ple,
    )


def load_qwen38_streaming(
    model_id_or_dir: str,
    plan: OffloadPlan,
    *,
    slab_dir: str = "/content/qwen38_moe_slabs",
    ple_dir: str | None = None,
    tokenized_prompts_for_preextract: Sequence[torch.Tensor] | None = None,
    revision: str | None = None,
    hf_token: str | None = None,
    compute_dtype: torch.dtype = torch.bfloat16,
    progress_callback: Callable[[str], None] | None = None,
) -> LoadedQwen38Model:
    """Stream-load Qwen3.8-Flash-Next-FP8 (from HF Hub or local dir) directly into our 3-tier hierarchy.

    Guarantees:
      - Zero Vision/MTP shard downloads.
      - Zero RAM allocated for the 51B PLE n-gram table (mmap or pre-extracted cache).
      - Peak disk stays well under Colab's 235 GiB limit via shard-by-shard pack-and-evict.
    """
    is_local = os.path.isdir(model_id_or_dir)
    device = torch.device("cuda:0") if plan.profile.has_cuda else torch.device("cpu")
    os.makedirs(slab_dir, exist_ok=True)

    def _log(msg: str) -> None:
        logger.info(msg)
        if progress_callback is not None:
            progress_callback(msg)

    def _fetch_file(filename: str, local_dir: str | None = None) -> str:
        if is_local:
            return os.path.join(model_id_or_dir, filename)
        from huggingface_hub import hf_hub_download

        kwargs: dict[str, Any] = {"repo_id": model_id_or_dir, "filename": filename}
        if revision is not None:
            kwargs["revision"] = revision
        if hf_token is not None:
            kwargs["token"] = hf_token
        if local_dir is not None:
            kwargs["local_dir"] = local_dir
        return hf_hub_download(**kwargs)

    # 1. Load config.json and model.safetensors.index.json
    config_path = _fetch_file("config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)
    cfg = Qwen38TextConfig.from_config_dict(raw_cfg)

    index_path = None
    try:
        index_path = _fetch_file("model.safetensors.index.json")
    except Exception:
        pass

    if index_path is not None and os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]
    else:
        # Single-file safetensors model (e.g. unit tests / tiny synthetic model)
        single_path = _fetch_file("model.safetensors")
        with safe_open(single_path, framework="pt", device="cpu") as sf:
            weight_map = {k: "model.safetensors" for k in sf.keys()}

    # 2. Classify shards: PLE shards vs text backbone shards vs ignored (vision/MTP)
    shard_to_keys: dict[str, list[str]] = {}
    ple_shard_names: set[str] = set()
    ple_scale_keys: dict[int, tuple[str, str]] = {}  # layer_idx -> (shard_name, raw_key)
    layer_expert_shards: dict[int, set[str]] = {l: set() for l in range(cfg.num_hidden_layers)}
    needed_shards: set[str] = set()

    for raw_key, shard_name in weight_map.items():
        norm_key = _normalize_text_key(raw_key)
        if norm_key is None:
            continue
        shard_to_keys.setdefault(shard_name, []).append(raw_key)
        needed_shards.add(shard_name)

        if _PLE_SHARD_RE.search(norm_key):
            ple_shard_names.add(shard_name)
        elif norm_key.endswith("ngram_embedding.weight_scale"):
            m_l = _LAYER_RE.match(norm_key)
            if m_l:
                ple_scale_keys[int(m_l.group(1))] = (shard_name, raw_key)
        elif norm_key.endswith("ngram_embedding.weight"):
            ple_shard_names.add(shard_name)
        else:
            m_sp = _EXPERT_SPLIT_RE.match(raw_key)
            m_fu = _EXPERT_FUSED_RE.match(raw_key)
            if m_sp:
                layer_expert_shards[int(m_sp.group(1))].add(shard_name)
            elif m_fu:
                layer_expert_shards[int(m_fu.group(1))].add(shard_name)

    total_shards = len(set(weight_map.values()))
    skipped_shards = total_shards - len(needed_shards)
    _log(
        f"Index parsed: {len(needed_shards)}/{total_shards} shards needed "
        f"({len(ple_shard_names)} PLE shards, {skipped_shards} Vision/MTP shards skipped)."
    )

    # 3. Allocate layer buffers & tracking counters
    weight_dtype = torch.float8_e4m3fn if cfg.is_fp8_experts else compute_dtype
    dense_top: dict[str, torch.Tensor] = {}
    dense_layers: dict[int, dict[str, torch.Tensor]] = {l: {} for l in range(cfg.num_hidden_layers)}
    moe_tier_store: dict[int, MoeExpertPayload | str] = {}
    layer_buffers: dict[int, MoeExpertPayload] = {}
    layer_remaining_shards: dict[int, set[str]] = {
        l: set(s_set) for l, s_set in layer_expert_shards.items()
    }

    def _get_or_alloc_layer_buf(layer_idx: int) -> MoeExpertPayload:
        if layer_idx in layer_buffers:
            return layer_buffers[layer_idx]
        tier = plan.layer_tiers[layer_idx]
        if tier == StorageTier.GPU_HBM:
            buf = allocate_moe_payload(
                cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size,
                device=device, pin_memory=False, weight_dtype=weight_dtype, block_size=cfg.fp8_block_size,
            )
        elif tier == StorageTier.HOST_PINNED:
            buf = allocate_moe_payload(
                cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size,
                device="cpu", pin_memory=plan.profile.has_cuda, weight_dtype=weight_dtype, block_size=cfg.fp8_block_size,
            )
        else:
            # HOST_PAGEABLE or temporary CPU build buffer for NVME_SLAB (flushed & freed once layer completes!)
            buf = allocate_moe_payload(
                cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size,
                device="cpu", pin_memory=False, weight_dtype=weight_dtype, block_size=cfg.fp8_block_size,
            )
        layer_buffers[layer_idx] = buf
        return buf

    def _finalize_layer_if_done(layer_idx: int) -> None:
        if layer_idx in moe_tier_store:
            return
        if layer_remaining_shards[layer_idx]:
            return
        buf = layer_buffers.pop(layer_idx)
        tier = plan.layer_tiers[layer_idx]
        if tier == StorageTier.NVME_SLAB:
            slab_path = os.path.join(slab_dir, f"layer_{layer_idx:02d}.q38slab")
            write_moe_slab(slab_path, buf)
            del buf
            gc.collect()
            moe_tier_store[layer_idx] = slab_path
        else:
            moe_tier_store[layer_idx] = buf

    def _consume_non_ple_tensors(shard_name: str, shard_path: str) -> None:
        keys = shard_to_keys.get(shard_name, [])
        touched_layers: set[int] = set()
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            for raw_key in keys:
                norm_key = _normalize_text_key(raw_key)
                if norm_key is None or _PLE_SHARD_RE.search(norm_key) or norm_key.endswith("ngram_embedding.weight"):
                    continue
                if norm_key.endswith("ngram_embedding.weight_scale"):
                    continue

                m_sp = _EXPERT_SPLIT_RE.match(raw_key)
                if m_sp:
                    l_idx = int(m_sp.group(1))
                    e_idx = int(m_sp.group(2))
                    proj = m_sp.group(3)
                    kind = m_sp.group(4)
                    buf = _get_or_alloc_layer_buf(l_idx)
                    t = sf.get_tensor(raw_key)
                    inter = cfg.moe_intermediate_size
                    if proj == "gate_proj":
                        if kind == "weight":
                            buf.gate_up_proj[e_idx, :inter].copy_(t, non_blocking=True)
                        elif buf.gate_up_scale is not None:
                            s_half = buf.gate_up_scale.shape[1] // 2
                            buf.gate_up_scale[e_idx, :s_half].copy_(t.float(), non_blocking=True)
                    elif proj == "up_proj":
                        if kind == "weight":
                            buf.gate_up_proj[e_idx, inter:].copy_(t, non_blocking=True)
                        elif buf.gate_up_scale is not None:
                            s_half = buf.gate_up_scale.shape[1] // 2
                            buf.gate_up_scale[e_idx, s_half:].copy_(t.float(), non_blocking=True)
                    else:  # down_proj
                        if kind == "weight":
                            buf.down_proj[e_idx].copy_(t, non_blocking=True)
                        elif buf.down_scale is not None:
                            buf.down_scale[e_idx].copy_(t.float(), non_blocking=True)
                    touched_layers.add(l_idx)
                    continue

                m_fu = _EXPERT_FUSED_RE.match(raw_key)
                if m_fu:
                    l_idx = int(m_fu.group(1))
                    field_name = m_fu.group(2)
                    buf = _get_or_alloc_layer_buf(l_idx)
                    t = sf.get_tensor(raw_key)
                    if field_name == "gate_up_proj":
                        buf.gate_up_proj.copy_(t, non_blocking=True)
                    elif field_name == "down_proj":
                        buf.down_proj.copy_(t, non_blocking=True)
                    elif field_name == "gate_up_proj_scale_inv" and buf.gate_up_scale is not None:
                        buf.gate_up_scale.copy_(t.float(), non_blocking=True)
                    elif field_name == "down_proj_scale_inv" and buf.down_scale is not None:
                        buf.down_scale.copy_(t.float(), non_blocking=True)
                    touched_layers.add(l_idx)
                    continue

                # Dense weight (always BF16/FP32)
                t = sf.get_tensor(raw_key)
                m_l = _LAYER_RE.match(norm_key)
                if m_l:
                    l_idx = int(m_l.group(1))
                    sub = m_l.group(2)
                    dense_layers[l_idx][sub] = t
                else:
                    dense_top[norm_key] = t

        for l_idx in touched_layers:
            layer_remaining_shards[l_idx].discard(shard_name)
            _finalize_layer_if_done(l_idx)

    # 4. Stage 1: Handle PLE shards FIRST (so if ple_mode == "preextract", we delete 48 GiB PLE shards BEFORE MoE!)
    ple_providers: dict[int, Any] = {}
    ple_local_paths: list[str] = []
    consumed_shards: set[str] = set()

    if cfg.ple_layer_ids and ple_shard_names:
        _log(f"Downloading / opening {len(ple_shard_names)} PLE shards (mode={plan.ple_mode})...")
        for s_name in sorted(ple_shard_names):
            p = _fetch_file(s_name, local_dir=ple_dir)
            ple_local_paths.append(p)
            # Extract any co-located non-PLE dense/expert tensors from this shard too
            _consume_non_ple_tensors(s_name, p)
            consumed_shards.add(s_name)

        # Also fetch scale shards if separate from PLE shards
        ple_scales: dict[int, float] = {}
        for l_idx, (s_name, r_key) in ple_scale_keys.items():
            sp = _fetch_file(s_name, local_dir=ple_dir if s_name in ple_shard_names else None)
            with safe_open(sp, framework="pt", device="cpu") as sf:
                ple_scales[l_idx] = float(sf.get_tensor(r_key).float().item())
            if s_name not in consumed_shards:
                _consume_non_ple_tensors(s_name, sp)
                consumed_shards.add(s_name)
                if plan.evict_raw_shards and not is_local and s_name not in ple_shard_names:
                    _safe_delete_hf_file(sp)

        for ple_order, one_based_lid in enumerate(cfg.ple_layer_ids):
            l_idx = one_based_lid - 1
            spec = PleHashSpec(
                vocab_size=cfg.vocab_size,
                ngram_size=cfg.ngram_size,
                heads_per_ngram=cfg.heads_per_ngram,
                ngram_vocab_size_base=cfg.ngram_vocab_size_base,
                make_ngram_vocab_size_divisible_by=cfg.make_ngram_vocab_size_divisible_by,
                ple_embed_dim=cfg.ple_embed_dim,
                seed=cfg.seed,
                eos_token_id=cfg.eos_token_id,
                ple_layer_index=ple_order,
            )
            mmap_tbl = MmapPleTable.from_safetensors_files(
                spec, ple_local_paths, weight_scale=ple_scales.get(l_idx)
            )
            if plan.ple_mode == "preextract":
                if tokenized_prompts_for_preextract is None:
                    raise ValueError(
                        "ple_mode='preextract' requires `tokenized_prompts_for_preextract`."
                    )
                _log(f"Pre-extracting PLE embeddings for {len(tokenized_prompts_for_preextract)} prompts...")
                provider = PreextractedPleProvider.from_mmap_and_prompts(
                    mmap_tbl, tokenized_prompts_for_preextract
                )
                mmap_tbl.close()
                ple_providers[l_idx] = provider
            else:
                ple_providers[l_idx] = mmap_tbl

        if plan.ple_mode == "preextract" and not is_local:
            _log("Evicting 48 GiB PLE shards from disk after pre-extraction...")
            for p in ple_local_paths:
                _safe_delete_hf_file(p)

    # 5. Stage 2: Stream remaining backbone shards in layer order (packing Tier 0/1/2 + immediate eviction!)
    remaining_shards = [s for s in sorted(needed_shards) if s not in consumed_shards]
    # Order remaining shards by earliest layer they contribute to -> minimizes simultaneous open layer buffers!
    shard_first_layer: dict[str, int] = {}
    for l_idx in range(cfg.num_hidden_layers):
        for s_name in layer_expert_shards[l_idx]:
            shard_first_layer.setdefault(s_name, l_idx)
    remaining_shards.sort(key=lambda s: (shard_first_layer.get(s, 999), s))

    _log(f"Streaming {len(remaining_shards)} backbone/MoE shards into 3-tier hierarchy...")
    for idx, s_name in enumerate(remaining_shards, start=1):
        sp = _fetch_file(s_name)
        _consume_non_ple_tensors(s_name, sp)
        if plan.evict_raw_shards and not is_local:
            _safe_delete_hf_file(sp)
        if idx % 10 == 0 or idx == len(remaining_shards):
            done_layers = len(moe_tier_store)
            _log(f"  shards {idx}/{len(remaining_shards)} unpacked ({done_layers}/{cfg.num_hidden_layers} MoE layers packed)")

    # Finalize any edge-case layers
    for l_idx in range(cfg.num_hidden_layers):
        _finalize_layer_if_done(l_idx)

    # 6. Assemble top-level & per-layer dense GPU weight structs
    embed_tokens = dense_top.pop("embed_tokens.weight").to(device=device, dtype=compute_dtype)
    if "lm_head.weight" in dense_top:
        lm_head = dense_top.pop("lm_head.weight").to(device="cpu", dtype=compute_dtype).contiguous()
    else:
        lm_head = embed_tokens.detach().cpu().contiguous()

    hc_mixer = GatedResidualWeights(
        hc_norm_weight=dense_top.pop("hyper_connection_mixer.hc_norm.weight").to(device=device, dtype=compute_dtype),
        input_mix_weight_down=dense_top.pop("hyper_connection_mixer.input_mix_weight_down.weight").to(device=device, dtype=compute_dtype),
        input_mix_weight_up=dense_top.pop("hyper_connection_mixer.input_mix_weight_up.weight").to(device=device, dtype=compute_dtype),
        block_inject_weight=None,
        hc_count=cfg.hc_count,
        hidden_size=cfg.hidden_size,
        eps=cfg.rms_norm_eps,
    )

    assembled_layers = [
        _assemble_layer_dense(l_idx, dense_layers[l_idx], cfg, device, compute_dtype)
        for l_idx in range(cfg.num_hidden_layers)
    ]

    if plan.profile.has_cuda:
        torch.cuda.synchronize()
    gc.collect()

    _log("Qwen3.8-Flash-Next 3-tier model load complete!")
    return LoadedQwen38Model(
        config=cfg,
        plan=plan,
        device=device,
        compute_dtype=compute_dtype,
        embed_tokens=embed_tokens,
        hyper_connection_mixer=hc_mixer,
        lm_head=lm_head,
        layers=assembled_layers,
        moe_tier_store=moe_tier_store,
        ple_providers=ple_providers,
    )
