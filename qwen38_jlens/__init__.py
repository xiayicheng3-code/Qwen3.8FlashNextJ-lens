"""Ultra-fast Jacobian Lens (J-lens) fitting engine for Qwen3.8-Flash-Next (qwen4_exp)."""

from qwen38_jlens.checkpoint import (
    AsyncCheckpointManager,
    CheckpointConfig,
    ConvergenceLogger,
    export_neuronpedia_bundle,
    load_jlens_safetensors,
    peek_checkpoint_metadata,
    publish_bundle_to_hf,
    save_jlens_safetensors,
)
from qwen38_jlens.engine import (
    FitStats,
    Qwen38JLensEngine,
    compute_aligned_bwd_start,
    ensure_vram_headroom,
    model_vram_breakdown,
    valid_position_mask,
)
from qwen38_jlens.hardware import HardwareProfile, OffloadPlan, StorageTier, build_offload_plan, detect_hardware
from qwen38_jlens.kernels import (
    GatedDeltaRuleBackend,
    dequantize_fp8_blockwise,
    grouped_mm,
    swiglu_vjp_pack,
    verify_cuda_triton_kernels,
)
from qwen38_jlens.lens import Qwen38JacobianLens, render_slice_html
from qwen38_jlens.loader import LoadedQwen38Model, Qwen38TextConfig, load_qwen38_streaming
from qwen38_jlens.ple_mmap import InMemoryPleProvider, MmapPleTable, PleHashSpec, PreextractedPleProvider

__all__ = [
    "AsyncCheckpointManager",
    "CheckpointConfig",
    "ConvergenceLogger",
    "FitStats",
    "GatedDeltaRuleBackend",
    "HardwareProfile",
    "InMemoryPleProvider",
    "LoadedQwen38Model",
    "MmapPleTable",
    "OffloadPlan",
    "PleHashSpec",
    "PreextractedPleProvider",
    "Qwen38JLensEngine",
    "Qwen38JacobianLens",
    "Qwen38TextConfig",
    "StorageTier",
    "build_offload_plan",
    "compute_aligned_bwd_start",
    "dequantize_fp8_blockwise",
    "ensure_vram_headroom",
    "detect_hardware",
    "export_neuronpedia_bundle",
    "grouped_mm",
    "load_jlens_safetensors",
    "load_qwen38_streaming",
    "model_vram_breakdown",
    "peek_checkpoint_metadata",
    "publish_bundle_to_hf",
    "render_slice_html",
    "save_jlens_safetensors",
    "swiglu_vjp_pack",
    "valid_position_mask",
    "verify_cuda_triton_kernels",
]
