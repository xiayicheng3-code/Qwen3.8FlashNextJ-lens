"""Ultra-fast Jacobian Lens (J-lens) fitting engine for Qwen3.8-Flash-Next (qwen4_exp)."""

from qwen38_jlens.engine import FitStats, Qwen38JLensEngine, valid_position_mask
from qwen38_jlens.hardware import HardwareProfile, OffloadPlan, StorageTier, build_offload_plan, detect_hardware
from qwen38_jlens.kernels import GatedDeltaRuleBackend, dequantize_fp8_blockwise, grouped_mm, swiglu_vjp_pack
from qwen38_jlens.lens import Qwen38JacobianLens, render_slice_html
from qwen38_jlens.loader import LoadedQwen38Model, Qwen38TextConfig, load_qwen38_streaming
from qwen38_jlens.ple_mmap import InMemoryPleProvider, MmapPleTable, PleHashSpec, PreextractedPleProvider

__all__ = [
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
    "dequantize_fp8_blockwise",
    "detect_hardware",
    "grouped_mm",
    "load_qwen38_streaming",
    "render_slice_html",
    "swiglu_vjp_pack",
    "valid_position_mask",
]
