#!/usr/bin/env python3
"""Deterministically build the self-contained Google Colab notebook `qwen38_jlens_colab.ipynb`
directly from the pytest-verified `qwen38_jlens/` package modules."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "qwen38_jlens"
OUT_NB = ROOT / "qwen38_jlens_colab.ipynb"


def _read_module(name: str) -> str:
    return (PKG / name).read_text(encoding="utf-8").rstrip() + "\n"


def build_notebook() -> nbformat.NotebookNode:
    cells = []

    # ------------------------------------------------------------------ #
    # Cell 0: Title & Architecture Overview
    # ------------------------------------------------------------------ #
    cells.append(
        new_markdown_cell(
            """# ⚡ Fastest Qwen3.8-Flash-Next Jacobian Lens (J-Lens) Fitting Notebook

**Auto-adapts to randomly assigned Google Colab High-RAM runtimes: `G4` (RTX PRO 6000 Blackwell 96 GB), `A100 80 GB`, and `A100 40 GB`.**

---

### Why Naive J-Lens Fitting Fails on Qwen3.8-Flash-Next (`qwen4_exp`)
1. **4-Branch Gated Residual (`hc_count=4`)**: Decoder layers output `[B, T, 10240]` (`4 × 2560`), collapsed to `[B, T, 2560]` by a final `hyper_connection_mixer` before `lm_head` (there is no `model.norm`). Standard `jlens.from_hf` crashes immediately.
   - Our exact J-lens targets `h_final = hyper_connection_mixer(H_47)` (`2560` dims) w.r.t. each layer's full 4-branch residual `H_l` (`10240` dims), yielding **`J_l ∈ R^{2560 × 10240}`** in the **exact same number of output cotangents (`K = 2560`)**!
2. **172.78 GiB FP8 Checkpoint (`112.5 GiB` MoE + `47.7 GiB` 51B PLE + `8.2 GiB` dense BF16)**:
   - Naive `dim_batch=8` backprop streams `112.5 GiB` across PCIe **320× per prompt** (`36 TB/prompt` ≈ 24 minutes/prompt!) and OOM-kills A100-40G (83.5 GiB RAM) immediately.
   - Naive HF `AutoModelForCausalLM` triggers `transformers#48349` (silently corrupting BF16 dense layers) and downloads unneeded Vision + MTP shards.

---

### Core Breakthroughs in This Notebook
| Optimization | Impact |
|---|---|
| **Single-Sweep Layer-by-Layer Reverse Propagation** | Forward runs ONCE at `B = 1` (~120 MB saved inputs). Backward holds ALL `K = 2560` cotangents in ONE in-place GPU buffer (`6.25 GiB`) and computes all `2560` VJPs of layer `l` **before** moving to layer `l-1` -> **1× weight sweep/prompt** (40×–120× less PCIe/NVMe traffic!). |
| **Fused Active-Expert MoE VJP (`grouped_mm` + Triton)** | Each layer dequantizes ONLY hit experts **ONCE** via a custom 128×128 Triton FP8 kernel (~2.5 ms), runs Tensor-Core-saturated CUTLASS/cuBLAS `grouped_mm` across `n_e × K_blk` rows, fuses SwiGLU backward in Triton, and includes the exact top-k softmax router VJP without materializing `[T, K, 512]`. |
| **Zero-RAM 51B PLE Mmap / Pre-Extraction** | `MADV_RANDOM` mmap directly on PLE safetensors shards (`0 GiB` VRAM, `0 GiB` pinned RAM, `< 2 ms`/prompt), with automatic two-stage pre-extraction (`~64 MB`) if disk is tight. |
| **Interleaved 3-Tier Async Double-Buffering** | Permanently pins dense BF16 (`8.2 GiB`) + max MoE layers in **Tier 0 GPU HBM**, spills to **Tier 1 Pinned/Pageable RAM**, and (on A100 40G) interleaves **Tier 2 contiguous NVMe slabs** every ~2.4 layers so background NVMe preads & H2D DMAs hide **100% behind backward compute**! |
| **Hardware-Optimal Kernels** | **G4 Blackwell (SM120)**: `FlashQLA v0.1.3` (TileLang SM120 fused GDN fwd+bwd). **A100 (SM80)**: `FLA` Triton fused GDN fwd+bwd. **QSA (all GPUs)**: exact indexer bypass for `T ≤ 2048` + L2-cached exact `B=1` attention VJP (cuBLAS batched GEMMs). |

---

### Concrete Auto-Planned Profiles (Default `T = 128`, `Qwen/Qwen3.8-Flash-Next-FP8`)
| Colab Runtime | GPU Arch | Tier 0 GPU HBM | Tier 1 Host RAM | Tier 2 NVMe Slabs | GDN Kernel | Est. Speed |
|---|---|---|---|---|---|---|
| **G4 High RAM** (95.6 GiB / 177 GiB RAM) | SM120 | 8.2G dense + **26 MoE layers** | **22 MoE layers** (pinned, interleaved) | **0 layers** | **FlashQLA SM120** | **~7–10 s/prompt** |
| **A100 80G High RAM** (79.2 GiB / 167 GiB RAM) | SM80 | 8.2G dense + **19 MoE layers** | **29 MoE layers** (pinned, interleaved) | **0 layers** | **FLA Triton SM80** | **~18–22 s/prompt** |
| **A100 40G High RAM** (39.6 GiB / 83.5 GiB RAM) | SM80 | 8.2G dense + **5 MoE layers** | **23 MoE layers** (14 pin + 9 page) | **20 MoE layers** (interleaved) | **FLA Triton SM80** | **~22–28 s/prompt** |
"""
        )
    )

    # ------------------------------------------------------------------ #
    # Cell 1: Environment Setup & GPU-Adaptive Kernel Installer
    # ------------------------------------------------------------------ #
    cells.append(
        new_markdown_cell("## 1. Environment Setup & GPU-Adaptive Kernel Installation")
    )
    cells.append(
        new_code_cell(
            """import os, subprocess, sys, torch

# Critical CUDA allocator & fast HF transfer settings (set before heavy allocations)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

assert torch.cuda.is_available(), "No CUDA GPU detected! Select A100 or G4 High-RAM runtime in Colab."
props = torch.cuda.get_device_properties(0)
cc = (props.major, props.minor)
print(f"Detected GPU: {props.name} | Compute Capability: SM{cc[0]}{cc[1]} | VRAM: {props.total_memory / 1024**3:.1f} GiB")
print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")

def pip_install(pkgs, extra_args=(), optional=False):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *extra_args, *pkgs]
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        if optional:
            print(f"  [optional install skipped with code {res.returncode}]: {pkgs}")
        else:
            raise RuntimeError(f"pip install failed for {pkgs}")

# Core dependencies
pip_install([
    "transformers>=5.17.0",
    "safetensors>=0.4.5",
    "huggingface_hub>=0.26.0",
    "hf_transfer",
    "datasets>=3.0.0",
    "psutil",
    "tqdm",
])

# Architecture-specific optimal GDN linear-attention kernel:
#   - SM90 / SM100 / SM120 (Colab G4 Blackwell, H100, B200): FlashQLA v0.1.3 (TileLang fused fwd+bwd) + FLA fallback
#   - SM80 (Colab A100 40G / 80G): FLA Triton fused chunk_gated_delta_rule (FlashQLA requires SM90+)
if cc in ((9, 0), (10, 0), (10, 3), (12, 0), (12, 1)):
    print(f"SM{cc[0]}{cc[1]} supports FlashQLA (2-3x faster GDN fwd/bwd)! Installing flash-qla + flash-linear-attention...")
    pip_install(["flash-qla"], optional=True)
    pip_install(["flash-linear-attention"], optional=True)
elif cc >= (8, 0):
    print(f"SM{cc[0]}{cc[1]} detected (A100 Ampere): installing flash-linear-attention (Triton GDN fused fwd+bwd)...")
    pip_install(["flash-linear-attention"], optional=True)
else:
    print(f"SM{cc[0]}{cc[1]}: using PyTorch chunked GDN backend.")

print("Setup complete!")
"""
        )
    )

    # ------------------------------------------------------------------ #
    # Cell 2: User Configuration & Optional Google Drive Persistence
    # ------------------------------------------------------------------ #
    cells.append(
        new_markdown_cell("## 2. Fitting Configuration & Checkpoint Persistence")
    )
    cells.append(
        new_code_cell(
            """#@title Qwen3.8-Flash-Next J-Lens Configuration { display-mode: "form" }
MODEL_ID = "Qwen/Qwen3.8-Flash-Next-FP8"  #@param {type:"string"}
N_PROMPTS = 100          #@param {type:"integer"}  # 100 = usable lens (paper §9.3); 1000 = full paper lens
PROMPT_OFFSET = 0        #@param {type:"integer"}  # Shift start index to shard across parallel Colab sessions & merge()
MAX_SEQ_LEN = 128        #@param {type:"integer"}  # Paper default: 128 tokens (keeps full K=2560 in 1 sweep!)
SKIP_FIRST = 16          #@param {type:"integer"}  # Exclude attention-sink prefix tokens (paper default: 16)
LAYER_STRIDE = 1         #@param {type:"integer"}  # 1 = all 48 layers (0..47); 2 = every 2nd layer
CHECKPOINT_EVERY = 5     #@param {type:"integer"}  # Atomic checkpoint cadence (prompts)
PLE_MODE = "auto"        #@param ["auto", "mmap", "preextract"]
MOUNT_GOOGLE_DRIVE = True #@param {type:"boolean"}
DRIVE_OUT_DIR = "/content/drive/MyDrive/qwen38_jlens_out" #@param {type:"string"}
LOCAL_WORK_DIR = "/content/qwen38_work"  #@param {type:"string"}
HF_TOKEN = ""            #@param {type:"string"}

import os
os.makedirs(LOCAL_WORK_DIR, exist_ok=True)
SLAB_DIR = os.path.join(LOCAL_WORK_DIR, "moe_slabs")
PLE_DIR = os.path.join(LOCAL_WORK_DIR, "ple_shards")
os.makedirs(SLAB_DIR, exist_ok=True)
os.makedirs(PLE_DIR, exist_ok=True)

# Resolve optional HF_TOKEN from Colab Secrets if not pasted directly
if not HF_TOKEN:
    try:
        from google.colab import userdata
        HF_TOKEN = userdata.get("HF_TOKEN") or ""
    except Exception:
        HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_TOKEN = HF_TOKEN or None

if MOUNT_GOOGLE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        OUT_DIR = DRIVE_OUT_DIR
    except Exception as exc:
        print(f"Google Drive mount unavailable ({exc}); saving outputs to {LOCAL_WORK_DIR}")
        OUT_DIR = LOCAL_WORK_DIR
else:
    OUT_DIR = LOCAL_WORK_DIR

os.makedirs(OUT_DIR, exist_ok=True)
TAG = f"off{PROMPT_OFFSET}_n{N_PROMPTS}_s{LAYER_STRIDE}"
CKPT_PATH = os.path.join(OUT_DIR, f"qwen38_jlens_ckpt_{TAG}.pt")
LENS_PATH = os.path.join(OUT_DIR, f"qwen38_flash_next_jlens_{TAG}.pt")

print(f"Outputs & checkpoints -> {OUT_DIR}")
print(f"  Checkpoint : {CKPT_PATH}")
print(f"  Final Lens : {LENS_PATH}")
"""
        )
    )

    # ------------------------------------------------------------------ #
    # Cells 3..9: Self-Contained Verified Engine Modules
    # ------------------------------------------------------------------ #
    cells.append(
        new_markdown_cell(
            """## 3. Install Self-Contained `qwen38_jlens` Engine Modules
Writing the verified, modular engine directly into `/content/qwen38_jlens/` so this notebook runs 100% standalone on Colab without needing external repo clones."""
        )
    )
    cells.append(new_code_cell("import os\nos.makedirs('qwen38_jlens', exist_ok=True)\n"))

    module_order = [
        ("hardware.py", "Hardware Auto-Detection & Concrete 3-Tier Interleaved Offloading Planner"),
        ("ple_mmap.py", "Zero-RAM 51B PLE N-Gram Mmap Table & Exact Hasher"),
        ("kernels.py", "GPU Kernels: FlashQLA (SM120/90), FLA (SM80), Triton FP8 Block-128 Dequant & SwiGLU VJP, grouped_mm"),
        ("model_ops.py", "Exact Qwen3.8-Flash-Next (qwen4_exp) Forward & Batched-Cotangent Analytical VJP Operators"),
        ("loader.py", "Streaming Shard-by-Shard Safetensors Packer & 3-Tier MoE Slab Manager"),
        ("lens.py", "Qwen38JacobianLens Container, 4-Branch Diagnostics & Interactive Slice Renderer"),
        ("engine.py", "Async Double-Buffered 3-Tier MoE Streamer & Single-Sweep Layer-by-Layer J-Lens Engine"),
        ("__init__.py", "Package Exports"),
    ]
    for mod_name, title in module_order:
        code = _read_module(mod_name)
        # Validate Python syntax of every module before embedding
        ast.parse(code, filename=mod_name)
        cells.append(new_code_cell(f"%%writefile qwen38_jlens/{mod_name}\n{code}"))

    # ------------------------------------------------------------------ #
    # Cell 10: Live Hardware Probe, Concrete Offload Plan & Kernel Micro-Benchmark
    # ------------------------------------------------------------------ #
    cells.append(
        new_markdown_cell("## 4. Live Hardware Auto-Detection, Concrete Offload Plan & Kernel Micro-Benchmark")
    )
    cells.append(
        new_code_cell(
            """import logging, time, torch
from qwen38_jlens import (
    GatedDeltaRuleBackend,
    StorageTier,
    build_offload_plan,
    dequantize_fp8_blockwise,
    detect_hardware,
    grouped_mm,
    swiglu_vjp_pack,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)

SOURCE_LAYERS = list(range(0, 48, LAYER_STRIDE))
if 47 not in SOURCE_LAYERS:
    SOURCE_LAYERS.append(47)

profile = detect_hardware(LOCAL_WORK_DIR)
plan = build_offload_plan(
    profile,
    num_layers=48,
    d_model=2560,
    hc_count=4,
    num_experts=512,
    moe_intermediate_size=640,
    max_seq_len=MAX_SEQ_LEN,
    num_source_layers=len(SOURCE_LAYERS),
    force_ple_mode=None if PLE_MODE == "auto" else PLE_MODE,
)
print(plan.summary())

# Show exact interleaved layer -> tier map (visual check!)
tier_sym = {
    StorageTier.GPU_HBM: "G",
    StorageTier.HOST_PINNED: "P",
    StorageTier.HOST_PAGEABLE: "R",
    StorageTier.NVME_SLAB: "D",
}
tier_map_str = " ".join(f"{l:02d}:{tier_sym[plan.layer_tiers[l]]}" for l in range(48))
print("\\nInterleaved Layer Placement Map (G=GPU HBM, P=Pinned RAM, R=Pageable RAM, D=NVMe Slab):")
for row_start in range(0, 48, 12):
    print("  " + " ".join(f"L{l:02d}:{tier_sym[plan.layer_tiers[l]]}" for l in range(row_start, row_start + 12)))

# ---- 3-Second Live Kernel Verification & Speed Micro-Benchmark ----
print("\\nRunning live GPU kernel parity & bandwidth benchmark...")
gdn_bench = GatedDeltaRuleBackend(preferred=plan.gdn_backend, verify_on_init=True)
print(f"  [OK] Active GDN kernel : {gdn_bench.name} (direct={gdn_bench._use_direct})")

dev = torch.device("cuda:0")
# 1) Benchmark Triton 128x128 FP8 block dequant (380 active experts ~ 1 typical prompt layer)
E_act = min(380, plan.expert_dequant_chunk)
w_fp8 = torch.randn(E_act, 1280, 2560, device=dev).to(torch.float8_e4m3fn)
s_fp32 = torch.ones(E_act, 10, 20, device=dev, dtype=torch.float32) * 0.05
_ = dequantize_fp8_blockwise(w_fp8, s_fp32, use_triton=plan.use_triton_fp8_dequant)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    w_bf16 = dequantize_fp8_blockwise(w_fp8, s_fp32, use_triton=plan.use_triton_fp8_dequant)
torch.cuda.synchronize()
dt_deq = (time.perf_counter() - t0) / 5
bytes_rw = E_act * 1280 * 2560 * (1 + 2)
print(f"  [OK] Triton FP8->BF16 dequant ({E_act} experts): {dt_deq*1e3:.2f} ms ({bytes_rw / dt_deq / 1e9:.0f} GB/s)")

# 2) Benchmark CUTLASS/cuBLAS grouped_mm backward GEMM (gate_up_proj VJP at K_blk = moe_k_chunk)
K_blk = plan.moe_k_chunk
S_rows = 128 * 10
counts = torch.full((E_act,), max(1, S_rows // E_act), device=dev, dtype=torch.int32)
offs_K = torch.cumsum(counts * K_blk, dim=0, dtype=torch.int32)
M_tot = int(offs_K[-1].item())
dGU = torch.randn(M_tot, 1280, device=dev, dtype=torch.bfloat16)
_ = grouped_mm(dGU, w_bf16, offs_K)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(3):
    out_gmm = grouped_mm(dGU, w_bf16, offs_K)
torch.cuda.synchronize()
dt_gmm = (time.perf_counter() - t0) / 3
tflops = (2.0 * M_tot * 1280 * 2560) / dt_gmm / 1e12
print(f"  [OK] MoE grouped_mm VJP (M={M_tot}, K=1280, N=2560): {dt_gmm*1e3:.2f} ms ({tflops:.1f} TFLOPS)")

del w_fp8, s_fp32, w_bf16, dGU, out_gmm
torch.cuda.empty_cache()
"""
        )
    )

    # ------------------------------------------------------------------ #
    # Cell 11: Load Tokenizer & Fitting Corpus (WikiText-103)
    # ------------------------------------------------------------------ #
    cells.append(
        new_markdown_cell(
            """## 5. Load Tokenizer & Pretraining-Like Fitting Corpus (`Salesforce/wikitext`)
Matches `anthropics/jacobian-lens`: streams `Salesforce/wikitext` (`wikitext-103-raw-v1`, `min_chars=600`), truncates to `MAX_SEQ_LEN=128` tokens, and prepends BOS when missing."""
        )
    )
    cells.append(
        new_code_cell(
            """from datasets import load_dataset
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=False)

def encode_prompt(text: str, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=False)
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    if bos_id is not None and (not ids or ids[0] != bos_id):
        ids = [bos_id] + ids
    ids = ids[:max_len]
    return torch.tensor([ids], dtype=torch.long)

def load_wikitext_slice(offset: int, n_prompts: int, min_chars: int = 600) -> list[str]:
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    out = []
    seen = 0
    for rec in ds:
        txt = rec["text"]
        if len(txt.strip()) >= min_chars:
            if seen >= offset:
                out.append(txt)
                if len(out) >= n_prompts:
                    break
            seen += 1
    return out

raw_prompts = load_wikitext_slice(PROMPT_OFFSET, N_PROMPTS)
tokenized_prompts = [encode_prompt(p, MAX_SEQ_LEN) for p in raw_prompts]

# Also tokenize evaluation prompts up front (needed if tight-disk `ple_mode='preextract'` triggers)
EVAL_PROMPTS = [
    "Fact: The currency used in the country shaped like a boot is",
    "The capital of the country where Mount Everest is located is",
    "     _______     \\n   /         \\\\   \\n  /  ~     ~  \\\\  \\n (   o     o   ) \\n |      ^      | \\n  \\\\   \\u2014\\u2014\\u2014   /  \\n   \\\\_________/   \\n\\nWhat is this?",
]
eval_tokenized = [encode_prompt(p, max_len=256) for p in EVAL_PROMPTS]

lengths = [int(t.shape[1]) for t in tokenized_prompts]
print(f"Prepared {len(tokenized_prompts)} WikiText prompts (offset={PROMPT_OFFSET}): "
      f"min_len={min(lengths)}, median={sorted(lengths)[len(lengths)//2]}, max_len={max(lengths)}")
"""
        )
    )

    # ------------------------------------------------------------------ #
    # Cell 12: Streaming Shard-by-Shard 3-Tier Model Loader
    # ------------------------------------------------------------------ #
    cells.append(
        new_markdown_cell(
            """## 6. Stream-Load `Qwen/Qwen3.8-Flash-Next-FP8` Directly Into the 3-Tier Hierarchy
- Skips Vision encoder and MTP shards automatically.
- Loads all dense BF16 weights (`~8.2 GiB`) directly to GPU (immune to `transformers#48349`).
- Keeps 51B PLE n-gram table out of RAM (`MADV_RANDOM` NVMe mmap or tight-disk pre-extraction).
- Unpacks each raw safetensors shard straight into **Tier 0 GPU HBM / Tier 1 Pinned+Pageable RAM / Tier 2 contiguous NVMe slabs**, evicting raw HF blobs immediately so Colab disk stays well below 235 GiB."""
        )
    )
    cells.append(
        new_code_cell(
            """import shutil, time, psutil, torch
from qwen38_jlens import Qwen38JLensEngine, load_qwen38_streaming

t_load = time.perf_counter()
loaded_model = load_qwen38_streaming(
    MODEL_ID,
    plan,
    slab_dir=SLAB_DIR,
    ple_dir=PLE_DIR,
    tokenized_prompts_for_preextract=(tokenized_prompts + eval_tokenized) if plan.ple_mode == "preextract" else None,
    hf_token=HF_TOKEN,
    compute_dtype=torch.bfloat16,
    progress_callback=lambda msg: print(f"[Loader {time.perf_counter()-t_load:5.0f}s] {msg}", flush=True),
)
engine = Qwen38JLensEngine(loaded_model)

vram_used = torch.cuda.memory_allocated() / 1024**3
ram_used = psutil.virtual_memory().used / 1024**3
disk_free = shutil.disk_usage(LOCAL_WORK_DIR).free / 1024**3
print(f"\\nReady for J-Lens fitting in {time.perf_counter()-t_load:.1f}s!")
print(f"  Active GDN Kernel : {engine.gdn_backend.name} (direct={engine.gdn_backend._use_direct})")
print(f"  Live Telemetry    : VRAM alloc={vram_used:.1f} GiB | RAM used={ram_used:.1f} GiB | Disk free={disk_free:.1f} GiB")
"""
        )
    )

    # ------------------------------------------------------------------ #
    # Cell 13: Run Single-Sweep Layer-by-Layer J-Lens Fitting
    # ------------------------------------------------------------------ #
    cells.append(
        new_markdown_cell(
            """## 7. Fit Qwen3.8-Flash-Next Jacobian Lens (`J_l ∈ R^{2560 × 10240}`)
Runs our single-sweep layer-by-layer reverse propagator with live convergence telemetry and atomic checkpointing (`CKPT_PATH`). If Colab disconnects, simply re-run Cells 1–7 and fitting resumes seamlessly from the last checkpoint!"""
        )
    )
    cells.append(
        new_code_cell(
            """import time, torch
from tqdm.auto import tqdm

pbar = tqdm(total=len(tokenized_prompts), desc="Fitting Qwen3.8 J-Lens", unit="prompt")
times = []

def _on_prompt(stats):
    times.append(stats.elapsed_s)
    avg_s = sum(times[-10:]) / len(times[-10:])
    vram_pk = torch.cuda.max_memory_allocated() / 1024**3
    pbar.n = stats.prompt_idx + 1
    pbar.set_postfix(
        sec=f"{stats.elapsed_s:.1f}s",
        fwd=f"{stats.fwd_s:.1f}s",
        bwd=f"{stats.bwd_s:.1f}s",
        avg=f"{avg_s:.1f}s/p",
        d_mean=f"{stats.mean_rel_change:.1e}",
        vram_pk=f"{vram_pk:.1f}G",
    )
    pbar.refresh()

torch.cuda.reset_peak_memory_stats()
t_fit = time.perf_counter()

lens = engine.fit(
    tokenized_prompts,
    source_layers=SOURCE_LAYERS,
    skip_first=SKIP_FIRST,
    checkpoint_path=CKPT_PATH,
    checkpoint_every=CHECKPOINT_EVERY,
    resume=True,
    on_prompt_done=_on_prompt,
)
pbar.close()

lens.save(LENS_PATH, dtype=torch.float16, include_collapsed=True)
total_min = (time.perf_counter() - t_fit) / 60.0
peak_vram = torch.cuda.max_memory_allocated() / 1024**3
print(f"\\nSaved fitted lens: {LENS_PATH}")
print(f"  Summary   : {lens}")
print(f"  Total time: {total_min:.2f} min ({(total_min*60)/max(1, len(times)):.2f} s/prompt)")
print(f"  Peak VRAM : {peak_vram:.2f} GiB")
"""
        )
    )

    # ------------------------------------------------------------------ #
    # Cell 14: Verify Fitted J-Lens vs Logit-Lens + 4-Branch Diagnostics + Slice HTML
    # ------------------------------------------------------------------ #
    cells.append(
        new_markdown_cell(
            """## 8. Verify Fitted J-Lens: Multi-Hop Readout, 4-Branch Gated Residual Norms & Interactive Slice
Compares **J-Lens** (`lm_head(H_l @ J_l.T)`) vs **Logit-Lens** (`lm_head(hyper_connection_mixer(H_l))`) on two-hop reasoning prompts and inspects how signal distributes across Qwen3.8-Flash-Next's 4 Gated Residual branches."""
        )
    )
    cells.append(
        new_code_cell(
            """import torch
from IPython.display import HTML, display
from qwen38_jlens import render_slice_html

# 1. Inspect 4-branch Gated Residual Frobenius norms across depth
bnorms = lens.branch_norms()
print("4-Branch Gated Residual Jacobian Norms ||J_l[:, b, :]||_F / sqrt(2560):")
for l in [0, 8, 16, 24, 32, 40, 47]:
    if l in bnorms:
        b_str = "  ".join(f"b{b}={v:.3f}" for b, v in enumerate(bnorms[l]))
        print(f"  L{l:02d}: {b_str}")

# 2. Multi-hop J-lens vs Logit-lens readout at final token
test_prompt = EVAL_PROMPTS[0]
test_ids = eval_tokenized[0]
probe_layers = [l for l in (8, 16, 24, 32, 40, 46, 47) if l in lens.jacobians]

jl_logits, model_logits = engine.apply_lens(lens, test_ids, layers=probe_layers, positions=[-1], use_jacobian=True)
ll_logits, _ = engine.apply_lens(lens, test_ids, layers=probe_layers, positions=[-1], use_jacobian=False)

def top5(logits_1d):
    vals, idxs = torch.softmax(logits_1d.float(), dim=-1).topk(5)
    return [f"{tokenizer.decode([int(i)])!r}({float(v)*100:.0f}%)" for v, i in zip(vals, idxs)]

print(f"\\nPrompt: {test_prompt!r}")
for l in probe_layers:
    print(f"  L{l:02d} Logit-Lens: {top5(ll_logits[l][0])}")
    print(f"  L{l:02d} J-Lens    : {top5(jl_logits[l][0])}")
print(f"  Model Output   : {top5(model_logits[0])}")

# 3. Interactive position x layer HTML slice heatmap (every 4th layer for clean viewing)
slice_layers = [l for l in lens.source_layers if l % 4 == 0 or l == lens.source_layers[-1]]
all_jl, all_model = engine.apply_lens(lens, test_ids, layers=slice_layers, positions=None, use_jacobian=True)

tok_strs = [tokenizer.decode([int(t)]) for t in test_ids[0].tolist()]
layer_top1 = {}
for l in slice_layers:
    probs = torch.softmax(all_jl[l].float(), dim=-1)
    pv, pi = probs.max(dim=-1)
    layer_top1[l] = [(tokenizer.decode([int(i)]), float(v)) for v, i in zip(pv, pi)]

m_probs = torch.softmax(all_model.float(), dim=-1)
mpv, mpi = m_probs.max(dim=-1)
model_top1 = [(tokenizer.decode([int(i)]), float(v)) for v, i in zip(mpv, mpi)]

display(HTML(render_slice_html(tok_strs, layer_top1, model_top1, title=f"Qwen3.8-Flash-Next J-Lens (n={lens.n_prompts})")))
"""
        )
    )

    # ------------------------------------------------------------------ #
    # Cell 15: Multi-Session Sharded Merge & Optional HF Hub Upload
    # ------------------------------------------------------------------ #
    cells.append(
        new_markdown_cell(
            """## 9. (Optional) Merge Parallel Colab Shards (`n = 1000`) & Upload to Hugging Face Hub
Run multiple Colab sessions with `PROMPT_OFFSET = 0, 100, 200, ...` saving to the same Google Drive folder, then merge them with exact `n_prompts`-weighting via `Qwen38JacobianLens.merge(...)`."""
        )
    )
    cells.append(
        new_code_cell(
            """# Uncomment to merge multiple sharded lens files into one combined lens:
# import glob
# from qwen38_jlens import Qwen38JacobianLens
# shard_files = sorted(glob.glob(f"{OUT_DIR}/qwen38_flash_next_jlens_off*.pt"))
# print("Merging:", shard_files)
# merged = Qwen38JacobianLens.merge([Qwen38JacobianLens.load(p) for p in shard_files])
# merged_path = f"{OUT_DIR}/qwen38_flash_next_jlens_merged_n{merged.n_prompts}.pt"
# merged.save(merged_path)
# print("Saved merged lens:", merged_path, merged)

# Optional cleanup of temporary local NVMe slabs / PLE shards:
# engine.close()
"""
        )
    )

    nb = new_notebook(
        cells=cells,
        metadata={
            "colab": {
                "provenance": [],
                "gpuType": "A100",
                "machine_shape": "hm",
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
            "accelerator": "GPU",
        },
    )
    return nb


def main() -> None:
    nb = build_notebook()
    nbformat.validate(nb)
    OUT_NB.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote validated notebook: {OUT_NB} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
