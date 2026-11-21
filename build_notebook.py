#!/usr/bin/env python3
"""Deterministically build the clean, readable Google Colab notebook `qwen38_jlens_colab.ipynb`
that installs and imports `qwen38_jlens` directly from the public GitHub repository."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent
OUT_NB = ROOT / "qwen38_jlens_colab.ipynb"
REPO_URL = "https://github.com/xiayicheng3-code/Qwen3.8FlashNextJ-lens.git"


def build_notebook() -> nbformat.NotebookNode:
    cells = []

    # ------------------------------------------------------------------ #
    # Cell 0: Title & Architecture Overview
    # ------------------------------------------------------------------ #
    cells.append(
        new_markdown_cell(
            """# ⚡ Fastest Qwen3.8-Flash-Next Jacobian Lens (J-Lens) Fitting Notebook

**Auto-adapts to randomly assigned Google Colab High-RAM runtimes: `G4` (RTX PRO 6000 Blackwell 96 GB), `A100 80 GB`, and `A100 40 GB`.**
Imports [`qwen38_jlens`](https://github.com/xiayicheng3-code/Qwen3.8FlashNextJ-lens) directly from GitHub so the notebook stays short, readable, and always up to date.

---

### Why Naive J-Lens Fitting Fails on Qwen3.8-Flash-Next (`qwen4_exp`)
1. **4-Branch Gated Residual (`hc_count=4`)**: Decoder layers output `[B, T, 10240]` (`4 × 2560`), collapsed to `[B, T, 2560]` by a final `hyper_connection_mixer` before `lm_head` (there is no `model.norm`). Standard `jlens.from_hf` crashes immediately.
   - Our exact J-lens targets `h_final = hyper_connection_mixer(H_47)` (`2560` dims) w.r.t. each layer's full 4-branch residual `H_l` (`10240` dims), yielding **`J_l ∈ R^{2560 × 10240}`** in the **exact same number of output cotangents (`K = 2560`)**!
2. **172.78 GiB FP8 Checkpoint (`112.5 GiB` MoE + `47.7 GiB` 51B PLE + `8.0 GiB` dense BF16)**:
   - Naive `dim_batch=8` backprop streams `112.5 GiB` across PCIe **320× per prompt** (`36 TB/prompt` ≈ 24 minutes/prompt!) and OOM-kills A100-40G (83.5 GiB RAM) immediately.
   - Naive HF `AutoModelForCausalLM` triggers `transformers#48349` (silently corrupting BF16 dense layers) and downloads unneeded Vision + MTP shards.

---

### Core Breakthroughs
| Optimization | Impact |
|---|---|
| **Single-Sweep Reverse Propagation + 16-Aligned Causal Suffix Pruning** | Forward runs ONCE per micro-batch of `FWD_BATCH_SIZE = 8` prompts (caching ~120 MB/prompt in CPU RAM -> **0 GiB extra GPU VRAM, 8× fewer forward weight sweeps**!). Backward exploits block lower-triangular causality to prune masked prefix tokens `[0 : bwd_start]` (`T_v = 112` at `T = 128, skip_first = 16`), holding ALL `K = 2560` cotangents in ONE in-place GPU buffer (`5.47 GiB`, `+0.78 GiB` VRAM headroom) while preserving 128-wide MoE/projection/FlashQLA tiling -> **1× backward weight sweep/prompt** (40×–120× less PCIe/NVMe traffic!). |
| **Fused Active-Expert MoE VJP (`grouped_mm` + Triton)** | Each layer dequantizes ONLY hit experts **ONCE** via a custom 128×128 Triton FP8 kernel (native FP8 on SM89+ and uint8+LUT on SM80 A100), passes zero-copy transposed views into CUTLASS/cuBLAS `grouped_mm`, fuses SwiGLU backward in Triton, and computes the exact top-k softmax router VJP via batched `bmm` (`~0.07 GiB` scratch vs `6.84 GiB`). |
| **Zero-RAM 51B PLE Mmap / Pre-Extraction** | `MADV_RANDOM` mmap directly on PLE safetensors shards (`0 GiB` VRAM, `0 GiB` pinned RAM, `< 2 ms`/prompt), with automatic two-stage pre-extraction (`~64 MB`) if disk is tight. |
| **Interleaved 3-Tier Async Double-Buffering + Auto VRAM Self-Heal** | Permanently pins dense BF16 (`8.0 GiB`) + max MoE layers in **Tier 0 GPU HBM**, spills to **Tier 1 Pinned/Pageable RAM**, and (on A100 40G) interleaves **Tier 2 contiguous NVMe slabs** so background NVMe preads & H2D DMAs hide **100% behind backward compute**. After loading, `ensure_vram_headroom` verifies live free VRAM and auto-demotes Tier-0 layers to pinned RAM if headroom is ever below the runtime reserve. |
| **Hardware-Optimal Kernels** | **G4 Blackwell (SM120)**: `FlashQLA v0.1.3` (TileLang SM120 fused GDN fwd+bwd). **A100 (SM80)**: `FLA` Triton fused GDN fwd+bwd. **QSA (all GPUs)**: exact indexer bypass for `T ≤ 2048` + L2-cached exact `B=1` attention VJP (cuBLAS batched GEMMs). |

---

### Concrete Auto-Planned Profiles (Default `T = 128`, `Qwen/Qwen3.8-Flash-Next-FP8`)
| Colab Runtime | GPU Arch | Tier 0 GPU HBM | Tier 1 Host RAM | Tier 2 NVMe Slabs | Dequant Chunk | GDN Kernel | Est. Speed |
|---|---|---|---|---|---|---|---|
| **G4 High RAM** (95.6 GiB / 177 GiB RAM) | SM120 | 8.0G dense + **25 MoE layers** | **23 MoE layers** (pinned, interleaved) | **0 layers** | 512 | **FlashQLA SM120** | **~7–10 s/prompt** |
| **A100 80G High RAM** (79.2 GiB / 167 GiB RAM) | SM80 | 8.0G dense + **19 MoE layers** | **29 MoE layers** (pinned, interleaved) | **0 layers** | 256 | **FLA Triton SM80** | **~18–22 s/prompt** |
| **A100 40G High RAM** (39.6 GiB / 83.5 GiB RAM) | SM80 | 8.0G dense + **3 MoE layers** | **23 MoE layers** (14 pin + 9 page) | **22 MoE layers** (interleaved) | 256 | **FLA Triton SM80** | **~22–28 s/prompt** |
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

# Critical CUDA allocator & fast HF Xet transfer settings (set before heavy allocations)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
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
    # Cell 2: User Configuration & Multi-Tier Checkpoint Persistence
    # ------------------------------------------------------------------ #
    cells.append(
        new_markdown_cell(
            """## 2. Fitting Configuration, Async Drive Checkpointing & Neuronpedia Publish Settings
Clean **split-role persistence** (never wastes outbound bandwidth uploading to both Drive and HF mid-run, and keeps your Hugging Face repo history to a single clean commit):
1. **During fitting → Local NVMe (`< 1 s`) + Async Google Drive ONLY**:
   - Stores running mean `J_mean = jacobian_sum / n_done` in a single `.safetensors` file (`float32` = `4.69 GiB` bit-exact resume by default for Google AI Pro 5 TB Drive; `float16` = `2.34 GiB` available for free 15 GB Drive).
   - Background Drive copy + `convergence.csv` mirror run on a worker thread with **0 s GPU stall** and zero HF commit-rate-limit exposure.
   - Any interrupted checkpoint is immediately loadable via `Qwen38JacobianLens.load(ckpt_path)` and auto-resumes seamlessly across disconnects.
2. **After fitting (or merging shards) → One-Shot Neuronpedia-Ready Publish to Hugging Face Hub (Section 9)**:
   - Exports `Qwen3.8-Flash-Next-FP8_jacobian_lens.pt` (`fp16`, Neuronpedia drop-in schema + `provenance`), `Qwen3.8-Flash-Next-FP8_jacobian_lens_4branch.safetensors` (full lossless 4-branch `[2560, 10240]`), `convergence.csv`, `config.yaml`, and `CREDIT.md`, then uploads the folder in **one single clean commit**."""
        )
    )
    cells.append(
        new_code_cell(
            """#@title Qwen3.8-Flash-Next J-Lens, Checkpoint & Publish Configuration { display-mode: "form" }
MODEL_ID = "Qwen/Qwen3.8-Flash-Next-FP8"  #@param {type:"string"}
N_PROMPTS = 100              #@param {type:"integer"}  # 100 = usable lens; 1000 w/ STOP_AT_DELTA=0.002 = full Neuronpedia recipe
PROMPT_OFFSET = 0            #@param {type:"integer"}  # Shift start index to shard across parallel Colab sessions & merge()
MAX_SEQ_LEN = 128            #@param {type:"integer"}  # Paper default: 128 tokens (keeps full K=2560 in 1 sweep!)
SKIP_FIRST = 16              #@param {type:"integer"}  # Exclude attention-sink prefix tokens (paper default: 16)
FWD_BATCH_SIZE = 8           #@param {type:"integer"}  # Zero-VRAM forward-only micro-batching (8x fewer forward weight sweeps!)
BOS_PREFIX = "none"          #@param ["none", "endoftext"]  # "none" = exact jlens/Neuronpedia parity; "endoftext" = 1x <|endoftext|> (248044)
LAYER_STRIDE = 1             #@param {type:"integer"}  # 1 = all 48 layers (0..47); 2 = every 2nd layer
INCLUDE_EMBEDDING_LAYER = True #@param {type:"boolean"} # Include layer -1 (embed_tokens -> H_{-1} before Layer 0; +2.1% time)
STOP_AT_DELTA = 0.0          #@param {type:"number"}   # 0.0 = fixed N_PROMPTS; set 0.002 for Neuronpedia early stopping (after 100 prompts)
PLE_MODE = "auto"            #@param ["auto", "mmap", "preextract"]

#@markdown ### During Fitting: Async Google Drive Checkpointing (Zero GPU Stall)
SAVE_TO_GOOGLE_DRIVE = True  #@param {type:"boolean"}         # Recommended: Drive-only during fitting (no mid-run HF bandwidth split!)
DRIVE_OUT_DIR = "/content/drive/MyDrive/qwen38_jlens_out" #@param {type:"string"}
CKPT_DTYPE = "float32"       #@param ["float32", "float16"]  # float32 = 4.69 GiB bit-exact resume (best for 5TB Drive); float16 = 2.34 GiB
CHECKPOINT_EVERY_PROMPTS = 15 #@param {type:"integer"}        # Trigger checkpoint every N prompts
CHECKPOINT_EVERY_MINUTES = 10.0 #@param {type:"number"}       # OR every M minutes (whichever comes first)
LOCAL_WORK_DIR = "/content/qwen38_work" #@param {type:"string"}

#@markdown ### After Fitting: One-Shot Neuronpedia-Ready Publish to Hugging Face Hub (Section 9)
HF_PUBLISH_REPO = ""         #@param {type:"string"}          # e.g. "your-username/qwen3.8-flash-next-jlens" (single clean commit at the end!)
HF_PUBLISH_PRIVATE = False   #@param {type:"boolean"}
AUTHOR_NAME = ""             #@param {type:"string"}          # Recorded in CREDIT.md, config.yaml, and .pt provenance
HF_TOKEN = ""                #@param {type:"string"}          # Or store HF_TOKEN in Colab Secrets

import os
os.makedirs(LOCAL_WORK_DIR, exist_ok=True)
SLAB_DIR = os.path.join(LOCAL_WORK_DIR, "moe_slabs")
PLE_DIR = os.path.join(LOCAL_WORK_DIR, "ple_shards")
LOCAL_CKPT_DIR = os.path.join(LOCAL_WORK_DIR, "checkpoints")
for _d in (SLAB_DIR, PLE_DIR, LOCAL_CKPT_DIR):
    os.makedirs(_d, exist_ok=True)

# Resolve optional HF_TOKEN from Colab Secrets if not pasted directly
if not HF_TOKEN:
    try:
        from google.colab import userdata
        HF_TOKEN = userdata.get("HF_TOKEN") or ""
    except Exception:
        HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_TOKEN = HF_TOKEN or None
HF_PUBLISH_REPO = HF_PUBLISH_REPO.strip() or None

DRIVE_ENABLED = False
if SAVE_TO_GOOGLE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        os.makedirs(DRIVE_OUT_DIR, exist_ok=True)
        DRIVE_ENABLED = True
    except Exception as exc:
        print(f"Google Drive mount unavailable ({exc}); saving checkpoints to local NVMe only.")

TAG = f"off{PROMPT_OFFSET}_n{N_PROMPTS}_s{LAYER_STRIDE}{'_emb' if INCLUDE_EMBEDDING_LAYER else ''}"
CKPT_FILENAME = f"qwen38_jlens_ckpt_{TAG}.safetensors"
CONV_FILENAME = f"qwen38_jlens_convergence_{TAG}.csv"

LOCAL_CKPT_PATH = os.path.join(LOCAL_CKPT_DIR, CKPT_FILENAME)
DRIVE_CKPT_PATH = os.path.join(DRIVE_OUT_DIR, CKPT_FILENAME) if DRIVE_ENABLED else None
LOCAL_CONV_CSV = os.path.join(LOCAL_CKPT_DIR, CONV_FILENAME)
DRIVE_CONV_CSV = os.path.join(DRIVE_OUT_DIR, CONV_FILENAME) if DRIVE_ENABLED else None
OUT_DIR = DRIVE_OUT_DIR if DRIVE_ENABLED else LOCAL_CKPT_DIR
BUNDLE_REL_DIR = "qwen3.8-flash-next/jlens/Salesforce-wikitext"
BUNDLE_DIR = os.path.join(OUT_DIR, f"release_{TAG}", BUNDLE_REL_DIR)

print("Persistence & Publishing Plan:")
print(f"  Mid-run Local NVMe    : {LOCAL_CKPT_PATH} (<1s atomic write)")
print(f"  Mid-run Google Drive  : {DRIVE_CKPT_PATH or 'disabled'} (async background copy, {CKPT_DTYPE})")
print(f"  Convergence CSV       : {DRIVE_CONV_CSV or LOCAL_CONV_CSV}")
print(f"  Checkpoint Cadence    : every {CHECKPOINT_EVERY_PROMPTS} prompts or {CHECKPOINT_EVERY_MINUTES} min")
print(f"  Final Release Bundle  : {BUNDLE_DIR}")
print(f"  One-Shot HF Publish   : {f'{HF_PUBLISH_REPO}/{BUNDLE_REL_DIR}' if HF_PUBLISH_REPO else 'not set (configure HF_PUBLISH_REPO for Section 9)'}")
"""
        )
    )

    # ------------------------------------------------------------------ #
    # Section 3: Direct Clone / Editable Install & Import from Public GitHub Repo
    # ------------------------------------------------------------------ #
    cells.append(
        new_markdown_cell(
            f"""## 3. Clone / Update & Import `qwen38_jlens`
Clones (or fast-forwards) the public repository [`xiayicheng3-code/Qwen3.8FlashNextJ-lens`]({REPO_URL[:-4]}) and installs `qwen38_jlens` in editable mode (`--no-deps`) for direct import."""
        )
    )
    cells.append(
        new_code_cell(
            f"""import importlib, subprocess, sys
from pathlib import Path

REPO_URL = "{REPO_URL}"
REPO_BRANCH = "main"
USE_LOCAL_CWD = (Path.cwd() / "qwen38_jlens" / "__init__.py").is_file()
LOCAL_REPO = Path.cwd() if USE_LOCAL_CWD else Path("/content/Qwen3.8FlashNextJ-lens")

if not USE_LOCAL_CWD:
    if (LOCAL_REPO / ".git").is_dir():
        subprocess.run(["git", "-C", str(LOCAL_REPO), "fetch", "--depth", "1", "origin", REPO_BRANCH], check=True)
        subprocess.run(["git", "-C", str(LOCAL_REPO), "reset", "--hard", f"origin/{{REPO_BRANCH}}"], check=True)
    else:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, str(LOCAL_REPO)],
            check=True,
        )

if str(LOCAL_REPO) not in sys.path:
    sys.path.insert(0, str(LOCAL_REPO))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(LOCAL_REPO)], check=True)

# If re-running after a git pull, clear cached qwen38_jlens modules so fresh code is imported cleanly
for _m in [k for k in list(sys.modules) if k == "qwen38_jlens" or k.startswith("qwen38_jlens.")]:
    del sys.modules[_m]
importlib.invalidate_caches()

import qwen38_jlens

rev = subprocess.check_output(
    ["git", "-C", str(LOCAL_REPO), "log", "-1", "--format=%h %s"], text=True
).strip() if (LOCAL_REPO / ".git").is_dir() else "local"
print(f"Loaded qwen38_jlens from {{Path(qwen38_jlens.__file__).parent}} ({{rev}})")
"""
        )
    )

    # ------------------------------------------------------------------ #
    # Cell 11: Live Hardware Probe, Concrete Offload Plan & Kernel Micro-Benchmark
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

SOURCE_LAYERS = ([-1] if INCLUDE_EMBEDDING_LAYER else []) + list(range(0, 48, LAYER_STRIDE))
if 47 not in SOURCE_LAYERS:
    SOURCE_LAYERS.append(47)
print(
    f"Source layers ({len(SOURCE_LAYERS)}): {SOURCE_LAYERS[0]}..{SOURCE_LAYERS[-1]} "
    f"(embedding layer -1: {'ON' if INCLUDE_EMBEDDING_LAYER else 'OFF'})"
)

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
print("\\nInterleaved Layer Placement Map (G=GPU HBM, P=Pinned RAM, R=Pageable RAM, D=NVMe Slab):")
for row_start in range(0, 48, 12):
    print("  " + " ".join(f"L{l:02d}:{tier_sym[plan.layer_tiers[l]]}" for l in range(row_start, row_start + 12)))

# ---- 3-Second Live Kernel Verification & Speed Micro-Benchmark ----
print("\\nRunning live GPU kernel parity & bandwidth benchmark...")
gdn_bench = GatedDeltaRuleBackend(preferred=plan.gdn_backend, verify_on_init=True)
print(f"  [OK] Active GDN kernel : {gdn_bench.name} (direct={gdn_bench._use_direct})")

dev = torch.device("cuda:0")
E_act = min(380, plan.expert_dequant_chunk)
w_fp8 = torch.randn(E_act, 1280, 2560, device=dev).to(torch.float8_e4m3fn)
s_fp32 = torch.rand(E_act, 10, 20, device=dev, dtype=torch.float32) * 0.1 + 0.01
w_bf16 = dequantize_fp8_blockwise(w_fp8, s_fp32, use_triton=plan.use_triton_fp8_dequant)
w_ref = dequantize_fp8_blockwise(w_fp8[:2], s_fp32[:2], use_triton=False)
deq_err = (w_bf16[:2].float() - w_ref.float()).abs().max().item()
assert deq_err == 0.0, f"FP8 Triton dequant mismatch: max_diff={deq_err}"
del w_ref
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    w_bf16 = dequantize_fp8_blockwise(w_fp8, s_fp32, use_triton=plan.use_triton_fp8_dequant)
torch.cuda.synchronize()
dt_deq = (time.perf_counter() - t0) / 5
bytes_rw = E_act * 1280 * 2560 * (1 + 2)
deq_mode = "native-fp8" if profile.compute_capability >= (8, 9) else "sm80-u8-lut"
print(f"  [OK] Triton FP8->BF16 dequant ({deq_mode}, {E_act} experts): {dt_deq*1e3:.2f} ms ({bytes_rw / dt_deq / 1e9:.0f} GB/s, max_diff={deq_err:.1e})")

# Verify Triton fused SwiGLU backward + pack
_dA = torch.randn(16, 64, 640, device=dev, dtype=torch.bfloat16)
_dg = torch.randn(16, 640, device=dev, dtype=torch.bfloat16)
_du = torch.randn(16, 640, device=dev, dtype=torch.bfloat16)
_sw_tri = swiglu_vjp_pack(_dA, _dg, _du, use_triton=plan.use_triton_swiglu_vjp)
_sw_ref = swiglu_vjp_pack(_dA, _dg, _du, use_triton=False)
sw_err = (_sw_tri.float() - _sw_ref.float()).abs().max().item()
assert sw_err <= 1e-2, f"Triton SwiGLU mismatch: max_diff={sw_err}"
print(f"  [OK] Triton SwiGLU VJP pack : verified (max_diff={sw_err:.1e})")
del _dA, _dg, _du, _sw_tri, _sw_ref

K_blk = plan.moe_k_chunk
S_rows = 128 * 10
counts = torch.full((E_act,), max(1, S_rows // E_act), device=dev, dtype=torch.int32)
offs_fwd = torch.cumsum(counts, dim=0, dtype=torch.int32)
_x_fwd = torch.randn(int(offs_fwd[-1].item()), 2560, device=dev, dtype=torch.bfloat16)
_ = grouped_mm(_x_fwd, w_bf16.transpose(-2, -1), offs_fwd)
del offs_fwd, _x_fwd

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
print(f"  [OK] MoE grouped_mm fwd(T-view)+VJP (M={M_tot}, K=1280, N=2560): {dt_gmm*1e3:.2f} ms ({tflops:.1f} TFLOPS)")

del w_fp8, s_fp32, w_bf16, dGU, out_gmm
torch.cuda.empty_cache()
"""
        )
    )

    # ------------------------------------------------------------------ #
    # Cell 12: Load Tokenizer & Fitting Corpus (WikiText-103)
    # ------------------------------------------------------------------ #
    cells.append(
        new_markdown_cell(
            """## 5. Load Tokenizer & Pretraining-Like Fitting Corpus (`Salesforce/wikitext`)
Matches `anthropics/jacobian-lens`: streams `Salesforce/wikitext` (`wikitext-103-raw-v1`, `min_chars=600`) and truncates to `MAX_SEQ_LEN=128` tokens (`BOS_PREFIX="none"` uses `tokenizer(..., add_special_tokens=True)` directly with no synthetic `<|im_end|>`, letting positions `0..SKIP_FIRST-1` absorb the attention sink before masking)."""
        )
    )
    cells.append(
        new_code_cell(
            """from datasets import load_dataset
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=False)
_EOT_ID = tokenizer.convert_tokens_to_ids("<|endoftext|>")
if _EOT_ID is None or _EOT_ID == tokenizer.unk_token_id:
    _EOT_ID = 248044  # Qwen3.8 text_config.eos_token_id (pretraining boundary & PLE n-gram cut)

def encode_prompt(text: str, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    if BOS_PREFIX == "none":
        # Exact anthropics/jacobian-lens (jlens/hf.py) behavior: no synthetic BOS prepended
        # when tokenizer.bos_token is null; positions 0..SKIP_FIRST-1 absorb the sink and are masked out.
        ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_len)["input_ids"]
    elif BOS_PREFIX == "endoftext":
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids or ids[0] != _EOT_ID:
            ids = [_EOT_ID] + ids
        ids = ids[:max_len]
    else:
        raise ValueError(f"Unsupported BOS_PREFIX={BOS_PREFIX!r}")
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

EVAL_PROMPTS = [
    "Fact: The currency used in the country shaped like a boot is",
    "The capital of the country where Mount Everest is located is",
    "     _______     \\n   /         \\\\   \\n  /  ~     ~  \\\\  \\n (   o     o   ) \\n |      ^      | \\n  \\\\   \\u2014\\u2014\\u2014   /  \\n   \\\\_________/   \\n\\nWhat is this?",
]
eval_tokenized = [encode_prompt(p, max_len=256) for p in EVAL_PROMPTS]

lengths = [int(t.shape[1]) for t in tokenized_prompts]
first_tok = tokenizer.decode([int(tokenized_prompts[0][0, 0])]) if tokenized_prompts else ""
print(f"Prepared {len(tokenized_prompts)} WikiText prompts (offset={PROMPT_OFFSET}, bos_prefix={BOS_PREFIX!r}, skip_first={SKIP_FIRST}): "
      f"min_len={min(lengths)}, median={sorted(lengths)[len(lengths)//2]}, max_len={max(lengths)}, tok0={first_tok!r}")
"""
        )
    )

    # ------------------------------------------------------------------ #
    # Cell 13: Streaming Shard-by-Shard 3-Tier Model Loader
    # ------------------------------------------------------------------ #
    cells.append(
        new_markdown_cell(
            """## 6. Stream-Load `Qwen/Qwen3.8-Flash-Next-FP8` Directly Into the 3-Tier Hierarchy
- Skips Vision encoder and MTP shards automatically.
- Loads all dense BF16 weights (`~8.0 GiB`) directly to GPU (immune to `transformers#48349`).
- Keeps 51B PLE n-gram table out of RAM (`MADV_RANDOM` NVMe mmap or tight-disk pre-extraction).
- Unpacks each raw safetensors shard straight into **Tier 0 GPU HBM / Tier 1 Pinned+Pageable RAM / Tier 2 contiguous NVMe slabs** and prints a live per-component VRAM breakdown (auto-demoting Tier-0 layers to pinned RAM if live free VRAM is ever below the runtime reserve)."""
        )
    )
    cells.append(
        new_code_cell(
            """import gc, shutil, time, psutil, torch
from qwen38_jlens import Qwen38JLensEngine, load_qwen38_streaming

# Release any previously loaded engine/model if re-running this cell in the same session
for _old_name in ("engine", "loaded_model"):
    if _old_name in globals():
        _obj = globals().pop(_old_name)
        if hasattr(_obj, "close"):
            _obj.close()
        del _obj
gc.collect()
torch.cuda.empty_cache()

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

vb = engine.vram_breakdown()
ram_used = psutil.virtual_memory().used / 1024**3
disk_free = shutil.disk_usage(LOCAL_WORK_DIR).free / 1024**3
print(f"\\nReady for J-Lens fitting in {time.perf_counter()-t_load:.1f}s!")
print(f"  Active GDN Kernel  : {engine.gdn_backend.name} (direct={engine.gdn_backend._use_direct})")
if engine.demoted_layers:
    print(f"  VRAM Self-Heal     : auto-demoted {len(engine.demoted_layers)} Tier-0 layers -> pinned RAM: {engine.demoted_layers}")
print(
    f"  Live VRAM Breakdown: dense={vb['dense_gib']:.2f}G + tier0_moe={vb['tier0_moe_gib']:.2f}G "
    f"+ staging={vb['staging_gib']:.2f}G | alloc={vb['allocated_gib']:.2f}G "
    f"| reserved={vb['reserved_gib']:.2f}G | free={vb['free_gib']:.2f}/{vb['total_gib']:.1f}G "
    f"(reserve={engine.plan.runtime_vram_reserve_gib:.1f}G)"
)
print(f"  Live Host Telemetry: RAM used={ram_used:.1f} GiB | Disk free={disk_free:.1f} GiB")
"""
        )
    )

    # ------------------------------------------------------------------ #
    # Cell 14: Run Single-Sweep Layer-by-Layer J-Lens Fitting w/ Async Checkpointer
    # ------------------------------------------------------------------ #
    cells.append(
        new_markdown_cell(
            """## 7. Fit Qwen3.8-Flash-Next Jacobian Lens (`J_l ∈ R^{2560 × 10240}`) with Async Drive Checkpointing
- Writes `J_mean` `.safetensors` to local NVMe in `< 1 s`, then copies asynchronously to **Google Drive only** (+ `convergence.csv`) with **0 s GPU stall** and zero HF Hub commit clutter.
- Interrupting the cell (`Stop` / `KeyboardInterrupt`) cleanly flushes local NVMe + Google Drive before exiting.
- On completion, exports a complete **Neuronpedia-ready release bundle** (`.pt` fp16 + 4-branch `.safetensors` + `convergence.csv` + `config.yaml` + `CREDIT.md`)."""
        )
    )
    cells.append(
        new_code_cell(
            """import gc, os, sys, time, torch
from tqdm.auto import tqdm
from qwen38_jlens import (
    AsyncCheckpointManager,
    CheckpointConfig,
    export_neuronpedia_bundle,
    peek_checkpoint_metadata,
)

# Clear any GPU tensors trapped in stale Jupyter tracebacks from earlier interrupted runs
sys.last_traceback = None
if hasattr(sys, "last_exc"):
    sys.last_exc = None
gc.collect()
torch.cuda.empty_cache()

ckpt_mgr = AsyncCheckpointManager(
    CheckpointConfig(
        local_path=LOCAL_CKPT_PATH,
        drive_path=DRIVE_CKPT_PATH,
        local_convergence_csv=LOCAL_CONV_CSV,
        drive_convergence_csv=DRIVE_CONV_CSV,
        hf_repo_id=None,  # Keep HF Hub clean during fitting; publish in ONE commit in Cell 9!
        ckpt_dtype=CKPT_DTYPE,
        every_n_prompts=CHECKPOINT_EVERY_PROMPTS,
        every_minutes=CHECKPOINT_EVERY_MINUTES,
        model_name=MODEL_ID,
    )
)

for lbl, p in (("Local NVMe", LOCAL_CKPT_PATH), ("Google Drive", DRIVE_CKPT_PATH)):
    meta = peek_checkpoint_metadata(p) if p else None
    if meta:
        print(f"Found existing {lbl} checkpoint: next_idx={meta['next_idx']}, n_done={meta['n_done']} ({meta['ckpt_dtype']})")

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
        vram=f"{vram_pk:.1f}G",
        ckpt=ckpt_mgr.last_status,
    )
    pbar.refresh()

torch.cuda.reset_peak_memory_stats()
t_fit = time.perf_counter()

try:
    lens = engine.fit(
        tokenized_prompts,
        source_layers=SOURCE_LAYERS,
        skip_first=SKIP_FIRST,
        fwd_batch_size=FWD_BATCH_SIZE,
        checkpoint_manager=ckpt_mgr,
        stop_at_delta=(STOP_AT_DELTA if STOP_AT_DELTA > 0 else None),
        min_prompts=100,
        stop_window=10,
        resume=True,
        on_prompt_done=_on_prompt,
    )
finally:
    pbar.close()
    ckpt_mgr.close()

# Export complete Neuronpedia-ready bundle locally / to Google Drive
bundle_files = export_neuronpedia_bundle(
    lens,
    BUNDLE_DIR,
    np_model_id="qwen3.8-flash-next",
    hf_model_name=MODEL_ID,
    file_prefix="Qwen3.8-Flash-Next-FP8",
    author=AUTHOR_NAME,
    dataset_name="Salesforce/wikitext",
    dataset_config="wikitext-103-raw-v1",
    dataset_split="train",
    max_seq_len=MAX_SEQ_LEN,
    skip_first=SKIP_FIRST,
    prompt_offset=PROMPT_OFFSET,
    stop_at_delta=(STOP_AT_DELTA if STOP_AT_DELTA > 0 else None),
    convergence_csv_path=LOCAL_CONV_CSV,
    gpu_name=profile.gpu_name,
    gpu_vram_gib=profile.vram_total_gib,
    gdn_kernel=engine.gdn_backend.name,
    include_4branch_safetensors=True,
)

total_min = (time.perf_counter() - t_fit) / 60.0
peak_vram = torch.cuda.max_memory_allocated() / 1024**3
print(f"\\nExported Neuronpedia-ready release bundle -> {BUNDLE_DIR}")
for k, p in bundle_files.items():
    sz_mb = os.path.getsize(p) / 1024**2
    print(f"  [{k:24s}] {os.path.basename(p)} ({sz_mb:.1f} MiB)")
print(f"  Summary   : {lens}")
print(f"  Total time: {total_min:.2f} min ({(total_min*60)/max(1, len(times)):.2f} s/prompt)")
print(f"  Peak VRAM : {peak_vram:.2f} GiB")
"""
        )
    )

    # ------------------------------------------------------------------ #
    # Cell 15: Verify Fitted J-Lens vs Logit-Lens + 4-Branch Diagnostics + Slice HTML
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

# 1. Inspect 4-branch Gated Residual Frobenius norms across depth (including Emb -1 if enabled)
bnorms = lens.branch_norms()
print("4-Branch Gated Residual Jacobian Norms ||J_l[:, b, :]||_F / sqrt(2560):")
for l in [-1, 0, 8, 16, 24, 32, 40, 47]:
    if l in bnorms:
        lbl = "Emb(-1)" if l == -1 else f"L{l:02d}    "
        b_str = "  ".join(f"b{b}={v:.3f}" for b, v in enumerate(bnorms[l]))
        print(f"  {lbl}: {b_str}")

# 2. Multi-hop J-lens vs Logit-lens readout at final token
test_prompt = EVAL_PROMPTS[0]
test_ids = eval_tokenized[0]
probe_layers = [l for l in (-1, 8, 16, 24, 32, 40, 46, 47) if l in lens.jacobians]

jl_logits, model_logits = engine.apply_lens(lens, test_ids, layers=probe_layers, positions=[-1], use_jacobian=True)
ll_logits, _ = engine.apply_lens(lens, test_ids, layers=probe_layers, positions=[-1], use_jacobian=False)

def top5(logits_1d):
    vals, idxs = torch.softmax(logits_1d.float(), dim=-1).topk(5)
    return [f"{tokenizer.decode([int(i)])!r}({float(v)*100:.0f}%)" for v, i in zip(vals, idxs)]

print(f"\\nPrompt: {test_prompt!r}")
for l in probe_layers:
    lbl = "Emb(-1)" if l == -1 else f"L{l:02d}    "
    print(f"  {lbl} Logit-Lens: {top5(ll_logits[l][0])}")
    print(f"  {lbl} J-Lens    : {top5(jl_logits[l][0])}")
print(f"  Model Output       : {top5(model_logits[0])}")

# 3. Interactive position x layer HTML slice heatmap (Emb -1 + every 4th layer for clean viewing)
slice_layers = [l for l in lens.source_layers if l == -1 or l % 4 == 0 or l == lens.source_layers[-1]]
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
    # Cell 16: Merge Parallel Shards & One-Shot Publish to Hugging Face / Neuronpedia
    # ------------------------------------------------------------------ #
    cells.append(
        new_markdown_cell(
            """## 9. One-Shot Publish to Hugging Face Hub (Neuronpedia-Ready) & Optional Shard Merge
- Uploads the **entire release folder in ONE single clean commit** (`_jacobian_lens.pt` fp16 + `_4branch.safetensors` + `_convergence.csv` + `config.yaml` + `CREDIT.md`) — zero rate-limit risk and clean Git history for contributing to [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens).
- If you ran parallel Colab shards (`PROMPT_OFFSET = 0, 200, 400, ...`), uncomment the merge block first to combine all Drive checkpoints into one exact `n_prompts`-weighted bundle before publishing."""
        )
    )
    cells.append(
        new_code_cell(
            """import glob, os
from qwen38_jlens import Qwen38JacobianLens, export_neuronpedia_bundle, publish_bundle_to_hf

PUBLISH_DIR = BUNDLE_DIR
PUBLISH_LENS = lens

# ---- (Optional) Merge multiple parallel Drive shards before publishing ----
# shard_ckpts = sorted(glob.glob(f"{OUT_DIR}/qwen38_jlens_ckpt_off*.safetensors"))
# print("Merging shards:", shard_ckpts)
# merged_lens = Qwen38JacobianLens.merge([Qwen38JacobianLens.load(p) for p in shard_ckpts])
# PUBLISH_LENS = merged_lens
# PUBLISH_DIR = os.path.join(OUT_DIR, f"release_merged_n{merged_lens.n_prompts}", BUNDLE_REL_DIR)
# export_neuronpedia_bundle(
#     merged_lens,
#     PUBLISH_DIR,
#     np_model_id="qwen3.8-flash-next",
#     hf_model_name=MODEL_ID,
#     file_prefix="Qwen3.8-Flash-Next-FP8",
#     author=AUTHOR_NAME,
#     max_seq_len=MAX_SEQ_LEN,
#     skip_first=SKIP_FIRST,
#     gpu_name=profile.gpu_name,
#     gpu_vram_gib=profile.vram_total_gib,
#     gdn_kernel=engine.gdn_backend.name,
# )
# print("Exported merged Neuronpedia bundle:", PUBLISH_DIR, merged_lens)

# ---- One-shot single-commit publish to Hugging Face Hub ----
if HF_PUBLISH_REPO:
    commit_url = publish_bundle_to_hf(
        PUBLISH_DIR,
        HF_PUBLISH_REPO,
        path_in_repo=BUNDLE_REL_DIR,
        token=HF_TOKEN,
        private=HF_PUBLISH_PRIVATE,
        commit_message=f"Publish Qwen3.8-Flash-Next Jacobian Lens (n={PUBLISH_LENS.n_prompts}, Neuronpedia format)",
    )
    print(f"Published in a single clean HF commit -> {commit_url}")
else:
    print(f"HF_PUBLISH_REPO is empty. Your Neuronpedia-ready bundle is saved locally/on Drive at:\\n  {PUBLISH_DIR}")
    print("Set HF_PUBLISH_REPO = 'your-username/qwen3.8-flash-next-jlens' in Cell 2 and re-run this cell to publish!")

# Optional cleanup of temporary local NVMe slabs / PLE shards:
# engine.close()
"""
        )
    )

    for i, cell in enumerate(cells):
        cell["id"] = f"q38-cell-{i:02d}"

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
