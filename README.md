# Qwen3.8-Flash-Next Jacobian Lens (J-Lens) — Ultra-Fast Google Colab Engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xiayicheng3-code/Qwen3.8FlashNextJ-lens/blob/main/qwen38_jlens_colab.ipynb)

High-throughput, memory-safe **Jacobian Lens (J-Lens)** fitting engine and clean **Google Colab notebook** ([`qwen38_jlens_colab.ipynb`](./qwen38_jlens_colab.ipynb), importing `qwen38_jlens` directly from this public repo) for **`Qwen/Qwen3.8-Flash-Next-FP8`** (`qwen4_exp`: 125B MoE + 51B PLE n-gram table, 48 layers, 512 experts top-10, 4-branch Gated Residual, Gated DeltaNet + Qwen Sparse Attention).

Automatically detects whichever Google Colab High-RAM runtime is randomly assigned (**G4 RTX PRO 6000 Blackwell 96 GB**, **A100 80 GB**, or **A100 40 GB**) and builds an optimal interleaved **3-tier offloading plan** and **architecture-specific kernel stack** with automatic post-load VRAM headroom self-healing.

---

## Key Architectural & Systems Innovations

1. **Mathematically Faithful 4-Branch Gated Residual J-Lens (`J_l ∈ R^{2560 × 10240}`, Layers `-1..47`)**:
   - In `Qwen3.8-Flash-Next` (`qwen4_exp`), decoder layers operate on a 4-branch hyper-connection residual stream (`H_l ∈ R^{B × T × 10240}`), collapsed to `h_final ∈ R^{B × T × 2560}` by `hyper_connection_mixer` before `lm_head` (there is no `model.norm`).
   - Computing $J_\ell = \mathbb{E}[\partial h_{\text{final}} / \partial H_\ell] \in \mathbb{R}^{2560 \times 10240}$ via reverse-mode VJP requires the **exact same `K = 2560` output cotangents** as a single-stream model while capturing all 4 branches (plus free `J_collapsed = sum_b J_l[:, b, :]` and per-branch norms).
   - Optionally includes **embedding layer `-1`** (`INCLUDE_EMBEDDING_LAYER = True` by default in the notebook, +2.1% runtime), whose 4-branch sum $J_{\text{collapsed}}[-1] = \sum_{b=0}^{3} J_{-1}[:, b, :]$ equals the exact token-embedding Jacobian $\mathbb{E}[\partial h_{\text{final}} / \partial \mathrm{emb}] \in \mathbb{R}^{2560 \times 2560}$ (stored as `J.embed` in `.safetensors` and `J[-1]` in `.pt`).

2. **Single-Sweep Reverse Propagation + Batched Forward MoE + 16-Aligned Causal Suffix Pruning (40×–120× Less PCIe/NVMe Traffic)**:
   - Naive `anthropics/jacobian-lens` (`dim_batch = 8`) runs 320 backward passes through all 48 layers per prompt -> `320 × 112.5 GiB = 36 TB` over PCIe/prompt (~24 min/prompt).
   - **Zero-VRAM Forward-Only Micro-Batching (`fwd_batch_size = 8`) + Batched Forward MoE**: Amortizes the weight-transfer-bound forward sweep across 8 prompts at once and concatenates all `B = 8` prompts' tokens into a **single** `moe_block_fwd` call per layer (8× fewer FP8 expert dequantizations & `grouped_mm` launches while supporting variable prompt lengths without padding; caches `~120 MB/prompt` in CPU RAM -> **`0 GiB` extra GPU VRAM**).
   - **16-Aligned Causal Suffix-Only Backward**: Exploits strict block lower-triangular causality to prune the masked attention-sink prefix (`[0 : bwd_start]`, `T_v = 112` at `T = 128, skip_first = 16`) from the entire backward sweep! Holds all **`K = 2560` cotangents** in ONE in-place GPU buffer (`5.47 GiB` vs `6.25 GiB`), finishing ALL 2560 VJPs of layer $\ell$ **before** moving to layer $\ell - 1$ -> **1× backward weight sweep per prompt**!

3. **9 Custom Fused Triton VJP Kernels + `grouped_mm` Active-Expert MoE**:
   - **Zero-FP32-Temp Hyper-Connection & RMSNorm VJP**: Single-read 4-branch injection VJP (`_hc_inject_vjp_hc4_kernel`), fused Hyper-Connection prep (`_hc_bwd_prep_kernel`), in-place `addmm_`, and fused `RMSNorm` VJP (`_rmsnorm_vjp_kernel`) accumulating straight into `G` with `tl.int64` pointer arithmetic (for 2.94B-element `G`).
   - **Fused Active-Expert MoE VJP**: Dequantizes **only hit experts** (`E_hit ≤ 512`, chunked at `256` on A100 and `512` on G4) **ONCE** per layer via a custom **Triton 128×128 FP8 block dequantizer** (native FP8 PTX on SM89+ and L1-cached uint8+LUT on SM80 A100), fuses token gather+scale (`_moe_gather_scale_kernel`), runs CUTLASS/cuBLAS `grouped_mm` (211 TFLOPS on A100), fuses 2D-tiled SwiGLU VJP (`_swiglu_vjp_pack_kernel`), and uses a **zero-atomic deterministic FP32-accum Triton scatter-add** (`_moe_scatter_add_tk_kernel`) replacing slow BF16 `index_add_` atomics.
   - **Fused GDN Peripheral VJP**: Single-pass register-resident `RMSNormGated` VJP (`_rmsnorm_gated_vjp_kernel`), `L2Norm` VJP (`_l2norm_bwd_kernel`), and 4-tap causal `Conv1D + SiLU` VJP directly in `[K, T, C]` (`_causal_conv1d_silu_bwd_slice_kernel` — zero transposes, zero `conv_transpose1d`).
   - All 9 Triton kernels + WY GDN are verified live in `< 1 s` against `use_triton=False` PyTorch references via `verify_cuda_triton_kernels()`.

4. **L2-Cached Exact WY Gated DeltaNet & Exact QSA Attention VJP**:
   - **Gated DeltaNet (36/48 layers)**:
     - **`T ≤ 256` (default J-lens `T = 128`)**: **L2-Cached Exact WY (`l2_cached_wy`)** — precomputes the tiny L2-resident `[48, 128, 128]` WY matrices (`W_eff = A @ L^{-1}`, `U = L^{-1} @ (beta * v)`) ONCE in `B = 1` forward and runs ALL `K = 2560` cotangents directly on suffix `[bwd_start:]` via **6 pure Tensor-Core-saturated cuBLAS `torch.bmm`s** (~8× faster than FLA on A100 SM80 with **0 s Triton autotune compile stall**!).
     - **`T > 256`**: Auto-dispatches to **`FlashQLA v0.1.3`** (SM90/100/120 Blackwell/Hopper) > **`FLA` Triton** (SM80 A100) > exact chunked PyTorch.
   - **Qwen Sparse Attention (12/48 layers)**:
     - **Exact QSA Indexer Bypass** for `T ≤ 2048` + **L2-Cached Exact `B = 1` Attention VJP** for `T ≤ 512` (caches `P ∈ R^{24 × 128 × 128}` in GPU L2 and executes pure cuBLAS batched GEMMs).

5. **Concrete Interleaved 3-Tier Offloading, Zero-RAM 51B PLE Mmap & Auto VRAM Self-Heal**:
   - **51B PLE N-gram Table (`47.68 GiB` FP8)**: Never enters GPU VRAM or pinned RAM — served via `MADV_RANDOM` zero-copy mmap (`< 2 ms`/prompt), or automatically pre-extracted (`~64 MB`) if Colab disk is tight.
   - **Streaming Shard-by-Shard Pack-and-Evict Loader**: Skips Vision & MTP shards, avoids `transformers#48349` / `#48350`, packs each shard straight into Tier 0/1/2, and deletes consumed raw HF blobs when disk is tight.
   - **Self-Healing VRAM Guard (`ensure_vram_headroom`)**: After loading, checks live `torch.cuda.mem_get_info()` against `plan.runtime_vram_reserve_gib` and automatically demotes evenly spaced Tier-0 MoE layers to pinned RAM if live VRAM headroom is ever below the safety reserve.

---

## Concrete Auto-Planned Profiles (`Qwen/Qwen3.8-Flash-Next-FP8`, `T = 128`)

| Colab Runtime | GPU Arch | Tier 0 GPU HBM (Permanent) | Tier 1 Host RAM (Async DMA) | Tier 2 NVMe Slabs (Interleaved) | Dequant Chunk | GDN Kernel (`T ≤ 256`) | Est. Peak VRAM / RAM / Disk | Est. Time / Prompt |
|---|---|---|---|---|---|---|---|---|
| **G4 High RAM** (95.6 GiB / 177 GiB RAM) | SM120 | 8.0G dense + **25 MoE layers** (58.6G) | **23 MoE layers** (53.9G pinned) | **0 layers** | 512 | **L2-WY cuBLAS** (`>256`: FlashQLA) | ~87.9G / ~63.8G / ~65.2G | **~10–14 s** |
| **A100 80G High RAM** (79.2 GiB / 167 GiB RAM) | SM80 | 8.0G dense + **19 MoE layers** (44.5G) | **29 MoE layers** (68.0G pinned) | **0 layers** | 256 | **L2-WY cuBLAS** (`>256`: FLA) | ~71.5G / ~77.9G / ~65.2G | **~24–30 s** |
| **A100 40G High RAM** (39.6 GiB / 83.5 GiB RAM) | SM80 | 8.0G dense + **3 MoE layers** (7.0G) | **23 MoE layers** (14 pin + 9 page) | **22 MoE layers** (51.6G, every ~2.2 layers) | 256 | **L2-WY cuBLAS** (`>256`: FLA) | ~32.8G / ~68.5G / ~116.8G | **~28–36 s** |

---

## Crash-Proof Zero-Stall Checkpointing & Neuronpedia Publishing

Naively calling `torch.save(jacobian_sum, "/content/drive/...")` blocks the GPU for **80–180 s** over Google Drive FUSE, while double-uploading mid-run checkpoints to both Drive and Hugging Face splits Colab outbound NIC bandwidth and bloats Git commit history.

Instead, [`qwen38_jlens/checkpoint.py`](./qwen38_jlens/checkpoint.py) uses a clean **split-role architecture**:
1. **During fitting → Local NVMe (`< 1 s`) + Async Google Drive ONLY** (`AsyncCheckpointManager`):
   - Saves running mean `J_mean = jacobian_sum / n_done` in a single `.safetensors` file (`float32` = `4.69 GiB` bit-exact resume by default for Google AI Pro 5 TB Drive; `float16` = `2.34 GiB` for free 15 GB Drive) with `n_done`, `next_idx`, `source_layers`, and `branch_norms` in the header.
   - Streams layer-by-layer division (avoiding a 4.69 GiB transient CPU RAM spike on A100 40G), hardlinks locally, and copies asynchronously to Google Drive (+ `convergence.csv`) on a background thread with **0 s GPU stall** and zero HF commit-rate-limit risk.
   - Supports time + count triggers, optional Neuronpedia convergence early stopping (`stop_at_delta=0.002`), clean `KeyboardInterrupt` flush, and direct loading of any interrupted checkpoint via `Qwen38JacobianLens.load(ckpt_path)`.
2. **After fitting (or shard merge) → One-Shot Neuronpedia-Ready Publish to Hugging Face Hub** (`export_neuronpedia_bundle` + `publish_bundle_to_hf`):
   - Exports a clean release folder (`_jacobian_lens.pt` fp16 drop-in `[2560, 2560]` collapsed map + `provenance`, `_jacobian_lens_4branch.safetensors` full lossless `[2560, 10240]`, `_convergence.csv`, `config.yaml`, `CREDIT.md`) and uploads it in **one single clean commit**.

---

## Repository Layout

- [`qwen38_jlens_colab.ipynb`](./qwen38_jlens_colab.ipynb) — Clean, readable **Google Colab notebook** (9 code cells; clones & imports `qwen38_jlens` directly from GitHub).
- [`qwen38_jlens/hardware.py`](./qwen38_jlens/hardware.py) — Live GPU/RAM/Disk probe + interleaved 3-tier offloading planner + runtime VRAM reserve.
- [`qwen38_jlens/kernels.py`](./qwen38_jlens/kernels.py) — L2-cached exact WY GDN (`T ≤ 256`) + FlashQLA/FLA fallback, 9 custom fused Triton VJP & FP8 dequant kernels, `verify_cuda_triton_kernels()`, and CUTLASS/cuBLAS `grouped_mm`.
- [`qwen38_jlens/ple_mmap.py`](./qwen38_jlens/ple_mmap.py) — Zero-RAM `MmapPleTable`, exact SplitMix64 prime-vocab n-gram hasher, and `PreextractedPleProvider`.
- [`qwen38_jlens/model_ops.py`](./qwen38_jlens/model_ops.py) — Exact `B = 1` forward + low-VRAM batched-cotangent analytical VJP for every `qwen4_exp` submodule.
- [`qwen38_jlens/loader.py`](./qwen38_jlens/loader.py) — Streaming safetensors shard packer (Vision/MTP skip, split/fused FP8 expert handling, contiguous binary Tier-2 slabs, raw blob eviction).
- [`qwen38_jlens/checkpoint.py`](./qwen38_jlens/checkpoint.py) — Zero-GPU-stall `AsyncCheckpointManager` (NVMe + async Drive + `ConvergenceLogger`) & single-commit Neuronpedia exporter/publisher (`export_neuronpedia_bundle`, `publish_bundle_to_hf`).
- [`qwen38_jlens/engine.py`](./qwen38_jlens/engine.py) — `AsyncMoEStreamer`, live VRAM telemetry (`model_vram_breakdown`) + self-healing headroom guard (`ensure_vram_headroom`), and `Qwen38JLensEngine` fitter/applier (with batched forward MoE).
- [`qwen38_jlens/lens.py`](./qwen38_jlens/lens.py) — `Qwen38JacobianLens` (`save`, `load`, `merge`, `branch_norms`, `collapsed_jacobians`) & HTML slice renderer.
- [`build_notebook.py`](./build_notebook.py) — Deterministically rebuilds `qwen38_jlens_colab.ipynb`.
- [`tests/`](./tests/) — Pytest verification suite covering hardware plans, VRAM self-heal selection, exact VJP parity against `torch.autograd.grad` across all operators and full 4-layer J-lens fitting, PLE mmap parity, Tier-2 NVMe slabs, checkpoint resume/early-stopping/Neuronpedia export, and notebook integrity.

## Running Tests Locally

```bash
pip install -e ".[dev]"
pytest -v tests/
```
