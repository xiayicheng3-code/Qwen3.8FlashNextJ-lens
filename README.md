# Qwen3.8-Flash-Next Jacobian Lens (J-Lens) — Ultra-Fast Google Colab Engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xiayicheng3-code/Qwen3.8FlashNextJ-lens/blob/main/qwen38_jlens_colab.ipynb)

High-throughput, memory-safe **Jacobian Lens (J-Lens)** fitting engine and **100% self-contained Google Colab notebook** ([`qwen38_jlens_colab.ipynb`](./qwen38_jlens_colab.ipynb)) for **`Qwen/Qwen3.8-Flash-Next-FP8`** (`qwen4_exp`: 125B MoE + 51B PLE n-gram table, 48 layers, 512 experts top-10, 4-branch Gated Residual, Gated DeltaNet + Qwen Sparse Attention).

Automatically detects whichever Google Colab High-RAM runtime is randomly assigned (**G4 RTX PRO 6000 Blackwell 96 GB**, **A100 80 GB**, or **A100 40 GB**) and builds an optimal interleaved **3-tier offloading plan** and **architecture-specific kernel stack** with zero OOM risk.

---

## Key Architectural & Systems Innovations

1. **Mathematically Faithful 4-Branch Gated Residual J-Lens (`J_l ∈ R^{2560 × 10240}`)**:
   - In `Qwen3.8-Flash-Next` (`qwen4_exp`), decoder layers operate on a 4-branch hyper-connection residual stream (`H_l ∈ R^{B × T × 10240}`), collapsed to `h_final ∈ R^{B × T × 2560}` by `hyper_connection_mixer` before `lm_head` (there is no `model.norm`).
   - Computing $J_\ell = \mathbb{E}[\partial h_{\text{final}} / \partial H_\ell] \in \mathbb{R}^{2560 \times 10240}$ via reverse-mode VJP requires the **exact same `K = 2560` output cotangents** as a single-stream model while capturing all 4 branches (plus free `J_collapsed = sum_b J_l[:, b, :]` and per-branch norms).

2. **Single-Sweep Layer-by-Layer Reverse Propagation (40×–120× Less PCIe/NVMe Traffic)**:
   - Naive `anthropics/jacobian-lens` (`dim_batch = 8`) runs 320 backward passes through all 48 layers per prompt -> `320 × 112.5 GiB = 36 TB` over PCIe/prompt (~24 min/prompt).
   - Our engine runs ONE `B = 1` forward pass saving only layer inputs (`~120 MB` total), then holds all **`K = 2560` cotangents** in a single in-place GPU buffer (`6.25 GiB` at `T = 128`) and finishes ALL 2560 VJPs of layer $\ell$ **before** moving to layer $\ell - 1$ -> **1× weight sweep per prompt**!

3. **Fused Active-Expert MoE VJP (`grouped_mm` + Custom Triton Kernels)**:
   - Dequantizes **only hit experts** (`E_hit ≤ 512`) **ONCE** per layer via a single-pass **Triton 128×128 FP8 block dequantizer** (~2.5 ms, zero temp allocations).
   - Reuses active BF16 weights across all 2560 cotangents using **CUTLASS/cuBLAS `grouped_mm`** with `n_e × K_blk` rows/expert (100% Tensor-Core saturated, every offset a multiple of 128).
   - Fuses SwiGLU backward + `[dG | dU]` packing in a custom **Triton kernel** and computes the exact top-k softmax router VJP without materializing `[T, K, 512]`.

4. **Hardware-Optimal Attention & Linear-Attention Kernels**:
   - **Gated DeltaNet (36/48 layers)**:
     - **Colab G4 (SM120 Blackwell) / H100 (SM90) / B200 (SM100)**: **`FlashQLA v0.1.3`** (TileLang warp-specialized fused fwd + SM120 bwd, 2–3× faster than FLA Triton).
     - **Colab A100 40G/80G (SM80 Ampere)**: **`flash-linear-attention` (`fla`)** Triton fused chunk fwd + bwd.
     - Verified at startup via a 1-ms numerical parity self-test with automatic fallback.
   - **Qwen Sparse Attention (12/48 layers)**:
     - **Exact QSA Indexer Bypass** for `T ≤ 2048` (mathematically bit-identical since top-512 blocks cover all 2048 tokens; eliminates `O(B·T)` Python loops and `torch.nonzero` GPU syncs).
     - **L2-Cached Exact `B = 1` Attention VJP** for `T ≤ 512`: caches `P ∈ R^{24 × 128 × 128}` (`768 KB`) in GPU L2 and executes pure cuBLAS batched GEMMs across all `K_sub` cotangents with zero Q/K/V replication.

5. **Concrete Interleaved 3-Tier Offloading & Zero-RAM 51B PLE Mmap**:
   - **51B PLE N-gram Table (`47.68 GiB` FP8)**: Never enters GPU VRAM or pinned RAM — served via `MADV_RANDOM` zero-copy mmap (`< 2 ms`/prompt), or automatically pre-extracted (`~64 MB`) if Colab disk is tight.
   - **Streaming Shard-by-Shard Pack-and-Evict Loader**: Skips Vision & MTP shards, avoids `transformers#48349` / `#48350`, packs each shard straight into Tier 0/1/2, and deletes consumed raw HF blobs so peak Colab disk stays ~65–112 GiB (well inside Colab's 235 GiB disk).

---

## Concrete Auto-Planned Profiles (`Qwen/Qwen3.8-Flash-Next-FP8`, `T = 128`)

| Colab Runtime | GPU Arch | Tier 0 GPU HBM (Permanent) | Tier 1 Host RAM (Async DMA) | Tier 2 NVMe Slabs (Interleaved) | 51B PLE | GDN Kernel | Est. Peak VRAM / RAM / Disk | Est. Time / Prompt |
|---|---|---|---|---|---|---|---|---|
| **G4 High RAM** (95.6 GiB / 177 GiB RAM) | SM120 | 8.2G dense + **26 MoE layers** (61.0G) | **22 MoE layers** (51.6G pinned) | **0 layers** | NVMe mmap | **FlashQLA SM120** | ~89.1G / ~61.5G / ~65.2G | **~7–10 s** |
| **A100 80G High RAM** (79.2 GiB / 167 GiB RAM) | SM80 | 8.2G dense + **19 MoE layers** (44.5G) | **29 MoE layers** (68.0G pinned) | **0 layers** | NVMe mmap | **FLA Triton SM80** | ~72.7G / ~77.9G / ~65.2G | **~18–22 s** |
| **A100 40G High RAM** (39.6 GiB / 83.5 GiB RAM) | SM80 | 8.2G dense + **5 MoE layers** (11.7G) | **23 MoE layers** (14 pin + 9 page) | **20 MoE layers** (46.9G, every ~2.4 layers) | NVMe mmap | **FLA Triton SM80** | ~35.4G / ~68.5G / ~112.1G | **~22–28 s** |

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

- [`qwen38_jlens_colab.ipynb`](./qwen38_jlens_colab.ipynb) — **100% self-contained Google Colab notebook** (click Open in Colab above or upload directly).
- [`qwen38_jlens/hardware.py`](./qwen38_jlens/hardware.py) — Live GPU/RAM/Disk probe + interleaved 3-tier offloading planner.
- [`qwen38_jlens/kernels.py`](./qwen38_jlens/kernels.py) — FlashQLA / FLA dispatcher, Triton 128×128 FP8 block dequant, Triton fused SwiGLU VJP, CUTLASS/cuBLAS `grouped_mm`, causal conv1d VJP.
- [`qwen38_jlens/ple_mmap.py`](./qwen38_jlens/ple_mmap.py) — Zero-RAM `MmapPleTable`, exact SplitMix64 prime-vocab n-gram hasher, and `PreextractedPleProvider`.
- [`qwen38_jlens/model_ops.py`](./qwen38_jlens/model_ops.py) — Exact `B = 1` forward + batched-cotangent analytical VJP for every `qwen4_exp` submodule.
- [`qwen38_jlens/loader.py`](./qwen38_jlens/loader.py) — Streaming safetensors shard packer (Vision/MTP skip, split/fused FP8 expert handling, contiguous binary Tier-2 slabs, raw blob eviction).
- [`qwen38_jlens/checkpoint.py`](./qwen38_jlens/checkpoint.py) — Zero-GPU-stall `AsyncCheckpointManager` (NVMe + async Drive + `ConvergenceLogger`) & single-commit Neuronpedia exporter/publisher (`export_neuronpedia_bundle`, `publish_bundle_to_hf`).
- [`qwen38_jlens/engine.py`](./qwen38_jlens/engine.py) — `AsyncMoEStreamer` (background NVMe thread + pinned host ring + CUDA copy stream + GPU staging ring) & `Qwen38JLensEngine` fitter/applier.
- [`qwen38_jlens/lens.py`](./qwen38_jlens/lens.py) — `Qwen38JacobianLens` (`save`, `load`, `merge`, `branch_norms`, `collapsed_jacobians`) & HTML slice renderer.
- [`build_notebook.py`](./build_notebook.py) — Rebuilds `qwen38_jlens_colab.ipynb` deterministically from `qwen38_jlens/`.
- [`tests/`](./tests/) — Pytest verification suite covering hardware plans, exact VJP parity against `torch.autograd.grad` across all operators and full 4-layer J-lens fitting, PLE mmap parity, Tier-2 NVMe slabs, checkpoint resume/early-stopping/Neuronpedia export, and notebook integrity.

## Running Tests Locally

```bash
pip install -e ".[dev]"
pytest -v tests/
```
