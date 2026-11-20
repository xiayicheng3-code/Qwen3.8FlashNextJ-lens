"""Crash-proof, zero-GPU-stall checkpointing & Neuronpedia-ready HF Hub publishing for Qwen3.8-Flash-Next J-Lens.

Recommended Split-Role Architecture on Google Colab:
  1. During fitting (mutable checkpoints): Local NVMe (<1 s atomic write) -> Async Google Drive ONLY.
     - Never stalls the GPU waiting on Google Drive FUSE (~80-160 s background copy).
     - Avoids splitting Colab outbound NIC bandwidth across two remote uploads and keeps your
       Hugging Face Hub Git commit history completely clean (zero risk of HF commit rate limits).
     - Supports bit-exact `float32` (4.69 GiB, recommended on Google AI Pro 5 TB Drive) or compact
       `float16` (2.34 GiB, for free 15 GB Drive) stored as running mean `J_mean = jacobian_sum / n_done`
       in a single `.safetensors` file — immediately loadable via `Qwen38JacobianLens.load()`.
     - Mirrors `convergence.csv` alongside each checkpoint so restarts never break convergence logs.
  2. After fitting / merging (immutable release): One-shot `publish_bundle_to_hf(...)` uploads the
     Neuronpedia-ready bundle (`_jacobian_lens.pt` fp16 + `_4branch.safetensors` + `_convergence.csv`
     + `config.yaml` + `CREDIT.md`) to Hugging Face Hub in ONE single clean commit.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import logging
import math
import os
import platform
import shutil
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_ckpt_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    key = dtype.strip().lower()
    if key not in _DTYPE_MAP:
        raise ValueError(f"Unsupported checkpoint dtype {dtype!r}; choose from {sorted(_DTYPE_MAP)}")
    return _DTYPE_MAP[key]


def peek_checkpoint_metadata(path: str) -> dict[str, Any] | None:
    """Read checkpoint metadata from a `.safetensors` header in <1 ms (without loading 2-5 GiB weights).

    Also supports legacy `.pt` checkpoints transparently.
    """
    if not path or not os.path.isfile(path):
        return None
    if path.endswith(".safetensors"):
        try:
            with safe_open(path, framework="pt", device="cpu") as sf:
                raw_meta = sf.metadata() or {}
            if "n_done" not in raw_meta:
                return None
            return {
                "n_done": int(raw_meta["n_done"]),
                "next_idx": int(raw_meta.get("next_idx", raw_meta["n_done"])),
                "source_layers": json.loads(raw_meta["source_layers"]),
                "skip_first": int(raw_meta["skip_first"]),
                "d_model": int(raw_meta["d_model"]),
                "hc_count": int(raw_meta["hc_count"]),
                "model_name": raw_meta.get("model_name", "Qwen/Qwen3.8-Flash-Next-FP8"),
                "ckpt_dtype": raw_meta.get("ckpt_dtype", "float16"),
                "format": "safetensors",
            }
        except Exception as exc:
            logger.warning("Failed to peek safetensors checkpoint %s: %s", path, exc)
            return None

    # Legacy `.pt` fallback
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "n_done" not in state:
            return None
        return {
            "n_done": int(state["n_done"]),
            "next_idx": int(state.get("next_idx", state["n_done"])),
            "source_layers": list(state["source_layers"]),
            "skip_first": int(state["skip_first"]),
            "d_model": int(state["d_model"]),
            "hc_count": int(state.get("hc_count", 4)),
            "model_name": state.get("model_name", "Qwen/Qwen3.8-Flash-Next-FP8"),
            "ckpt_dtype": "float32",
            "format": "pt",
        }
    except Exception as exc:
        logger.warning("Failed to peek legacy pt checkpoint %s: %s", path, exc)
        return None


def _layer_tensor_key(prefix: str, layer: int) -> str:
    return f"{prefix}.embed" if int(layer) == -1 else f"{prefix}.layer_{int(layer):02d}"


def save_jlens_safetensors(
    path: str,
    jacobian_mean: dict[int, torch.Tensor],
    *,
    n_done: int,
    next_idx: int,
    skip_first: int,
    d_model: int,
    hc_count: int,
    model_name: str = "Qwen/Qwen3.8-Flash-Next-FP8",
    dtype: torch.dtype = torch.float16,
    include_collapsed: bool = False,
    divisor: float = 1.0,
) -> str:
    """Atomically write a single-file `.safetensors` J-lens / resumable checkpoint.

    Pass `divisor=float(n_done)` when supplying `jacobian_sum` to divide one layer at a time
    (avoiding a 4.69 GiB transient fp32 host RAM spike on 83.5 GiB A100 40G runtimes).
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    source_layers = sorted(int(l) for l in jacobian_mean.keys())
    tensors: dict[str, torch.Tensor] = {}
    branch_norms: dict[str, list[float]] = {}
    scale = d_model ** 0.5
    inv_div = 1.0 / float(divisor) if divisor != 1.0 else 1.0

    for l in source_layers:
        raw = jacobian_mean[l].detach().cpu()
        J_f = (raw.float() * inv_div) if divisor != 1.0 else raw.float()
        J4 = J_f.view(d_model, hc_count, d_model)
        branch_norms[str(l)] = [round(float(J4[:, b, :].norm().item() / scale), 5) for b in range(hc_count)]
        tensors[_layer_tensor_key("J", l)] = J_f.to(dtype=dtype).contiguous()
        if include_collapsed:
            tensors[_layer_tensor_key("J_collapsed", l)] = J4.sum(dim=1).to(dtype=dtype).contiguous()
        del J_f, J4

    metadata = {
        "format_version": "qwen38_jlens_v1",
        "model_name": str(model_name),
        "n_done": str(int(n_done)),
        "next_idx": str(int(next_idx)),
        "source_layers": json.dumps(source_layers),
        "skip_first": str(int(skip_first)),
        "d_model": str(int(d_model)),
        "hc_count": str(int(hc_count)),
        "d_source": str(int(d_model * hc_count)),
        "ckpt_dtype": str(dtype).replace("torch.", ""),
        "branch_norms": json.dumps(branch_norms),
    }

    tmp_path = f"{path}.tmp.{os.getpid()}"
    save_file(tensors, tmp_path, metadata=metadata)
    os.replace(tmp_path, path)
    return path


def load_jlens_safetensors(path: str) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Load `jacobian_mean` (`{layer: fp32 CPU Tensor}`) and metadata from a `.safetensors` file."""
    with safe_open(path, framework="pt", device="cpu") as sf:
        raw_meta = sf.metadata() or {}
        avail_keys = set(sf.keys())
        source_layers = json.loads(raw_meta["source_layers"])
        jacobian_mean: dict[int, torch.Tensor] = {}
        for l in source_layers:
            li = int(l)
            key = _layer_tensor_key("J", li)
            if key not in avail_keys:
                key = f"J.layer_{li:02d}"
            jacobian_mean[li] = sf.get_tensor(key).float()

    meta = {
        "n_done": int(raw_meta["n_done"]),
        "next_idx": int(raw_meta.get("next_idx", raw_meta["n_done"])),
        "source_layers": [int(x) for x in source_layers],
        "skip_first": int(raw_meta.get("skip_first", 16)),
        "d_model": int(raw_meta["d_model"]),
        "hc_count": int(raw_meta.get("hc_count", 4)),
        "model_name": raw_meta.get("model_name", "Qwen/Qwen3.8-Flash-Next-FP8"),
        "ckpt_dtype": raw_meta.get("ckpt_dtype", "float16"),
    }
    return jacobian_mean, meta


class ConvergenceLogger:
    """Append-safe, resume-deduped CSV logger matching Neuronpedia's `*_convergence.csv`."""

    FIELDNAMES = (
        "prompt_idx",
        "n_done",
        "seq_len",
        "n_valid",
        "max_norm",
        "mean_rel_change",
        "elapsed_s",
        "fwd_s",
        "bwd_s",
    )

    def __init__(self, path: str, *, resume_next_idx: int = 0) -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        kept_rows: list[dict[str, str]] = []
        if resume_next_idx > 0 and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if int(row["prompt_idx"]) < resume_next_idx:
                            kept_rows.append(row)
            except Exception as exc:
                logger.warning("Could not parse existing convergence CSV %s (%s); starting fresh.", path, exc)
                kept_rows = []

        # Rewrite cleanly up to `resume_next_idx` so crash tails after last checkpoint are never duplicated
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            for row in kept_rows:
                writer.writerow({k: row.get(k, "") for k in self.FIELDNAMES})
        os.replace(tmp, path)

        self._recent_deltas: list[float] = []
        for row in kept_rows:
            val_str = row.get("mean_rel_change", "")
            if val_str:
                try:
                    v = float(val_str)
                    if math.isfinite(v):
                        self._recent_deltas.append(v)
                except ValueError:
                    pass

    def append(self, n_done: int, stats: Any) -> None:
        rel = float(stats.mean_rel_change)
        rel_str = f"{rel:.8e}" if math.isfinite(rel) else ""
        if math.isfinite(rel):
            self._recent_deltas.append(rel)
        with open(self.path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(
                {
                    "prompt_idx": int(stats.prompt_idx),
                    "n_done": int(n_done),
                    "seq_len": int(stats.seq_len),
                    "n_valid": int(stats.n_valid),
                    "max_norm": f"{float(stats.max_norm):.6f}",
                    "mean_rel_change": rel_str,
                    "elapsed_s": f"{float(stats.elapsed_s):.3f}",
                    "fwd_s": f"{float(stats.fwd_s):.3f}",
                    "bwd_s": f"{float(stats.bwd_s):.3f}",
                }
            )
            f.flush()

    def tail_deltas(self, window: int) -> list[float]:
        if window <= 0:
            return []
        return self._recent_deltas[-window:]

    def last_summary(self) -> tuple[float | None, float | None]:
        """Return `(final_mean_rel_change, final_max_norm)` from the CSV if available."""
        if not os.path.isfile(self.path):
            return None, None
        last_rel: float | None = None
        last_norm: float | None = None
        with open(self.path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("mean_rel_change"):
                    try:
                        v = float(row["mean_rel_change"])
                        if math.isfinite(v):
                            last_rel = v
                    except ValueError:
                        pass
                if row.get("max_norm"):
                    try:
                        v = float(row["max_norm"])
                        if math.isfinite(v):
                            last_norm = v
                    except ValueError:
                        pass
        return last_rel, last_norm


@dataclass
class CheckpointConfig:
    """Asynchronous checkpointing policy (recommended: Local NVMe + Google Drive ONLY during fitting).

    Prefer setting ONLY ONE remote destination (`drive_path` recommended on Google AI Pro 5 TB Drive;
    reserve Hugging Face Hub for one-shot single-commit publishing via `publish_bundle_to_hf`).
    """

    local_path: str                               # Fast local NVMe `.safetensors` path (~0.8s write!)
    drive_path: str | None = None                 # Recommended remote mirror: Google Drive (async background copy)
    local_convergence_csv: str | None = None      # Optional local per-prompt convergence CSV
    drive_convergence_csv: str | None = None      # Optional Drive mirror of convergence CSV
    hf_repo_id: str | None = None                 # Optional (discouraged alongside Drive due to shared NIC & Git history bloat)
    hf_path_in_repo: str | None = None
    hf_token: str | None = None
    hf_private: bool = True
    ckpt_dtype: str = "float32"                   # "float32" (4.69 GiB bit-exact resume) or "float16" (2.34 GiB compact)
    every_n_prompts: int = 15                     # Trigger checkpoint every N prompts
    every_minutes: float = 10.0                   # OR every M minutes (whichever comes first!)
    max_sync_retries: int = 3
    model_name: str = "Qwen/Qwen3.8-Flash-Next-FP8"


class AsyncCheckpointManager:
    """Non-blocking checkpoint manager: fast local NVMe write + background remote sync (Drive recommended).

    Guarantees:
      - Zero GPU stalls waiting on Google Drive FUSE (~80-160 s background copy).
      - Never piles up queued background uploads (if a previous remote sync is still running,
        local NVMe is still updated and the next completed window triggers the remote sync;
        `flush()` waits cleanly on completion or `KeyboardInterrupt`).
      - Multi-source resume automatically chooses the freshest checkpoint across Local NVMe,
        Google Drive (and HF Hub if configured), and restores `convergence.csv` transparently!
    """

    def __init__(self, cfg: CheckpointConfig) -> None:
        self.cfg = cfg
        self.dtype = resolve_ckpt_dtype(cfg.ckpt_dtype)
        if cfg.drive_path and cfg.hf_repo_id:
            logger.warning(
                "Both drive_path and hf_repo_id are enabled for intermediate checkpoints. "
                "They share outbound Colab NIC bandwidth and create extra HF Git commits; "
                "prefer Drive-only during fitting and publish_bundle_to_hf() once at the end."
            )
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jlens-ckpt")
        self._remote_future: Future | None = None
        self._last_ckpt_time = time.monotonic()
        self._last_local_next_idx = -1
        self._last_synced_next_idx = -1
        self._hf_repo_ready = False
        self.last_status: str = "idle"

    @property
    def has_remote(self) -> bool:
        return bool(self.cfg.drive_path or self.cfg.hf_repo_id)

    def should_checkpoint(self, prompts_since_last: int, *, force: bool = False) -> bool:
        if force:
            return True
        if self.cfg.every_n_prompts > 0 and prompts_since_last >= self.cfg.every_n_prompts:
            return True
        if self.cfg.every_minutes > 0 and (time.monotonic() - self._last_ckpt_time) >= self.cfg.every_minutes * 60.0:
            return True
        return False

    def _copy_to_drive(self, src_path: str, dst_path: str) -> None:
        """Atomic copy to Google Drive FUSE with retry for transient FUSE errors."""
        parent = os.path.dirname(os.path.abspath(dst_path))
        os.makedirs(parent, exist_ok=True)
        tmp_dst = f"{dst_path}.uploading"
        last_exc: Exception | None = None
        for attempt in range(1, self.cfg.max_sync_retries + 1):
            try:
                shutil.copyfile(src_path, tmp_dst)
                os.replace(tmp_dst, dst_path)
                return
            except Exception as exc:
                last_exc = exc
                logger.warning("Drive checkpoint sync attempt %d/%d failed: %s", attempt, self.cfg.max_sync_retries, exc)
                time.sleep(min(5.0, attempt * 2.0))
        if last_exc is not None:
            raise last_exc

    def _upload_to_hf(self, src_path: str, next_idx: int, n_done: int) -> None:
        """Optional mid-run HF Hub upload (prefer Drive-only + single-commit publish_bundle_to_hf)."""
        assert self.cfg.hf_repo_id is not None
        from huggingface_hub import HfApi

        api = HfApi(token=self.cfg.hf_token)
        if not self._hf_repo_ready:
            api.create_repo(
                repo_id=self.cfg.hf_repo_id,
                repo_type="model",
                private=self.cfg.hf_private,
                exist_ok=True,
            )
            self._hf_repo_ready = True

        path_in_repo = self.cfg.hf_path_in_repo or os.path.basename(self.cfg.local_path)
        last_exc: Exception | None = None
        for attempt in range(1, self.cfg.max_sync_retries + 1):
            try:
                api.upload_file(
                    path_or_fileobj=src_path,
                    path_in_repo=path_in_repo,
                    repo_id=self.cfg.hf_repo_id,
                    repo_type="model",
                    commit_message=f"Checkpoint J-lens (n_done={n_done}, next_idx={next_idx})",
                )
                return
            except Exception as exc:
                last_exc = exc
                logger.warning("HF Hub checkpoint upload attempt %d/%d failed: %s", attempt, self.cfg.max_sync_retries, exc)
                time.sleep(min(8.0, attempt * 2.5))
        if last_exc is not None:
            raise last_exc

    def _sync_remote_task(self, staged_path: str, next_idx: int, n_done: int) -> None:
        t0 = time.perf_counter()
        targets: list[str] = []
        try:
            if self.cfg.drive_path:
                self._copy_to_drive(staged_path, self.cfg.drive_path)
                if (
                    self.cfg.drive_convergence_csv
                    and self.cfg.local_convergence_csv
                    and os.path.isfile(self.cfg.local_convergence_csv)
                ):
                    self._copy_to_drive(self.cfg.local_convergence_csv, self.cfg.drive_convergence_csv)
                targets.append("Drive")
            if self.cfg.hf_repo_id:
                self._upload_to_hf(staged_path, next_idx, n_done)
                targets.append(f"HF:{self.cfg.hf_repo_id}")
            self._last_synced_next_idx = next_idx
            dt = time.perf_counter() - t0
            self.last_status = f"synced@{next_idx} ({'+'.join(targets)} in {dt:.1f}s)"
            logger.info("Async remote checkpoint synced: %s", self.last_status)
        except Exception as exc:
            self.last_status = f"remote_err@{next_idx}: {exc}"
            logger.error("Async remote checkpoint sync failed (local NVMe copy is still safe!): %s", exc)
        finally:
            if staged_path != self.cfg.local_path and os.path.exists(staged_path):
                try:
                    os.unlink(staged_path)
                except OSError:
                    pass

    def save_async(
        self,
        jacobian_sum: dict[int, torch.Tensor],
        *,
        n_done: int,
        next_idx: int,
        skip_first: int,
        d_model: int,
        hc_count: int,
        wait_remote: bool = False,
    ) -> None:
        """Write local NVMe `.safetensors` immediately (<1 s), then push to Drive in background."""
        if n_done <= 0:
            return

        if self._last_local_next_idx != next_idx or not os.path.isfile(self.cfg.local_path):
            t0 = time.perf_counter()
            save_jlens_safetensors(
                self.cfg.local_path,
                jacobian_sum,
                n_done=n_done,
                next_idx=next_idx,
                skip_first=skip_first,
                d_model=d_model,
                hc_count=hc_count,
                model_name=self.cfg.model_name,
                dtype=self.dtype,
                include_collapsed=False,
                divisor=float(n_done),
            )
            self._last_local_next_idx = next_idx
            self._last_ckpt_time = time.monotonic()
            dt_local = time.perf_counter() - t0
            self.last_status = f"local@{next_idx} ({dt_local:.1f}s)"
            logger.info("Local NVMe checkpoint saved: %s (n_done=%d, next_idx=%d, %.2fs)", self.cfg.local_path, n_done, next_idx, dt_local)

        if not self.has_remote:
            return

        if self._remote_future is not None and not self._remote_future.done():
            if not wait_remote:
                logger.info("Previous remote sync still in flight; keeping local NVMe checkpoint fresh without stalling GPU.")
                return
            self._remote_future.result()

        staged_path = f"{self.cfg.local_path}.sync.{next_idx}"
        if os.path.exists(staged_path):
            os.unlink(staged_path)
        try:
            os.link(self.cfg.local_path, staged_path)
        except OSError:
            shutil.copyfile(self.cfg.local_path, staged_path)

        self._remote_future = self._pool.submit(self._sync_remote_task, staged_path, next_idx, n_done)
        if wait_remote:
            self._remote_future.result()

    def load_best_resume(
        self,
        *,
        expected_source_layers: Sequence[int],
        expected_skip_first: int,
    ) -> tuple[dict[int, torch.Tensor], int, int] | None:
        """Inspect Local NVMe, Google Drive (and optional HF Hub); resume from highest `next_idx`."""
        candidates: list[tuple[int, str, str]] = []

        for label, path in (("local", self.cfg.local_path), ("drive", self.cfg.drive_path)):
            if path and os.path.isfile(path):
                meta = peek_checkpoint_metadata(path)
                if meta is not None:
                    candidates.append((meta["next_idx"], label, path))

        if self.cfg.hf_repo_id:
            try:
                from huggingface_hub import hf_hub_download

                path_in_repo = self.cfg.hf_path_in_repo or os.path.basename(self.cfg.local_path)
                dl_path = hf_hub_download(
                    repo_id=self.cfg.hf_repo_id,
                    filename=path_in_repo,
                    repo_type="model",
                    token=self.cfg.hf_token,
                )
                meta = peek_checkpoint_metadata(dl_path)
                if meta is not None:
                    candidates.append((meta["next_idx"], f"hf:{self.cfg.hf_repo_id}", dl_path))
            except Exception as exc:
                logger.info("No resumable checkpoint found on HF Hub %s (%s).", self.cfg.hf_repo_id, exc)

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_next, best_label, best_path = candidates[0]
        logger.info("Selected freshest checkpoint from %s (%s, next_idx=%d).", best_label, best_path, best_next)

        # Restore Drive convergence CSV to local NVMe when resuming on a fresh VM
        if (
            self.cfg.drive_convergence_csv
            and self.cfg.local_convergence_csv
            and os.path.isfile(self.cfg.drive_convergence_csv)
        ):
            local_sz = os.path.getsize(self.cfg.local_convergence_csv) if os.path.isfile(self.cfg.local_convergence_csv) else -1
            drive_sz = os.path.getsize(self.cfg.drive_convergence_csv)
            if drive_sz > local_sz:
                parent = os.path.dirname(os.path.abspath(self.cfg.local_convergence_csv))
                if parent:
                    os.makedirs(parent, exist_ok=True)
                shutil.copyfile(self.cfg.drive_convergence_csv, self.cfg.local_convergence_csv)

        if best_path.endswith(".safetensors"):
            j_mean, meta = load_jlens_safetensors(best_path)
            if list(meta["source_layers"]) != list(expected_source_layers) or meta["skip_first"] != expected_skip_first:
                raise ValueError(
                    f"Checkpoint at {best_path} has incompatible source_layers/skip_first "
                    f"({meta['source_layers']}/{meta['skip_first']} vs {list(expected_source_layers)}/{expected_skip_first}); "
                    "pass resume=False to discard it."
                )
            n_done = meta["n_done"]
            next_idx = meta["next_idx"]
            jacobian_sum = {l: (J * float(n_done)) for l, J in j_mean.items()}
            return jacobian_sum, n_done, next_idx

        state = torch.load(best_path, map_location="cpu", weights_only=True)
        if state.get("source_layers") != list(expected_source_layers) or state.get("skip_first") != expected_skip_first:
            raise ValueError(f"Legacy checkpoint at {best_path} has incompatible source_layers/skip_first.")
        return state["jacobian_sum"], int(state["n_done"]), int(state["next_idx"])

    def flush(self) -> None:
        """Wait for any in-flight remote upload to finish cleanly."""
        if self._remote_future is not None:
            self._remote_future.result()

    def close(self) -> None:
        self.flush()
        self._pool.shutdown(wait=True)


def _yaml_scalar(val: Any) -> str:
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    return json.dumps(str(val))


def export_neuronpedia_bundle(
    lens: Any,
    bundle_dir: str,
    *,
    np_model_id: str = "qwen3.8-flash-next",
    hf_model_name: str = "Qwen/Qwen3.8-Flash-Next-FP8",
    file_prefix: str = "Qwen3.8-Flash-Next-FP8",
    author: str = "",
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    dataset_split: str = "train",
    max_seq_len: int = 128,
    skip_first: int = 16,
    prompt_offset: int = 0,
    stop_at_delta: float | None = None,
    min_prompts: int = 100,
    stop_window: int = 10,
    convergence_csv_path: str | None = None,
    gpu_name: str = "unknown",
    gpu_vram_gib: float = 0.0,
    gdn_kernel: str = "unknown",
    include_4branch_safetensors: bool = True,
) -> dict[str, str]:
    """Export a clean Neuronpedia-compatible J-Lens release folder ready for a single-commit HF Hub upload.

    Writes inside `bundle_dir`:
      1. `<prefix>_jacobian_lens.pt`                  — Neuronpedia drop-in fp16 `.pt` (`J`: `[d_model, d_model]`
                                                        equal-perturbation collapsed map over the 4 GR streams,
                                                        loadable via `torch.load(..., weights_only=True)` + `provenance`).
      2. `<prefix>_jacobian_lens_4branch.safetensors` — Full lossless 4-branch fp16 `.safetensors` (`[d_model, 4*d_model]`
                                                        + collapsed `[d_model, d_model]`, loadable via `Qwen38JacobianLens.load`).
      3. `<prefix>_convergence.csv`                   — Per-prompt convergence history (if provided).
      4. `config.yaml`                                — Neuronpedia-style provenance, dataset, hardware, and convergence summary.
      5. `CREDIT.md`                                  — Contributor credit & 4-branch Gated Residual usage note.
    """
    os.makedirs(bundle_dir, exist_ok=True)
    created_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    written: dict[str, str] = {}

    pt_filename = f"{file_prefix}_jacobian_lens.pt"
    st_filename = f"{file_prefix}_jacobian_lens_4branch.safetensors"
    csv_filename = f"{file_prefix}_convergence.csv"

    pt_path = os.path.join(bundle_dir, pt_filename)
    st_path = os.path.join(bundle_dir, st_filename)
    csv_dst = os.path.join(bundle_dir, csv_filename)
    cfg_path = os.path.join(bundle_dir, "config.yaml")
    credit_path = os.path.join(bundle_dir, "CREDIT.md")

    # 1. Copy convergence CSV first so we can summarize final metrics into config.yaml
    final_rel: float | None = None
    final_norm: float | None = None
    if convergence_csv_path and os.path.isfile(convergence_csv_path):
        shutil.copyfile(convergence_csv_path, csv_dst)
        written["convergence_csv"] = csv_dst
        final_rel, final_norm = ConvergenceLogger(csv_dst, resume_next_idx=10**9).last_summary()

    # 2. Neuronpedia-compatible fp16 `.pt` (`J[l]: [d_model, d_model]`, weights_only=True safe!)
    collapsed = lens.collapsed_jacobians()
    has_embed_layer = -1 in lens.source_layers
    provenance = {
        "source_model": hf_model_name,
        "np_model_id": np_model_id,
        "method": "jacobian_lens",
        "capture_point": "embed_tokens_and_decoder_layer_output" if has_embed_layer else "decoder_layer_output",
        "include_embedding_layer": has_embed_layer,
        "stream_reduce": "equal_perturbation_sum_over_4_gr_branches",
        "hc_count": int(lens.hc_count),
        "full_4branch_file": st_filename if include_4branch_safetensors else None,
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "dataset_split": dataset_split,
        "max_seq_len": int(max_seq_len),
        "skip_first": int(skip_first),
        "prompt_offset": int(prompt_offset),
        "stop_at_delta": float(stop_at_delta) if stop_at_delta else None,
        "dtype": "float16",
        "author": author.strip() if author else "community",
        "created_at_utc": created_at,
    }
    pt_payload = {
        "J": {int(l): collapsed[l].to(torch.float16).cpu().contiguous() for l in lens.source_layers},
        "source_layers": [int(l) for l in lens.source_layers],
        "d_model": int(lens.d_model),
        "n_prompts": int(lens.n_prompts),
        "branch_norms": {int(l): [round(v, 5) for v in norms] for l, norms in lens.branch_norms().items()},
        "provenance": provenance,
    }
    tmp_pt = f"{pt_path}.tmp.{os.getpid()}"
    torch.save(pt_payload, tmp_pt)
    os.replace(tmp_pt, pt_path)
    written["neuronpedia_pt"] = pt_path

    # 3. Full lossless 4-branch fp16 `.safetensors` (`[d_model, 4*d_model]` + collapsed `[d_model, d_model]`)
    if include_4branch_safetensors:
        save_jlens_safetensors(
            st_path,
            lens.jacobians,
            n_done=lens.n_prompts,
            next_idx=lens.n_prompts,
            skip_first=skip_first,
            d_model=lens.d_model,
            hc_count=lens.hc_count,
            model_name=hf_model_name,
            dtype=torch.float16,
            include_collapsed=True,
        )
        written["full_4branch_safetensors"] = st_path

    # 4. Neuronpedia-compatible `config.yaml`
    yaml_lines = [
        "# Jacobian Lens fit — Qwen3.8-Flash-Next (4-branch Gated Residual, qwen4_exp)",
        "# Compatible with neuronpedia/jacobian-lens and anthropics/jacobian-lens (Apache-2.0).",
        f"# Generated: {created_at}",
        f"np_model_id: {_yaml_scalar(np_model_id)}",
        f"hf_model_name: {_yaml_scalar(hf_model_name)}",
        "dataset:",
        f"  name: {_yaml_scalar(dataset_name)}",
        f"  config: {_yaml_scalar(dataset_config)}",
        f"  split: {_yaml_scalar(dataset_split)}",
        '  text_field: "text"',
        "  min_chars: 600",
        f"  prompt_offset: {int(prompt_offset)}",
        "fit:",
        f"  n_prompts: {int(lens.n_prompts)}",
        f"  max_seq_len: {int(max_seq_len)}",
        f"  skip_first: {int(skip_first)}",
        f"  d_model: {int(lens.d_model)}",
        f"  hc_count: {int(lens.hc_count)}",
        f"  d_source_4branch: {int(lens.d_source)}",
        f"  include_embedding_layer: {_yaml_scalar(has_embed_layer)}",
        f"  n_source_layers: {len(lens.source_layers)}",
        f"  source_layers: {json.dumps([int(l) for l in lens.source_layers])}",
        "  target_layer: null",
        '  compute_dtype: "bfloat16"',
        '  saved_dtype: "float16"',
        f"  stop_at_delta: {_yaml_scalar(float(stop_at_delta) if stop_at_delta else None)}",
        f"  min_prompts: {int(min_prompts)}",
        f"  stop_window: {int(stop_window)}",
        "gpus:",
        "  gpu_0:",
        f"    name: {_yaml_scalar(gpu_name)}",
        f"    total_gib: {round(float(gpu_vram_gib), 2)}",
        "environment:",
        f"  python: {_yaml_scalar(platform.python_version())}",
        f"  torch: {_yaml_scalar(torch.__version__)}",
        f"  platform: {_yaml_scalar(platform.platform())}",
        f"  gdn_kernel: {_yaml_scalar(gdn_kernel)}",
        "results:",
        f"  prompts_fitted: {int(lens.n_prompts)}",
        f"  final_mean_rel_change: {_yaml_scalar(round(final_rel, 8) if final_rel is not None else None)}",
        f"  final_max_norm: {_yaml_scalar(round(final_norm, 6) if final_norm is not None else None)}",
        "artifacts:",
        f"  neuronpedia_pt: {_yaml_scalar(pt_filename)}",
        f"  full_4branch_safetensors: {_yaml_scalar(st_filename if include_4branch_safetensors else None)}",
        f"  convergence_csv: {_yaml_scalar(csv_filename if 'convergence_csv' in written else None)}",
        f"author: {_yaml_scalar(author.strip() if author else 'community')}",
        'attribution: "Jacobian lens (Gurnee et al., 2026, anthropics/jacobian-lens, Apache-2.0) fitted with Qwen3.8FlashNextJ-lens."',
        "",
    ]
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines))
    written["config_yaml"] = cfg_path

    # 5. Contributor `CREDIT.md`
    author_label = author.strip() if author else "Community contributor"
    emb_note = (
        "\n- **Embedding Layer (`-1`)**: Included (`J[-1]` in `.pt` equals the exact token-embedding Jacobian "
        "`E[d h_final / d emb] = sum_{b=0..3} J_{-1}[:, b, :]`; stored as `J.embed` in `.safetensors`)."
        if has_embed_layer
        else ""
    )
    credit_md = f"""# Qwen3.8-Flash-Next Jacobian Lens

- **Model**: `{hf_model_name}` (`qwen4_exp`, 48 layers, 4-branch Gated Residual `hc_count=4`, `d_model=2560`, 51B PLE n-gram table at layer 2)
- **Fitted by**: {author_label}
- **Prompts**: {lens.n_prompts} (`{dataset_name}` / `{dataset_config}`, `max_seq_len={max_seq_len}`, `skip_first={skip_first}`)
- **Method**: Exact Jacobian Lens (`J_l = E[d h_final / d H_l]`, Gurnee et al., 2026){emb_note}

## Files

| File | Shape / Format | Purpose |
|---|---|---|
| `{pt_filename}` | `J[l]: [2560, 2560]` (`fp16` `.pt`, `weights_only=True`) | Drop-in Neuronpedia readout (`unembed(h @ J[l].T)` where `h` is the mean/equal-perturbation residual across the 4 Gated Residual streams: `J_collapsed = sum_{{b=0..3}} J_4branch[:, b, :]`). |
| `{st_filename}` | `J.layer_XX` / `J.embed`: `[2560, 10240]` + `J_collapsed.*`: `[2560, 2560]` (`fp16` `.safetensors`) | Full lossless 4-branch Jacobian (`unembed(H_l @ J_l.T)` where `H_l` is the concatenated `[4 * 2560]` Gated Residual state). |
| `config.yaml` | YAML | Reproducible fitting configuration, hardware, kernel stack, and convergence summary. |
| `{csv_filename}` | CSV | Per-prompt convergence log (`mean_rel_change`, `max_norm`, wall-clock timing). |
"""
    with open(credit_path, "w", encoding="utf-8") as f:
        f.write(credit_md)
    written["credit_md"] = credit_path
    return written


def publish_bundle_to_hf(
    bundle_dir: str,
    repo_id: str,
    *,
    path_in_repo: str = "",
    token: str | None = None,
    private: bool = False,
    commit_message: str | None = None,
) -> str:
    """Upload the entire Neuronpedia-ready bundle to Hugging Face Hub in ONE single clean commit."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    msg = commit_message or f"Publish Qwen3.8-Flash-Next Jacobian Lens bundle ({os.path.basename(os.path.abspath(bundle_dir))})"
    info = api.upload_folder(
        folder_path=bundle_dir,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=msg,
    )
    return str(getattr(info, "commit_url", f"https://huggingface.co/{repo_id}"))

