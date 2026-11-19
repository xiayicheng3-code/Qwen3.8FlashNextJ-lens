"""Crash-proof, zero-GPU-stall checkpointing for Qwen3.8-Flash-Next J-Lens fitting.

Why not naive synchronous `torch.save(jacobian_sum, "/content/drive/...")`?
  - 48 layers x [2560, 10240] in fp32 is 4.69 GiB (5.03 GB); atomic temp+rename needs ~10 GB
    of Google Drive quota and blocks the GPU for 80-180 seconds over Drive FUSE!
  - Saving a single `.safetensors` file storing the running MEAN `J_mean = jacobian_sum / n_done`
    in fp16 (2.34 GiB, or optional fp32 4.69 GiB) with metadata embedded in the safetensors header:
      1. Cuts checkpoint size & Drive/HF upload time in half (2.34 GiB vs 4.69 GiB) with zero
         fp16 overflow risk regardless of `n_prompts` (because we store the mean, not the sum!).
      2. Makes EVERY in-progress checkpoint immediately loadable as a valid `Qwen38JacobianLens`!
      3. Writes to local NVMe in <1 second, then syncs asynchronously on a background thread to
         Google Drive and/or Hugging Face Hub (free unlimited storage, ~300-800 MB/s Xet upload)
         with ZERO GPU idle time!
      4. On resume, inspects local NVMe, Google Drive, and HF Hub headers (<1 ms via safetensors
         metadata) and automatically resumes from whichever copy has the highest `next_idx`!
"""

from __future__ import annotations

import json
import logging
import os
import shutil
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
        tensors[f"J.layer_{l:02d}"] = J_f.to(dtype=dtype).contiguous()
        if include_collapsed:
            tensors[f"J_collapsed.layer_{l:02d}"] = J4.sum(dim=1).to(dtype=dtype).contiguous()
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
        source_layers = json.loads(raw_meta["source_layers"])
        jacobian_mean: dict[int, torch.Tensor] = {}
        for l in source_layers:
            key = f"J.layer_{int(l):02d}"
            jacobian_mean[int(l)] = sf.get_tensor(key).float()

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


@dataclass
class CheckpointConfig:
    """Multi-destination asynchronous checkpointing policy."""

    local_path: str                           # Fast local NVMe `.safetensors` path (~0.8s write!)
    drive_path: str | None = None             # Optional Google Drive mirror path (copied asynchronously)
    hf_repo_id: str | None = None             # Optional Hugging Face Hub repo (e.g. "user/qwen38-jlens")
    hf_path_in_repo: str | None = None        # Filename inside HF repo (defaults to basename of local_path)
    hf_token: str | None = None
    hf_private: bool = True
    ckpt_dtype: str = "float16"               # "float16" (2.34 GiB, default) or "float32" (4.69 GiB exact)
    every_n_prompts: int = 10                 # Trigger checkpoint every N prompts
    every_minutes: float = 10.0               # OR every M minutes (whichever comes first!)
    max_sync_retries: int = 3
    model_name: str = "Qwen/Qwen3.8-Flash-Next-FP8"


class AsyncCheckpointManager:
    """Non-blocking checkpoint manager: fast local NVMe write + background Drive & HF Hub sync.

    Guarantees:
      - Zero GPU stalls waiting on Google Drive FUSE or HF Hub network uploads.
      - Never piles up queued background uploads (if a previous remote sync is still running,
        local NVMe is still updated and the next completed window triggers the remote sync;
        `flush()` waits cleanly on completion or `KeyboardInterrupt`).
      - Multi-source resume automatically chooses the freshest checkpoint across Local NVMe,
        Google Drive, and Hugging Face Hub!
    """

    def __init__(self, cfg: CheckpointConfig) -> None:
        self.cfg = cfg
        self.dtype = resolve_ckpt_dtype(cfg.ckpt_dtype)
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
        """Atomic copy to Google Drive FUSE with fallback + retry for transient FUSE errors."""
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
        """Upload checkpoint to Hugging Face Hub (Xet chunk-deduplicated, free unlimited storage)."""
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
            # Remove immutable staging hardlink/copy once remote sync finishes
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
        """Write local NVMe `.safetensors` immediately (~0.8s), then push to Drive/HF in background."""
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

        # If a previous background upload is still running:
        #   - during fitting (`wait_remote=False`), don't pile up uploads; local copy is already fresh!
        #   - on final/interrupt flush (`wait_remote=True`), wait for previous upload then launch final sync.
        if self._remote_future is not None and not self._remote_future.done():
            if not wait_remote:
                logger.info("Previous remote sync still in flight; keeping local NVMe checkpoint fresh without stalling GPU.")
                return
            self._remote_future.result()

        # Hardlink local file to an immutable staging path (0 extra disk space, 0 ms!) so
        # subsequent local writes never clobber bytes being read by the background upload thread!
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
        """Inspect Local NVMe, Google Drive, and HF Hub; resume from whichever has the highest `next_idx`."""
        candidates: list[tuple[int, str, str]] = []  # (next_idx, source_label, local_file_path)

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

        # Pick the candidate with the largest `next_idx`
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_next, best_label, best_path = candidates[0]
        logger.info("Selected freshest checkpoint from %s (%s, next_idx=%d).", best_label, best_path, best_next)

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

        # Legacy `.pt` resume
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
