"""Zero-RAM mmap reader & exact n-gram hasher for Qwen3.8-Flash-Next's 51B PLE embedding table.

In Qwen/Qwen3.8-Flash-Next-FP8:
  - PLE is active at layer 2 (1-indexed -> decoder layer index 1).
  - The n-gram table has ~320,001,536 rows x 160 dims = 51.2B params (~47.68 GiB in FP8 E4M3)
    split across `split_ngram_parts = 128` shard tensors (`ngram_embedding.shard_0..127.weight`),
    plus a single per-tensor scale (`ngram_embedding.weight_scale`, shape `[1]`).
  - Each token only reads 16 rows x 160 bytes = 2.5 KB! For a 128-token J-lens prompt, that is
    only 320 KB of reads -> < 2 ms via MADV_RANDOM mmap with ZERO RAM footprint.
  - Because `ple_embedding(input_ids)` depends ONLY on `input_ids` (never on hidden states),
    when disk is tight we can also pre-extract embeddings (~640 KB/prompt) and delete the 48 GiB
    PLE shards before downloading MoE shards!
"""

from __future__ import annotations

import json
import math
import mmap as _mmap
import os
import re
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch


# SplitMix64 constants matching transformers/models/qwen4_exp/modeling_qwen4_exp.py exactly
_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007

_SHARD_IDX_RE = re.compile(r"ngram_embedding\.shard_(\d+)\.weight$")

_SAFETENSORS_DTYPE_INFO: dict[str, tuple[torch.dtype, int]] = {
    "F8_E4M3": (torch.float8_e4m3fn, 1),
    "F8_E5M2": (torch.float8_e5m2, 1),
    "BF16": (torch.bfloat16, 2),
    "F16": (torch.float16, 2),
    "F32": (torch.float32, 4),
}


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def build_layer_multipliers(
    unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int
) -> torch.Tensor:
    """Bit-identical to Qwen4Exp `_build_layer_multipliers`."""
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return torch.tensor(multipliers, dtype=torch.long)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def parse_safetensors_header(path: str) -> tuple[dict, int]:
    """Return `(header_dict, data_start_byte_offset)` of a safetensors file."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header, 8 + header_len


@dataclass
class PleHashSpec:
    """Deterministic prime-vocab n-gram hashing specification matching Qwen4ExpTextNGramEmbedding."""

    vocab_size: int
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    ple_embed_dim: int = 2560
    seed: int = 1234
    eos_token_id: int = 248044
    ple_layer_index: int = 0

    def __post_init__(self) -> None:
        self.context_len = self.ngram_size - 1
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.head_dim_per_ngram = self.ple_embed_dim // self.ngram_heads

        self.head_vocab_sizes: list[int] = []
        self.head_offsets: list[int] = []
        total = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
            size = _find_nth_prime_after(self.ngram_vocab_size_base - 1, global_head_idx + 1)
            self.head_vocab_sizes.append(size)
            self.head_offsets.append(total)
            total += size
        self.total_vocab_size = total
        div = self.make_ngram_vocab_size_divisible_by
        self.padded_vocab_size = math.ceil(total / div) * div

        self.layer_multipliers = build_layer_multipliers(
            self.vocab_size, self.ngram_size, self.ple_layer_index, self.seed
        )
        self.ngram_heads_vocab_sizes = torch.tensor(self.head_vocab_sizes, dtype=torch.long)
        self.ngram_heads_offsets = torch.tensor(self.head_offsets, dtype=torch.long)

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1
        )
        segment_start = previous_eos + 1
        position_in_segment = positions.unsqueeze(0) - segment_start
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted = token_ids.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def compute_ngram_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute `[B, T, ngram_heads]` global row indices bit-identical to Qwen4Exp."""
        device = input_ids.device
        input_ids = input_ids.long()
        prev_ctx = input_ids.new_full((input_ids.shape[0], self.context_len), self.eos_token_id)
        token_history = torch.cat([prev_ctx, input_ids], dim=-1)
        shifted_tokens = [
            self._shift_right_ignore_eos(token_history, shift) for shift in range(self.ngram_size)
        ]

        mult = self.layer_multipliers.to(device)
        vocab_sizes = self.ngram_heads_vocab_sizes.to(device)
        offsets = self.ngram_heads_offsets.to(device)

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start_idx = (ngram - 2) * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            mixed_ids = shifted_tokens[0] * mult[0]
            for pos in range(1, ngram):
                mixed_ids = torch.bitwise_xor(mixed_ids, shifted_tokens[pos] * mult[pos])
            h_sizes = vocab_sizes[start_idx:end_idx]
            h_offsets = offsets[start_idx:end_idx]
            ngram_ids = torch.remainder(mixed_ids.unsqueeze(-1), h_sizes.view(1, 1, -1))
            blocks.append(ngram_ids + h_offsets.view(1, 1, -1))

        return torch.cat(blocks, dim=-1)[:, -input_ids.shape[1] :]


class MmapPleTable:
    """Zero-RAM NVMe mmap lookup table for Qwen3.8-Flash-Next's sharded FP8/BF16 PLE embedding.

    Supports both:
      - Sharded `ngram_embedding.shard_0..N.weight` safetensors layout (real Qwen3.8-Flash-Next-FP8),
      - Single-tensor `ngram_embedding.weight` safetensors or in-memory tensor (unit tests / small checkpoints).
    """

    def __init__(
        self,
        hash_spec: PleHashSpec,
        shards: Sequence[tuple[str, int, int]],
        row_dim: int,
        torch_dtype: torch.dtype,
        weight_scale: float | torch.Tensor | None = None,
        workers: int = 16,
    ) -> None:
        self.hash_spec = hash_spec
        self.row_dim = int(row_dim)
        self.torch_dtype = torch_dtype
        self.elem_bytes = torch.tensor([], dtype=torch_dtype).element_size()
        self.row_bytes = self.row_dim * self.elem_bytes
        if weight_scale is None:
            self.weight_scale = 1.0
        elif torch.is_tensor(weight_scale):
            self.weight_scale = float(weight_scale.float().item())
        else:
            self.weight_scale = float(weight_scale)

        self._mmaps: list[np.memmap] = []
        self._shard_starts: list[int] = []
        self._shard_ends: list[int] = []
        cursor = 0
        for path, byte_offset, n_rows in shards:
            mm = np.memmap(path, dtype=np.uint8, mode="r", offset=byte_offset, shape=(n_rows, self.row_bytes))
            # Avoid Linux readahead polluting page cache on random 160-byte hashed rows
            raw_mm = getattr(mm, "_mmap", None)
            if raw_mm is not None and hasattr(_mmap, "MADV_RANDOM"):
                try:
                    raw_mm.madvise(_mmap.MADV_RANDOM)
                except OSError:
                    pass
            self._mmaps.append(mm)
            self._shard_starts.append(cursor)
            cursor += n_rows
            self._shard_ends.append(cursor)
        self.total_rows = cursor
        self._shard_starts_np = np.array(self._shard_starts, dtype=np.int64)
        self._pool = ThreadPoolExecutor(max_workers=max(1, min(workers, len(self._mmaps))))

    @classmethod
    def from_safetensors_files(
        cls,
        hash_spec: PleHashSpec,
        shard_paths: Iterable[str],
        weight_scale: float | torch.Tensor | None = None,
        workers: int = 16,
    ) -> "MmapPleTable":
        """Discover all PLE n-gram shards from safetensors headers and open zero-copy mmaps."""
        indexed_shards: dict[int, tuple[str, int, int]] = {}
        single_entry: tuple[str, int, int] | None = None
        row_dim: int | None = None
        torch_dtype: torch.dtype | None = None
        discovered_scale = weight_scale

        for path in shard_paths:
            header, data_start = parse_safetensors_header(path)
            for key, meta in header.items():
                if key.endswith("ngram_embedding.weight_scale") and discovered_scale is None:
                    off0, off1 = meta["data_offsets"]
                    dt_str = meta["dtype"]
                    dt, _ = _SAFETENSORS_DTYPE_INFO[dt_str]
                    with open(path, "rb") as f:
                        f.seek(data_start + off0)
                        raw = f.read(off1 - off0)
                    t = torch.frombuffer(bytearray(raw), dtype=dt)
                    discovered_scale = float(t.float().item())
                    continue

                m = _SHARD_IDX_RE.search(key)
                if m is not None:
                    s_idx = int(m.group(1))
                    n_rows, r_dim = meta["shape"]
                    dt, _ = _SAFETENSORS_DTYPE_INFO[meta["dtype"]]
                    off0 = data_start + meta["data_offsets"][0]
                    indexed_shards[s_idx] = (path, off0, int(n_rows))
                    row_dim = int(r_dim)
                    torch_dtype = dt
                elif key.endswith("ngram_embedding.weight"):
                    n_rows, r_dim = meta["shape"]
                    dt, _ = _SAFETENSORS_DTYPE_INFO[meta["dtype"]]
                    off0 = data_start + meta["data_offsets"][0]
                    single_entry = (path, off0, int(n_rows))
                    row_dim = int(r_dim)
                    torch_dtype = dt

        if indexed_shards:
            ordered = [indexed_shards[i] for i in sorted(indexed_shards)]
        elif single_entry is not None:
            ordered = [single_entry]
        else:
            raise FileNotFoundError("No `ngram_embedding.shard_*.weight` or `ngram_embedding.weight` found in shards.")

        assert row_dim is not None and torch_dtype is not None
        return cls(
            hash_spec=hash_spec,
            shards=ordered,
            row_dim=row_dim,
            torch_dtype=torch_dtype,
            weight_scale=discovered_scale,
            workers=workers,
        )

    def gather_rows(self, row_ids: np.ndarray) -> torch.Tensor:
        """Gather `row_ids` (`[...]` int64) -> dequantized `bfloat16` tensor `[..., row_dim]` on CPU."""
        orig_shape = row_ids.shape
        flat_ids = np.ascontiguousarray(row_ids, dtype=np.int64).reshape(-1)
        if flat_ids.size == 0:
            return torch.empty((*orig_shape, self.row_dim), dtype=torch.bfloat16)

        uniq, inverse = np.unique(flat_ids, return_inverse=True)
        if uniq[0] < 0 or uniq[-1] >= self.total_rows:
            raise IndexError(f"PLE row id out of range [{uniq[0]}, {uniq[-1]}] vs total_rows={self.total_rows}")

        # Map each unique global row id to (shard_idx, local_row)
        shard_ids = np.searchsorted(self._shard_starts_np, uniq, side="right") - 1
        local_rows = uniq - self._shard_starts_np[shard_ids]

        out_bytes = np.empty((uniq.size, self.row_bytes), dtype=np.uint8)
        bounds = np.flatnonzero(np.diff(shard_ids)) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [uniq.size]))

        if uniq.size <= 512 or len(starts) == 1:
            for s, e in zip(starts.tolist(), ends.tolist()):
                si = int(shard_ids[s])
                out_bytes[s:e] = self._mmaps[si][local_rows[s:e]]
        else:
            def _fetch(span: tuple[int, int]) -> None:
                s, e = span
                si = int(shard_ids[s])
                out_bytes[s:e] = self._mmaps[si][local_rows[s:e]]

            list(self._pool.map(_fetch, zip(starts.tolist(), ends.tolist())))

        # Re-expand to original order and reinterpret bytes as native torch dtype (FP8 / BF16)
        gathered_bytes = out_bytes[inverse]
        raw_tensor = torch.from_numpy(gathered_bytes).view(self.torch_dtype).view(*orig_shape, self.row_dim)
        if self.torch_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            return raw_tensor.to(torch.bfloat16).mul_(self.weight_scale)
        return raw_tensor.to(torch.bfloat16)

    def lookup(self, input_ids: torch.Tensor, device: torch.device | str | None = None) -> torch.Tensor:
        """Compute n-gram IDs for `input_ids` `[B, T]` and return `[B, T, ple_embed_dim]` BF16."""
        ngram_ids = self.hash_spec.compute_ngram_ids(input_ids.cpu())  # [B, T, ngram_heads]
        emb = self.gather_rows(ngram_ids.numpy()).flatten(-2)  # [B, T, ple_embed_dim]
        target_dev = input_ids.device if device is None else torch.device(device)
        return emb.to(target_dev, non_blocking=True)

    def preextract_corpus(self, tokenized_prompts: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Pre-gather PLE embeddings (`[1, T, ple_embed_dim]` BF16 CPU) for an entire prompt list.

        ~640 KB per 128-token prompt -> 64 MB for 100 prompts, 640 MB for 1000 prompts.
        Allows safely deleting the 48 GiB PLE shards on disk-constrained Colab VMs!
        """
        return [self.lookup(ids, device="cpu").contiguous() for ids in tokenized_prompts]

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        for mm in self._mmaps:
            raw_mm = getattr(mm, "_mmap", None)
            if raw_mm is not None:
                try:
                    raw_mm.close()
                except Exception:
                    pass
        self._mmaps.clear()


class PreextractedPleProvider:
    """Zero-disk, zero-latency PLE embedding provider backed by pre-extracted CPU tensors."""

    def __init__(self, cache_by_key: dict[bytes, torch.Tensor]) -> None:
        self._cache = cache_by_key

    @staticmethod
    def _key(input_ids: torch.Tensor) -> bytes:
        return input_ids.detach().cpu().to(torch.int64).contiguous().numpy().tobytes()

    @classmethod
    def from_mmap_and_prompts(
        cls, mmap_table: MmapPleTable, tokenized_prompts: Sequence[torch.Tensor]
    ) -> "PreextractedPleProvider":
        cache: dict[bytes, torch.Tensor] = {}
        for ids in tokenized_prompts:
            k = cls._key(ids)
            if k not in cache:
                cache[k] = mmap_table.lookup(ids, device="cpu").contiguous()
        return cls(cache)

    def lookup(self, input_ids: torch.Tensor, device: torch.device | str | None = None) -> torch.Tensor:
        k = self._key(input_ids)
        if k not in self._cache:
            raise KeyError(
                "Prompt not in PreextractedPleProvider cache. Use `ple_mode='mmap'` to support arbitrary prompts."
            )
        target_dev = input_ids.device if device is None else torch.device(device)
        return self._cache[k].to(target_dev, non_blocking=True)

    def save(self, path: str) -> None:
        torch.save(self._cache, path)

    @classmethod
    def load(cls, path: str) -> "PreextractedPleProvider":
        return cls(torch.load(path, map_location="cpu", weights_only=True))


class InMemoryPleProvider:
    """Simple in-memory PLE provider for unit tests and tiny synthetic models."""

    def __init__(
        self,
        hash_spec: PleHashSpec,
        weight: torch.Tensor,
        weight_scale: float = 1.0,
    ) -> None:
        self.hash_spec = hash_spec
        self.weight = weight
        self.weight_scale = float(weight_scale)

    def lookup(self, input_ids: torch.Tensor, device: torch.device | str | None = None) -> torch.Tensor:
        ngram_ids = self.hash_spec.compute_ngram_ids(input_ids.to(self.weight.device))
        rows = torch.nn.functional.embedding(ngram_ids, self.weight)
        if rows.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            rows = rows.to(torch.bfloat16) * self.weight_scale
        out = rows.flatten(-2)
        target_dev = input_ids.device if device is None else torch.device(device)
        return out.to(target_dev)
