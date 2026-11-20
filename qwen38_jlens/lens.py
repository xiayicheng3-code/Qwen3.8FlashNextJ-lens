"""Qwen38JacobianLens container, readout evaluation, and interactive HTML slice visualizer.

Compatible with `anthropics/jacobian-lens` (`jlens`) while natively supporting Qwen3.8-Flash-Next's
4-branch Gated Residual (`hc_count = 4`, residual stream width `10240 -> 2560`).

Format saved to `.pt`:
  {
    "J": {layer_idx: Tensor[2560, 10240] (fp16)},
    "J_collapsed": {layer_idx: Tensor[2560, 2560] (fp16)},   # sum across 4 branches (identical-perturbation J)
    "branch_norms": {layer_idx: [norm_b0, norm_b1, norm_b2, norm_b3]},
    "n_prompts": int,
    "source_layers": list[int],
    "d_model": 2560,
    "d_source": 10240,
    "hc_count": 4,
    "model_name": str,
  }
"""

from __future__ import annotations

import html as _html
import json
import os
from typing import Any, Sequence

import torch
import torch.nn.functional as F


class Qwen38JacobianLens:
    """Fitted Jacobian Lens for Qwen3.8-Flash-Next (`qwen4_exp`, 4-branch Gated Residual)."""

    def __init__(
        self,
        jacobians: dict[int, torch.Tensor],
        *,
        n_prompts: int,
        d_model: int = 2560,
        hc_count: int = 4,
        model_name: str = "Qwen/Qwen3.8-Flash-Next-FP8",
    ) -> None:
        self.jacobians = {int(layer): J.float().cpu() for layer, J in jacobians.items()}
        self.source_layers = sorted(self.jacobians.keys())
        self.n_prompts = int(n_prompts)
        self.d_model = int(d_model)
        self.hc_count = int(hc_count)
        self.d_source = self.d_model * self.hc_count
        self.model_name = model_name

    def __repr__(self) -> str:
        rng = f"{self.source_layers[0]}..{self.source_layers[-1]}" if self.source_layers else "empty"
        return (
            f"Qwen38JacobianLens(model={self.model_name!r}, d_out={self.d_model}, "
            f"d_src={self.d_source} ({self.hc_count}x{self.d_model}), "
            f"n_prompts={self.n_prompts}, layers=[{rng}] ({len(self.source_layers)}))"
        )

    def branch_norms(self) -> dict[int, list[float]]:
        """Per-branch Frobenius norms `||J_l[:, b, :]||_F / sqrt(d_model)` across the 4 GR branches."""
        scale = self.d_model ** 0.5
        out: dict[int, list[float]] = {}
        for l in self.source_layers:
            J4 = self.jacobians[l].view(self.d_model, self.hc_count, self.d_model)
            out[l] = [float(J4[:, b, :].norm().item() / scale) for b in range(self.hc_count)]
        return out

    def collapsed_jacobians(self) -> dict[int, torch.Tensor]:
        """Sum across the 4 branches: `J_sum[l] = sum_{b=0..3} d h_final / d H_l^(b)` (`[d_model, d_model]`)."""
        return {
            l: self.jacobians[l].view(self.d_model, self.hc_count, self.d_model).sum(dim=1)
            for l in self.source_layers
        }

    def transport(self, H_l: torch.Tensor, layer: int) -> torch.Tensor:
        """Transport 4-branch residual `H_l` (`[..., hc_count * d_model]`) -> `[..., d_model]`."""
        J = self.jacobians[layer].to(device=H_l.device, dtype=torch.float32)
        return F.linear(H_l.float(), J)

    def save(self, path: str, *, dtype: torch.dtype = torch.float16, include_collapsed: bool = True) -> None:
        """Atomically save fitted lens to `.safetensors` (recommended, fast, zero-pickle) or `.pt`."""
        if path.endswith(".safetensors"):
            from qwen38_jlens.checkpoint import save_jlens_safetensors

            save_jlens_safetensors(
                path,
                self.jacobians,
                n_done=self.n_prompts,
                next_idx=self.n_prompts,
                skip_first=16,
                d_model=self.d_model,
                hc_count=self.hc_count,
                model_name=self.model_name,
                dtype=dtype,
                include_collapsed=include_collapsed,
            )
            return

        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload: dict[str, Any] = {
            "J": {l: J.to(dtype) for l, J in self.jacobians.items()},
            "branch_norms": self.branch_norms(),
            "n_prompts": self.n_prompts,
            "source_layers": self.source_layers,
            "d_model": self.d_model,
            "d_source": self.d_source,
            "hc_count": self.hc_count,
            "model_name": self.model_name,
        }
        if include_collapsed:
            payload["J_collapsed"] = {l: J.to(dtype) for l, J in self.collapsed_jacobians().items()}
        tmp = f"{path}.tmp.{os.getpid()}"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "Qwen38JacobianLens":
        """Load a saved lens or ANY in-progress `.safetensors` checkpoint directly as a usable lens!"""
        if path.endswith(".safetensors"):
            from qwen38_jlens.checkpoint import load_jlens_safetensors

            jacobians, meta = load_jlens_safetensors(path)
            return cls(
                jacobians=jacobians,
                n_prompts=meta["n_done"],
                d_model=meta["d_model"],
                hc_count=meta["hc_count"],
                model_name=meta["model_name"],
            )

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if "J" not in ckpt:
            raise ValueError(f"{path} is not a saved Qwen38JacobianLens file (keys={sorted(ckpt)})")
        return cls(
            jacobians=ckpt["J"],
            n_prompts=ckpt["n_prompts"],
            d_model=ckpt["d_model"],
            hc_count=ckpt.get("hc_count", ckpt["J"][ckpt["source_layers"][0]].shape[1] // ckpt["d_model"]),
            model_name=ckpt.get("model_name", "Qwen/Qwen3.8-Flash-Next-FP8"),
        )

    @classmethod
    def merge(cls, lenses: Sequence["Qwen38JacobianLens"]) -> "Qwen38JacobianLens":
        """Combine lenses fitted on disjoint prompt slices (n_prompts-weighted mean)."""
        if not lenses:
            raise ValueError("merge() requires at least one lens.")
        first = lenses[0]
        for other in lenses[1:]:
            if (
                other.source_layers != first.source_layers
                or other.d_model != first.d_model
                or other.hc_count != first.hc_count
            ):
                raise ValueError("Cannot merge lenses with mismatched layers / d_model / hc_count.")
        n_total = sum(l.n_prompts for l in lenses)
        merged = {}
        for layer in first.source_layers:
            acc = sum(l.jacobians[layer] * l.n_prompts for l in lenses)
            merged[layer] = acc / n_total
        return cls(
            jacobians=merged,
            n_prompts=n_total,
            d_model=first.d_model,
            hc_count=first.hc_count,
            model_name=first.model_name,
        )


def render_slice_html(
    tokens: Sequence[str],
    layer_top1: dict[int, list[tuple[str, float]]],
    model_top1: list[tuple[str, float]],
    title: str = "Qwen3.8-Flash-Next Jacobian Lens Slice",
) -> str:
    """Build a lightweight, self-contained interactive HTML layer x position heatmap for Colab display."""
    layers = sorted(layer_top1.keys())
    rows_html = []
    for l in layers:
        lbl = "Emb" if l == -1 else f"L{l}"
        cells = []
        for tok_str, prob in layer_top1[l]:
            alpha = min(0.85, max(0.08, prob))
            bg = f"rgba(56, 139, 253, {alpha:.2f})"
            esc = _html.escape(tok_str.replace("\n", "\\n"))
            cells.append(
                f'<td style="background:{bg};padding:4px 6px;border:1px solid #2d333b;'
                f'font-family:monospace;font-size:12px;white-space:nowrap;" '
                f'title="{lbl} p={prob:.3f}">{esc}<sup style="opacity:.7">{int(prob*100)}%</sup></td>'
            )
        rows_html.append(
            f'<tr><th style="position:sticky;left:0;background:#161b22;color:#8b949e;padding:4px 8px;'
            f'border:1px solid #2d333b;font-family:monospace;font-size:12px;">{lbl}</th>{"".join(cells)}</tr>'
        )
    # Final model output row
    m_cells = []
    for tok_str, prob in model_top1:
        alpha = min(0.90, max(0.12, prob))
        bg = f"rgba(63, 185, 80, {alpha:.2f})"
        esc = _html.escape(tok_str.replace("\n", "\\n"))
        m_cells.append(
            f'<td style="background:{bg};padding:4px 6px;border:1px solid #2d333b;'
            f'font-family:monospace;font-size:12px;font-weight:bold;white-space:nowrap;" '
            f'title="Model p={prob:.3f}">{esc}<sup style="opacity:.7">{int(prob*100)}%</sup></td>'
        )
    rows_html.append(
        f'<tr><th style="position:sticky;left:0;background:#161b22;color:#3fb950;padding:4px 8px;'
        f'border:1px solid #2d333b;font-family:monospace;font-size:12px;">OUT</th>{"".join(m_cells)}</tr>'
    )

    header_cells = "".join(
        f'<th style="background:#0d1117;color:#c9d1d9;padding:5px 6px;border:1px solid #2d333b;'
        f'font-family:monospace;font-size:12px;white-space:nowrap;">{_html.escape(t.replace(chr(10), "⏎"))}</th>'
        for t in tokens
    )
    return (
        f'<div style="background:#0d1117;color:#e6edf3;padding:12px;border-radius:8px;overflow-x:auto;">'
        f'<div style="font-family:sans-serif;font-weight:600;margin-bottom:8px;">{_html.escape(title)}</div>'
        f'<table style="border-collapse:collapse;">'
        f'<thead><tr><th style="background:#0d1117;color:#8b949e;padding:5px 8px;border:1px solid #2d333b;">Layer</th>'
        f'{header_cells}</tr></thead><tbody>{"".join(rows_html)}</tbody></table></div>'
    )
