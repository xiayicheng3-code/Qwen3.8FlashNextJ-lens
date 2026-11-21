"""Validate qwen38_jlens_colab.ipynb structure, syntax, direct-import cleanliness, and API sync."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import nbformat

import build_notebook
import qwen38_jlens
from qwen38_jlens.hardware import HardwareProfile, OffloadPlan

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "qwen38_jlens_colab.ipynb"


def test_notebook_valid_direct_import_and_in_sync():
    assert NB_PATH.is_file(), f"Missing {NB_PATH}"
    nb = nbformat.read(str(NB_PATH), as_version=4)
    nbformat.validate(nb)

    # Verify notebook on disk matches build_notebook.py deterministically
    expected_nb = build_notebook.build_notebook()
    assert json.loads(NB_PATH.read_text(encoding="utf-8")) == json.loads(
        json.dumps(expected_nb, ensure_ascii=False)
    ), "qwen38_jlens_colab.ipynb drifted from build_notebook.py; re-run build_notebook.py"

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) == 9, f"Expected 9 clean code cells (Sections 1..9), got {len(code_cells)}"

    all_code = "\n".join(c.source for c in code_cells)
    assert "%%writefile" not in all_code, "Notebook should import qwen38_jlens directly instead of %%writefile cells"
    assert build_notebook.REPO_URL in all_code, "Notebook must reference the public GitHub repo URL"

    profile_attrs = set(dir(HardwareProfile)) | set(HardwareProfile.__dataclass_fields__)
    plan_attrs = set(dir(OffloadPlan)) | set(OffloadPlan.__dataclass_fields__)

    imported_from_pkg: set[str] = set()
    for idx, cell in enumerate(code_cells):
        tree = ast.parse(cell.source, filename=f"code_cell_{idx}")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "qwen38_jlens":
                for alias in node.names:
                    imported_from_pkg.add(alias.name)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "profile":
                    assert node.attr in profile_attrs, f"Unknown HardwareProfile attribute: profile.{node.attr}"
                elif node.value.id == "plan":
                    assert node.attr in plan_attrs, f"Unknown OffloadPlan attribute: plan.{node.attr}"

    assert imported_from_pkg, "Notebook must directly import symbols from qwen38_jlens"
    for sym in sorted(imported_from_pkg):
        assert hasattr(qwen38_jlens, sym), f"Notebook imports missing symbol qwen38_jlens.{sym}"
