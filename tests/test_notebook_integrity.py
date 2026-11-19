"""Validate qwen38_jlens_colab.ipynb structure, syntax, and byte-exact sync with qwen38_jlens/."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "qwen38_jlens_colab.ipynb"
PKG_DIR = ROOT / "qwen38_jlens"


def test_notebook_valid_and_in_sync():
    assert NB_PATH.is_file(), f"Missing {NB_PATH}"
    nb = nbformat.read(str(NB_PATH), as_version=4)
    nbformat.validate(nb)

    embedded_modules: dict[str, str] = {}
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        src = cell.source
        if src.startswith("%%writefile "):
            first_line, body = src.split("\n", 1)
            rel_path = first_line[len("%%writefile ") :].strip()
            ast.parse(body, filename=rel_path)
            embedded_modules[rel_path] = body
        else:
            ast.parse(src, filename=f"cell_{idx}")

    expected_files = sorted(f"qwen38_jlens/{p.name}" for p in PKG_DIR.glob("*.py"))
    assert sorted(embedded_modules.keys()) == expected_files
    for rel_path, body in embedded_modules.items():
        disk_text = (ROOT / rel_path).read_text(encoding="utf-8").rstrip() + "\n"
        assert body == disk_text, f"Notebook cell for {rel_path} drifted from disk; re-run build_notebook.py"
