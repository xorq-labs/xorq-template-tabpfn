"""A function outside expr.py, imported by scripts/src_import_expr.py's UDF."""

from __future__ import annotations


def label(x: float) -> str:
    return "big" if x >= 2 else "small"
