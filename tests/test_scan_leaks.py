"""Leak scanner: number spellings and default scope (no NDA data needed)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _scanner():
    spec = importlib.util.spec_from_file_location("scan_leaks", ROOT / "public" / "scan_leaks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_numbers_are_matched_in_every_spelling():
    v = _scanner()._num_variants(123457)
    assert {"123457", "123,457", "123 457", "123_457"} <= v  # 123_457 = Python / YAML literal


def test_default_scope_covers_config_docs_tests_but_not_ignored_files():
    files = {p.relative_to(ROOT).as_posix() for p in _scanner().pushable()}
    assert {"config/business.yaml", "docs/business.md", "src/business/suppliers.py", "README.md"} <= files
    assert "config/suppliers.local.yaml" not in files and not any(f.startswith("data/raw") for f in files)
