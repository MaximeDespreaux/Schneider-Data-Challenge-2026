"""Supplier names stay in a git-ignored file: the tracked config holds none, and a missing file fails clearly."""

import pytest
import yaml

from src.business import suppliers
from src.business.config import CONFIG, load_business_config


def test_tracked_config_has_no_supplier():
    tracked = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["suppliers"]
    assert not {"companies", "expected_keur", "expected_rule"} & set(tracked)


def test_local_file_is_merged_and_missing_file_is_explained(tmp_path):
    p = tmp_path / "suppliers.local.yaml"
    cfg = load_business_config(suppliers_path=p)
    assert cfg["suppliers"]["companies"] == []
    with pytest.raises(FileNotFoundError, match="suppliers.example.yaml"):
        suppliers.load_supplier_financials(cfg)
    p.write_text("companies:\n  - {file_keyword: A, name: A}\nexpected_keur: {A: 1}\n", encoding="utf-8")
    sc = load_business_config(suppliers_path=p)["suppliers"]
    assert sc["companies"][0]["name"] == "A" and sc["expected_keur"] == {"A": 1} and sc["expected_rule"] == {}
