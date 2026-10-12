"""Teste só é liberado com comparação completa, revisão e configuração íntegras."""
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/rag"))
from evaluate_v3 import approved_validation, CONDITIONS
from ft11_backend import sha


def fixture(tmp_path):
    def save(name, value):
        (tmp_path / name).write_text(json.dumps(value))
    frozen = {"version": "v3-fixture"}
    save("chain_config.json", frozen)
    save("protocol.json", {"split": "validation", "partial_probe": False, "case_count": 40,
         "conditions": CONDITIONS, "chain_config_sha256": sha(tmp_path / "chain_config.json")})
    save("documentary_review.json", {"records": [dict(condition=c, id=str(n)) for c in CONDITIONS for n in range(40)]})
    save("reviewed_summary.json", {"conditions": [dict(condition=c, count=40) for c in CONDITIONS]})
    save("decision.json", {"advance_to_reserved_test": True, "documentary_ai_signoff": True, "checks": {"all": True},
         "review_sha256": sha(tmp_path / "documentary_review.json"), "summary_sha256": sha(tmp_path / "reviewed_summary.json")})
    return frozen


def test_complete_unchanged_validation_is_accepted(tmp_path):
    approved_validation(tmp_path, fixture(tmp_path))


def test_partial_probe_is_rejected_even_if_decision_claims_approval(tmp_path):
    frozen = fixture(tmp_path)
    path = tmp_path / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol["partial_probe"] = True
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="teste não lido"):
        approved_validation(tmp_path, frozen)


def test_modified_review_or_configuration_is_rejected(tmp_path):
    frozen = fixture(tmp_path)
    (tmp_path / "documentary_review.json").write_text('{"records": []}')
    with pytest.raises(ValueError, match="teste não lido"):
        approved_validation(tmp_path, frozen)
