"""Consulta livre exige recibo de uma versão avaliada, sem consultar QAs."""
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/rag"))
import query_document
from ft11_backend import sha


def receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(query_document, "INDEX", tmp_path)
    decision = tmp_path / "decision.json"
    decision.write_text('{"rag_preparation_accepted": true}')
    (tmp_path / "frozen_config.json").write_text('{}')
    value = {"test_decision_path": str(decision), "test_decision_sha256": sha(decision),
             "runtime_script_hashes": {}, "index_config_sha256": sha(tmp_path / "frozen_config.json"), "packages": {}}
    (tmp_path / "readiness.json").write_text(json.dumps(value))
    return value


def test_free_query_readiness_uses_only_receipt_decision_and_configuration(tmp_path, monkeypatch):
    value = receipt(tmp_path, monkeypatch)
    assert query_document.check_readiness() == value
    assert not (tmp_path / "test.jsonl").exists()


def test_changed_decision_is_rejected_before_loading_model(tmp_path, monkeypatch):
    receipt(tmp_path, monkeypatch)
    (tmp_path / "decision.json").write_text('{"rag_preparation_accepted": false}')
    with pytest.raises(ValueError, match="decisão alterada"):
        query_document.check_readiness()


def test_changed_runtime_module_is_rejected(tmp_path, monkeypatch):
    value = receipt(tmp_path, monkeypatch)
    value["runtime_script_hashes"] = {"document_chain_v3.py": "bad-sha"}
    (tmp_path / "readiness.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Módulo diverge"):
        query_document.check_readiness()
