"""Gate do teste reservado: rejeição antes de abrir perguntas ou carregar GPU."""

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/rag"))
from evaluate_reserved import approved_config, REQUIRED_CRITERIA, VALIDATION_RUN


def test_rejected_or_partial_validation_cannot_open_test(tmp_path):
    decision = {"validation_complete": True, "advance_to_reserved_test": True, "partial_probe": True,
                "records_reviewed": 40, "criteria": {key: True for key in REQUIRED_CRITERIA}}
    (tmp_path / "decision.json").write_text(json.dumps(decision))
    with pytest.raises(ValueError, match="antes de abrir o teste"):
        approved_config(tmp_path)


def test_corrupted_validation_artifact_is_rejected(tmp_path):
    decision = {"validation_complete": True, "advance_to_reserved_test": True, "partial_probe": False,
                "records_reviewed": 160, "criteria": {key: True for key in REQUIRED_CRITERIA}, "review_sha256": "wrong"}
    (tmp_path / "decision.json").write_text(json.dumps(decision))
    (tmp_path / "documentary_review.json").write_text('{}')
    with pytest.raises(ValueError, match="Artefato de validação divergente"):
        approved_config(tmp_path)


def test_accepted_frozen_config_without_reading_test_questions():
    assert approved_config(VALIDATION_RUN)["version"] == "document_chain_v2"
