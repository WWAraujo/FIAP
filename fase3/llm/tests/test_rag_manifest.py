import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src/rag/prepare_manifest.py"
spec = importlib.util.spec_from_file_location("prepare_manifest", SCRIPT)
manifest_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manifest_module)


def catalog():
    return json.loads((ROOT / "data/rag_pilot_v1/catalog.json").read_text(encoding="utf-8"))


def test_pending_sources_and_superseded_version_are_not_eligible():
    result = manifest_module.build_manifest(catalog())
    assert result["counts"] == {"documents": 10, "available": 7, "pending": 3,
                                "eligible_documents": 6, "sections": 14, "eligible_sections": 12}
    for document in result["documents"]:
        if document["admission_status"] == "pending":
            assert document["content_sha256"] is None
            assert not document["retrieval_eligible"]
    old_sections = [s for s in result["sections"]
                    if s["document_id"] == "SIM-RAG-PENDENCIAS" and s["document_version"] == "1.0"]
    assert old_sections and all(not s["retrieval_eligible"] for s in old_sections)


def test_hashes_preserve_text_and_detect_changes_without_changing_section_identity():
    source = catalog()
    before = manifest_module.build_manifest(source)
    assert before == manifest_module.build_manifest(copy.deepcopy(source))
    source["documents"][3]["sections"][0]["text"] += " Registro de conferência adicional."
    after = manifest_module.build_manifest(source)
    assert before["catalog_sha256"] != after["catalog_sha256"]
    assert before["documents"][3]["content_sha256"] != after["documents"][3]["content_sha256"]
    for section in before["sections"]:
        assert section["text_sha256"] == hashlib.sha256(section["text"].encode("utf-8")).hexdigest()
    assert [s["id"] for s in before["sections"]] == [s["id"] for s in after["sections"]]


@pytest.mark.parametrize("invalid", ["duplicate_document", "duplicate_section", "pending_text", "pending_license"])
def test_invalid_inventory_is_rejected(invalid):
    source = catalog()
    if invalid == "duplicate_document":
        source["documents"].append(copy.deepcopy(source["documents"][0]))
    elif invalid == "duplicate_section":
        source["documents"][0]["sections"].append(copy.deepcopy(source["documents"][0]["sections"][0]))
    elif invalid == "pending_text":
        source["documents"][-1]["sections"] = source["documents"][0]["sections"]
    else:
        source["documents"][0]["license"]["status"] = "pending"
    with pytest.raises(ValueError):
        manifest_module.build_manifest(source)


def test_saved_manifest_matches_catalog_and_evaluation_spec_is_separate():
    root = ROOT / "data/rag_pilot_v1"
    result = manifest_module.build_manifest(catalog())
    assert json.loads((root / "manifest.json").read_text()) == result
    assert not {"question", "reference_quote", "expected_behavior"} & result.keys()
    evaluation = json.loads((root / "evaluation_spec.json").read_text())
    assert evaluation["status"] == "criteria_registered_questions_not_created"
    for split in evaluation["splits"].values():
        assert sum(split["categories"].values()) == split["count"] == 40


def test_cli_does_not_overwrite_evidence_and_check_detects_tampering(tmp_path):
    output = tmp_path / "manifest.json"
    command = [sys.executable, "-B", str(SCRIPT), "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    original = output.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert output.read_bytes() == original
    subprocess.run(command + ["--check"], check=True, capture_output=True)
    altered = json.loads(original)
    altered["sections"][0]["text"] = "Texto alterado sem atualizar a versão."
    output.write_text(json.dumps(altered))
    assert subprocess.run(command + ["--check"], capture_output=True).returncode != 0
