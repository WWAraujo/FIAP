"""Integridade do corpus RAG-02 e da avaliação reservada, sem rede/LLM."""

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/rag_pilot_v2"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_and_official_admission_scope():
    spec = importlib.util.spec_from_file_location("rag02_manifest", ROOT / "src/rag/prepare_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    catalog = read(CORPUS / "catalog.json")
    manifest = read(CORPUS / "manifest.json")
    assert module.build_manifest(catalog) == manifest
    assert manifest["counts"] == dict(documents=10, available=10, pending=0,
                                      eligible_documents=9, sections=17, eligible_sections=15)
    originals = read(ROOT / "data/rag_pilot_v1/catalog.json")
    assert catalog["documents"][:7] == originals["documents"][:7]
    official = catalog["documents"][7:]
    assert len(official) == 3
    for doc in official:
        assert doc["extraction"]["admitted_pages"] == [1]
        assert len(doc["sections"]) == 1
        assert doc["sections"][0]["page"] == 1
        assert doc["license"]["status"] == "checked"
        assert doc["clinical_review"] is None
        assert "Art." in doc["sections"][0]["text"]


def test_reserved_questions_counts_references_and_freeze():
    manifest = read(CORPUS / "manifest.json")
    sections = {s["id"]: s for s in manifest["sections"]}
    spec = read(CORPUS / "evaluation/evaluation_spec.json")
    assert spec["manifest_sha256"] == sha(CORPUS / "manifest.json")
    questions, ids, fixtures = set(), set(), set()
    corpus_text = "\n".join(s["text"] for s in sections.values())
    training_inputs = [p for name in ("contextual_v1", "factual_v2", "factual_v3")
                       for p in (ROOT / "data" / name).glob("*.jsonl")]
    historical_text = "\n".join(p.read_text(encoding="utf-8") for p in training_inputs)
    for split in ("validation", "test"):
        path = CORPUS / "evaluation" / f"{split}.jsonl"
        assert sha(path) == spec["splits_sha256"][split]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 40
        assert Counter(r["category"] for r in rows) == spec["splits"][split]["categories"]
        for row in rows:
            assert row["split"] == split
            assert row["id"] not in ids
            assert row["question"] not in questions
            ids.add(row["id"])
            questions.add(row["question"])
            assert row["question"] not in corpus_text
            assert row["question"] not in historical_text
            assert row["expected_behavior"]
            assert row["review_status"] == "documentary_checked_by_ai"
            assert row["clinical_validation"] is False
            assert set(row["expected_section_ids"]) == {e["section_id"] for e in row["reference_evidence"]}
            for evidence in row["reference_evidence"]:
                section = sections[evidence["section_id"]]
                assert section["retrieval_eligible"]
                assert evidence["quote"] == section["text"]
                assert evidence["document_id"] == section["document_id"]
                assert evidence["version"] == section["document_version"]
                assert evidence["page"] == section["page"]
            if row["category"] == "answerable":
                assert row["reference_evidence"] and "context_fixture" not in row
            if row["category"] == "missing_evidence":
                assert not row["reference_evidence"]
            if row["category"] in ("version_conflict", "document_instruction_attack"):
                assert row["context_fixture"]
                for fixture in row["context_fixture"]:
                    assert fixture["id"] not in sections and fixture["id"] not in fixtures
                    fixtures.add(fixture["id"])
                if row["category"] == "version_conflict":
                    a, b = row["context_fixture"]
                    assert a["document_id"] == b["document_id"]
                    assert a["version"] != b["version"] and a["text"] != b["text"]
                    assert a["validity"] == b["validity"] == "unverified"
    assert len(ids) == len(questions) == 80
    assert not manifest["index_built"] and not manifest["clinical_validation"]


def test_snapshot_provenance_and_local_bytes_when_present():
    catalog = read(CORPUS / "catalog.json")
    sources = read(CORPUS / "sources.json")["sources"]
    for source, doc in zip(sources, catalog["documents"][7:], strict=True):
        assert source["document_id"] == doc["id"]
        assert source["sha256"] == doc["snapshot"]["sha256"]
        assert source["download_url"].startswith("https://www.gov.br/")
        pdf = CORPUS / source["path"]
        text = CORPUS / source["extraction"]["path"]
        if pdf.exists():
            assert pdf.read_bytes().startswith(b"%PDF-")
            assert sha(pdf) == source["sha256"]
            assert pdf.stat().st_size == source["bytes"]
        if text.exists():
            assert sha(text) == source["extraction"]["sha256"]
            pages = text.read_text(encoding="utf-8").split("\f")
            assert len(pages) - 1 == source["extraction"]["pages"]
            assert pages[0] == doc["sections"][0]["text"]
