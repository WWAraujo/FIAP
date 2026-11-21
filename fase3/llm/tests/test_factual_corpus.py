import json
import sys
from pathlib import Path
from urllib.parse import urlparse

SCRIPTS = Path(__file__).resolve().parents[1] / "src/finetuning/scripts"
sys.path.insert(0, str(SCRIPTS))

from prepare_factual_corpus import first_sentence, quality_reason


def test_translation_checks_reject_lost_uncertainty_and_changed_numbers():
    english = "Changes in the ABC1 gene can cause a rare inherited disorder."
    faithful = "Alterações no gene ABC1 podem causar uma doença hereditária rara."
    assert quality_reason(english, faithful) is None
    assert quality_reason(english, faithful.replace("podem causar", "causam")) == "qualifier_mismatch"
    assert quality_reason(english, faithful.replace("ABC1", "ABC2")) == "numbers_mismatch"
    assert quality_reason(english, faithful.replace("ABC1", "XYZ1")) == "gene_mismatch"


def test_sentence_boundary_preserves_abbreviated_titles():
    sentence = "Clinical review by Dr. Silva describes a congenital disorder. Additional details follow."
    assert first_sentence(sentence) == "Clinical review by Dr. Silva describes a congenital disorder."


def test_factual_splits_keep_related_diseases_and_prompts_separate():
    root = Path(__file__).resolve().parents[1] / "data/factual_v2"
    manifest = json.loads((root / "data/splits/manifest.json").read_text())
    previous_groups, previous_prompts, previous_ids = set(), set(), set()
    for split in ("train", "validation", "test", "challenge"):
        path = root / "challenge.jsonl" if split == "challenge" else root / f"data/splits/{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        groups = {row["source_group"] for row in rows}
        groups.update(source["source_group"] for row in rows for source in row.get("provenance", {}).get("context_sources", []))
        prompts = {row["instruction"] + row["input"] for row in rows}
        ids = {row["source_id"] for row in rows}
        assert len(rows) == manifest["splits"][split]["count"]
        assert 100 < manifest["splits"][split]["max_tokens"] <= 512
        assert len(prompts) == len(ids) == len(rows)
        assert not groups & previous_groups
        assert not prompts & previous_prompts
        assert not ids & previous_ids
        for row in rows:
            if "provenance" in row:
                provenance = row["provenance"]
                assert urlparse(provenance["url"]).hostname in {"ghr.nlm.nih.gov", "rarediseases.info.nih.gov"}
                assert quality_reason(provenance["primary_sentence"], provenance["curated_sentence"]) is None
                if provenance["context_sources"]:
                    context = json.loads(row["input"])["documentos"]
                    assert {doc["id"] for doc in context} == {doc["id"] for doc in provenance["context_sources"]}
                    for source in provenance["context_sources"]:
                        assert quality_reason(source["primary_sentence"], source["curated_sentence"]) is None
        previous_groups.update(groups)
        previous_prompts.update(prompts)
        previous_ids.update(ids)
