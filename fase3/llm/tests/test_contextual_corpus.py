import importlib.util
import json
import random
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "src/finetuning/scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_alert_includes_both_active_and_inactive_cases():
    generator = load("prepare_contextual_corpus")
    rng = random.Random(42)
    seen = set()
    for i in range(100):
        row = generator.example("alerta", i, "validation", rng)
        answer = json.loads(row["output"])
        active = answer["dados"]["alerta_ativo"]
        seen.add(active)
        assert answer["status"] == ("alerta" if active else "informativo")
        assert answer["pendencias"] == (["atendimento imediato"] if active else [])
    assert seen == {True, False}


def test_scoring_does_not_accept_explanatory_text_or_extra_facts():
    score = load("score_contextual_results")
    assert score.parse_response('Texto antes: {"status":"modelo"}') is None
    assert score.parse_response('```json\n{"status":"modelo"}\n```') == {"status": "modelo"}
    assert score.parse_response('{"validacao_humana":"true"}') != {"validacao_humana": True}
    assert not score.matches({"validacao_humana": 1}, {"validacao_humana": True})
    assert score.matches({"a": 1, "b": None}, {"b": None, "a": 1})


def test_split_domain_partitions_and_prompts_do_not_overlap():
    root = Path(__file__).resolve().parents[1] / "data/contextual_v1/data/splits"
    manifest = json.loads((root / "manifest.json").read_text())
    previous_prompts, previous_ids, previous_domains = set(), set(), set()
    for split in ("train", "validation", "test"):
        rows = [json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines()]
        prompts = {r["instruction"] + r["input"] for r in rows}
        ids = {r["source_id"] for r in rows}
        domains = set(manifest["domain_partition"][split])
        assert len(prompts) == len(rows)
        assert not (prompts & previous_prompts or ids & previous_ids or domains & previous_domains)
        previous_prompts.update(prompts)
        previous_ids.update(ids)
        previous_domains.update(domains)
