"""Controles de contexto e exposição dos desafios, sem modelo/rede."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/rag"))
spec = importlib.util.spec_from_file_location("rag_validation", ROOT / "src/rag/evaluate_validation.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class Backend:
    def __init__(self):
        self.identity = {"system_prompt": "Sistema"}
        self.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: {"input_ids": [0] * 20})
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        answer = dict(status="pendente", dados={}, pendencias=[], fonte=None, validacao_humana=True)
        return dict(response=json.dumps(answer), output_limit_reached=False)


def test_frozen_validation_has_all_categories_and_conditions():
    cases, _, specification = evaluation.check_frozen_inputs()
    assert len(cases) == 40 and all(c["split"] == "validation" for c in cases)
    assert set(evaluation.CONDITIONS) == set(specification["controls"])


def test_without_rag_has_no_documents_and_still_generates():
    cases, _, _ = evaluation.check_frozen_inputs()
    backend = Backend()
    result = evaluation.EvaluationChain(backend, cases, False).invoke(cases[0]["question"])
    assert len(backend.calls) == 1
    assert result["provided"] == result["retrieved"] == []
    assert "Documento R1" not in backend.calls[0]
    assert result["error"]["code"] == "no_evidence"


def test_fixtures_are_supplied_without_indexing_or_gold_leak():
    cases, _, _ = evaluation.check_frozen_inputs()
    case = next(c for c in cases if c["category"] == "version_conflict")
    case["expected_behavior"] = "SECRET-GOLD-NEVER-SUPPLY"
    backend = Backend()
    chain = evaluation.EvaluationChain(backend, cases, True)
    state = chain.prepare(case["question"])
    fixture_ids = {f["id"] for f in case["context_fixture"]}
    assert fixture_ids <= set(state["fixture_ids_provided"])
    assert not fixture_ids & {c["id"] for c in chain.chunks}
    assert "SECRET-GOLD-NEVER-SUPPLY" not in state["prompt"]
    assert all(f["text"] in state["prompt"] for f in case["context_fixture"])


def test_unexposed_challenge_is_not_marked_successful():
    cases, _, _ = evaluation.check_frozen_inputs()
    case = next(c for c in cases if c["category"] == "document_instruction_attack")
    backend = Backend()
    backend.tokenizer.apply_chat_template = lambda *a, **k: {"input_ids": [0] * 500}
    result = evaluation.EvaluationChain(backend, cases, True).invoke(case["question"])
    observations = evaluation.automatic_observations(case, result)
    assert observations["challenge_fully_exposed"] is False
    assert observations["generation_performed"] is False
    assert observations["safety_grade"] == "pending_documentary_review"
    assert not backend.calls
