"""Lacuna válida, orçamento, fontes e conflitos sem consultar teste reservado."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/rag"))
from document_chain_v2 import DocumentChainV2, OUTPUT_TOKENS, TOTAL_TOKENS
from evaluate_validation import check_frozen_inputs, automatic_observations
from evaluate_validation_v2 import EvaluationChainV2


class Backend:
    def __init__(self):
        self.identity = {"system_prompt": "Sistema"}
        self.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: {"input_ids": [0] * 900})
        self.answer = DocumentChainV2.gap_answer()
        self.limited = False
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return {"response": json.dumps(self.answer), "output_limit_reached": self.limited}


def test_explicit_absence_is_a_valid_outcome_without_source():
    result = DocumentChainV2(Backend()).invoke("Identificação do revisor de laudo")
    assert "error" not in result and result["outcome"] == "no_evidence"
    assert not result["sources"] and result["answer"]["fonte"] is None
    assert len(result["provided"]) == 3 and result["input_tokens"] > 512


def test_budget_preserves_whole_chunks_and_excludes_oversized_context():
    backend = Backend()
    backend.tokenizer.apply_chat_template = lambda *a, **k: {"input_ids": [0] * TOTAL_TOKENS}
    result = DocumentChainV2(backend).invoke("Transcrição de resultados")
    assert result["error"]["code"] == "context_budget_exceeded"
    assert result["omitted"] and not result["provided"] and not backend.calls


def test_without_rag_still_calls_model_without_documents():
    cases, _, _ = check_frozen_inputs()
    backend = Backend()
    result = EvaluationChainV2(backend, cases, False).invoke(cases[0]["question"])
    assert len(backend.calls) == 1 and backend.calls[0][1]["max_new_tokens"] == OUTPUT_TOKENS
    assert result["provided"] == result["retrieved"] == [] and "Documento R1" not in backend.calls[0][0]


def test_conflict_records_both_editions_without_selecting_one_or_feeding_gold():
    cases, _, _ = check_frozen_inputs()
    case = next(c for c in cases if c["category"] == "version_conflict")
    case["expected_behavior"] = "GOLD-SECRET"
    backend = Backend()
    chain = EvaluationChainV2(backend, cases, True)
    result = chain.invoke(case["question"])
    assert result["outcome"] == "version_conflict" and "error" not in result
    assert {s["version"] for s in result["sources"]} == {"A", "B"}
    assert result["answer"]["status"] == "pendente" and len(result["answer"]["dados"]["conflito"]) == 2
    assert "GOLD-SECRET" not in backend.calls[0][0]
    assert not set(result["fixture_ids_provided"]) & {c["id"] for c in chain.chunks}
    assert automatic_observations(case, result)["challenge_fully_exposed"] is True


@pytest.mark.parametrize("limited,answer", [(True, None), (False, {"status": "pendente"})])
def test_generation_failure_never_becomes_approved_conflict(limited, answer):
    cases, _, _ = check_frozen_inputs()
    case = next(c for c in cases if c["category"] == "version_conflict")
    backend = Backend()
    backend.limited = limited
    if answer is not None:
        backend.answer = answer
    result = EvaluationChainV2(backend, cases, True).invoke(case["question"])
    assert "error" in result and not result["sources"]


def test_invented_quote_is_rejected_and_raw_is_preserved():
    backend = Backend()
    backend.answer = dict(status="informativo", dados={"trecho": "Responsável: Pessoa Inventada."},
                          pendencias=[], fonte="R1", validacao_humana=True)
    result = DocumentChainV2(backend).invoke("Identificação do revisor de laudo")
    assert result["error"]["code"] == "unsupported_quote" and not result["sources"]
    assert "Pessoa Inventada" in result["generation"]["response"]


def test_all_attack_texts_are_delivered_without_indexing_or_truncation():
    cases, _, _ = check_frozen_inputs()
    backend = Backend()
    chain = EvaluationChainV2(backend, cases, True)
    for case in cases:
        if case["category"] == "document_instruction_attack":
            state = chain.prepare(case["question"])
            assert not state["omitted"]
            assert all(f["text"] in state["prompt"] for f in case["context_fixture"])
            assert automatic_observations(case, state)["challenge_fully_exposed"] is True
