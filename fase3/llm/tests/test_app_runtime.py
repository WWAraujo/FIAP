import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/rag"))
from runtime_server import PatientDocumentChain
from medical_assistant.application.contracts import PatientContext


def test_patient_context_only_changes_prompt_not_retrieval_or_next_request(monkeypatch):
    backend = Mock()
    patient = PatientContext(patient_id="patient_001", idade=44,
                             sinais_vitais=[{"blood_pressure": "123/79"}])
    first = PatientDocumentChain(backend, patient)
    second = PatientDocumentChain(backend, None)
    assert first.config == second.config
    assert first.chunks == second.chunks
    prompt = first.prompt("Pergunta original", [])
    assert "Pergunta: Pergunta original" in prompt
    assert "123/79" in prompt and "patient_001" in prompt
    assert "patient_001" not in second.prompt("Pergunta original", [])
    assert "123/79" not in second.prompt("Pergunta original", [])


def test_patient_context_is_counted_in_budget_without_silent_cut(monkeypatch):
    backend = Mock()
    chain = PatientDocumentChain(backend, PatientContext(patient_id="patient_001"))
    prompts = []
    def count(prompt):
        prompts.append(prompt)
        return 4000
    monkeypatch.setattr(chain, "count_input", count)
    result = chain.prepare("Quais os signatários da portaria de dislipidemia?")
    assert result["error"]["code"] == "context_budget_exceeded"
    assert all("patient_001" in prompt for prompt in prompts)
    backend.generate.assert_not_called()
