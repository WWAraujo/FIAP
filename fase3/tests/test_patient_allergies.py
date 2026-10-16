from unittest.mock import Mock

import pytest

from medical_assistant.application.assistant import MedicalAssistantLangChain
from medical_assistant.application.contracts import PatientContext
from medical_assistant.clinical.patient_database import PatientDatabase
from medical_assistant.clinical.seed_patients import seed_patients


@pytest.fixture
def patients(tmp_path):
    db = PatientDatabase(tmp_path / "patients.db")
    seed_patients(db)
    try:
        yield db
    finally:
        db.close()


@pytest.mark.parametrize("patient_id,expected", [
    ("sim_002", "Alergia a penicilina"),
    ("sim_004", "Sem alergias conhecidas"),
    ("sim_009", "Alergias não informadas"),
])
def test_patient_allergy_uses_records_without_model(patients, tmp_path, patient_id, expected):
    assistant = MedicalAssistantLangChain()
    assistant.audit_logger.audit_file = tmp_path / "audit.log"
    assistant.backend.invoke = Mock(side_effect=AssertionError("Must not invoke LLM"))
    context = PatientContext.from_database(patients, patient_id)
    result = assistant.process_query("Esse paciente tem alguma alergia?", context)
    assert expected in result["response"]
    assert result["model_usage"] is None
    source = result["sources"][0]
    assert source["source_type"] == "prontuario" and patient_id in source["document_id"]
    assert source["text"][source["char_start"]:source["char_end"]] == source["quoted_text"]
    assert result["requires_validation"]
    assistant.backend.invoke.assert_not_called()


def test_patient_missing_and_absent_records_do_not_claim_no_allergy(patients, tmp_path):
    assistant = MedicalAssistantLangChain()
    assistant.audit_logger.audit_file = tmp_path / "audit.log"
    assistant.backend.invoke = Mock(side_effect=AssertionError("Must not invoke LLM"))
    assert "Selecione" in assistant.process_query("Esse paciente tem alguma alergia?")["response"]
    result = assistant.process_query("Quais são as alergias desse paciente?",
                                    PatientContext.from_database(patients, "patient_001"))
    assert "não confirma ausência" in result["response"]
    assert result["sources"] == []


def test_allergy_protocol_still_uses_documentary_route(patients):
    from medical_assistant.application.contracts import AssistantRequest
    from medical_assistant.clinical.patient_queries import allergy_query
    request = AssistantRequest(query="Qual o protocolo para alergia a penicilina?",
                               patient_context=PatientContext.from_database(patients, "sim_002"))
    assert allergy_query(request, "test") is None
