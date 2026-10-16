from unittest.mock import Mock

import pytest

from medical_assistant.application.assistant import MedicalAssistantLangChain
from medical_assistant.application.contracts import AssistantRequest, PatientContext
from medical_assistant.clinical.patient_queries import bmi_query


def context(weight=74, height=1.62, age=62):
    return PatientContext(patient_id="sim_001", idade=age,
        sinais_vitais=[{"id": 4, "weight": weight, "height": height, "recorded_at": "2026-10-13T09:00:00"}])


def test_original_question_calculates_from_patient_without_llm(tmp_path):
    assistant = MedicalAssistantLangChain()
    assistant.audit_logger.audit_file = tmp_path / "audit.log"
    assistant.backend.invoke = Mock(side_effect=AssertionError("Must not invoke LLM"))
    result = assistant.process_query("esse paciente está a cima do peso?", context())
    assert "28,20 kg/m²" in result["response"]
    assert "60 anos ou mais" in result["response"] and "Sim," in result["response"]
    assert "SISVAN" in result["response"] and "não um diagnóstico" in result["response"]
    assert result["sources"][0]["id"] == "sqlite:vital_signs:4"
    assert result["model_usage"] is None and result["requires_validation"]
    assistant.backend.invoke.assert_not_called()


@pytest.mark.parametrize("age,weight,expected", [(40, 100, "sobrepeso"), (40, 120, "faixa de obesidade"),
    (40, 99.99, "faixa adequada"), (62, 108, "sobrepeso"), (62, 107.99, "faixa adequada"),
    (62, 88, "faixa adequada"), (62, 87.99, "baixo peso"), (19, 100, "Não apliquei classificação"),
    (None, 100, "Não apliquei classificação")])
def test_classification_uses_age_and_unrounded_bmi(age, weight, expected):
    result = bmi_query(AssistantRequest(query="Qual o IMC desse paciente?", patient_context=context(weight, 2, age)), "test")
    assert expected in result.response


@pytest.mark.parametrize("weight,height", [(None, 1.62), (74, None), (74, 0), (74, 162),
                                         (float("nan"), 1.62), (74, float("inf")), (-1, 1.62)])
def test_invalid_latest_measurement_never_uses_older_values(weight, height):
    patient = context(weight, height)
    patient.sinais_vitais.append({"id": 3, "weight": 74, "height": 1.62, "recorded_at": "2026-10-01"})
    result = bmi_query(AssistantRequest(query="Esse paciente está acima do peso?", patient_context=patient), "test")
    assert "não tem peso e altura válidos" in result.response
    assert result.sources == []


def test_missing_patient_and_measurements_and_unrelated_question():
    assert "Selecione" in bmi_query(AssistantRequest(query="Qual o IMC?"), "test").response
    request = AssistantRequest(query="Qual o IMC?", patient_context=PatientContext(patient_id="missing"))
    assert "Não há medição" in bmi_query(request, "test").response
    assert bmi_query(AssistantRequest(query="Qual protocolo de tratamento do sobrepeso?"), "test") is None
