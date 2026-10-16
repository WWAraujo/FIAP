import pytest

from medical_assistant.application.contracts import PatientContext
from medical_assistant.tools import TOOLS, execute_tool


def test_tool_call_uses_only_patient_bound_by_application():
    call = {"name": "calcular_imc", "arguments": {}}
    first = PatientContext(patient_id="first", idade=62,
        sinais_vitais=[{"id": 1, "weight": 74, "height": 1.62}])
    second = PatientContext(patient_id="second", idade=40,
        sinais_vitais=[{"id": 2, "weight": 80, "height": 2}])
    result = execute_tool(**call, patient_context=first, query_id="query_first")
    assert "28,20 kg/m²" in result.response
    assert result.query_id == "query_first" and result.requires_validation
    assert result.sources[0].document_id == "prontuario:first"
    next_result = execute_tool(**call, patient_context=second, query_id="query_second")
    assert "20,00 kg/m²" in next_result.response
    assert next_result.sources[0].document_id == "prontuario:second"


@pytest.mark.parametrize("arguments", [{"patient_id": "other"}, {"weight": 90}, [], None])
def test_tool_rejects_model_supplied_patient_or_measurements(arguments):
    with pytest.raises(ValueError, match="objeto vazio"):
        execute_tool("calcular_imc", arguments, patient_context=None, query_id="test")


def test_unknown_tool_is_rejected():
    with pytest.raises(ValueError, match="não permitida"):
        execute_tool("buscar_documentos", {}, patient_context=None, query_id="test")


def test_missing_patient_returns_explicit_pending_message():
    result = execute_tool("calcular_imc", {}, patient_context=None, query_id="test")
    assert "Selecione um paciente" in result.response and not result.sources


def test_only_bmi_is_exposed_without_model_arguments():
    assert len(TOOLS) == 1
    function = TOOLS[0]["function"]
    assert function["name"] == "calcular_imc"
    assert function["parameters"] == {
        "type": "object", "properties": {}, "additionalProperties": False}
