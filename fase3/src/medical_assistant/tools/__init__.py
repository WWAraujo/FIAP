from medical_assistant.application.contracts import AssistantResponse, PatientContext
from medical_assistant.tools.bmi import CALCULAR_IMC_SCHEMA, calcular_imc

TOOLS = [CALCULAR_IMC_SCHEMA]


def execute_tool(name: str, arguments: dict, *, patient_context: PatientContext | None,
                 query_id: str) -> AssistantResponse:
    if name != "calcular_imc":
        raise ValueError(f"Ferramenta não permitida: {name}")
    if not isinstance(arguments, dict) or arguments:
        raise ValueError("calcular_imc aceita somente um objeto vazio de argumentos.")
    return calcular_imc(patient_context, query_id)
