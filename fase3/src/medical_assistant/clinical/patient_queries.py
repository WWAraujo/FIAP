from datetime import datetime
import re
import unicodedata

from medical_assistant.application.contracts import AssistantResponse, Source
from medical_assistant.tools import execute_tool


def bmi_query(request, query_id):
    question = "".join(c for c in unicodedata.normalize("NFD", request.query.lower())
                       if unicodedata.category(c) != "Mn")
    question = " ".join(question.strip(" ?!.\n").split()).replace("a cima", "acima")
    patterns = (
        r"(?:(?:(?:esse|este|essa|esta|o|a) )?paciente |ele |ela )?(?:esta|tem) (?:acima do peso|sobrepeso|excesso de peso)",
        r"(?:qual (?:e )?(?:o )?|calcule (?:o )?)?imc(?: (?:desse|deste|dessa|desta|do|da) paciente| dele| dela)?",
    )
    if not any(re.fullmatch(pattern, question) for pattern in patterns):
        return None
    return execute_tool("calcular_imc", {}, patient_context=request.patient_context, query_id=query_id)


def allergy_query(request, query_id):
    question = "".join(c for c in unicodedata.normalize("NFD", request.query.lower())
                       if unicodedata.category(c) != "Mn")
    question = " ".join(question.strip(" ?!.\n").split())
    patterns = (
        r"(?:(?:(?:esse|este|essa|esta|o|a) )?paciente |ele |ela )?(?:tem|possui) (?:alguma |algum tipo de )?alergias?",
        r"(?:quais|qual|que) (?:sao |e )?(?:as |a )?alergias?(?: (?:desse|deste|dessa|desta|do|da) paciente| dele| dela)?",
    )
    if not any(re.fullmatch(pattern, question) for pattern in patterns):
        return None
    context = request.patient_context
    if context is None:
        return AssistantResponse(query_id=query_id, response="Selecione um paciente para consultar as alergias registradas.")
    records = [item for item in context.historico
               if str(item.get("condition", "")).lower().startswith(("alergia", "sem alergia"))]
    if not records:
        return AssistantResponse(query_id=query_id,
            response="Não há registro de alergias no prontuário fornecido. Isso não confirma ausência de alergias.")
    sources = []
    for item in records:
        quote = item["condition"]
        if item.get("notes"):
            quote += ". " + item["notes"]
        sources.append(Source(
            source_type="prontuario", id=f"sqlite:medical_history:{item['id']}",
            document_id=f"prontuario:{context.patient_id}",
            title=f"Prontuário {context.patient_id} — alergias",
            version="Consulta em " + datetime.now().isoformat(timespec="seconds"),
            section_id="medical_history", section="Histórico médico — alergias",
            text=quote, quoted_text=quote, char_start=0, char_end=len(quote),
            evidence_id=f"medical_history:{item['id']}",
        ))
    return AssistantResponse(query_id=query_id,
        response="Consta no prontuário:\n\n" + "\n\n".join(source.quoted_text for source in sources),
        sources=sources)
