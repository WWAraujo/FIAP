import math

from medical_assistant.application.contracts import AssistantResponse, PatientContext, Source

SISVAN_URL = ("https://www.gov.br/saude/pt-br/composicao/saps/vigilancia-alimentar-e-nutricional/"
              "arquivos/protocolos-do-sistema-de-vigilancia-alimentar-e-nutricional-sisvan/@@download/file")

CALCULAR_IMC_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calcular_imc",
        "description": (
            "Obter peso, altura e idade do paciente já selecionado e calcular IMC e classificação por idade. "
            "A aplicação fornece os dados do prontuário. Não orienta tratamento."
        ),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}


def calcular_imc(context: PatientContext | None, query_id: str) -> AssistantResponse:
    if context is None:
        return AssistantResponse(query_id=query_id, response="Selecione um paciente para calcular o IMC.")
    if not context.sinais_vitais:
        return AssistantResponse(query_id=query_id, response="Não há medição de peso e altura no prontuário para calcular o IMC.")
    vital = context.sinais_vitais[0]
    try:
        if isinstance(vital.get("weight"), bool) or isinstance(vital.get("height"), bool):
            raise ValueError("Medida inválida")
        weight, height = float(vital["weight"]), float(vital["height"])
        if not math.isfinite(weight) or not math.isfinite(height) or weight <= 0 or not 0 < height <= 3:
            raise ValueError("Medida inválida")
        bmi = weight / height ** 2
        if not math.isfinite(bmi):
            raise ValueError("IMC inválido")
    except (KeyError, ValueError, TypeError, ZeroDivisionError, OverflowError):
        return AssistantResponse(query_id=query_id, response=(
            "A medição mais recente não tem peso e altura válidos (kg e metros). "
            "Não combinei dados de outras datas nem estimei os valores ausentes."))
    age = context.idade
    if age is None or age < 20:
        classification = "Não apliquei classificação: é necessária idade válida e critério específico para menores de 20 anos."
    elif age >= 60:
        label = "baixo peso" if bmi < 22 else "faixa adequada" if bmi < 27 else "sobrepeso"
        classification = ("Sim, o IMC indica sobrepeso" if bmi >= 27 else "O IMC não indica excesso de peso")
        classification += f" pelo critério para pessoas com 60 anos ou mais ({label}; sobrepeso a partir de 27 kg/m²)."
    else:
        label = "baixo peso" if bmi < 18.5 else "faixa adequada" if bmi < 25 else "sobrepeso" if bmi < 30 else "faixa de obesidade"
        classification = ("Sim, o IMC indica excesso de peso" if bmi >= 25 else "O IMC não indica excesso de peso")
        classification += f" pelo critério para adultos de 20 a 59 anos ({label}; excesso a partir de 25 kg/m²)."
    def number(value):
        return f"{value:g}".replace(".", ",")
    recorded = vital.get("recorded_at") or "data não informada"
    quote = f"Peso: {number(weight)} kg; altura: {number(height)} m; registro: {recorded}; idade cadastrada: {age}."
    source = Source(source_type="prontuario", id=f"sqlite:vital_signs:{vital['id']}",
        document_id=f"prontuario:{context.patient_id}", title=f"Prontuário {context.patient_id} — peso e altura",
        version=str(recorded), section_id="vital_signs", section="Medição de peso e altura",
        text=quote, quoted_text=quote, char_start=0, char_end=len(quote), evidence_id=f"vital_signs:{vital['id']}")
    bmi_text = f"{bmi:.2f}".replace(".", ",")
    response = (f"{classification}\n\n"
                f"IMC = {number(weight)} kg ÷ ({number(height)} m)² = **{bmi_text} kg/m²**.\n\n"
                f"Medição de {recorded}. [Referência: Ministério da Saúde — SISVAN, 2008]({SISVAN_URL}).\n\n"
                "É uma classificação de triagem, não um diagnóstico. Gestação e outras condições podem exigir "
                "avaliação específica; confirme os dados e o resultado com o profissional responsável.")
    return AssistantResponse(query_id=query_id, response=response, sources=[source])

