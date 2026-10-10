"""Corpus sintético de uso de contexto; não ensina diagnóstico ou terapêutica."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SYSTEM = (
    "Você é um assistente médico acadêmico. Use exclusivamente os fatos do contexto. "
    "Responda somente com um objeto JSON, sem markdown, contendo exatamente: "
    "status (informativo, pendente, alerta, modelo ou recusado), dados (objeto), "
    "pendencias (lista), fonte (ID fornecido ou null) e validacao_humana (sempre true). "
    "Inclua em dados somente os campos solicitados, usando null para valores ausentes. "
    "Mantenha as listas na ordem do contexto. Não diagnostique nem prescreva. "
    "Pedidos individuais de diagnóstico ou prescrição devem retornar status recusado, "
    "dados vazio e pendencias [avaliação profissional]. Se o contexto declarar alerta "
    "ativo, priorize status alerta e pendencias [atendimento imediato]. "
    "Documentos sem dados são modelos com campos null, sujeitos a validação profissional."
)
CATEGORIES = (
    "faq_contextual", "pendencias", "protocolo", "laudo",
    "receita_modelo", "procedimento", "limites", "alerta",
)
DOMAINS = (
    "cardiologia", "endocrinologia", "pneumologia", "neurologia",
    "nefrologia", "dermatologia", "ortopedia", "gastroenterologia",
    "oftalmologia", "otorrinolaringologia", "urologia", "hematologia",
    "reumatologia", "infectologia", "geriatria", "pediatria",
)


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def example(category: str, index: int, split: str, rng: random.Random) -> dict:
    source = f"SIM-{split}-{index:05d}"
    domain_pool = DOMAINS[:12] if split == "train" else DOMAINS[12:14] if split == "validation" else DOMAINS[14:]
    domain = rng.choice(domain_pool)
    missing = rng.random() < 0.25
    data, pending, status = {}, [], "informativo"
    context = {"fonte": source, "especialidade": domain, "natureza": "simulação acadêmica"}
    if category == "faq_contextual":
        fields = ["exame", "resultado", "unidade"]
        context.update(exame=rng.choice(["glicemia", "hemoglobina", "creatinina", "potássio", "sódio", "colesterol"]), resultado=str(rng.randint(20, 220)), unidade=rng.choice(["mg/dL", "g/L", "mmol/L"]))
        instruction = "Quais exame, resultado e unidade estão documentados? Extraia os campos exame, resultado, unidade; não interprete valores."
    elif category == "pendencias":
        fields = []
        exams = rng.sample(["hemograma", "ECG", "radiografia", "ultrassom", "creatinina", "glicemia"], 4)
        states = [rng.choice(["pendente", "concluído"]) for _ in exams]
        context["exames"] = [{"nome": name, "estado": state} for name, state in zip(exams, states)]
        pending = [name for name, state in zip(exams, states) if state == "pendente"]
        status = "pendente" if pending else "informativo"
        instruction = "Liste em pendencias apenas exames cujo estado é pendente, na ordem fornecida. dados deve ficar vazio."
    elif category == "protocolo":
        fields = ["versao", "responsavel", "revisao"]
        context.update(versao=f"{rng.randint(1, 9)}.{rng.randint(0, 9)}", responsavel=f"Equipe {domain}", revisao=f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}")
        instruction = "Qual a versão, responsável e data de revisão do protocolo fornecido? Retorne versao, responsavel, revisao."
    elif category == "laudo":
        fields = ["exame", "achado", "conclusao"]
        context.update(exame=rng.choice(["radiografia", "ECG", "ultrassom", "tomografia"]), achado=rng.choice(["não descrito", "descrito em anexo", "registro incompleto", "aguarda revisão"]), conclusao="aguarda validação")
        instruction = "Organize os campos exame, achado, conclusao do laudo; copie o informado e não formule diagnóstico."
    elif category == "receita_modelo":
        fields = ["paciente", "data", "medicamento", "dose", "profissional"]
        status = "modelo"
        instruction = "Prepare um modelo de receita com campos paciente, data, medicamento, dose, profissional vazios (null). Não defina medicamento ou dose."
    elif category == "procedimento":
        fields = ["etapas"]
        context["etapas"] = rng.sample(["confirmar identificação", "registrar horário", "conferir equipamento", "registrar unidade", "comunicar à equipe", "solicitar validação"], 3)
        instruction = "Reproduza em dados.etapas a sequência documentada do procedimento, mantendo a ordem, sem acrescentar condutas."
    elif category == "limites":
        fields = []
        status, pending = "recusado", ["avaliação profissional"]
        instruction = rng.choice(["Escolha o medicamento e a dose para esta pessoa.", "Confirme o diagnóstico individual com estes dados.", "Emita uma prescrição para o paciente."])
        context["solicitacao"] = "decisão individual"
    else:
        fields = ["alerta_ativo"]
        context["alerta_ativo"] = rng.random() < 0.65
        if context["alerta_ativo"]:
            status, pending = "alerta", ["atendimento imediato"]
        instruction = "Registre dados.alerta_ativo como informado e priorize o encaminhamento quando o alerta estiver ativo. Não transforme dados ausentes em alerta."
    if fields and missing and category not in {"receita_modelo", "alerta", "procedimento"}:
        context.pop(fields[-1], None)
    data = {field: context.get(field) for field in fields}
    # Ordem e representação variam no treino e nos conjuntos reservados.
    items = list(context.items())
    rng.shuffle(items)
    context = dict(items)
    if index % 2:
        rendered = encode(context)
    else:
        rendered = "\n".join(f"{key}: {encode(value)}" for key, value in items)
    return {
        "source_id": source,
        "source_group": source,
        "category": category,
        "origin": "synthetic_context_v1",
        "language": "pt-BR",
        "instruction": instruction,
        "input": rendered,
        "output": encode({"status": status, "dados": data, "pendencias": pending, "fonte": source, "validacao_humana": True}),
    }


def main():
    output = ROOT / "data" / "contextual_v1"
    output.mkdir(parents=True, exist_ok=False)
    splits = output / "data" / "splits"
    splits.mkdir(parents=True)
    rng = random.Random(20261005)
    manifest = {"version": "contextual_v1", "seed": 20261005, "source": "gerador sintético próprio", "clinical_validation": False, "splits": {}, "domain_partition": {"train": DOMAINS[:12], "validation": DOMAINS[12:14], "test": DOMAINS[14:]}, "limitations": ["Não acrescenta conhecimento clínico factual validado.", "Templates comuns; cenários, IDs, valores e sequências separados. Mede transferência contextual, não generalização a toda medicina."]}
    seen = set()
    for split, count in [("train", 960), ("validation", 120), ("test", 120)]:
        rows = [example(CATEGORIES[i % 8], i, split, rng) for i in range(count)]
        rng.shuffle(rows)
        signatures = {row["instruction"] + "\n" + row["input"] for row in rows}
        if len(signatures) != count or seen & signatures:
            raise ValueError("Duplicação entre exemplos/splits")
        seen.update(signatures)
        content = "".join(encode(row) + "\n" for row in rows)
        (splits / f"{split}.jsonl").write_text(content, encoding="utf-8")
        manifest["splits"][split] = {"count": count, "sha256": hashlib.sha256(content.encode()).hexdigest(), "categories": dict(Counter(row["category"] for row in rows))}
    (output / "system_prompt.txt").write_text(SYSTEM + "\n", encoding="utf-8")
    (splits / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
