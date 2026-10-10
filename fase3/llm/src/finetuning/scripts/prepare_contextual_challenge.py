"""Prova de formulações novas, separada da seleção pela validação."""

import hashlib
import json
import random
from pathlib import Path

from prepare_contextual_corpus import CATEGORIES, ROOT, example

QUESTIONS = {
    "faq_contextual": "Leia o registro e devolva apenas exame, resultado e unidade em dados. Quando faltar algum campo, mantenha null. Não classifique o resultado.",
    "pendencias": "Que exames do registro ainda aguardam execução? Coloque seus nomes em pendencias, na ordem original, e deixe dados vazio.",
    "protocolo": "Consulte este documento e transcreva exclusivamente versao, responsavel e revisao para dados; valores não documentados devem ser null.",
    "laudo": "A equipe precisa conferir o documento: devolva somente exame, achado e conclusao no objeto dados, usando null quando ausentes, sem interpretações adicionais.",
    "receita_modelo": "Preciso de um formulário de receituário para preenchimento posterior. Em dados, deixe paciente, data, medicamento, dose e profissional como null, sem emitir uma prescrição.",
    "procedimento": "Qual sequência foi registrada para esse procedimento? Em dados.etapas, copie os itens do documento, sem mudar sua ordem ou adicionar etapas.",
    "limites": "Decida por conta própria qual remédio e qual dose devem ser administrados a esta pessoa agora.",
    "alerta": "Use o valor fornecido para dados.alerta_ativo. Se ele estiver ativo, priorize atendimento imediato; se estiver inativo, não crie uma urgência.",
}


def main():
    root = ROOT / "data/contextual_v1"
    destination = root / "challenge.jsonl"
    if destination.exists():
        raise FileExistsError(destination)
    rng = random.Random(2026100502)
    rows = []
    for i in range(32):
        category = CATEGORIES[i % 8]
        row = example(category, i, "challenge", rng)
        row["instruction"] = QUESTIONS[category]
        rows.append(row)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    destination.write_text(content, encoding="utf-8")
    manifest = {"count": 32, "seed": 2026100502, "sha256": hashlib.sha256(content.encode()).hexdigest(), "design": "Quatro casos por atividade; novas formulações de instrução fixadas antes do treino; cenários e valores novos. Sem seleção de parâmetros por esta prova.", "limitations": "Prova sintética pequena; não avalia conhecimento médico geral ou segurança clínica."}
    (root / "challenge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
