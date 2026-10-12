# RAG-09 — validação v3, 12/10/2026

Comparação completa: 40 questões por condição, 160 registros, código/configuração/pesos congelados. Todos os critérios documentais do gate passaram; `decision.json` libera somente o novo teste reservado.

| Condição | Respondíveis completas | Parciais | Ausência correta | Conflitos | Instruções hostis |
|---|---:|---:|---:|---:|---:|
| FT-11 + RAG | 22/24 | 2 | 8/8 | 4/4 | 4/4 |
| Base + RAG | 22/24 | 2 | 8/8 | 4/4 | 4/4 |
| FT-11 sem RAG | 0/24 | 0 | 8/8 | 0/4 | 0/4 |
| Base sem RAG | 0/24 | 0 | 8/8 | 0/4 | 0/4 |

Os casos 14 e 18 citam regras verdadeiras, mas omitem respectivamente identificação como modelo sujeito a validação e proibição explícita de reutilizar dados. Permanecem parciais. As portarias 19–24 foram conferidas semanticamente: o gabarito contém página inteira, enquanto a resposta cita somente o parágrafo que responde. Justificativas em `annotations.json`; registros completos em `documentary_review.json`.

BM25 sobre 15 seções integrais elegíveis, normalização lexical e escopo documental derivados exclusivamente do corpus. Todos os 24 spans esperados chegaram ao contexto. Nenhuma omissão, erro de execução, fonte inexistente ou citação final sem suporte. Todos os desafios foram expostos integralmente; lacunas indevidas não contaram como resistência. Conferência de offsets, hashes e estado PEFT em `integrity_verification.json`. Orçamento total 2.048 tokens, reserva 256; VRAM alocada pico 7,91 GiB. Medianas totais: FT+RAG 8,24 s, base+RAG 5,70 s, FT sem RAG 5,43 s, base sem RAG 2,77 s.

O seletor usa FT-11 ou base conforme a condição. A conferência de suporte usa a mesma base quantizada, adapter desativado, em ambas. Decodificação limita escolhas a IDs dos parágrafos fornecidos/ausência e contratos JSON; a aplicação resolve o texto literal e aplica regras para conflitos e diretivas hostis. Os ganhos pertencem ao pipeline completo, não ao FT-11 isolado. Base+RAG empatou: não há evidência de superioridade FT neste conjunto. Zero sem RAG decorre do modo documental extrativo sem fontes fornecidas e não mede conhecimento geral.

O novo reservado foi fixado antes da calibração: 40 novas formulações sobre os mesmos conceitos/corpus, com famílias de ataques herdadas da validação. É holdout de paráfrases, não independência temática ou clínica. Não reutiliza o teste consumido RAG-08. Revisão documental por IA; nenhuma aprovação de profissional de saúde. Corpus oficial limitado à primeira página das três portarias, sem anexos clínicos admitidos. Pesos FT-11, seleção e versões/decisões negativas anteriores preservados. Sem treino, API, prontuário ou LangGraph nesta tarefa.

Comandos a partir de `fase3/llm/`, SDK Python 3.14.4 ROCm:

```bash
.venv/bin/python -B src/rag/evaluate_v3.py
.venv/bin/python -B src/rag/review_v3.py artifacts/experiments/20261012T122108946656Z_rag09_validation --overrides artifacts/experiments/20261012T122108946656Z_rag09_validation/annotations.json --documentary-signoff
```

A opção de signoff exige revisão prévia dos registros; não autoriza aprovação automática de novas execuções. As revisões usam criação exclusiva e não sobrescrevem evidência anterior.
