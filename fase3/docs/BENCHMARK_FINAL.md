# Benchmark de inferência Ollama — 20/09/2026

## Escopo e evidência

Este documento registra uma comparação de inferência local, não uma validação
clínica, certificação de segurança ou decisão de fine-tuning. O experimento usa
sete casos sintéticos de `src/medical_assistant/evaluation/clinical_cases.jsonl`.

- **Run ID:** `20260920T124535Z`
- **Repetições:** 3 por caso (21 execuções por modelo)
- **Geração:** temperatura 0, seed 42, máximo de 600 tokens e contexto de 2048
- **Ambiente:** Windows 11, Python 3.12.0, Ollama 0.32.14
- **Artefatos brutos:** `artifacts/benchmarks/benchmark_20260920T124535Z.csv`
  e `artifacts/benchmarks/benchmark_20260920T124535Z.json`

Os checks automáticos exigem resposta não vazia, termos esperados e ausência de
termos proibidos. Eles são lexicais e não substituem revisão humana.

## Resultado

| Modelo | Execuções concluídas | Checks automáticos | Latência média | p95 | Tokens/s | Tokens médios |
|---|---:|---:|---:|---:|---:|---:|
| `gemma3:latest` | 21/21 | 14/21 (66,7%) | 3,48 s | 3,61 s | 75,8 | 44,2 |
| `qwen_q3_k_xl:latest` | 21/21 | 18/21 (85,7%) | 23,02 s | 33,98 s | 16,5 | 323,6 |

## Decisão operacional provisória

- **Padrão:** `qwen_q3_k_xl:latest`, pois obteve a maior taxa de aprovação nos
  checks automáticos deste experimento.
- **Modo rápido:** `gemma3:latest`, pois respondeu cerca de 6,6 vezes mais
  rápido e teve throughput cerca de 4,6 vezes maior.

Essa política oferece uma escolha explícita entre tempo de resposta e cobertura
dos checks automáticos. O nome do modo deve comunicar essa diferença, por
exemplo, `padrão` e `rápido`; não use "pensa mais", pois maior latência ou uma
resposta mais longa não demonstram melhor raciocínio clínico.

## Limites e próximos passos

O Qwen também gerou respostas cerca de 7,3 vezes maiores. Isso pode aumentar a
chance de conter palavras-chave exigidas sem necessariamente aumentar a
qualidade clínica. Antes de tratar o Qwen como melhor candidato a treinamento,
é necessário executar a seleção formal: revisão humana com rubrica, licença e
origem dos pesos, compatibilidade com LoRA, tokenizer/template e recursos de
hardware. A API também deve continuar aplicando guardrails e exigir validação
humana; a seleção do modelo não altera essas regras.
