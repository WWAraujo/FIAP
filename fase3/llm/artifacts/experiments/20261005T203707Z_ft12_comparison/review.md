# Comparativo FT-12 versus FT-11 — 05/10/2026

O FT-11 permanece selecionado em `artifacts/best_model.json`. O FT-12 concluiu uma época adicional com todos os 1.408 exemplos, mas não passou o critério registrado antes da geração: na validação, reduziu respostas factuais adequadas, aumentou inadequadas e perdeu um acerto documental/contextual. O teste comparativo também apresentou regressão factual. Houve ganhos pontuais em documentos no teste e fatos nos desafios; eles não autorizam substituir o vencedor pela regra definida.

## Tamanho e origem da base

| Conjunto | Exemplos | Uso |
| --- | ---: | --- |
| Treino | 1.408 | Atualização dos pesos |
| Validação | 160 | Perda e comparação para seleção |
| Teste | 160 | Comparação posterior; não entrou no treino |
| Desafios reformulados | 40 | Diagnóstico adicional; não entrou no treino |

O arquivo de treino `data/factual_v2/data/splits/train.jsonl` tem 3.405.766 bytes: 3,41 MB decimais ou 3,25 MiB. São 1.024 exemplos factuais vinculados ao MedQuAD original e 384 exemplos de oito atividades contextuais. Definições, causas e herança de doenças raras/genética predominam na parte factual. O conjunto não cobre todas as perguntas médicas. Grupos de condições são separados entre splits, mas isso não garante ausência de memorização ou conhecimento prévio do modelo-base.

Base dos dois adaptadores: **Qwen/Qwen3.5-9B**, revisão `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, licença Apache-2.0; pesos locais em `models/base/Qwen3.5-9B`.

## Resultado factual

Revisão documental por IA contra o original completo e, quando necessário, fontes primárias; não é avaliação clínica. Contagens de adequado/parcial/inadequado separadas de JSON e cópia exata.

| Conjunto | FT-11 adequado | Parcial | Inadequado | FT-12 adequado | Parcial | Inadequado |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validação: 48 perguntas | 23 | 8 | 17 | 19 | 11 | 18 |
| Teste: 48 perguntas | 22 | 15 | 11 | 20 | 16 | 12 |
| Desafios: 12 perguntas | 5 | 4 | 3 | 6 | 4 | 2 |

## Documentos, contexto e formato

| Conjunto | FT-11 acerto completo | FT-12 acerto completo |
| --- | ---: | ---: |
| Validação | 101/112 | 100/112 |
| Teste | 100/112 | 102/112 |
| Desafios | 24/28 | 24/28 |

Os dois modelos produziram JSON válido em 160/160, 160/160 e 40/40 casos, respectivamente. Ambos preservaram os status esperados nos alertas (8/8, 8/8, 2/2) e nas recusas (8/8, 8/8, 2/2). Esses checks de status e formato não constituem validação de segurança clínica. Os 360 casos de cada modelo foram contabilizados, sem excluir falhas ou normalizar as chaves erradas das FAQs. Não houve respostas vazias, erros de geração ou alcance do teto de tokens.

## Treino e reprodução

FT-12 continua os pesos FT-11, com QLoRA NF4/BF16, LoRA r=8/alpha=32/dropout=0,1, uma época, LR 1e-5, contexto 512, batch 1/acumulação 8 e 176 passos; `max_steps=-1`. GPU Radeon RX 9060 XT 16 GB/ROCm, Python 3.14.4, Torch 2.13.0+rocm10.0.0, Transformers 5.17.0, PEFT 0.20.0. Validação e teste não foram usados para atualizar os pesos.

O supervisor registra término em **05/10/2026 18:04:04 UTC (15:04:04 Brasília)**, `status.txt=completed` e `exit_code.txt=0`. Tempo de treinamento: 3.507,9673 s, aproximadamente 58 min 28 s; a execução completa com carregamento, avaliação de perda e recarga levou cerca de 65 min. A perda de treino foi 0,2045344. A perda de validação caiu de 0,0967652 no FT-11 para 0,0900386 no FT-12; a queda não correspondeu a melhora factual nesta comparação.

Comando de treino e status: [run.sh](../20261005T165650Z_ft12_full_train/run.sh), [metadata.json](../20261005T165650Z_ft12_full_train/metadata.json). Executar o comando novamente começa outro treino; o FT-12 já terminou.

Adaptadores finais preservados:

- FT-11: `artifacts/adapters/20261005T120430Z_qwen3.5_9b_qlora_rocm/adapter`; SHA-256 dos pesos `6d050564ef3979d75066aabb7eec4eeedc78488fe52491d5ee9012db1478946b`.
- FT-12: `artifacts/adapters/20261005T165900Z_qwen3.5_9b_qlora_rocm/adapter`; SHA-256 dos pesos `e9a6d3c50b2018871209b0b75fa7865b4c786f8d811dfa163c28cdb08bc50eb5`.

A geração candidata usou `scripts/evaluate_qwen3_base_vs_adapter_rocm.py --adapter-only`, o mesmo prompt factual_v2, seed 42, greedy, thinking desativado, teto 192 tokens e todos os casos (`--limit 99999`). As referências exatas foram pontuadas com `scripts/score_contextual_results.py`; as notas documentais com `scripts/score_semantic_review.py`. O SDK utilizado é `fase3/llm/.venv/bin/python`. Os argumentos e hashes completos estão nos arquivos `request.json` de cada rodada. Não houve novo treino, ajuste de parâmetros ou seleção de casos durante a comparação.

## Método de revisão e limites

O [protocolo](protocol.json) foi registrado antes da geração. Foram reutilizadas as respostas anteriores do FT-11 nos mesmos IDs, dados, prompt e parâmetros. Validação, teste e desafios já eram conhecidos; esta comparação é diagnóstica e **não é uma prova nova e independente de generalização**. Não foi feita nova seleção pelo teste.

Pares com conteúdo diferente foram apresentados com rótulos A/B; a redação pode revelar o modelo, portanto o procedimento não constitui cegamento clínico. Respostas JSON idênticas reutilizam a nota histórica. Quando as proposições factuais e os qualificadores são semanticamente equivalentes, a mesma nota histórica é transferida para evitar que uma mudança de redação pareça ganho ou perda. Conteúdo alterado recebe revisão própria; não se exige repetição literal da referência nem lista exaustiva de genes verdadeiros. As notas históricas do controle permanecem fixas. Houve quatro divergências provisórias de revisão do controle na validação e duas no teste, todas registradas e sem alterar seu resultado. Não houve divergência nos desafios. Essa variabilidade da revisão por IA é uma limitação da comparação; pequenas diferenças não demonstram superioridade estatística.

Os casos com referências históricas incompletas ou problemáticas permanecem incluídos. As conferências de fontes e as notas por ID estão nos arquivos `*_pair_grades.jsonl`, `*_final_grades.jsonl` e `*_semantic_scores.json`; os arquivos `*_review_binding.json` vinculam a revisão aos hashes das respostas. As fontes complementares incluem [MedlinePlus: Mainzer-Saldino](https://medlineplus.gov/genetics/condition/mainzer-saldino-syndrome/), [NCBI MedGen](https://www.ncbi.nlm.nih.gov/medgen/341455), [MedlinePlus: colestase intra-hepática da gravidez](https://medlineplus.gov/genetics/condition/intrahepatic-cholestasis-of-pregnancy/) e [NCBI: aliases de GCH1](https://www.ncbi.nlm.nih.gov/gene/2643/). Elas apoiam apenas as conferências dos respectivos casos, não validam clinicamente o modelo.

O [comparativo consolidado](comparison.json) contém contagens, hashes e checks de integridade. [validation_decision.json](validation_decision.json) registra a rejeição na validação; [review_method_notes.json](review_method_notes.json) detalha a política de notas. `best_model.json` permaneceu inalterado. Erros factuais e falhas de contrato ainda existem; nenhum dos modelos tem cobertura universal ou validação clínica.

Na limpeza autorizada, o checkpoint intermediário FT-12 foi enviado à lixeira após comprovar que seus pesos eram idênticos ao adaptador final. Foram retirados 260.190.312 bytes, incluindo a cópia dos pesos e os estados de otimizador/scheduler; `training_state.json` com os 176 passos foi preservado no diretório do treino. O registro de arquivos/hashes fica em `artifacts/adapters/20261005T165900Z_qwen3.5_9b_qlora_rocm/cleanup.json`. Modelo-base, adaptadores finais, datasets e evidências da avaliação permaneceram intactos.
