# FT-11 — ampliação factual e contextual

A validação escolheu o adaptador FT-11 antes da geração do novo teste reservado. O teste confirmou os critérios operacionais registrados. A prova adicional mostrou ganho contextual e uma regressão no número de respostas factuais adequadas; a generalização permanece parcial. Não há validação clínica.

Base oficial: `Qwen/Qwen3.5-9B`, revisão `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. A FT-11 continuou os pesos LoRA da FT-10, sem alterar o arquivo do adaptador anterior. Uma época, LR 1e-5, comprimento 512, NF4/BF16, embeddings congelados BF16. Foram 176 etapas, 3.541,4 segundos, perda de treino 0,2470 e de validação 0,0968. Perda não comprova correção factual. Pesos finais salvos e recarregados em processo novo.

Corpus: 1.408 exemplos de treino, 160 de validação, 160 de teste e prova adicional de 40. Inclui 1.024 questões factuais no treino e 384 casos das oito atividades contextuais. Proveniência liga traduções ao original MedQuAD fixo; são referências históricas, concentradas em doenças raras/genética, com revisão semântica amostral por IA. Grupos transitivos por condição/sinônimos/CUIs são separados entre conjuntos, inclusive distratores. Todas as sequências de treino/validação/teste e EOS cabem no limite de tokens. Não se pode garantir ausência absoluta de memorização ou conhecimento prévio do corpus pelo modelo base.

## Validação usada para seleção

As 48 perguntas gerais foram revisadas documentalmente por IA com nomes dos modelos ocultados por caso; formato foi pontuado separadamente. A referência curta não é lista exaustiva, e afirmações adicionais foram confrontadas com o original completo ou fontes primárias. A revisão não é clínica; estilo pode permitir reconhecimento de um sistema. Contagens pequenas não demonstram não inferioridade estatística.

| Sistema | Factual adequado | Parcial | Inadequado | Documentos/contexto: acerto completo |
| --- | ---: | ---: | ---: | ---: |
| Base 9B | 21/48 | 2/48 | 25/48 | 29/112 |
| FT-10 anterior | 15/48 | 6/48 | 27/48 | 75/112 |
| FT-11 novo | 23/48 | 8/48 | 17/48 | 101/112 |

A FT-11 preservou os oito status de alerta e as oito recusas do base. Seus 160 resultados são JSON válido. Os 17 erros factuais impedem tratar o modelo como confiável para toda pergunta médica. Nas oito FAQs da validação, os valores foram majoritariamente preservados, mas houve renomeação indevida de `exame` para `tripla_exame`; são erros completos de contrato. Também houve falhas isoladas de fonte ausente, laudo e protocolo. Não normalizamos essas falhas para inflar a pontuação.

Critérios registrados em `selection_protocol.json`: pelo menos tantos adequados e no máximo tantos inadequados quanto o base nas perguntas gerais; acertos documentais/contextuais e status de alerta/recusa não inferiores. A FT-11 atende nesta validação; a FT-10 falha no critério factual. A decisão e hashes das notas estão em `validation_selection.json`, registrada antes da geração do teste. As notas por caso e fontes estão em `validation_semantic_scores.json`; respostas brutas e pontuação de contrato ficam nas três pastas de validação.

## Teste reservado concluído

O teste novo não escolheu hiperparâmetros ou checkpoint. `reserved_evaluation.json` registra a geração do base e do adaptador escolhido no mesmo prompt, decodificação greedy, thinking desativado e máximo de 192 tokens. As notas das 48 perguntas factuais foram concluídas com rótulos ocultados antes de consolidar os resultados; nenhum caso foi excluído.

| Sistema | Factual adequado | Parcial | Inadequado | Documentos/contexto: acerto completo |
| --- | ---: | ---: | ---: | ---: |
| Base 9B | 20/48 | 7/48 | 21/48 | 29/112 |
| FT-11 novo | 22/48 | 15/48 | 11/48 | 100/112 |

FT-11: JSON válido em 160/160, status de alerta 8/8 contra 6/8 do base, recusas 8/8 em ambos. Os 12 erros completos documentais/contextuais foram oito FAQs, dois documentos de causas genéticas e dois protocolos. Acertos de formato/conteúdo documental não são interpretados como respostas clínicas corretas. As onze respostas factuais inadequadas e quinze parciais limitam o resultado; o ganho de apenas dois adequados não demonstra superioridade universal ou não inferioridade estatística. `test_confirmation.json` confirma todos os critérios operacionais registrados. As notas e evidências completas ficam em `test_semantic_scores.json`.

## Prova adicional concluída

A prova de 40 casos usa grupos separados e perguntas definidas antes do treino. As 12 perguntas educativas foram revisadas com nomes ocultados, sem exclusões. O adaptador teve 5 adequados, 4 parciais e 3 inadequados, contra 6, 1 e 5 do base. Houve regressão de um adequado: o ganho observado no teste principal não se confirmou nesse critério da prova adicional. Reduzir erros graves sem aumentar adequados em toda prova não demonstra superioridade universal. A escolha permanece a registrada pela validação e confirmada no teste principal; não foi refeita usando esta prova.

Nos 28 casos documentais/contextuais, FT-11 teve 24 acertos completos contra 9 do base, JSON válido em 40/40 contra 39/40, alertas 2/2 e recusas 2/2 em ambos. Os quatro erros completos foram duas FAQs, um caso de fonte ausente e um laudo. Entre erros factuais, persistem troca de modos de herança e genes; algumas respostas corretas são mais vagas que as do base. Resultados por caso: `challenge_semantic_scores.json`, `base_challenge/scores.json` e `adapter_challenge/scores.json`. Essa prova não autoriza afirmar cobertura de todas as perguntas.

A conferência das referências, sem observar o conteúdo das respostas dos modelos, encontrou uma imprecisão no original histórico do colangiocarcinoma intra-hepático: ele mistura locais dentro e fora do fígado. O caso e os dados são mantidos; a avaliação factual usará a distinção anatômica do [NCI](https://www.cancer.gov/publications/dictionaries/cancer-terms/def/liver-and-bile-duct-cancer). O registro está em `test_reference_audit.json`. Igualdade textual com fonte histórica não demonstra verdade clínica.

## Artefatos e limpeza

Adaptador FT-11: `../../adapters/20261005T120430Z_qwen3.5_9b_qlora_rocm/adapter/`, SHA-256 dos pesos `6d050564ef3979d75066aabb7eec4eeedc78488fe52491d5ee9012db1478946b`. `../../best_model.json` identifica o escolhido sem duplicar seus pesos. O checkpoint final tinha pesos idênticos e foi enviado à lixeira. Os ambientes das sondagens ficam nas pastas dos resultados, em `probe_environment.json`; pastas temporárias vazias foram retiradas. Logs concluídos, snapshots externos, caches e benchmarks de inferência não essenciais foram enviados à lixeira. Base, dois adaptadores treinados, dados, código necessário, ambiente, manifestos e evidências são preservados. Resultados históricos de modelos treinados também foram mantidos; a revisão original da FT-09 permanece em `artifacts/evaluations/20261001_ft09_9b_adapter/20261001T102530Z_eval03/review.md`.

Não há RAG neste ensaio. Documentos presentes no contexto são fornecidos diretamente à avaliação. O campo histórico `source_available=false` do executor indica ausência de recuperação, e não ausência desses documentos. `validacao_humana=true` é campo do contrato; não significa revisão por um profissional.
