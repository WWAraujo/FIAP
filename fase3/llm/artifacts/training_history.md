# Histórico dos treinamentos de modelos

Resumo criado a partir dos manifestos e relatórios disponíveis antes da limpeza de `artifacts/experiments`. As perdas são métricas do objetivo de treino/validação e não são diretamente comparáveis entre bases, modelos e configurações diferentes. As avaliações de conteúdo foram revisões documentais/experimentais; nenhuma representa validação clínica.

| Etapa / execução | Modelo e dados de treino | Resultado do treino (loss treino / validação) | Resultado de avaliação e decisão |
|---|---|---:|---|
| FT-01 — `20260927T020236Z_qwen3_8b_qlora_rocm` | Qwen3-8B; 3 épocas, 153 passos | 0,9034 / 0,6740 | Adapter recarregado em processo novo e gerou resposta não vazia. |
| FT-03 — `20260928T100950Z_qwen3_8b_qlora_rocm` | Qwen3-8B; 4.165 exemplos, 1 época | 1,8799 / 1,8164 | Comparação revisada: aderência adequada em 9/50 versus 25/50 no modelo-base; segurança adequada 27/50 versus 34/50; 18 regressões materiais versus 6. Rejeitado. |
| Execução Qwen3-8B — `20260928T230104Z_qwen3_8b_qlora_rocm` | Qwen3-8B; 4.165 exemplos, 1 época | 1,9245 / 1,8648 | Métricas iguais às da execução FT-04 abaixo; não foi localizado relatório de avaliação separado para este ID. |
| FT-04 — `20260930T102822Z_qwen3_8b_qlora_rocm` | Qwen3-8B; 4.165 exemplos, 1 época | 1,9245 / 1,8648 | 12/50 respostas entraram em loop de repetição, com respostas longas/cortadas e erros factuais. Rejeitado. |
| FT-05 — `20260930T164016Z_qwen3_8b_qlora_rocm` | Qwen3-8B; 145 exemplos sintéticos, 2 épocas | 2,7483 / 2,5003 | Recarga aprovada; avaliação encontrou falhas em alertas de urgência e loops graves. Rejeitado. |
| FT-06 — `20260930T185850Z_qwen3_8b_qlora_rocm` | Qwen3-8B; 145 exemplos sintéticos, 2 épocas | 3,6569 / 3,2960 | 7 adequadas, 32 parciais, 8 inadequadas e 3 não avaliáveis; sem ganho consistente sobre os controles. Rejeitado. |
| FT-07 — `20260930T212304Z_qwen3_14b_bnb_4bit_qlora_rocm` | Qwen3-14B quantizado; 145 exemplos sintéticos, 2 épocas | 3,3501 / 2,9741 | 11 adequadas, 31 parciais, 5 inadequadas e 3 não avaliáveis, igual ao controle nos mesmos casos; sem ganho. Rejeitado. |
| FT-08 — `20260930T223317Z_qwen3_8b_qlora_rocm` | Qwen3-8B; 190 exemplos, 2 épocas | 2,9947 / 3,1344 | 15 adequadas, 25 parciais, 7 inadequadas e 3 não avaliáveis; sem ganho de categoria sobre o controle. Rejeitado. |
| FT-09 — `20261001T001634Z_qwen3.5_9b_qlora_rocm` | Qwen3.5-9B; 190 exemplos, 2 épocas | 2,1417 / 2,7200 | 10 adequadas, 30 parciais, 7 inadequadas e 3 não avaliáveis; controle 13/31/3/3. Rejeitado. |
| FT-10 — `20261005T101013Z_qwen3.5_9b_qlora_rocm` | Qwen3.5-9B; 960 exemplos contextuais, 1 época | 0,0085 / 0,0002 | Avaliação contextual: 120/120 exatos na validação e teste; prova de novas formulações 25/32 versus 15/32 do controle. Foi a origem histórica do FT-11; seus pesos foram removidos nesta limpeza. |
| FT-11 — `20261005T120430Z_qwen3.5_9b_qlora_rocm` | Qwen3.5-9B; 1.408 exemplos, 1 época | 0,2470 / 0,0968 | Selecionado em `best_model.json`. Teste: 22/48 factual adequado, 15 parcial, 11 inadequado; acerto documental/contextual 100/112 (base 29/112). Prova adicional factual 5/12 adequadas versus 6/12 base. Permanecem limitações; sem validação clínica. Adapter mantido. |
| FT-12 — `20261005T165900Z_qwen3.5_9b_qlora_rocm` | Continuação do FT-11; 1.408 exemplos, 1 época | 0,2045 / 0,0900 | Na validação factual: 19/48 adequadas versus 23/48 do FT-11; documentos/contexto 100/112 versus 101/112. Regressões no teste factual; rejeitado, FT-11 permaneceu selecionado. |
| FT-13 piloto — `20261006T232654Z_qwen3.5_9b_qlora_rocm` | Qwen3.5-9B; 1.920 no corpus, 48 passos (20% da época) | 0,0905 / 0,1622 | Validação: 21/24 factuais adequadas versus 22/24 base; 2 inadequadas versus 1 base. Piloto não avançou para treino completo nem substituiu FT-11. |
| FT-14 amostra — `20261019T232200Z_ft14_medpt` | Candidato MedPT com amostra; interrupção solicitada ao ampliar para todos os registros aproveitáveis | Não concluído | Interrompido antes de gerar adapter final; não promovido. |
| FT-14 corpus completo — `20261019T234300Z_ft14_medpt_full` | Continuação de FT-11; 34.833 exemplos previstos, 4.355 passos | Parcial até passo 254/4.355; sem métrica final | Interrompido pelo desligamento. Não havia checkpoint retomável nem adapter final. |
| FT-14 corpus completo — `20261020T_ft14_medpt_full_restart` | Continuação de FT-11; 34.833 exemplos, 4.355 passos previstos | Em andamento; progresso visto durante esta limpeza: 2.586/4.355 passos | Ainda sem resultado final ou avaliação. A execução e seus logs foram preservados em `artifacts/experiments/20261020T_ft14_medpt_full_restart/`. |

O corpus MedPT completo tem 41.828 exemplos aproveitáveis: 34.449 treino, 3.295 validação e 4.084 teste. O treino inclui mais 384 exemplos sintéticos de replay. O candidato FT-14 não é promovido automaticamente; aguarda conclusão e revisão semântica.

FT-02 foi apenas a validação de recarga/serviço do adapter FT-01, sem novo treinamento. Os experimentos de recuperação, RAG e integração da aplicação não são modelos treinados e não fazem parte desta tabela.
