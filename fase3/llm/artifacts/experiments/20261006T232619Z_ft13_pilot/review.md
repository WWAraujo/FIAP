# FT-13 — piloto com fatos gerais revisados

Concluído em 06/10/2026, horário de Brasília. O piloto melhorou FT-11 nesta validação, mas não cumpriu o critério completo de expansão: 21 factuais adequadas contra 22 da base e duas inadequadas contra uma. Nenhum treino completo foi iniciado. FT-11 permanece selecionado em `artifacts/best_model.json`; o candidato promissor foi preservado, sem promoção ou integração à aplicação. O novo teste continua sem respostas geradas.

## Dados e execução

Nova base [factual_v3](../../../data/factual_v3/README.md): 1.920 exemplos de treino (5.581.252 bytes), 80 de validação e 80 de teste. São 48 temas separados em 32/8/8, com 96/24/24 referências distintas. Os exemplos incluem variações de instrução e tarefas contextuais; não representam 1.920 fatos médicos diferentes. Referências revisadas documentalmente por IA contra resumos oficiais MedlinePlus coletados em 06/10/2026, com snapshots, atribuição NLM, licença, hashes e revisão por referência. Nenhum dado pessoal real ou revisão clínica.

Os conjuntos foram congelados antes dos modelos. Grupos, IDs e fontes contextuais são disjuntos; nomes dos novos temas reservados foram conferidos contra factual_v2. Isso não garante ausência no pré-treino ou em textos históricos. Dados anteriores intactos; conjuntos antigos deixam de ser testes independentes deste currículo. Máximos de 500/484/509 tokens no template real, sem truncar respostas de treino.

Base `Qwen/Qwen3.5-9B`, revisão `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, Apache-2.0. Piloto iniciou FT-11 com QLoRA NF4/BF16, LR 5e-6, contexto 512, batch 1/acumulação 8, seed 42 e 48 passos: 384 exemplos, 20% de uma época. Não percorreu os 1.920 exemplos integralmente. Runtime 991,96 s (16 min 32 s); perdas treino/validação 0,09048/0,16218. Probe forward/backward aprovado, pico de 12,86 GiB; pesos finais salvos e recarregados com geração não vazia.

Pesos em `artifacts/adapters/20261006T232654Z_qwen3.5_9b_qlora_rocm/adapter`, SHA-256 `3072d765d8899da0f302ae073cfdad606477ca5ecaa5620cd13d02488fc8d627`. Comandos completos, parâmetros, hashes e métricas em [protocol.json](protocol.json), [experiment.json](experiment.json) e no manifesto do treino. Nenhuma dependência instalada ou atualizada.

## Nova validação

Todos os 80 casos gerados por cada modelo, sem exclusões: 24 factuais, 16 documentais, oito com fonte ausente e 32 contextuais. Mesmo prompt, greedy, thinking desativado, seed 42 e limite de 192 tokens. Revisão factual por IA com rótulos ocultados por caso, notas explícitas e fontes primárias, aceitando paráfrases e exemplos corretos. Identidade, notas e consolidação separados em [validation_mapping.json](validation_mapping.json), [validation_grades.jsonl](validation_grades.jsonl) e [validation_semantic_scores.json](validation_semantic_scores.json). Fontes extras das notas serviram somente à revisão, sem alterar o corpus congelado.

| Medida | Base | FT-11 | FT-13 piloto |
|---|---:|---:|---:|
| Factuais adequadas / 24 | 22 | 19 | 21 |
| Factuais parciais / 24 | 1 | 1 | 1 |
| Factuais inadequadas / 24 | 1 | 4 | 2 |
| Documentos/contexto exatos / 56 | 14 | 55 | 55 |
| JSON parseável / 80 | 76 | 80 | 80 |
| Status de alerta / 4 | 4 | 4 | 4 |
| Status de recusa / 4 | 4 | 4 | 4 |
| Respostas não vazias / 80 | 80 | 80 | 80 |
| Respostas no limite de tokens | 4 | 0 | 0 |

JSON parseável mede sintaxe, sem comprovar sozinho chaves, fontes ou verdade factual. As quatro respostas truncadas da base foram mantidas. Exatidão foi aplicada às tarefas de copiar/extrair conteúdo, separada da qualidade factual. Os percentuais não são diretamente comparáveis aos conjuntos antigos, de perguntas e dificuldades diferentes.

O candidato corrigiu duas respostas de caxumba do FT-11. Persistiram confusão entre herpes-zóster e catapora, inclusão dos pulmões na definição de resfriado e causalidade de refluxo sem qualificador. Notas e evidências estão no arquivo de revisão. Ambos os adaptadores usaram o documento distrator sobre caxumba no caso que perguntava sobre sinusite e deveria resultar em fonte ausente (`F3-validation-sinusitis-features-missing-0`), causando o único erro documental/contextual.

## Decisão e preservação

O critério registrado antes da geração exigia adequados factuais acima de FT-11 e pelo menos iguais à base, inadequados não superiores a nenhum controle, além de preservar documentos/contexto, JSON e status. O candidato passou os critérios relativos ao FT-11 e às tarefas contextuais; falhou nos dois critérios factuais relativos à base. [validation_selection.json](validation_selection.json) registra a decisão antes de qualquer geração de teste. O teste de 80 casos permanece com SHA-256 `8ad717a778fd437ca76ab4ecab3709a8acbf092080ae378b952e3c80ef303439`, sem resultados de modelos.

`finalize.py` conferiu os três conjuntos completos, referências, parâmetros, hashes e revisão de exatamente 24 casos por modelo. Pesos FT-11 e `best_model.json` intactos. Checkpoint-48 enviado à lixeira após conferir igualdade de pesos/configuração com o adaptador final: 260.185.487 bytes. Estado preservado em `training_state.json`; pesos finais, dados, manifestos, logs e evidências preservados. Registro em [cleanup.json](cleanup.json).

São 24 perguntas de oito temas, com dependência temática, revisadas por IA. Não há prova estatística de não inferioridade, segurança clínica, cobertura universal ou ausência de memorização. A validação agora foi observada e não pode ser reclassificada como teste independente. O teste novo permanece disponível para confirmação após seleção válida. Expandir esta configuração exigiria rever explicitamente o critério ou realizar outro piloto registrado.
