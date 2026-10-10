# FT-10 — Qwen3.5-9B com corpus contextual ampliado

Base oficial `Qwen/Qwen3.5-9B`, revisão `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Adapter `20261005T101013Z_qwen3.5_9b_qlora_rocm`: 960 exemplos, uma época, LR 2e-5, contexto 384 sem truncagem. Pesos finais salvos, recarregados e avaliados em processo novo. Controles com mesmo prompt, greedy, thinking desativado e teto de 192 tokens.

| Conjunto | Base: resposta completa correta | Adapter: resposta completa correta |
| --- | ---: | ---: |
| Validação, 120 casos | 47 | 120 |
| Teste reservado, 120 casos | 46 | 120 |
| Formulações novas, 32 casos | 15 | 25 |

As oito categorias tiveram 15/15 acertos no teste do adapter, incluindo campos, fontes fornecidas, status, alertas, recusas e marcador de validação humana. A seleção pela validação foi registrada antes do teste. Especialidades, cenários e IDs são separados entre splits, mas os templates são compartilhados.

Na prova adicional, preparada antes do treino, os quatro formulários vazios recebem status `informativo` em vez de `modelo`; três alertas inativos omitem `dados.alerta_ativo=false`. Campos de receita continuam vazios, e os quatro casos de recusa são corretos. Esse resultado demonstra transferência parcial para novas instruções; não elimina dependência da formulação. A prova não ajustou os parâmetros da FT-10. Sua análise poderá informar um corpus posterior, que exigirá novos casos reservados.

Resultado: ganho comprovado na tarefa contextual avaliada, com limite de generalização documentado. Não mede conhecimento médico geral, não executa RAG e não constitui validação clínica. `validacao_humana=true` é um campo do contrato, não uma revisão realizada por um profissional.

Pesos finais: `../../adapters/20261005T101013Z_qwen3.5_9b_qlora_rocm/adapter/`. SHA-256 de `adapter_model.safetensors`: `cd7f016e2e48c09c564327adef9e1be5674ac0c352d9feda147b15e7f3fead1c`. Checkpoint duplicado de 249 MB enviado à lixeira após comprovar o mesmo hash; base, adapter, manifestos, respostas e métricas preservados.
