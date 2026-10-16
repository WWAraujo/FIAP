# Artefatos da entrega

A seleção para o Git está no `.gitignore` e em [versioning_policy.json](versioning_policy.json). A política inclui execução e prova acadêmica; substitui a seleção temporária de apenas quatro arquivos de runtime.

| Conjunto | Conteúdo versionado |
|---|---|
| `best_model.json` | Seleção FT-11 e identidade. |
| Adapters FT-10/FT-11 | Manifestos, ambientes e configuração LoRA; pesos locais. |
| Sete execuções selecionadas | Configurações, métricas, respostas brutas, revisões e decisões de treino/recuperação/RAG. |
| Outras execuções com relatório selecionado | Somente `review.md`; detalhes permanecem locais. |
| Integração atual | Resumo `20261016T101610484760Z_delivery_live/summary.json`, sem prontuário ou decisões humanas pessoais. |
| Limpezas | Registros de arquivos removidos nas limpezas autorizadas anteriores. |

Não mover os arquivos exigidos pelo recibo `../data/rag_index_v3_final/readiness.json`. O teste reservado está consumido. Falhas e recibos revogados nos relatórios são resultados históricos, sem aprovação vigente.

Os relatórios por run_id e os model cards gerados são históricos imutáveis: descrevem a versão da época, não instruções atuais da aplicação. Consulte [relatório técnico](../../docs/RELATORIO_TECNICO.md) e [README LLM](../README.md). Links de relatórios antigos para detalhes não selecionados dependem da cópia local.

Novas consultas e experimentos são ignorados por padrão. Avaliações/diagnósticos intermediários, snapshots de código, pesos, checkpoints e dados de pacientes/revisões continuam locais. Remoção do índice usa `git rm --cached`, sem apagar o arquivo local; não elimina cópias de commits anteriores. Nenhum commit é criado automaticamente.
