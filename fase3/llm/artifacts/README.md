# Artefatos da entrega

A seleção para o Git está no `.gitignore` e em [versioning_policy.json](versioning_policy.json). A política descreve os arquivos selecionados para versionamento; a limpeza local autorizada pelo usuário removeu os diretórios antigos de `experiments` da cópia de trabalho.

| Conteúdo local | Localização / finalidade |
|---|---|
| Modelo selecionado FT-11 | `best_model.json` identifica a seleção; os pesos do adapter permanecem em `adapters/20261005T120430Z_qwen3.5_9b_qlora_rocm/adapter/`. |
| Execução FT-14 concluída | `experiments/20261020T_ft14_medpt_full_restart/`, com candidato aguardando revisão semântica. |
| Histórico consolidado dos treinamentos | [training_history.md](training_history.md), criado antes da remoção dos resultados detalhados antigos. |
| Avaliações e auditorias históricas | [evaluation_audit_summary.md](evaluation_audit_summary.md), resumo dos resultados e decisões de `evaluations/` e `data_audits/`. |
| Pacote multiplataforma para o grupo | `shares/FT14_MedPT_20261020-Q4_K_M.gguf` (5,3 GB), com checksum/prompt; instruções em `../../docs/COMPARTILHAR_FT14.md`. |
| Dados e dependências do runtime | Permanecem em `../data/`, no SDK local e nos diretórios de modelo necessários. |

As demais pastas e arquivos que estavam dentro de `experiments/` foram removidos localmente, exceto a execução FT-14 em andamento. Também foram removidas as pastas de adapters diferentes do FT-11 selecionado. O conteúdo detalhado removido não pode ser recuperado desta cópia; o resumo preserva as métricas e decisões de treinamento que estavam disponíveis. O adapter FT-11 não foi alterado. A limpeza local não altera commits anteriores; nenhuma nova commit foi criada.

Os resultados brutos antigos de `evaluations/` e os arquivos de auditoria de `data_audits/` também foram consolidados em [evaluation_audit_summary.md](evaluation_audit_summary.md) e removidos. `evaluations/` permanece vazio como destino padrão para novas avaliações; `data_audits/` não é necessário para executar o runtime ou o treino FT-14 atual.
