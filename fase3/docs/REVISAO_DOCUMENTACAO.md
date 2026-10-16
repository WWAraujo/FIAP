# Revisão da documentação Markdown

Revisão de 16/10/2026, após APP-02/SAFE-01 e consolidação técnica ENT-01. Escopo: todos os `.md` próprios existentes na Fase 3, incluindo evidências locais ignoradas. Bibliotecas dos SDKs e documentação do modelo-base são de terceiros e não foram reescritas. O AGENTS da raiz foi conferido/atualizado nas referências à Fase 3; não havia referência à fase nos demais Markdown da raiz que exigisse alteração.

## Resultado e regras

README principal, README LLM/pacotes, relatório técnico, guia, roteiro e plano distinguem execução real, testes mockados e pendências humanas. AGENTS da fase agora contém tabela explícita de responsabilidade e atualização dos demais documentos. A proposta conserva o enunciado de cinco páginas; o texto antigo de implementação foi preservado em snapshot local histórico. Registros de arquivo/atendimento e README anterior estão identificados como históricos.

Relatórios por execução e model cards são evidências da versão da época. Não reescrever neles “API pendente” ou decisões negativas como sucesso atual; o estado vigente está no [relatório técnico](RELATORIO_TECNICO.md). Arquivos históricos locais podem citar detalhes não selecionados para o Git. A política acadêmica vigente está em `llm/artifacts/versioning_policy.json`; ela substitui a seleção anterior de runtime mínimo. Pesos, bancos, logs e consultas operacionais permanecem locais.

Conferência automática dos links relativos Markdown: nenhum destino inexistente nos documentos ativos. Os 11 links herdados do snapshot `README_COMPLETO_ANTES_GIT_MINIMO.md` conservam a resolução original da raiz e não são instruções vigentes; o arquivo está arquivado localmente. URLs externas e âncoras não foram verificadas por rede. Caminhos em comandos/backticks foram confrontados com a estrutura/código; comandos históricos permanecem datados. Não foi executado novo treinamento ou reaberto teste reservado.

## Inventário completo

64 documentos próprios lidos/conferidos. A lista inclui este índice.

| Arquivo relativo à Fase 3 | Classificação | Resultado |
|---|---|---|
| `.pytest_cache/README.md` | Ativo | Instruções/resultados atuais conferidos; links locais válidos. |
| `AGENTS.md` | Ativo | Instruções/resultados atuais conferidos; links locais válidos. |
| `PROPOSTA.md` | Ativo | Instruções/resultados atuais conferidos; links locais válidos. |
| `README.md` | Ativo | Instruções/resultados atuais conferidos; links locais válidos. |
| `docs/GUIA_VALIDACAO.md` | Ativo | Instruções/resultados atuais conferidos; links locais válidos. |
| `docs/PLANO_EXECUCAO.md` | Ativo | Instruções/resultados atuais conferidos; links locais válidos. |
| `docs/PLANO_RAG.md` | Ativo | Plano histórico identificado; encaminhamento atual e preservação do piloto atualizados. |
| `docs/README_COMPLETO_ANTES_GIT_MINIMO.md` | Histórico | Snapshot anterior; identificado como histórico. |
| `docs/RELATORIO_TECNICO.md` | Ativo | Instruções/resultados atuais conferidos; links locais válidos. |
| `docs/RESUMO_ENTREGA_GRUPO.md` | Ativo | Instruções/resultados atuais conferidos; links locais válidos. |
| `docs/REVISAO_ARQUIVOS_FASE3.md` | Histórico | Snapshot anterior; identificado como histórico. |
| `docs/REVISAO_ATENDIMENTO_PROPOSTA.md` | Histórico | Snapshot anterior; identificado como histórico. |
| `docs/REVISAO_DOCUMENTACAO.md` | Ativo | Instruções/resultados atuais conferidos; links locais válidos. |
| `docs/ROTEIRO_VIDEO.md` | Ativo | Instruções/resultados atuais conferidos; links locais válidos. |
| `docs/historico/PROPOSTA_IMPLEMENTACAO_ANTIGA.md` | Histórico | Snapshot anterior; identificado como histórico. |
| `llm/README.md` | Ativo | Instruções/resultados atuais conferidos; links locais válidos. |
| `llm/artifacts/README.md` | Ativo | Instruções/resultados atuais conferidos; links locais válidos. |
| `llm/artifacts/adapters/20261005T101013Z_qwen3.5_9b_qlora_rocm/adapter/README.md` | Gerado/histórico | Model card PEFT gerado; preservado sem reescrever. |
| `llm/artifacts/adapters/20261005T120430Z_qwen3.5_9b_qlora_rocm/adapter/README.md` | Gerado/histórico | Model card PEFT gerado; preservado sem reescrever. |
| `llm/artifacts/data_audits/20260930_dat07_resume/RESULTADOS.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/evaluations/20260930T175422Z_eval03/reference_quality_audit.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/evaluations/20260930T175422Z_eval03/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/evaluations/20260930T182438Z_eval03/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/evaluations/20260930T192204Z_eval03/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/evaluations/20260930T215546Z_eval03/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/evaluations/20260930T230117Z_eval03/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/evaluations/20260930T232622Z_eval03/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/evaluations/20260930T233923Z_eval03/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/evaluations/20260930_eval06_conditional_prompt/comparison_review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/evaluations/20261001_ft09_9b_adapter/20261001T102530Z_eval03/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261005T095351Z_contextual_v1/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261005T120254Z_factual_v2/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261005T203707Z_ft12_comparison/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261006T232619Z_ft13_pilot/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261011T212752609941Z_rag03/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261011T212901816585Z_rag03/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261011T214416417521Z_rag04/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261011T215705006500Z_rag05/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261011T215906287410Z_rag05/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261011T225210885223Z_rag06_validation/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261011T234356000996Z_rag07_validation/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261011T235027290179Z_rag07_validation/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261012T001057419178Z_rag08_test/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261012T104756171442Z_rag09_validation/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261012T105941618852Z_rag09_validation/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261012T112732865793Z_rag09_validation/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261012T121124698742Z_rag09_validation/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261012T122108946656Z_rag09_validation/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261012T170701149180Z_rag09_test/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261012T174142028524Z_rag09_validation/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261012T175916205183Z_rag09_test/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261012T182715001180Z_rag09_validation/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261012T184438077141Z_rag09_test/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261012T190305897293Z_rag_query_v3/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261016T000851104289Z_ft11_tool_selection/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/artifacts/experiments/20261016T002602034519Z_three_routes/review.md` | Histórico | Evidência de execução; bytes/decisões preservados. |
| `llm/data/factual_v3/README.md` | Histórico | Snapshot anterior; identificado como histórico. |
| `llm/data/rag_pilot_v2/README.md` | Ativo | Corpus congelado; etapa RAG-02 identificada e integração atual referenciada. |
| `llm/docs/DADOS_EXTERNOS.md` | Histórico | Snapshot anterior; identificado como histórico. |
| `llm/src/finetuning/evaluation/README.md` | Ativo | Instruções/resultados atuais conferidos; links locais válidos. |
| `llm/src/finetuning/training/README.md` | Ativo | Instruções/resultados atuais conferidos; links locais válidos. |
| `llm/src/rag/README.md` | Ativo | Instruções/resultados atuais conferidos; links locais válidos. |
| `src/medical_assistant/application/README.md` | Ativo | Instruções/resultados atuais conferidos; links locais válidos. |
| `src/medical_assistant/tools/README.md` | Ativo | Instruções/resultados atuais conferidos; links locais válidos. |
