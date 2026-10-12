# Artefatos LLM

O treino selecionado e a preparação do piloto documental RAG estão concluídos. A integração à aplicação é uma etapa separada; não há validação clínica.

| Caminho | Finalidade | Git |
|---|---|---|
| `best_model.json` | Seleção FT-11, identidade e hashes dos pesos | Sim |
| `adapters/20261005T101013Z_qwen3.5_9b_qlora_rocm/` | FT-10, origem do treino FT-11 | Manifesto, ambiente e configuração; pesos locais |
| `adapters/20261005T120430Z_qwen3.5_9b_qlora_rocm/` | FT-11 selecionado | Manifesto, ambiente e configuração; pesos locais |
| `experiments/` | Respostas, métricas, decisões e histórico, inclusive falhas | JSON, JSONL e Markdown; logs e cópias históricas de scripts locais |
| `evaluations/`, `environment/`, `data_audits/` | Avaliações e diagnósticos históricos | Locais, ignorados |
| `cleanup_20261010.json` | Registro da limpeza dos pesos rejeitados e caches | Sim |

## Evidências atuais

- [Treino e seleção FT-11](experiments/20261005T120254Z_factual_v2/review.md).
- [Comparação BM25/E5](experiments/20261011T212901816585Z_rag03/review.md).
- [Validação RAG final](experiments/20261012T182715001180Z_rag09_validation/review.md).
- [Teste RAG final e limitações](experiments/20261012T184438077141Z_rag09_test/review.md).
- [Consulta e recarga reais pela CLI](experiments/20261012T190305897293Z_rag_query_v3/review.md).

Preserve os IDs das execuções e as decisões anteriores. As avaliações reprovadas e os recibos revogados explicam a seleção atual e não devem ser tratados como resultados vigentes. Não mover esses diretórios: o recibo em `../data/rag_index_v3_final/readiness.json` verifica caminhos e hashes.

Os modelos-base ficam em `../models/`, fora do Git. [Download da base e reprodução dos adapters](../README.md#baixar-os-arquivos-grandes). Os adapters do projeto não têm download público registrado: copie a execução completa ou realize FT-10 e FT-11 conforme o README. Um novo treino exige nova avaliação; não herda a aprovação do recibo histórico.
