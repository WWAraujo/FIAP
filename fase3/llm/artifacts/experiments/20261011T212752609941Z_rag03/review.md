# Execução técnica preliminar RAG-03

Esta execução de 11/10/2026 foi preservada, mas a leitura de memória por `resource.getrusage` incluiu pico pré-exec do launcher, tornando o RSS lexical inadequado para comparação. Não utilizar sua memória nem sua seleção automática por seção como decisão final.

A comparação foi repetida integralmente após corrigir a medição para VmHWM Linux. Resultados finais, revisão dos spans e configuração congelada em [20261011T212901816585Z_rag03](../20261011T212901816585Z_rag03/review.md). O ranking foi igual entre execuções; a revisão de evidência selecionou BM25 (24/24 contra 23/24), apesar de ambos obterem 24/24 por ID de seção.
