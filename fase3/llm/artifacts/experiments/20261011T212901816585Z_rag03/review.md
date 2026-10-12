# RAG-03 — comparação na validação, 11/10/2026

**Decisão: congelar BM25 para o piloto**, em `frozen_config.json`. A revisão documental encontrou suporte no top-3 em 24/24 casos para BM25 e 23/24 para multilingual-E5-small/CPU. Ambos superam o mínimo de 22/24 nesta validação; não há execução do teste reservado ou geração de respostas.

| Medida | BM25 lexical | E5 semântico CPU |
|---|---:|---:|
| Seção esperada no top-3 | 24/24 | 24/24 |
| Trecho com informação necessária no top-3 | **24/24** | **23/24** |
| MRR@3 por seção | 0,8958 | 0,9583 |
| MRR@3 por evidência efetiva | 0,8472 | 0,8750 |
| Mediana de consulta | 0,040 ms | 416,011 ms |
| p95 de consulta | 0,056 ms | 532,270 ms |
| Construção/carregamento no worker | 0,0016 s | 7,7859 s |
| Pico de RSS do worker | 23,23 MiB | 1.646,20 MiB |
| Erros / truncamentos de entrada | 0 / 0 | 0 / 0 |

O hit automático de seção é insuficiente: `RAG02-validation-19` pede número/data da portaria de hipertensão. E5 recuperou `MS-PCDT-HAS@portaria_49_2025#portaria~001`, mas o cabeçalho com esses dados está em `~000`, ausente do seu top-3. BM25 trouxe `~000` na terceira posição. A classificação automática por seção/MRR havia indicado E5; esse resultado permanece em `selected_config.json`/`summary.json` como candidato **anterior à revisão**. A configuração final válida é `frozen_config.json`, que registra a decisão por suporte efetivo, com hash do candidato e da revisão.

`evidence_review.json` registra os 24 casos por método, spans literais, offsets no documento de origem, chunks de suporte e ranks. As 18 perguntas sintéticas usam seções curtas integralmente recuperadas; os seis atos oficiais exigem cabeçalho, sistema, revogação ou artigo pertinente. A verificação de presença de spans complementa inspeção documental por IA; não é revisão humana nem avaliação de resposta clínica. Nenhum dado de revisão foi usado na busca ou acrescentado ao índice.

## Configuração e reprodução

Corpus `rag_pilot_v2`, nove documentos/15 seções elegíveis, dividido em 21 chunks rastreáveis. Tamanho alvo 192 tokens E5, sobreposição 32, expansão aos limites de palavras, offsets e SHA-256 do texto original. Título/seção entram como metadados de busca para ambos os métodos. Top-3 são chunks, podendo conter mais de um trecho da mesma seção. BM25 k1=1,5/b=0,75, normalização de acentos/minúsculas e stopwords PT fixas. Versão substituída, avaliação/gabaritos e fixtures não indexados.

Modelo de embeddings [multilingual-E5-small](https://huggingface.co/intfloat/multilingual-e5-small/blob/614241f622f53c4eeff9890bdc4f31cfecc418b3/README.md), revisão `614241f622f53c4eeff9890bdc4f31cfecc418b3`, MIT. Prefixos query/passage, média com máscara de atenção e normalização L2 conforme documentação oficial. CPU, float32, quatro threads, batch 8, entradas conferidas em até 512 tokens, sem truncamento. Tokenização da página integral pode emitir aviso de sequência longa antes da divisão; a página integral não é entregue ao modelo.

Ambiente: Linux, SDK PyCharm Python 3.14.4, torch 2.13.0+rocm10.0.0, transformers 5.17.0, numpy 2.4.6, huggingface_hub 1.33.0; nenhuma dependência instalada/alterada. Medidas de consulta após uma consulta de aquecimento; E5 inclui codificação da pergunta e ranking, BM25 inclui score/ranking. Tempos de construção excluem download e divisão comum dos documentos. RSS usa VmHWM Linux em workers separados: inclui imports/modelo, exclui memória do launcher; não representa memória incremental ou latência da API. GPU e FT-11 não utilizados.

Em `fase3/llm/`:

```bash
.venv/bin/python -B src/rag/compare_retrieval.py
.venv/bin/python -B -m pytest tests/test_rag_retrieval.py tests/test_rag_manifest.py tests/test_rag_corpus.py -q -p no:cacheprovider
```

O comando cria novo run_id e baixa/confere a revisão fixa em `models/embeddings/multilingual-e5-small/`, ignorada pelo Git. Arquivos pequenos de evidência/índice/configuração ficam neste experimento; `embeddings.npy` fica local. O benchmark produz candidato automático por seção, que exige revisão dos trechos antes de congelar a configuração final. Não sobrescrever estes artefatos.

Resultados: 14 testes aprovados em 0,27 s; verificados filtros, offsets, cobertura, preservação de negações/unidades, ranking, corpus e referências. Integridade dos 21 chunks, hash dos embeddings, script executado e configuração conferida. `environment.json` registra hashes dos arquivos do modelo; `comparison_config.json`, `index.json`, `lexical.json`, `semantic.json` e `summary.json` preservam configuração, texto e resultados das 40 perguntas. SHA-256 de `frozen_config.json`: `36ef967ca411cb8fcd55b3939de3bad91a5ea0b26aa4131a4dd2ddd58e639e99`.

Execução técnica preliminar `20261011T212752609941Z_rag03` preservada, mas sua leitura RSS por getrusage incorporou pico pré-exec do launcher. Não usar aquela memória como comparação entre métodos. Esta execução completa repetiu ambos os métodos após corrigir a medição; ranking e hits foram iguais.

## Limites e próxima etapa

Sem limiar de abstenção calibrado: vizinho recuperado não comprova evidência, especialmente nas oito perguntas sem informação. Conflitos/ataques têm resultados de busca registrados, mas seus fixtures não foram inseridos na busca e o comportamento da LLM não foi avaliado. Não há aprovação dos critérios de segurança, fontes na resposta, orçamento do prompt ou ganho do FT-11 com RAG.

O corpus é pequeno e documental, sem anexos clínicos; perguntas de validação e teste compartilham conceitos. A seleção não demonstra superioridade universal de BM25. Próxima etapa do plano: comprovar o acesso ao FT-11 antes da cadeia documental, usando esta configuração e preservando o teste reservado até congelar o fluxo avaliado.
