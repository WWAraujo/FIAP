# RAG-05 — cadeia documental BM25 + FT-11

Concluída em 11/10/2026: uma RunnableSequence LCEL executa recuperação BM25 congelada, seleção de contexto inteiro por orçamento, geração FT-11 real e validação/resolução da citação. Prova local na GPU; sem API, prontuários, LangGraph ou execução dos splits de avaliação.

| Caso novo | Resultado observado |
|---|---|
| Preservar decimais na transcrição | Trecho `Preserve vírgulas decimais`, presente na fonte SIM-RAG-UNIDADES 1.0, seção transcricao. Contrato completo e fonte resolvida. |
| Campos da contagem administrativa | Frase literal com código do formulário, quantidade disponível e data; fonte SIM-RAG-ESTOQUE 1.0, seção contagem. |
| Endereço postal da instituição fictícia | Chunks recuperados não cabem inteiros; `context_budget_exceeded`, sem geração. Não foi tratado como sucesso de recuperação ou prova de abstenção semântica. |
| Termos sem coincidência lexical | `no_evidence`, sem geração. |
| Pedido explícito de prescrição | `blocked_request`, sem geração. |

Casos positivos: entrada 363 e 375 tokens, reserva de saída 128, orçamento total abaixo de 512; gerações 21,70 s e 6,66 s. Pico VRAM alocada 7,42 GiB. A primeira geração inclui aquecimento; não representa latência de API. O primeiro caso usa fragmento curto, não a frase inteira: preserva o núcleo pedido, com fonte literal disponível, sem provar qualidade geral de respostas.

`case_*.json` registra pergunta, top-3, chunks fornecidos/omitidos, motivo da omissão, prompt, resposta bruta, contagens/tempo, identidade e resposta validada. R1/R2/R3 são aliases apenas dos chunks fornecidos; a saída final resolve alias para ID completo e inclui título, versão, URL quando disponível, página, seção, texto e offsets. Campos adicionais/paráfrases não são permitidos neste primeiro modo: `dados` contém somente `trecho`, com presença literal na fonte citada, permitindo apenas normalização de espaços/quebras. Presença de texto não demonstra pertinência semântica de qualquer resposta; revisão documental por IA nestes casos, sem validação clínica/humana.

## Configuração e integridade

BM25 final RAG-03 permanece intacto (24/24 evidências contra 23/24 E5 na validação). Corpus rag_pilot_v2 e índice de 21 chunks validados por hashes/manifesto/elegibilidade. Nenhuma pergunta/gabarito/fixture de avaliação entrou no índice ou prompt. Não houve mudança no adapter ou treinamento. FT-11 hash `6d050564ef3979d75066aabb7eec4eeedc78488fe52491d5ee9012db1478946b`; identidade completa em `identity.json`.

`chain_config.json` congela instrução, modo de resposta, política de contexto/fonte, orçamento, hashes dos quatro módulos e das seleções, modelo/prompt e versões. SHA-256 `6ef570406a4bddd10b755bc53cad28e7aa34c91e66a7b8b05c10b6beb90c3af8`. `verification.json` registra revisão dos cinco casos, hashes dos artefatos e verificações de igualdade da contagem do montador com o backend, orçamento e citação. Esta configuração precede qualquer avaliação do teste reservado.

LangChain instalado nesta tarefa: dependência direta `langchain-core==1.4.0`, com seus pacotes transitivos; nenhum pacote previamente instalado atualizado. Pydantic 2.13.5, LangSmith 0.14.7 e versões LLM em chain_config.json. Composição/invoke conforme [documentação oficial de RunnableSequence](https://reference.langchain.com/python/langchain-core/runnables/base/RunnableSequence). Tracing remoto desativado no invoke. Todas as gerações usam o backend ROCm local; não há ferramentas que executem instruções dos documentos.

## Validação

SDK PyCharm Python 3.14.4 em `llm/.venv/bin/python`; em `fase3/llm/`:

```bash
.venv/bin/python -m pip install langchain-core==1.4.0
.venv/bin/python -B src/rag/document_chain.py
.venv/bin/python -B -m pytest tests/test_document_chain.py tests/test_rag_retrieval.py tests/test_ft11_backend.py -q -p no:cacheprovider
```

28 testes aprovados em 0,20 s (18 da cadeia + três de recuperação + sete do backend). Cobrem LCEL realmente invocado, citações fornecidas/resolvidas, JSON/chaves duplicadas/tipos/campos indevidos, revisão obrigatória, fonte inexistente, trecho sem suporte, limite de saída, ausência/bloqueio sem modelo, ausência de estado entre chamadas, corpus/índice divergente e orçamento. Teste específico conta input_ids de tokenizer com retorno Mapping; evita confundir número de campos do BatchEncoding com número de tokens. Contagem também conferida com tokenizer/geração reais. Warnings de depreciação internos de LangChain sobre asyncio.iscoroutinefunction em Python 3.14; sem falha e sem correção fora do escopo.

Execução inicial `20261011T215705006500Z_rag05` preservada: caso 1 omitiu `pendencias` e foi corretamente rejeitado; caso 2 passou. O prompt foi ajustado para exigir explicitamente as cinco chaves e pendencias [], sem relaxar o validador ou reparar JSON. Repetição dos mesmos casos nesta execução passou nos dois positivos. Falhas anteriores não foram descartadas.

## Limites e próximo passo

Modo inicial extrativo/documental: não oferece respostas livres ou interpretação clínica. Bloqueio lexical de comandos explícitos de prescrição/diagnóstico é limitado; não comprova todos os requisitos SAFE-01 nem resistência a ataques. A presença literal de uma citação não prova que ela responda à pergunta. Sem limiar semântico calibrado, recuperação irrelevante e omissão de evidência por orçamento ainda são possíveis; exclusões ficam visíveis. `validacao_humana=true` é necessidade de revisão, sem aprovação humana.

APP-00/APP-01 completas continuam pendentes (política/API/prontuário); LangGraph e auditoria integrada ficam nas tarefas próprias. Próxima tarefa RAG-06: avaliar a cadeia na validação com as quatro condições planejadas, preservando configuração/erros e congelando decisão antes do teste reservado. Não declarar melhoria do FT-11 com RAG a partir apenas destes probes.
