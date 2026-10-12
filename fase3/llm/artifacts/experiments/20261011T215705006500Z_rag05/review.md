# RAG-05 — execução inicial preservada

Em 11/10/2026, a cadeia LCEL recuperou os documentos e chamou FT-11: caso 2 retornou citação de estoque válida; caso 1 omitiu `pendencias` e foi rejeitado como `invalid_contract`. Não houve reparo silencioso ou apresentação daquela resposta como documentada. Os demais casos retornaram limite de contexto, ausência lexical e bloqueio explícitos sem geração.

O prompt passou a exigir explicitamente as cinco chaves e `pendencias=[]`; os mesmos cinco casos foram repetidos. Configuração final e evidências em [20261011T215906287410Z_rag05](../20261011T215906287410Z_rag05/review.md). Esta execução permanece como evidência da falha inicial, sem substituir seus artefatos.
