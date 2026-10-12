# Consulta livre — recarga final da cadeia documental

Novo processo com `query_document.py`, recibo e pesos avaliados, sem abrir perguntas/gabaritos. Pergunta: “Quais são os nomes dos dois signatários ao final da portaria de dislipidemia do snapshot?”

**Prova positiva:** cita os dois nomes, FRANCISCO DE ASSIS FIGUEIREDO e DENIZAR VIANNA, no bloco original da portaria de dislipidemia. Fonte/versão/seção/página corretas; offsets 3165–3261 conferidos contra a seção original. Sem erro ou omissão. FT-11 ativo na seleção; base sem adapter na conferência; identidade igual ao recibo.

`verification.json` confirma nomes, offsets, pesos, estados PEFT e ausência de splits de avaliação. `request`, `identity`, `result` e `resources` registram contexto completo, gerações/verificações, resultado e recursos. A falha anterior da mesma consulta está preservada em `20261012T181702101790Z_rag_query_v3`, com correção do diagnóstico: a última linha não chegava ao prompt antes do ajuste de EOF. O recibo antigo foi revogado e preservado.

Partindo de `fase3/llm/`:

```bash
.venv/bin/python -B src/rag/query_document.py --question 'Quais são os nomes dos dois signatários ao final da portaria de dislipidemia do snapshot?'
```

Este caso comprova recarga/consulta documental local; não é revisão clínica, integração à API/prontuário nem garantia de acerto em novas perguntas.
