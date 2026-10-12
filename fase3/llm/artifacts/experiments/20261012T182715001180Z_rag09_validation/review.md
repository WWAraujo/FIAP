# RAG-09 — validação após corrigir o último parágrafo, 12/10/2026

**Todos os critérios documentais passaram; reservado v5 autorizado.** 160 registros novos, quatro condições, mesmos pesos/índice/corpus, orçamento 3.072/reserva 256. 42 testes de software aprovados no SDK Python 3.14.4/ROCm.

| Condição | Completas respondíveis | Parciais | Ausências | Conflitos | Instruções hostis |
|---|---:|---:|---:|---:|---:|
| FT-11 + RAG | 22/24 | 2 | 8/8 | 4/4 | 4/4 |
| Base + RAG | 22/24 | 2 | 8/8 | 4/4 | 4/4 |
| FT-11 sem RAG | 0/24 | 0 | 8/8 | 0/4 | 0/4 |
| Base sem RAG | 0/24 | 0 | 8/8 | 0/4 | 0/4 |

Os 160 resultados recém-gerados foram confrontados com a revisão anterior. Única diferença de resposta/fonte: FT-11 caso 24 cita a ementa, que contém o mesmo título solicitado no artigo 1º; equivalência documental conferida. Casos 14 e 18 permanecem parciais. Anotações e revisão por caso preservam essas limitações, sem alterar o critério após abrir o novo teste.

Correção efetiva: `paragraphs()` agora inclui o último parágrafo terminado em whitespace. A consulta livre anterior às duas assinaturas tinha um nome em `provided.text` que não chegava ao prompt; diagnóstico/recibo antigos foram corrigidos/revogados em artefatos separados. O teste de corpus verifica cobertura de todos os caracteres úteis das 15 seções, além de offsets e presença de cada bloco no prompt. A regex nova preserva nomes, unidades e negações no fim do texto. Todas as 455 combinações de três seções foram medidas com pergunta genérica fora das avaliações: máximo 2.577 tokens de entrada, mantendo a reserva de 256 dentro de 3.072.

`integrity_verification.json` confirma **cobertura completa no prompt realmente salvo**, módulos/snapshots, índice, identidade dos pesos, offsets e estados PEFT. Todos os 24 spans e desafios entregues, nenhuma omissão, erro ou citação final sem suporte segundo revisão por IA. Entrada máxima desta validação 1.791; 160 gerações de seleção e 94 de conferência; pico VRAM alocada 7,91 GiB. Medianas por pergunta, sem carga: FT+RAG 8,36 s, base+RAG 5,75 s, FT sem RAG 5,44 s, base sem RAG 2,77 s.

O seletor usa FT-11/base conforme condição; verificador comum na mesma base quantizada sem adapter. Conflitos/diretivas e JSON restrito são políticas do pipeline; não demonstram resistência do modelo isolado. O empate não prova superioridade do FT-11. Zeros sem RAG decorrem do modo extrativo sem fontes fornecidas, não ausência de conhecimento geral.

O reservado v5 foi fixado antes da correção de EOF e inclui as duas assinaturas com referência na mesma página admitida do corpus. É novo holdout de paráfrases e cobertura de regressão, com conceitos/corpus/famílias de desafios compartilhados; não é independência temática/clínica. Testes anteriores estão consumidos e preservados, incluindo a revogação do recibo da v4 após a prova CLI. Sem treino, mudança de pesos/seleção, API, prontuário, LangGraph ou revisão por profissional de saúde.

Comando a partir de `fase3/llm/`: `.venv/bin/python -B src/rag/evaluate_v3.py --conditions ft11_with_rag base_with_rag ft11_without_rag base_without_rag`. Revisão semântica e signoff por IA explícitos, sem aprovação clínica; decisões/arquivos históricos não são sobrescritos.
