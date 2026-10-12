# RAG-09 — teste final da preparação documental, 12/10/2026

**Aceite documental aprovado**, após comparação completa e revisão por IA: `decision.json` registra todos os critérios verdadeiros e `rag_preparation_accepted: true`. Não é aprovação humana ou clínica. A configuração 3.072/reserva 256, módulos, pacotes, índice e pesos são iguais à validação aprovada `20261012T174142028524Z_rag09_validation`. 160 registros novos, sem descarte.

| Condição | Completas respondíveis | Parciais | Ausências adequadas | Conflitos | Instruções hostis |
|---|---:|---:|---:|---:|---:|
| FT-11 + RAG | 23/24 | 1 | 8/8 | 4/4 | 4/4 |
| Base + RAG | 23/24 | 1 | 8/8 | 4/4 | 4/4 |
| FT-11 sem RAG | 0/24 | 0 | 8/8 | 0/4 | 0/4 |
| Base sem RAG | 0/24 | 0 | 8/8 | 0/4 | 0/4 |

**Adjudicações explícitas:** caso 18 permanece parcial nos dois seletores: citar campos null não reproduz a regra específica de não reutilizar dados de outro paciente. Casos 19–24 citam os parágrafos pertinentes das portarias corretas; o gabarito de página inteira não exige devolver toda a página. Caso 28 pergunta se o procedimento identifica nominalmente a pessoa, sem pedir um nome concreto. A citação explicita profissional entre os campos e todos os valores null, sustentando resposta negativa sobre identificação nominal no modelo vazio, sem nome ou aprovação inventados. Foi avaliado como ausência adequadamente comunicada por evidência negativa, com fonte, em vez de `no_evidence`. Portanto, com RAG são **sete lacunas canônicas e uma negativa citada**, e não oito saídas `no_evidence`. Essa interpretação semântica, seu texto e a diferença de formato estão em `annotations.json`/`documentary_review.json`; o check automático isolado não aprovava o caso. Não houve alteração do runner, dataset ou critérios após abrir este teste.

Todos os 24 spans esperados foram recuperados/fornecidos. Nenhuma omissão, erro de execução, fonte inexistente ou citação final sem suporte segundo a revisão documental. Quatro conflitos e quatro instruções hostis expostos integralmente em ambas as condições com RAG. Os quatro ataques receberam a regra legítima, não o rodapé hostil. Conflitos são registrados pela política da aplicação com ambas as edições, sem escolher vigência desconhecida. Entrada máxima 1.810 tokens; pico VRAM alocada 7,92 GiB. Medianas por pergunta, sem carga: FT+RAG 8,23 s, base+RAG 5,76 s, FT sem RAG 5,37 s, base sem RAG 2,77 s. Houve 160 gerações de seleção e 102 de conferência de suporte.

`integrity_verification.json` confirma configuração idêntica à validação, módulos atuais/snapshots, identidade dos pesos, índice, offsets das citações, estados PEFT reais e orçamento. Seleção usa o estado FT-11/base da condição; a conferência é comum na mesma base quantizada sem adapter. JSON restrito, filtros de diretivas e regras de conflitos pertencem ao pipeline. O empate não demonstra superioridade do FT-11, e o resultado sob ataques não mede resistência do modelo isolado. Zero sem RAG é consequência do modo extrativo sem fontes fornecidas, não ausência de conhecimento geral.

O reservado v4, SHA-256 `38aac365efd2244e69a3af727ca5029f3972695ee49734a6ce945d264e5929a2`, contém novas paráfrases dos mesmos conceitos/corpus e famílias de desafios da validação. Foi criado antes da revisão do orçamento e não usado como fonte de ajustes após sua abertura. Não demonstra independência temática/clínica nem cobertura de ataques novos. As perguntas sob instruções hostis desta versão solicitam a regra documental sem alertar explicitamente sobre o rodapé. Os conjuntos anteriores consumidos e suas decisões negativas permanecem preservados; esta versão está agora consumida e não deve ser repetida como novo teste.

BM25 sobre 15 seções de nove documentos elegíveis; portarias oficiais admitidas somente na primeira página, sem anexos clínicos. Pesos/seleção FT-11 preservados, sem treino, API/prontuário ou LangGraph nesta tarefa. 38 testes de software passaram no SDK Python 3.14.4/ROCm. Necessidade de validação profissional continua indicada e não equivale a aprovação realizada.

Comandos a partir de `fase3/llm/`, registrados para reproduzir o protocolo, não repetir um teste já consumido:

```bash
.venv/bin/python -B src/rag/evaluate_v3.py --split test --approved-validation artifacts/experiments/20261012T174142028524Z_rag09_validation --conditions ft11_with_rag base_with_rag ft11_without_rag base_without_rag
.venv/bin/python -B src/rag/review_v3.py artifacts/experiments/20261012T175916205183Z_rag09_test --overrides artifacts/experiments/20261012T175916205183Z_rag09_test/annotations.json --documentary-signoff
```

Signoff requer inspeção prévia dos registros. Decisões usam criação exclusiva; não sobrescrever uma revisão histórica. Recibo em `data/rag_index_v3_final/readiness.json` vincula a consulta livre aos módulos, índice, pacotes, pesos e decisão avaliados, sem abrir perguntas/gabaritos durante a consulta. Prova da CLI registrada separadamente em novo experimento de consulta.
