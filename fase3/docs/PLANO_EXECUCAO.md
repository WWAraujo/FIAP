# Plano de execução — Fase 3

Atualizado em 20/09/2026. Este documento controla o trabalho da Fase 3. Uma tarefa só muda para concluída após satisfazer seus critérios de aceite e registrar evidência abaixo. Não marque uma implementação parcial como concluída.

## Fluxo de dependências

```text
ORG-01 ─┬─ DAT-01 ─ DAT-02 ─ DAT-03 ─ DAT-04 ─ DAT-05 ─┬─ FT-01 ─ FT-02 ─ EVAL-03 ─┐
        ├─ EVAL-01 ─ EVAL-02 ─────────┘        └─ APP-00 ─ APP-01 ─ APP-02 ─ SAFE-01 ─ ENT-01
        └─ ORG-02 ────────────────────────────────────────────────┘
ORG-03 ─ ORG-04 ──────────────────────┘
```

## Regras de acompanhamento

- Status permitido: `PENDENTE`, `EM ANDAMENTO`, `BLOQUEADA`, `CONCLUÍDA`.
- Cada tarefa deve registrar data, comando/ambiente, resultado e caminhos dos artefatos. Falhas também são evidência.
- Só uma tarefa fica `EM ANDAMENTO` por vez.
- Alterações que revelem requisito adicional viram uma nova tarefa; não ampliam silenciosamente a tarefa atual.
- Dados, resultados e pesos não devem conter dados pessoais. Versionar dados sintéticos pequenos e metadados; manter pesos grandes e logs fora do Git.

## Tarefas

### ORG-01 — Documentar objetivos, módulos e status

**Status:** CONCLUÍDA

**Objetivo:** substituir alegações de conclusão por inventário verificável e tornar as responsabilidades explícitas.

**Aceite:** README, AGENTS e documentação do benchmark distinguem estado atual, planejado e comprovado; plano criado.

**Evidência (20/09/2026):** `README.md`, `AGENTS.md`, `src/medical_assistant/evaluation/README.md`, `src/medical_assistant/training/README.md` e este arquivo.

### ORG-02 — Aplicar estrutura orientada por objetivos de aprendizado

**Status:** CONCLUÍDA

**Objetivo:** separar dados, avaliação, treinamento, recuperação, domínio clínico, aplicação e interfaces em módulos explícitos.

**Depende de:** ORG-01.

**Aceite:** imports, comandos e documentação apontam aos novos caminhos; comportamento existente preservado; testes de importação passam. Não corrigir lógica de recuperação ou assistente nesta tarefa.

**Evidência (20/09/2026):** código movido para `src/medical_assistant/`; dados processados em `data/curated/` e `data/splits/`; resultados em `artifacts/benchmarks/`; CLI, Streamlit, Docker e documentação atualizados.

### ORG-03 — Reproduzir ambiente mínimo

**Status:** CONCLUÍDA

**Objetivo:** validar uma instalação limpa das dependências exigidas por cada tarefa e documentar versões de Python, PyTorch, PEFT, Transformers, Ollama e hardware.

**Aceite:** comando de instalação reproduzível, inventário de versões e limitações de CPU/GPU documentados. Não adicionar dependências não utilizadas.

### ORG-04 — Validar acelerador para treinamento

**Status:** PENDENTE

**Objetivo:** validar um ambiente acelerado para treinamento. A rota prioritária é Colab com GPU CUDA; Linux nativo + ROCm para a AMD Radeon RX 9060 XT (`gfx1200`) permanece alternativa. O Windows local usa PyTorch CPU-only e não tem backward pass suportado.

**Depende de:** ORG-03.

**Aceite:** registrar provedor, GPU, VRAM, driver, PyTorch e versões; `torch.cuda.is_available()` é verdadeiro; um forward e backward pass mínimos passam na GPU; limitações de memória para LoRA são medidas. No Colab, validar persistência de checkpoint no Drive. Se o acelerador falhar, registrar a evidência e não iniciar FT-01.

### DAT-01 — Tornar dados sintéticos reproduzíveis

**Status:** CONCLUÍDA

**Objetivo:** versionar o gerador e um dataset sintético pequeno, ajustando `.gitignore` com regras específicas.

**Depende de:** ORG-01.

**Aceite:** clone obtém gerador e dados de exemplo; pesos, banco operacional e logs continuam excluídos; execução do gerador cria a mesma estrutura esperada.

**Evidência (21/09/2026):** `src/medical_assistant/data/generate_synthetic.py` foi validado inicialmente com 12 registros sintéticos e passou a gerar 25 registros em DAT-03. `.gitignore` permite somente essa fonte; `data/curated/`, `data/splits/` e `data/prontuarios.db` continuam ignorados. Validação: geração e contagem das cinco coleções.

### DAT-02 — Corrigir preprocessing e curadoria

**Status:** CONCLUÍDA

**Objetivo:** impedir remoção de texto clínico pela anonimização, cobrir PII relevante e preservar protocolos, FAQs, laudos, receitas e procedimentos.

**Depende de:** DAT-01.

**Aceite:** testes de PII e frases clínicas passam; nenhum dado bruto é alterado por efeito colateral; conjunto curado tem metadados de origem, versão e finalidade.

**Evidência (21/09/2026):** `src/medical_assistant/data/preprocessing.py` usa cópia profunda e anonimização contextual; `tests/test_preprocessing.py` cobre PII e preservação de termos clínicos. A execução gerou `data/curated/dataset_medico_anonimizado.json` com metadados e exemplos de protocolos, FAQs, laudos, receita e procedimentos. Validação inicial: `python -m pytest fase3/tests/test_preprocessing.py` (2 aprovados); a DAT-03 ampliou a suíte para 3 testes.

### DAT-03 — Criar splits para treino e avaliação

**Status:** CONCLUÍDA

**Objetivo:** produzir treino, validação e teste sem vazamento e com cobertura de cada tipo de exemplo.

**Depende de:** DAT-02.

**Aceite:** split determinístico por seed, distribuição registrada, receita incluída quando aplicável e teste reservado não consumido no treinamento.

**Evidência (21/09/2026):** `src/medical_assistant/data/split_dataset.py --seed 42` gera `data/splits/train.jsonl`, `validation.jsonl`, `test.jsonl` e `manifest.json`. A fonte sintética foi ampliada para 25 exemplos (cinco por tipo), resultando em 15/5/5 exemplos e cobertura de todas as categorias em cada split. `RECEITA_004` está reservado no teste e não aparece no treino. Validação: `pytest fase3/tests/test_preprocessing.py` (3 aprovados) e manifesto sem categorias esparsas.

### DAT-04 — Ingerir amostra externa de MedQuAD

**Status:** CONCLUÍDA

**Objetivo:** baixar e curar uma amostra determinística de 1.000 pares QA do MedQuAD oficial, mantendo proveniência, licença e separação dos dados sintéticos.

**Depende de:** DAT-03.

**Aceite:** download reprodutível da fonte oficial; manifesto registra URL, revisão, hash, licença CC BY 4.0, seed, idioma e contagens; dados externos permanecem fora do Git; exemplos externos e sintéticos recebem IDs distintos; o corpus combinado é dividido sem vazamento.

**Evidência (21/09/2026):** `src/medical_assistant/data/ingest_medquad.py` baixou o MedQuAD oficial na revisão `577bd37b96c02d1833b2c9eed2de9f96964e96cb`, SHA-256 `161d948f8ce8f82accca1f5769a512dff2a01af2b3004d48f2b3da2758a38a09`. A amostra de 1.000 pares únicos foi gravada em `data/external/medquad/` com manifesto CC BY 4.0; `split_dataset.py` gerou 815/105/105 exemplos sem interseção de `source_id` ou pergunta normalizada. Reprodução e atribuição: `docs/DADOS_EXTERNOS.md`.

### DAT-05 — Compor corpus de treinamento balanceado

**Status:** CONCLUÍDA

**Objetivo:** definir e gerar uma mistura explícita entre QA externo em inglês e exemplos sintéticos em português, sem diluir idioma, limites de atuação e validação humana.

**Depende de:** DAT-04.

**Aceite:** manifesto registra proporção por origem, idioma e categoria; exemplos sintéticos de guarda/português têm representação deliberada no treino; validação e teste mantêm cobertura por origem e categoria; nenhuma duplicação atravessa splits. A escolha não pode resultar em treino uniforme de 1.000 exemplos externos contra 25 internos sem justificativa.

**Evidência (21/09/2026):** fonte sintética expandida para 225 exemplos: 25 de domínio e 200 comportamentais em português, organizados em grupos de cenário para impedir vazamento de variações. `split_dataset.py --external-limit 300` criou 405/60/60 exemplos: no treino, 165 sintéticos `pt-BR` (40,7%) e 240 MedQuAD `en` (59,3%). `manifest.json` registra distribuições por origem, idioma e categoria. Validação: `pytest fase3/tests/test_preprocessing.py` (5 aprovados), sem interseção de `source_group` ou pergunta normalizada entre splits.

### EVAL-01 — Corrigir benchmark de inferência

**Status:** CONCLUÍDA

**Objetivo:** transformar `benchmark_ollama.py` em experimento reprodutível de inferência, sem confundir latência com qualidade clínica.

**Depende de:** ORG-01.

**Aceite:** CSV inclui configuração completa e identificadores de modelo; resultados preservam respostas; checks automáticos não aprovam resposta vazia; falhas de download/execução são registradas; comandos do README refletem a CLI.

### EVAL-02 — Definir seleção de candidato treinável

**Status:** CONCLUÍDA

**Objetivo:** estabelecer scorecard para qualidade, segurança, licença, pesos-base, template/tokenizer, hardware e compatibilidade com LoRA.

**Depende de:** EVAL-01 e ORG-03.

**Aceite:** lista curta de candidatos com fontes verificáveis e decisão registrada; nenhum modelo é escolhido apenas por p95 ou tokens/s.

**Evidência (20/09/2026):** `docs/SCORECARD_MODELOS.md` e `artifacts/benchmarks/benchmark_20260920T190753Z.{csv,json,md}`. `Qwen/Qwen3-8B` foi priorizado condicionalmente para validação em CUDA; FT-01 continua bloqueado por DAT-04 e ORG-04.

### FT-01 — Implementar fine-tuning LoRA real

**Status:** PENDENTE

**Objetivo:** carregar modelo-base elegível, preparar dataset, aplicar PEFT/LoRA, treinar e salvar adapter/checkpoints.

**Depende de:** DAT-05, EVAL-02, ORG-02 e ORG-04.

**Aceite:** execução reproduzível gera `adapter_config.json`, pesos do adapter, métricas de treino/validação e manifesto com modelo-base, commit/revisão, dataset, seed e hiperparâmetros.

### FT-02 — Recarregar e servir o adapter

**Status:** PENDENTE

**Objetivo:** carregar base + adapter em inferência e expor uma interface única que será usada pelo assistente.

**Depende de:** FT-01.

**Aceite:** teste de recarga em processo novo, resposta do adapter identificada pelo `run_id`, erro claro se modelo ou adapter não existir.

### EVAL-03 — Comparar base versus modelo adaptado

**Status:** PENDENTE

**Objetivo:** executar o mesmo conjunto reservado para ambos, registrar qualidade, segurança, fontes e latência, com revisão humana.

**Depende de:** FT-02 e EVAL-01.

**Aceite:** artefatos brutos, rubrica preenchida, regressões explícitas e decisão de adoção/rejeição documentada. Sem alegações de segurança total sem evidência.

### APP-00 — Aplicar política de seleção de modelo na API

**Status:** PENDENTE

**Objetivo:** fazer a aplicação consumir a política declarada em `config.yaml`: Qwen como modo padrão e Gemma como modo rápido, sem aceitar identificadores arbitrários fornecidos pelo cliente.

**Depende de:** FT-02.

**Aceite:** contrato tipado aceita somente os modos `standard` e `fast`; ausência de modo seleciona Qwen; `fast` seleciona Gemma; a resposta e o log de auditoria registram modo, nome e digest do modelo efetivamente usado; erro de modelo indisponível é explícito; testes mockados cobrem seleção, default e erro. Não chamar modo lento de “pensa mais” nem inferir adequação clínica pela seleção.

### APP-01 — Integrar LangChain, RAG e prontuário

**Status:** PENDENTE

**Objetivo:** substituir resposta simulada por cadeia real, definir contratos tipados de paciente/contexto e retornar fontes verificáveis.

**Depende de:** ORG-02, APP-00 e DAT-03.

**Aceite:** caso com paciente usa dados atuais sem erro; LLM recebe contexto/retrieval; fontes incluem ID, título/versão e trecho; interfaces usam os mesmos contratos.

### APP-02 — Implementar LangGraph e isolar estado

**Status:** PENDENTE

**Objetivo:** converter fluxo sequencial em grafo LangGraph compilado e separar estado por execução/paciente.

**Depende de:** APP-01.

**Aceite:** `StateGraph` compilado e invocado; duas execuções não compartilham decisões/alertas; exames e sinais vitais do banco afetam o resultado esperado.

### SAFE-01 — Aplicar guardrails, auditoria e revisão humana ao fluxo real

**Status:** PENDENTE

**Objetivo:** bloquear ou encaminhar entradas/respostas inseguras antes da geração final, registrar decisão e exigir aprovação humana onde aplicável.

**Depende de:** APP-02.

**Aceite:** testes cobrem prescrições, diagnóstico definitivo, dados insuficientes e fontes ausentes; resposta bloqueada nunca alcança o modelo; log correlaciona query, modelo, fontes, decisão e aprovação.

### ENT-01 — Consolidar entrega e vídeo

**Status:** PENDENTE

**Objetivo:** revisar documentação histórica, relatório técnico e roteiro de demonstração com artefatos reais.

**Depende de:** EVAL-03 e SAFE-01.

**Aceite:** relatório explica fine-tuning executado, arquitetura real, avaliação e limites; vídeo mostra treino, resposta contextualizada, fluxo, fonte, log e validação; repositório contém somente artefatos permitidos.

## Histórico de execução

| Data | Tarefa | Resultado | Evidência |
|---|---|---|---|
| 20/09/2026 | ORG-01 | Concluída | Documentação reorganizada e diagnóstico anterior incorporado sem alegações não comprovadas. |
| 20/09/2026 | ORG-02 | Concluída | Estrutura por objetivos aplicada sem corrigir regras de negócio pendentes. |
| 20/09/2026 | EVAL-01 | Concluída | Reaberta após o run `20260920T124535Z` revelar falso negativo lexical em `HTN-002` (`validar` versus `validação`). O caso agora declara alternativas explícitas, validadas com `keyword_checks` e `py_compile`; um novo run terá seu próprio `run_id`. |
| 20/09/2026 | ORG-03 | Concluída | Ambiente limpo `fase3/.venv-org03` instalado com `requirements.txt` fixado; `pip check` passou; CLI do benchmark carregou. Evidência: `artifacts/environment/org03_20260920.json`. AMD Radeon RX 9060 XT detectada, mas PyTorch 2.14.0+cpu não expõe CUDA/HIP; treino local é CPU-only. `fase3/tests` não contém testes. |
| 20/09/2026 | EVAL-02 | Concluída | Scorecard verificou pesos, licença, template e restrição de hardware. `Qwen/Qwen3-8B` foi priorizado para validação condicional em CUDA, sem inferir qualidade clínica de checks lexicais. Evidência: `docs/SCORECARD_MODELOS.md`. |
| 21/09/2026 | DAT-01 | Concluída | Fonte sintética de 12 registros e gerador tornados reproduzíveis. Apenas `data/raw/dataset_medico.json` foi liberado do `.gitignore`; banco, dados derivados e splits permanecem locais. |
| 21/09/2026 | DAT-02 | Concluída | Anonimização contextual substituiu regex genérico, preservando frases clínicas; curadoria registra metadados e inclui todos os cinco tipos de conteúdo. Evidência: `tests/test_preprocessing.py` (2 aprovados). |
| 21/09/2026 | DAT-03 | Concluída | Fonte expandida para 25 exemplos; splits determinísticos por `source_id` criados em 15 treino, 5 validação e 5 teste, com cobertura de cinco categorias e manifesto de distribuição. |
| 21/09/2026 | DAT-04 | Concluída | MedQuAD oficial, CC BY 4.0, amostrado em 1.000 pares únicos na revisão fixada. Corpus combinado dividido em 815/105/105 sem repetição de ID ou pergunta normalizada. Evidência: `docs/DADOS_EXTERNOS.md` e manifestos locais ignorados pelo Git. |
| 21/09/2026 | DAT-05 | Concluída | Corpus balanceado com 225 exemplos sintéticos em português e seleção determinística de 300 MedQuAD em inglês. Splits 405/60/60 sem vazamento de grupo ou pergunta. |

## Decisões registradas

| ID | Decisão | Motivo | Reavaliar em |
|---|---|---|---|
| DEC-01 | RAG será complementar ao fine-tuning | O edital exige fine-tuning e RAG não altera pesos do modelo | FT-01 |
| DEC-02 | Benchmark de Ollama é avaliação de inferência, não seleção final de treino | Quantização/latência não garantem pesos treináveis, licença ou qualidade adaptada | EVAL-02 |
| DEC-03 | Manter uma aplicação modular, sem microserviços | Escopo acadêmico não justifica custo operacional extra | ENT-01 |
| DEC-04 | Validar Colab/CUDA antes de FT-01; manter Linux/ROCm como alternativa | RX 9060 XT é `gfx1200`, mas o PyTorch atual no Windows é CPU-only e a matriz Radeon do Windows não oferece backward pass | ORG-04 |
| DEC-05 | Priorizar `Qwen/Qwen3-8B` para validação LoRA/QLoRA em CUDA | Licença Apache-2.0, pesos oficiais disponíveis e custo mais compatível que 27B; decisão condicionada a ORG-04 e não baseada apenas em latência | FT-01 |
| DEC-06 | Limitar MedQuAD a 300 exemplos e incluir 200 exemplos comportamentais sintéticos em português | Evitar que 1.000 QA externos em inglês diluam idioma, fontes e guardrails do assistente acadêmico | FT-01 |
