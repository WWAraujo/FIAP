# Plano de execução — Fase 3

Atualizado em 28/09/2026. Este documento controla o trabalho da Fase 3. Uma tarefa só muda para concluída após satisfazer seus critérios de aceite e registrar evidência abaixo. Não marque uma implementação parcial como concluída.

## Fluxo de dependências

```text
ORG-01 ─┬─ DAT-01 ─ DAT-02 ─ DAT-03 ─ DAT-04 ─ DAT-05 ─┬─ FT-01 ─ FT-02 ─ EVAL-03 ─┐
        │                                                └─ DAT-06 ─ FT-03 ─ EVAL-04 ─┤
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

**Status:** CONCLUÍDA

**Objetivo:** validar um ambiente acelerado para treinamento. A rota prioritária é Colab com GPU CUDA; Linux nativo + ROCm para a AMD Radeon RX 9060 XT (`gfx1200`) permanece alternativa. O Windows local usa PyTorch CPU-only e não tem backward pass suportado.

**Depende de:** ORG-03.

**Aceite:** registrar provedor, GPU, VRAM, driver, PyTorch e versões; `torch.cuda.is_available()` é verdadeiro; um forward e backward pass mínimos passam na GPU; limitações de memória para LoRA são medidas. No Colab, validar persistência de checkpoint no Drive. Se o acelerador falhar, registrar a evidência e não iniciar FT-01.

**Evidência parcial (23/09/2026):** implementado `scripts/verify_accelerator.py`, com relatório exclusivo por `run_id`, forward/backward e pico de memória do tensor mínimo. No Linux/Python 3.14.4 (`fase3/.venv/bin/python`), `-m unittest discover -s fase3/tests -p test_verify_accelerator.py -v` passou os 3 testes sem rede. `fase3/.venv/bin/python fase3/scripts/verify_accelerator.py --provider local` retornou código 1: PyTorch ausente; `nvidia-smi` e `rocminfo` não encontrados. Relatório: `artifacts/environment/org04_20260923T123828Z_35427478.json`. A sessão também não expõe `/dev/kfd` ou `/dev/dri`. Pendente executar em ambiente com GPU, medir memória para LoRA do candidato e validar persistência no Drive caso se use Colab. FT-01 não foi iniciada.

**Preparação para Colab (26/09/2026):** criado `notebooks/treinamento_qwen3_8b_colab.ipynb`. Ele monta o Drive, valida GPU CUDA com forward/backward, reproduz corpus e splits, baixa `Qwen/Qwen3-8B` na revisão `9c925d64d72725edaf899c6cb9c377fd0709d9c5`, mede memória QLoRA, treina LoRA e persiste adapter, checkpoints, métricas, manifestos e evidência de ambiente no Drive. A execução no Colab e a recarga em uma nova sessão continuam necessárias para concluir ORG-04 e então iniciar FT-01.

**Preparação local ROCm (26/09/2026):** criado `scripts/train_qwen3_8b_rocm.py`, cuja validação exige `torch.version.hip` e GPU disponível; o script realiza forward/backward mínimo, mede memória QLoRA e, sem `--validate-only`, reproduz dados, treina, salva adapter/checkpoints/métricas/manifesto e recarrega o adapter. `py_compile` e `--help` passaram no ambiente atual. A sessão automatizada não expõe dispositivos ROCm, portanto a validação real deve ser executada no host com ROCm 10.0 antes de concluir ORG-04.

**Pré-check local (26/09/2026):** criado `scripts/check_rocm_before_training.py`, que não baixa modelo nem treina; ele registra permissões de `/dev/kfd` e `/dev/dri/renderD128`, `rocminfo`, PyTorch HIP e disponibilidade da GPU. A execução na sessão automatizada falhou nos três checks esperadamente, pois ela não expõe dispositivos ROCm e o ambiente Python contém PyTorch CUDA. Execute-o no terminal do host com a venv ROCm antes de `train_qwen3_8b_rocm.py --validate-only`.

**Evidência local ROCm (27/09/2026):** host Ubuntu 26.04 com RX 9060 XT (`gfx1200`) executou forward/backward QLoRA com pico de 9,30 GiB e treinamento de 153 passos. Ambiente e métricas: `llm/artifacts/adapters/20260927T020236Z_qwen3_8b_qlora_rocm/environment.json` e `manifest.json`.

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

### DAT-06 — Preparar corpus externo em português para a segunda tentativa

**Status:** CONCLUÍDA

**Objetivo:** substituir MedQuAD no corpus de fine-tuning por uma amostra reproduzível de 5.000 pares MedPT em português brasileiro, preservando os exemplos sintéticos de segurança e mantendo benchmarks externos fora de todos os splits.

**Depende de:** DAT-05. Os achados parciais de EVAL-03 orientam esta correção, mas não são um pré-requisito concluído.

**Aceite:** MedPT baixado em revisão SHA registrada; manifesto contém licença, hashes, filtros, seed e contagens; os splits não contêm MedQuAD, HealthQA-BR ou DrBodeBench; não há vazamento por `source_id` ou `source_group`; o plano de treino registra época, learning rate e limite de geração revisados.

**Evidência (28/09/2026):** `ingest_medpt.py --revision 24aeac298da0f1b46e6e7360eeeb0d36e711779d --sample-size 5000 --seed 42` gerou a nova amostra MedPT. Dos 384.095 pares, 46.919 permaneceram após deduplicação e remoção das categorias Diagnosis/Tratamento; a amostra possui SHA-256 `d728ea5048d91a1b3d1b934a0ab83dbdf0be089805eed6273304b151f4b88512`. `split_dataset.py` gerou 4.165/530/530 exemplos (treino/validação/teste), todos `pt-BR`: respectivamente 4.000/500/500 MedPT e 165/30/30 sintéticos. A verificação confirmou ausência de `MEDQUAD`, sem interseção por ID ou grupo. Fontes de avaliação e PCDTs para RAG: `llm/docs/DADOS_EXTERNOS.md`. O treinamento passa a usar por padrão 1 época e learning rate `1e-4`; EVAL-03 passa a gerar até 512 tokens.

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

**Status:** CONCLUÍDA

**Objetivo:** carregar modelo-base elegível, preparar dataset, aplicar PEFT/LoRA, treinar e salvar adapter/checkpoints.

**Depende de:** DAT-05, EVAL-02, ORG-02 e ORG-04.

**Aceite:** execução reproduzível gera `adapter_config.json`, pesos do adapter, métricas de treino/validação e manifesto com modelo-base, commit/revisão, dataset, seed e hiperparâmetros.

**Evidência (27/09/2026):** QLoRA local do `Qwen/Qwen3-8B` concluiu 3 épocas e 153 passos. Métricas: `train_loss=0,9034` e `eval_loss=0,674`. Artefatos: `llm/artifacts/adapters/20260927T020236Z_qwen3_8b_qlora_rocm/adapter/adapter_model.safetensors`, `adapter_config.json` e `manifest.json`.

### FT-02 — Recarregar e servir o adapter

**Status:** CONCLUÍDA

**Objetivo:** carregar base + adapter em inferência e expor uma interface única que será usada pelo assistente.

**Depende de:** FT-01.

**Aceite:** teste de recarga em processo novo, resposta do adapter identificada pelo `run_id`, erro claro se modelo ou adapter não existir.

**Evidência (27/09/2026):** `llm/scripts/run_qwen3_adapter_rocm.py --adapter-dir llm/artifacts/adapters/20260927T020236Z_qwen3_8b_qlora_rocm/adapter` recarregou o adapter em processo novo e retornou resposta não vazia, identificada pelo `run_id` `20260927T020236Z_qwen3_8b_qlora_rocm`. Latência total de carga e primeira geração: 77,33 s.

### FT-03 — Treinar a versão portuguesa do adapter

**Status:** CONCLUÍDA

**Objetivo:** executar uma nova tentativa QLoRA com os splits DAT-06, uma época, preenchimento fixo até 384 tokens, learning rate `1e-4`, otimizador `adamw_torch`, atenção `eager`, BLAS `hipblas` e checkpointing não reentrante, sem substituir os artefatos da primeira tentativa.

**Depende de:** DAT-06 e ORG-04.

**Aceite:** novo `run_id` contém adapter, checkpoints, métricas, manifesto com hashes dos splits MedPT e validação de recarga; não reutiliza ou sobrescreve o adapter de 27/09/2026.

**Evidência (28/09/2026):** a execução de diagnóstico com `--max-steps 30` ultrapassou os passos 12 e 22, que antes encerravam com `HSA_STATUS_ERROR_MEMORY_FAULT`, e concluiu treino, avaliação, predição do teste e recarga. A execução integral de uma época concluiu sem falha HSA, OOM ou reinicialização: `train_loss=1,8799`, `eval_loss=1,8164`, duração de treino de 7.366 s. Adapter, métricas, manifesto e recarga: `llm/artifacts/adapters/20260928T100950Z_qwen3_8b_qlora_rocm/`.

### EVAL-03 — Comparar base versus modelo adaptado

**Status:** BLOQUEADA

**Objetivo:** executar o mesmo conjunto reservado para ambos, registrar qualidade, segurança, fontes e latência, com revisão humana.

**Depende de:** FT-02 e EVAL-01.

**Aceite:** artefatos brutos, rubrica preenchida, regressões explícitas e decisão de adoção/rejeição documentada. Sem alegações de segurança total sem evidência.

**Evidência parcial (28/09/2026):** `llm/scripts/evaluate_qwen3_base_vs_adapter_rocm.py` comparou os 60 casos reservados. Ambos retornaram resposta não vazia em 100% dos casos; latência média do base foi 7,38 s e do adapter 10,18 s. Artefatos brutos: `llm/artifacts/evaluations/20260928T005225Z_eval03/{base,adapter}.jsonl` e `summary.json`. Inspeção inicial de `ALERTA_URGENCIA_01_*`: o base lista hipóteses diagnósticas, enquanto o adapter encaminha para avaliação imediata e inclui uma fonte textual. A fonte não foi recuperada por RAG, portanto permanece `source_available=false`. Falta rubrica e revisão humana dos 60 pares antes da decisão de adoção.

**Bloqueio (28/09/2026):** a revisão assistida apontou aderências inadequadas, regressões factuais e mistura de idiomas no corpus que contém MedQuAD. A decisão provisória é rejeitar esta tentativa para adoção e registrar uma nova comparação em EVAL-04 após FT-03.

### EVAL-04 — Avaliar a versão portuguesa do adapter

**Status:** CONCLUÍDA

**Objetivo:** comparar o adapter FT-03 e o modelo-base em uma amostra estratificada de até 50 casos do split reservado DAT-06, com até 512 tokens, e preparar revisão humana de regressões, idioma e segurança.

**Depende de:** FT-03.

**Aceite:** respostas brutas, parâmetros, latência e rubrica documentam a comparação no conjunto reservado; benchmarks HealthQA-BR e DrBodeBench permanecem fora do treino e só entram após métrica própria de múltipla escolha.

**Evidência parcial (28/09/2026):** `evaluate_qwen3_base_vs_adapter_rocm.py` comparou base e FT-03 em 50 casos estratificados do teste reservado, com seed 42 e até 512 tokens. Ambos retornaram respostas não vazias em 100%; latência média: base 22,850 s e adapter 18,536 s. Artefatos: `llm/artifacts/evaluations/20260928T174344Z_eval03/{base,adapter}.jsonl` e `summary.json`. O sufixo `_eval03` é o identificador histórico emitido pelo script. Falta anexar os três arquivos ao prompt de revisão e registrar a avaliação humana antes da decisão acadêmica.

**Revisão humana (28/09/2026):** a validação confirmou que o adapter não ficou adequado. A rubrica baseada nas respostas brutas registrou aderência adequada em 9 casos, contra 25 no base; segurança adequada em 27, contra 34; 18 regressões materiais, contra 6. Foram identificados loops de repetição, perda de conteúdo, conclusões categóricas e procedimentos inventados no adapter. A menor latência não altera a decisão de qualidade. O adapter FT-03 foi rejeitado para adoção; essa evidência não constitui validação clínica.

### FT-04 — Corrigir truncagem e retreinar de forma controlada

**Status:** CONCLUÍDA — adapter rejeitado

**Objetivo:** corrigir a preparação de exemplos longos, preservando prompt e resposta supervisionada, e executar nova tentativa QLoRA com os mesmos splits MedPT da DAT-06 e learning rate de `5e-5`.

**Depende de:** EVAL-04.

**Aceite:** teste unitário comprova que um exemplo longo não é convertido em resposta isolada; novo `run_id` registra adapter, checkpoints, métricas, manifesto, hashes dos mesmos splits e recarga em processo novo. A decisão entre configurações usa somente a validação; o teste DAT-06 já revisado em EVAL-04 não será reutilizado para selecionar hiperparâmetros.

**Evidência parcial (28/09/2026):** `llm/scripts/train_qwen3_8b_rocm.py` passou a reter uma parte do prompt e da resposta em exemplos acima de 384 tokens; o padrão de `learning_rate` mudou de `1e-4` para `5e-5`. Validação: `fase3/llm/.venv/bin/python -m pytest fase3/llm/tests/test_training_data.py fase3/llm/tests/test_preprocessing.py` (8 aprovados). A execução em GPU e a comparação na validação permanecem pendentes.

**Evidência e decisão (30/09/2026):** o preflight ROCm passou em uma AMD Radeon RX 9060 XT de 16 GiB; o forward/backward QLoRA atingiu 9,21 GiB alocados e 10,62 GiB reservados. O treino de uma época (`learning_rate=5e-5`, 4.165 exemplos, seed 42, `max_length=384`) concluiu em 7.368,89 s, com `train_loss=1,9245` e `eval_loss=1,8648`. O adapter, manifesto, hashes dos três splits e recarga em processo novo estão em `llm/artifacts/adapters/20260930T102822Z_qwen3_8b_qlora_rocm/`.

Na seleção de configuração foi usada somente a validação DAT-06: `llm/scripts/evaluate_qwen3_base_vs_adapter_rocm.py --adapter-dir .../20260930T102822Z_qwen3_8b_qlora_rocm/adapter --test-file data/splits/validation.jsonl --limit 50 --seed 42 --max-new-tokens 512`. Os pares brutos e o resumo estão em `llm/artifacts/evaluations/20260930T131842Z_eval03/`. Base e adapter responderam aos 50 casos, com latência média de 23,851 s e 22,108 s, respectivamente; o campo `test_count` do resumo é um nome histórico do script, mas o arquivo de entrada foi explicitamente `validation.jsonl` (530 registros). O teste reservado não foi reutilizado.

O adapter é rejeitado: houve loop de repetição em 12/50 respostas, várias delas atingindo cerca de 60 s e o teto de geração. Exemplos reproduzíveis: `FAQ_001` repete “dor nas costas”; `MEDPT:106932` repete a regra de 50%; `MEDPT:138850` entra em ciclo de “falta de absorção”; `MEDPT:723679` afirma, sem sustentação, que rejeição infantil pode gerar retardo mental; `MEDPT:95411` transforma pneumonia viral em indicação repetida de antibióticos; e `MEDPT:223961` inventa medidas preventivas para torção testicular. A inspeção de tokenização mostrou que só 127/4.165 exemplos (3,0%) excedem 384 tokens; portanto, a correção de truncagem não é explicação suficiente para o padrão. Não iniciar outro fine-tuning nos mesmos alvos MedPT sem curadoria semântica e validação independente da qualidade dos dados. Esta conclusão é experimental e não constitui validação clínica.

### DAT-07 — Curadoria controlada antes de novo fine-tuning

**Status:** CONCLUÍDA — triagem e revisão textual; MedPT não aprovado

**Objetivo:** auditar somente o treino DAT-06 quanto a duplicatas, repetição, encerramento, comprimentos e proveniência; definir critérios reproduzíveis de exclusão e um corpus candidato menor, rastreável, sujeito a revisão semântica independente antes de novo treinamento.

**Depende de:** FT-04 concluída com adapter rejeitado.

**Aceite:** preservar splits e artefatos anteriores; registrar hashes das entradas, regras, contagens por origem, decisões por exemplo e distribuição de comprimentos; distinguir triagem automática de revisão clínica; manter validação para seleção e não inspecionar nem usar o conteúdo do teste reservado; não retreinar alvos MedPT brutos. A aprovação do corpus exige evidência verificável de revisão semântica, não apenas filtros automáticos.

**Início (30/09/2026):** leitura dos sete documentos solicitados e `git diff -- fase3/docs/PLANO_EXECUCAO.md` confirmaram FT-04 rejeitada. A ferramenta PyCharm `get_python_environment` não está exposta nesta sessão; nenhum comando Python será executado sem resolver essa condição. Manter `smtek/Qwen3.8-27B:Q3_K_M` como candidato principal de inferência. A auditoria não altera a seleção de modelos nem autoriza deploy.

**Execução retomada (30/09/2026):** o ambiente PyCharm foi consultado com sucesso: Python 3.14.4 em `llm/.venv`. A auditoria `audit_training_corpus.py --output-dir artifacts/data_audits/20260930_dat07_resume --seed 42 --medpt-limit 50` encontrou 4.165 registros, 161 respostas redundantes, 810 sem pontuação terminal, 127 acima de 384 tokens e 7 com repetição forte. A triagem gerou 79 candidatos (29 sintéticos e 50 MedPT), sem aprovação clínica. Os artefatos incluem hashes e decisões técnicas por registro. Respostas idênticas para perguntas diferentes não serão descartadas automaticamente no piloto de política: podem representar o mesmo limite de atuação em diferentes situações. A revisão textual dos cinco grupos sintéticos será registrada separadamente dos filtros; MedPT e exemplos clínicos factuais permanecerão fora do próximo treino.

**Decisão (30/09/2026):** `policy_semantic_review.jsonl` registra a revisão textual por IA de 150 exemplos sintéticos de política, separada da triagem automática. Foi autorizado apenas um piloto de comportamento com 145 registros (29 perguntas únicas, 25 alvos), removendo citações de fontes internas não fornecidas e excluindo cinco variantes de pedido de diagnóstico de infarto cujo alvo omitia encaminhamento imediato. O corpus, hashes, alterações e limitações estão em `llm/artifacts/data_audits/20260930_dat07_resume/policy_pilot_manifest.json` e `policy_pilot/data/splits/train.jsonl`. Nenhum alvo MedPT foi aprovado semanticamente ou incluído. Esta revisão não é clínica; aprovação clínica do MedPT continua pendente e não bloqueia o ensaio limitado de política.

### FT-05 — Piloto sintético de limites de atuação

**Status:** CONCLUÍDA — adapter rejeitado para adoção

**Objetivo:** partir novamente de Qwen/Qwen3-8B, sem carregar adapters rejeitados, e testar 145 exemplos sintéticos revisados de política, duas épocas, LR `2e-5`, comprimento 192, seed 42, QLoRA r=8 e acumulação 8. Hipótese: reduzir a interferência dos alvos clínicos e das citações fabricadas, preservando a resposta útil da base.

**Depende de:** DAT-07 (corpus de política, sem MedPT).

**Aceite:** novo adapter, manifesto e hashes; recarga independente; comparação em 50 casos estratificados da validação DAT-06 com os parâmetros anteriores (512 tokens, thinking desativado, greedy, seed 42), revisão de loops, urgência, recusa e regressões. Decisão por validação, sem pontuar o teste reservado. A validade externa é limitada por 29 perguntas únicas e templates correlacionados; latência não decide qualidade.

**Comando (30/09/2026, em `fase3/llm`):** `.venv/bin/python scripts/train_qwen3_8b_rocm.py --project-dir artifacts/data_audits/20260930_dat07_resume/policy_pilot --model-dir models/base/Qwen3-8B --output-dir artifacts/adapters --epochs 2 --learning-rate 2e-5 --max-length 192`. Validação e teste são links aos splits originais para compatibilidade com o executor. O executor lê o teste para verificar sobreposição e também o tokeniza em CPU; seus tensores não são usados pelo Trainer, perda de validação, geração ou seleção. Correção de registro em 30/09: a descrição anterior “somente para verificar sobreposição” estava incompleta; nenhum alvo de teste foi revisado semanticamente neste ciclo.

**Treino concluído (30/09/2026):** `llm/artifacts/adapters/20260930T164016Z_qwen3_8b_qlora_rocm/` contém o adapter, manifesto e ambiente. `train_runtime=850,0714 s`, `train_loss=2,7483`, `eval_loss=2,5003`; o segundo checkpoint melhorou a perda de validação frente à primeira época (2,649). A recarga interna retornou resposta não vazia. Log completo: `llm/artifacts/data_audits/20260930_dat07_resume/ft05_training.log`. Os 14 testes existentes de auditoria, truncagem e pré-processamento passaram. A seleção de 50 IDs da validação coincide exatamente com a FT-04. Comparação de geração em processo novo ainda em andamento; não adotar o adapter antes da revisão.

**Avaliação concluída (30/09/2026):** `llm/artifacts/evaluations/20260930T175422Z_eval03/`: recarga independente e 100 respostas não vazias. Base/adapter: latência média 23,39394/39,51028 s; teto atingido 32/19 casos. Adapter omite encaminhamento imediato nos dois alertas, mantém loop grave em MEDPT:522763 e confusão de receituário com tributação. Revisão por IA em `review.md`, `qualitative_review.jsonl` e `qualitative_summary.json`; três referências não avaliáveis, sem pontuação clínica de segurança. IDs e hash da validação conferidos. Decisão: preservar artefatos e não adotar o FT-05.

### EVAL-05 — Ablação do prompt de sistema na validação

**Status:** CONCLUÍDA — concisão comprovada; adapter e prompt v2 rejeitados para adoção

**Objetivo:** avaliar base Qwen3-8B e FT-05 com prompt curto que explicita contexto médico, limites e encaminhamento em urgência, sem novo treinamento. Manter os mesmos 50 IDs da FT-05, seed 42, greedy, thinking desativado e teto 512. O pacote de instruções não permite atribuir efeito a uma frase isolada.

**Depende de:** FT-05.

**Aceite:** prompt com conteúdo e hash, respostas brutas, comparação dos alertas, recusa, receituário, truncagem e latência; decisão registrada. Validação reutilizada é desenvolvimento, não teste independente. Sem gerar ou consultar respostas do teste reservado.

**Execução iniciada (30/09/2026):** em `fase3/llm`, `.venv/bin/python -u scripts/evaluate_qwen3_base_vs_adapter_rocm.py --adapter-dir artifacts/adapters/20260930T164016Z_qwen3_8b_qlora_rocm/adapter --test-file data/splits/validation.jsonl --limit 50 --seed 42 --max-new-tokens 512 --system-prompt-file artifacts/evaluations/20260930_eval05_prompt_ablation/system_prompt.txt`. Log por caso: `llm/artifacts/evaluations/20260930_eval05_prompt_ablation/execution.log`. O prompt não contém respostas de referência nem os IDs da amostra.

**Resultado (30/09/2026):** `llm/artifacts/evaluations/20260930T182438Z_eval03/`: base/adapter com 50 respostas não vazias, zero tetos, média de tokens 87,38/75,52 e latência 5,03/9,08642 s. Encaminhamento imediato aparece nos dois alertas, mas há fatos inventados por cópia do sistema em FONTE_E_VALIDACAO_08_02 e prevenção fabricada em MEDPT:223961 no adapter. Revisão e decisão em `review.md`, `review.jsonl`, `review_summary.json`. Não adotar FT-05 nem prompt v2 sem correções.

### EVAL-06 — Instrução condicional e comparação com Q3 local

**Status:** CONCLUÍDA — Q3 candidato principal de inferência

**Objetivo:** distinguir regras do sistema de fatos efetivamente informados, preservar concisão e recusa explícita, e comparar Qwen3-8B base com `smtek/Qwen3.8-27B:Q3_K_M` nos mesmos 50 casos da validação. Não carregar adapters rejeitados nem treinar nas respostas de validação. Registrar digest/metadados do modelo Ollama, prompt, parâmetros e dados.

**Depende de:** EVAL-05.

**Aceite:** respostas brutas e logs, revisão de urgência, dados ausentes, fonte, receituário, conduta e falhas factuais; decisão documentada. Ollama GGUF e Transformers NF4 são sistemas diferentes: latência e diferenças não isolam tamanho ou quantização. Não consumir teste reservado.

**Execução iniciada (30/09/2026):** controle com `.venv/bin/python -u scripts/evaluate_qwen3_base_vs_adapter_rocm.py --base-only --test-file data/splits/validation.jsonl --limit 50 --seed 42 --max-new-tokens 512 --system-prompt-file artifacts/evaluations/20260930_eval06_conditional_prompt/system_prompt.txt`. Log: `llm/artifacts/evaluations/20260930_eval06_conditional_prompt/base_execution.log`. Novo executor Ollama reutiliza a seleção, registra digest/metadados e grava cada resposta imediatamente. Integração com API local simulada validou 15 casos, incluindo erro HTTP, resposta vazia e thinking que não conta como resposta (`software_check.json`). Q3 será executado após liberação da GPU pelo controle, sem coexistência dos dois modelos na VRAM.

**Progresso:** controle concluído em `llm/artifacts/evaluations/20260930T184206Z_eval03/`, latência média 6,53782 s, 50 respostas, zero tetos. Q3 iniciou em `llm/artifacts/evaluations/20260930T184234Z_ollama_validation/` com `.venv/bin/python -u scripts/evaluate_ollama_validation.py --model smtek/Qwen3.8-27B:Q3_K_M --test-file data/splits/validation.jsonl --limit 50 --seed 42 --max-new-tokens 512 --num-ctx 2048 --system-prompt-file artifacts/evaluations/20260930_eval06_conditional_prompt/system_prompt.txt`. Log `q3_execution.log` no diretório da instrução. API `/api/ps` após warmup informa tamanho carregado e VRAM iguais (13.450.073.865 bytes), contexto 2048; snapshot ROCm em `q3_hardware_snapshot.txt`. O nome local Qwen3.8 é um alias: metadados declaram família `qwen35`, não comprovam identidade/revisão oficial de pesos Qwen3.8.

**Resultado (30/09/2026):** base/Q3 com 50 respostas não vazias e zero tetos/erros. Revisão documental por IA nas 47 referências elegíveis: base 16 adequado/24 parcial/7 inadequado; Q3 24/23/0; três não avaliáveis em ambos. Latência 6,53782/9,40926 s. Q3 não inventa atendimento no resumo auditável, preserva encaminhamentos e recusa, mas mantém extrapolações e oito respostas acima de 120 palavras. Comparação em `llm/artifacts/evaluations/20260930_eval06_conditional_prompt/comparison_review.md`, `comparison_review.jsonl`, `comparison_summary.json`. Escolha provisória: Q3 para inferência; 8B controle treinável; FT-05 rejeitado, sem adoção na API ou validação clínica.

### FT-06 — Reduzir intensidade de adaptação no piloto de política

**Status:** CONCLUÍDA — adapter rejeitado

**Objetivo:** testar LR `5e-6` (quatro vezes menor que FT-05), mantendo base limpa Qwen3-8B, os mesmos 145 exemplos de política, duas épocas, comprimento 192, r=8, alpha 32 e demais parâmetros. Não carregar adapters rejeitados. Hipótese: preservar a utilidade factual da base e reduzir interferência, sem supor que templates acrescentem conhecimento médico.

**Depende de:** EVAL-06.

**Aceite:** adapter/manifesto/recarga, comparação nos mesmos 50 casos e prompt original da FT-05 para isolar a alteração de LR; decisão por urgência, recusa, repetição e regressões. Manter Q3 candidato de inferência. Não usar teste reservado nem MedPT no treino. Se persistir o padrão de falha, rejeitar e registrar que repetir intensidade/corpus não resolve o problema de dados/conhecimento.

**Execução iniciada (30/09/2026):** `.venv/bin/python -u scripts/train_qwen3_8b_rocm.py --project-dir artifacts/data_audits/20260930_dat07_resume/policy_pilot --model-dir models/base/Qwen3-8B --output-dir artifacts/adapters --epochs 2 --learning-rate 5e-6 --max-length 192`. Log `llm/artifacts/data_audits/20260930_dat07_resume/ft06_training.log`. Q3 descarregado via API, com `/api/ps` vazio antes do treino. Avaliador ROCm agora cria diretório/arquivo de parâmetros antes da geração e preserva cada resposta imediatamente, para registrar o progresso mesmo em interrupção manual; não muda prompt ou decodificação padrão.

**Progresso:** run `20260930T185850Z_qwen3_8b_qlora_rocm`. O avaliador passou integração com modelo simulado nos modos conjunto, `--base-only` e `--adapter-only`, incluindo persistência de parâmetros/JSONL e teto (`llm/artifacts/data_audits/20260930_dat07_resume/evaluator_incremental_smoke.json`). A comparação FT-06 reutilizará o controle base da FT-05 (`20260930T175422Z_eval03/base.jsonl`) e gerará todos os 50 casos do novo adapter com `--adapter-only`; registrar hashes/IDs e a origem das latências separadas, sem repetir geração do controle inalterado.

**Treino concluído:** `train_runtime=856,6552 s`, `train_loss=3,656886`, validação `3,295993`, segunda época selecionada; recarga interna com resposta não vazia em ALERTA_URGENCIA_08_01, processo finalizado com código 0. `ft06_training_comparison.json` confirma modelo, splits e versões idênticos à FT-05, com única mudança de configuração em LR. Controle conferido em `ft06_base_control_provenance.json`. Avaliação independente iniciada com `.venv/bin/python -u scripts/evaluate_qwen3_base_vs_adapter_rocm.py --adapter-only --adapter-dir artifacts/adapters/20260930T185850Z_qwen3_8b_qlora_rocm/adapter --test-file data/splits/validation.jsonl --limit 50 --seed 42 --max-new-tokens 512`; log `llm/artifacts/data_audits/20260930_dat07_resume/ft06_evaluation.log`.

**Resultado final:** avaliação `20260930T192204Z_eval03`, 50/50 respostas não vazias, 32 tetos, 455,3 tokens e 52,7242 s médios. Revisão documental: FT-06 7 adequados / 32 parciais / 8 inadequados / 3 não avaliáveis; controle histórico base 9/31/7/3 e FT-05 8/29/10/3. Todos os casos anotados com hash em `llm/artifacts/evaluations/20260930T192204Z_eval03/qualitative_review.jsonl`; decisão e limitações em `review.md` e `qualitative_summary.json`. Segurança clínica não pontuada. Rejeitar FT-06: a menor LR não trouxe ganho consistente; não repetir o mesmo piloto com outra intensidade sem nova hipótese.

### FT-07 — Verificar capacidade local para Qwen3-14B treinável

**Status:** CONCLUÍDA — treinamento viável; adapter rejeitado

**Objetivo:** ampliar capacidade da base mantendo arquitetura Qwen3 e pesos oficiais Apache-2.0. Fixar revisão, baixar somente arquivos necessários e medir forward/backward QLoRA real sem offload, antes de decidir treino. Não inferir capacidade de treino a partir do GGUF de inferência.

**Depende de:** FT-06.

**Aceite:** registrar proveniência, espaço, memória e resultado do probe; se couber, realizar treinamento e comparação base/adapter nos mesmos 50 IDs com prompt condicional v3. Se falhar, preservar evidência e escolher próxima hipótese sem sucessivas tentativas idênticas de OOM. Teste reservado permanece fora das métricas e seleção.

**Execução (30/09/2026):** modelo oficial `Qwen/Qwen3-14B`, Apache-2.0, revisão `40c069824f4251a91eefaf281ebe4c544efd3e18`. Download concluído: 29.536.614.400 bytes de tensores, oito shards, 14.768.307.200 parâmetros conferidos nos cabeçalhos e limites dos arquivos. Proveniência/checks em `llm/artifacts/data_audits/20260930_dat07_resume/ft07_*.json`; 37 GB livres após download. Trainer recebeu seleção explícita de modelo/revisão e nome de run, preservando padrão 8B; dois testes de preparo de treino e import/CLI passaram. Probe iniciado com `--model-id Qwen/Qwen3-14B --model-dir models/base/Qwen3-14B --project-dir artifacts/data_audits/20260930_dat07_resume/policy_pilot --max-length 192 --learning-rate 5e-6 --validate-only`; log `ft07_capacity_probe.log`. Pico incluindo otimizador é registrado separadamente do pico de backward.

**Limitação observada na carga:** run de probe `20260930T202450Z_qwen3_14b_qlora_rocm` apresentou pausas prolongadas; a máquina chegou a aproximadamente 29 GiB de RAM usada e 4,3 GiB de swap antes de completar a carga. `ft07_loading_pressure.txt` preserva uma amostra, não um pico contínuo. O código instalado já usa carga síncrona quando quantiza pesos, portanto não atribuir o problema a carregamento paralelo. Amostra de pilha via gdb indisponível por restrição ptrace; não houve mudança de privilégios nem dependências. Configuração exige todos os pesos na GPU; uso temporário de RAM na conversão não prova offload dos pesos no treino. Ainda pendente concluir forward/backward e otimizador.

**Resultado do primeiro probe:** terminou com `torch.OutOfMemoryError` no forward/LoRA dropout, antes de backward/otimizador. Erro: tentativa de 2 MiB, 12,38 GiB alocados, 3,03 GiB reservados sem uso, 6 MiB livres. Não houve treino nem adapter 14B produzido. Fragmentação é uma hipótese apoiada pelo erro, não causa comprovada. Processo encerrado e GPU liberada.

**Alternativa FT-07:** selecionar `unsloth/Qwen3-14B-bnb-4bit`, Apache-2.0, revisão `b455922269e4678c84f854fe38a6993d73614a1e`, NF4/double-quant sem offload declarado, 9.928.544.933 bytes de tensores pelo índice. Safetensors pré-quantizados permitem LoRA; não são GGUF. Origem da quantização é comunitária e não se comprovou equivalência bit a bit com a revisão oficial anterior. Não usar a variante Dynamic (`-unsloth-bnb-4bit`), que mantém mais pesos em BF16 e exige mais memória após preparação. Novo probe preservará 192/145/r8; cache é liberado após preparação, allocator expansível solicitado via ambiente. Executor agora registra ambiente antes da carga e antes do forward, exige todos os parâmetros na GPU e preserva etapa alcançada em caso de falha. Dois testes existentes de dados passaram após a alteração.

**Segundo probe aprovado:** run `20260930T212100Z_qwen3_14b_bnb_4bit_qlora_rocm`, estágio `qlora_probe_completed`, forward/backward e AdamW concluídos. 1.003 parâmetros registrados em `cuda:0`, nenhum em CPU. Pico incluindo otimizador: 14.664.270.848 bytes alocados (13,66 GiB), 16.236.150.784 reservados (15,12 GiB), GPU 15,92 GiB. Carregamento dos pesos pré-quantizados concluiu em aproximadamente cinco segundos; não atribuir causalmente o sucesso só ao allocator, pois cache e caminho dos pesos também mudaram. Treino completo iniciado com base limpa pré-quantizada, mesmos 145 exemplos, r8/alpha32, LR5e-6, duas épocas, 192 e acumulador8. Log `ft07_training.log`; pico do treino/validação será medido em janela própria, sem presumir equivalência ao probe.

**Progresso do treino:** run `20260930T212304Z_qwen3_14b_bnb_4bit_qlora_rocm`; primeira época/19 passos e validação dos 530 exemplos concluídas sem erro, `eval_loss ≈ 3,03`, 402,2 s de validação, checkpoint-19 salvo. Segunda época em andamento. Ocupação total de VRAM observada próxima de 16,95 GB e junção 90 °C; são amostras, não picos contínuos. Auditoria do tokenizer 14B: os 145 exemplos têm no máximo 119 tokens completos, zero truncagens e EOS preservado em todos (`ft07_token_audit.json`). Critério e seleção da revisão de geração fixados antes das respostas em `ft07_evaluation_design.json`.

**Duas épocas concluídas:** 38 passos, checkpoint-38 selecionado por `eval_loss=2,974144697189331`, contra `3,0301058292388916` no checkpoint-19. Validações por época 402,2408 e 401,4227 s; `train_runtime ≈ 1472 s`, `train_loss ≈ 3,35` no log (consultar manifesto final para precisão completa). Avaliação final do checkpoint selecionado, salvamento do adapter e recarga ainda pendentes. Não declarar adapter final validado antes de esses passos concluírem.

**Treino/recarga concluídos:** processo código 0, adapter salvo e recarregado com saída não vazia em ALERTA_URGENCIA_08_01 (teste de carga, sem julgamento clínico). Manifesto: `train_runtime=1471,5277 s`, `train_loss=3,3500915577537134`, `eval_loss=2,974144697189331`, avaliação final 401,7624 s. Janela de treino/validação: pico 15.699.555.328 bytes alocados (14,62 GiB) / 15.925.772.288 reservados (14,83 GiB); não confundir com ocupação total do dispositivo. Comparação independente base+adapter nos mesmos 50 IDs iniciada com `--model-dir models/base/Qwen3-14B-bnb-4bit --adapter-dir artifacts/adapters/20260930T212304Z_qwen3_14b_bnb_4bit_qlora_rocm/adapter --test-file data/splits/validation.jsonl --limit 50 --seed 42 --max-new-tokens 512 --system-prompt-file artifacts/evaluations/20260930_eval06_conditional_prompt/system_prompt.txt`, allocator expansível. Log `ft07_evaluation.log`; ainda sem decisão de adoção.

**Avaliação concluída:** run `20260930T215546Z_eval03`, código 0, 50 base + 50 adapter, mesmos IDs/hash de entrada/prompt v3 verificados. Ambos 11 adequados / 31 parciais / 5 inadequados / 3 não avaliáveis; 28 textos exatamente idênticos, nenhuma mudança de classificação. Base 6,8242 s / 102,92 tokens; adapter 16,87258 s / 103,14 tokens. Zero vazios/tetos e zero respostas acima de 120 palavras; ambos misturam caracteres chineses em PROTOC_002. Todas as respostas comparadas, com reutilização explícita de grade apenas por igualdade integral e hash; alterações lidas/anotadas. `review.md`, `qualitative_summary.json` e 100 registros em `qualitative_review.jsonl`. Rejeitar FT-07 para adoção: sem ganho de aderência e latência maior; melhorias pontuais e regressões materiais descritas. Q3 continua candidato principal; 8B controle treinável. Sem validação clínica ou implantação.

### DAT-08 — Ampliar cobertura documental do piloto sintético

**Status:** CONCLUÍDA — piloto documental sem validação clínica

**Objetivo:** corrigir cobertura de treino, não aumentar base ou repetir taxa. Selecionar documentos/registros próprios existentes no split de treino, retirar faixas clínicas fixas de laudos e manter campos vazios/validação humana. Excluir exemplos com diagnóstico, classificação de risco simplificada ou conduta terapêutica não revisada. Não adicionar MedPT, alvos/perguntas da validação ou teste.

**Depende de:** FT-07.

**Aceite:** revisões/decisões por source_id e origem; exemplos completos/EOS/sem truncagem, sobreposição exata/grupos/IDs conferida, hashes e proporções explícitos. Variações lexicais não contam como conhecimento novo. Escolher tamanho de contexto pela auditoria, com base8B limpa para o próximo treino. Comparação futura permanece nos mesmos50 IDs/promptv3; validação reutilizada é desenvolvimento, não prova independente.

**Resultado:** revisados15 registros próprios do treino, nove selecionados para estrutura documental e seis excluídos por orientação/classificação/procedimento clínico que exige revisão específica. Faixas fixas de laboratório/ECG retiradas; campos vazios e validação profissional acrescentados. Cinco formulações por cada um dos nove documentos geram45 registros, somados aos145 de política sem mudança:190 casos/74 perguntas distintas. Fonte única por documento, não45 fontes novas. `build_document_pilot.py` reproduz corpus; `audit_document_pilot.py` confirma máximo277 tokens, contexto384, EOS190/190, zero truncagem e zero sobreposição exata de prompts/IDs/grupos com splits reservados. Não revisa semanticamente alvos de teste, embora o loader os parseie. Hash final de treino `cf5ea971684471e5497a0a7e99bb4560e8ba8ec958468db8c9c44f4af58d1506`; evidências `dat08_manifest.json`, `dat08_source_review.jsonl`, `dat08_token_overlap_audit.json`, `dat08_per_example_tokens.jsonl`. Sem MedPT no treino, sem alvos/perguntas reservados adicionados. Duas variações de roteiro foram corrigidas antes da auditoria final para não exigir campos vazios de uma lista de etapas.

### FT-08 — Testar cobertura documental com base8B limpa

**Status:** CONCLUÍDA — ADAPTER REJEITADO

**Objetivo:** treinar piloto DAT-08, preservando base oficial8B, r8/alpha32, LR5e-6, duas épocas e acumulador8. Contexto384 é necessário para alvos documentais completos. Hipótese: melhor formato/validação de documentos e registros sem degradação factual. Não acrescenta conhecimento médico validado.

**Depende de:** DAT-08.

**Aceite:** probe384/otimizador, adapter/manifesto/pico real/recarga; 50 respostas com promptv3 e mesmos IDs/seed/teto, comparação com controle8B v3 histórico e candidatoQ3, revisão e decisão explícitas. Registrar mudança de dados/contexto, sem causalidade isolada. Rejeitar se não trouxer ganho de aderência ou criar novas falhas; não usar teste reservado na seleção.

**Execução:** probe `20260930T223126Z_qwen3_8b_qlora_rocm` aprovado, 903 parâmetros em `cuda:0`, forward/backward e AdamW, 9.875.265.536 bytes alocados/11.272.192.000 reservados com384. Treino `20260930T223317Z_qwen3_8b_qlora_rocm` concluído com `--project-dir artifacts/data_audits/20260930_dat07_resume/document_pilot --model-dir models/base/Qwen3-8B --epochs 2 --learning-rate 5e-6 --max-length 384`, allocator expansível, log `ft08_training.log`:48 passos,1245,5254s, perda de treino2,994712 e validação3,134430, checkpoint48; recarga/código0. Pico da janela de treino/validação11.158.411.264 bytes alocados/11.435.769.856 reservados. Avaliação adapter-only `20260930T230117Z_eval03` iniciada, mesmos50 IDs/promptv3/seed42/teto512; revisão pendente. Desenho pré-registrado `ft08_evaluation_design.json`. Usar controle8B v3 `20260930T184206Z_eval03/base.jsonl` histórico, explicitando origem da latência; comparar IDs/hash/prompt. Nenhum modelo rejeitado foi integrado à aplicação.

**Resultado:** avaliação `20260930T230117Z_eval03` concluída/código0; adapter15 adequados/25 parciais/7 inadequados e3 excluídos, contra controle8B16/24/7 eQ3 histórico24/23/0.16 textos idênticos ao8B, uma mudança de categoria (`MEDPT:591089`, adequado→parcial); nenhum ganho de categoria. Mantém falha de resumo auditável/laudo/receituário e perde identificação em procedimento.50/50 não vazias,0 cortes,0 acima de120 palavras,109,06 tokens médios,13,77568s. Revisão dos50 com perguntas/referências, notas e hashes; controle/texto/IDs/input conferidos. Evidências `artifacts/evaluations/20260930T230117Z_eval03/{review.md,qualitative_review.jsonl,qualitative_summary.json}` e `artifacts/data_audits/20260930_dat07_resume/ft08_final_result.json`. Rejeitar adoção; Q3 permanece candidato de inferência. Sem RAG/validação clínica; validação reutilizada e controles históricos.

### EVAL-07 — Verificar base Qwen3.5-9B treinável

**Status:** CONCLUÍDA

**Objetivo:** medir geração mais recente antes de investir em novo fine-tuning. Base oficial `Qwen/Qwen3.5-9B`, revisão `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, Apache2. Iniciar com inferência NF4/BF16 via classe causal de texto disponível em Transformers5.17; não usar visão/imagens. Contrastar sistemas nos mesmos50 IDs/promptv3/seed42/teto512/greedy/thinkingfalse, com8B/Q3 históricos.

**Depende de:** FT-08 concluída/rejeitada. Download cabe no disco medido; capacidade GPU ainda não comprovada.

**Aceite:** metadados/licença/revisão/shards verificados, processo GPU sem offload inesperado,50 respostas/erros registrados, revisão completa com a mesma rubrica47+3 e decisão sobre elegibilidade para treino. Registrar arquitetura/tokenizador/templates diferentes, memória/latência e limites; não usar teste reservado. Se carregar/inferir falhar, registrar evidência e ajustar apenas o necessário.

**Execução:** metadados/config oficiais salvos em `artifacts/data_audits/20260930_dat07_resume/qwen35_9b_{metadata,config}.json`.9.653.104.368 parâmetros totais declarados (inclui visão), quatro shards oficiais, aproximadamente19,31GB em tensores. AutoModelForCausalLM local mapeia `qwen3_5` para `Qwen3_5ForCausalLM` e converte text_config; isso é suporte de código, não inferência/treino comprovados. Download e avaliação pendentes.

**Progresso:** download concluído/código0 em cerca de322s; quatro arquivos/hashes SHA256/índice/limites/contagem verificados (`eval07_weights_check.json`),19.306.216.416 bytes de tensores. Classe construída em meta:8.953.803.264 parâmetros de texto, sem visão (`eval07_meta_constructor_check.json`). Template validado comthinkingfalse (`eval07_tokenizer_check.json`). Avaliação base-only `20260930T232622Z_eval03` iniciada pelo script existente, log `eval07_evaluation.log`. Carregamento GPU e revisão ainda pendentes;8,2GiB livres após download.

**Resultado:** geração/código0,50 respostas lidas com perguntas/referências;13 adequados/31 parciais/3 inadequados, três exclusões;8,62314s/145,84tokens médios,50 não vazias,0 cortes,4 respostas acima de120 palavras. Carregamento33s; VRAM física observada8,47GiB, sem offload configurado; fallbackPyTorch de causal_conv1d/FLA, sem dependências novas. Seis ganhos/sete perdas de categoria contra8B, incluindo recusas indevidas a informação educativa. Não substituirQ3 sobv3. Evidências `artifacts/evaluations/20260930T232622Z_eval03/{review.md,qualitative_review.jsonl,qualitative_summary.json}` e `eval07_final_result.json` na auditoria. InferênciaGPU comprovada; treino/backward/otimizador9B ainda não comprovados.

### EVAL-08 — Distinguir informação educativa de decisão individual no9B

**Status:** CONCLUÍDA — PROMPT REJEITADO

**Objetivo:** verificar se recusas excessivas são influenciadas pelo enquadramento do prompt. Acrescentar um parágrafo ao promptv3 que permite informação educativa/encaminhamento sem prontuário e evita localização/telefone não informado, mantendo limites de diagnóstico/prescrição individual. Não usar respostas ou regras específicas dos50 IDs.

**Depende de:** EVAL-07 concluída. Pesos9B/NF4/tokenizador/backend/IDs/seed42/thinkingfalse/teto512 preservados.

**Aceite:** prompt/hash/desenho registrados antes de gerar,50 respostas nos mesmosIDs, revisão47+3, comparação direta9Bv3/v4 com hashes e decisão; verificar fatos inventados, omissões, urgência e extensão. Não misturar controles8B/Q3v3 em ranking de causalidade do novo prompt. Não usar teste reservado.

**Execução:** preparação do promptv4; geração/revisão pendentes. Nenhum fine-tuning9B iniciado.

**Resultado:** `20260930T233923Z_eval03`, código0/50 lidas;v4=11/32/4 contra9Bv3=13/31/3, três exclusões. Quatro ganhos/seis perdas de categoria,0 cortes/erros,50 não vazias,159,78tokens/9,16824s,9 acima de120 palavras. Novas falhas factuais, incluindoLeiden/vacina verificadas emNIH/EMA. Rejeitarv4; manterQ3v3 candidato. Relatório/notas/hashes na avaliação e `eval08_final_result.json` na auditoria. Sem RAG/validação clínica/teste reservado.

### FT-09 — Piloto9B documental com prompt de treino alinhado

**Status:** CONCLUÍDA — ADAPTER REJEITADO

**Objetivo:** testar9B treinável comDAT-08 e o mesmo promptv3 da avaliação, para eliminar diferença de instruções de sistema entre treino e geração. Base oficial/revisão fixa,NF4,BF16,LoRA r8/alpha32/dropout0,1, incluindo projeçõesGatedDeltaNet específicas desta arquitetura;LR1e-5,duas épocas,batch1/acumulador8,seed42. Contexto será escolhido pela auditoria de tokens completos e probe, sem truncar alvos/EOS.

**Depende de:** EVAL-08 rejeitada; corpusDAT-08 inalterado190/74, sem exemplos extraídos dos50 casos. Alterações mínimas necessárias: permitir prompt de sistema no trainer e escolher revisão/projeções9B, preservando defaults8B.

**Aceite:** auditoria/probeGPU/backward/AdamW, treino/adapter/manifesto/hash/picos/recarga;50 respostas mesmosIDs/v3/512/seed42, comparadas diretamente ao9Bv3 controlado; revisão47+3 e decisão. Registrar múltiplas diferenças frente aFT-08 (base/prompt/targets/LR/contexto), sem causalidade isolada. Não usar teste reservado. Se capacidade falhar, registrar limite antes de ajustar.

**Execução:** duas tentativas de treino encerradas sem adapter: contexto640/OOM no backward/código1/zero passos; contexto576/avisos HIP OOM e segmentation fault/código139 após1/48 passos. Causa final da falha de segmentação não isolada. Evidências `ft09_first_training_failure.json`, `ft09_second_training_failure.json` e respectivos logs no diretório DAT-07. Próximo ciclo não é autorizado a adotar adapter sem concluir revisão.

**Progresso:** contexto576 preserva os190 alvos/EOS completos, máximo549tokens. Probe inicial passou, mas não garantiu capacidade do Trainer. Novo ajuste opcional `--frozen-embeddings-bf16` restaura BF16 apenas em embeddings/head congelados e usa autocast no probe como no Trainer; preserva demais camadas FP32 e defaults anteriores. Dois testes existentes passaram. Probe corrigido em andamento, `ft09_capacity_bf16_probe.log`; exigir conclusão/código0 antes do novo treino. Não houve geração nem avaliação de qualidade da FT-09.

**Treino e avaliação (01/10/2026):** o probe corrigido e o treino concluíram com `--frozen-embeddings-bf16`; o adapter foi recarregado no run `20261001T001634Z_qwen3.5_9b_qlora_rocm` (2 épocas, `train_loss=2,1417`, `eval_loss=2,7200`). A avaliação registrada gerou 50/50 respostas não vazias no split de validação, em `llm/artifacts/evaluations/20261001_ft09_9b_adapter/20261001T102530Z_eval03/`, com latência média de 6,63834 s e um corte pelo teto. A revisão documental 47+3 registrou 10 adequadas, 30 parciais, 7 inadequadas e 3 não avaliáveis, contra 13/31/3/3 do controle 9B. Regressões materiais: ambos os alertas de urgência pedem mais dados sem orientar ação imediata; `FAQ_001` repete “dor no peito” até o teto; `MEDPT:522763` afirma desenvolvimento adulto de síndrome congênita; `MEDPT:1390474` nega sem suporte possível efeito do cigarro; `MEDPT:252493` e `PROTOC_002` recusam informação educativa geral. O adapter foi rejeitado para adoção. Evidências: `qualitative_summary.json`, `review.md`, respostas brutas, `manifest.json` e `environment.json`. Pesos e checkpoints rejeitados foram removidos conforme a política de armazenamento, preservando manifestos, métricas e respostas brutas.

### DAT-09 — Ampliar corpus contextual e limpar artefatos

**Status:** CONCLUÍDA

**Objetivo:** atender à autorização de 05/10/2026: ampliar os exemplos para as atividades do assistente e limpar arquivos gerados dispensáveis em `llm`. Base treinável mantida: Qwen/Qwen3.5-9B na revisão fixa da FT-09. Produzir cerca de mil casos sintéticos com contexto explícito em oito categorias, referências verificáveis e cenários separados entre treino, validação e teste. Não afirmar cobertura universal nem conhecimento clínico validado.

**Aceite:** corpus reproduzível com hashes, seed, contagens, grupos e ausência de duplicatas entre splits; auditoria de tokens e EOS; preservação dos resultados históricos compactos e arquivos necessários para reproduzir treino/inferência. Limpeza recuperável registrada. Treino posterior compara base e adapter por categoria e em teste reservado, sem escolher hiperparâmetros pelo teste.

**Evidência (05/10/2026):** `llm/scripts/prepare_contextual_corpus.py` gerou `llm/data/contextual_v1/`: 960/120/120 casos em oito categorias, especialidades disjuntas por split, seed 20261005, hashes no manifesto, máximo 360 tokens no treino. Os templates são compartilhados e os alvos são sintéticos; mede uso de contexto, não conhecimento médico geral. Cinco testes focados passaram (`test_contextual_corpus.py`, `test_training_data.py`). Limpeza via `gio trash`: snapshots externos antigos, `data/curation`, caches, benchmarks de inferência, inventários históricos de ambiente e logs/auditorias volumosas foram enviados à lixeira. Cerca de 410 MB retirados da árvore ativa; resultados de modelos, avaliações brutas, manifestos, dados curados usados pela API, código e `.venv` preservados. Download da base 9B fixa iniciado; sem novo treino ainda.

### FT-10 — Treinar e comparar o corpus contextual ampliado

**Status:** CONCLUÍDA

**Objetivo:** base limpa Qwen3.5-9B fixa, 960 casos DAT-09, uma época, LR 2e-5, contexto 384 sem truncagem, NF4/BF16 e embeddings congelados BF16. Controle base e adapter no mesmo prompt e decodificação greedy/thinking desativado. Reusar executor local e manter apenas pesos finais após concluir avaliação.

**Aceite:** probe forward/backward/AdamW, treino e recarga em processos novos; 120 casos completos por modelo na validação, seleção registrada antes de 120 casos novos de teste; métricas exatas por categoria, fontes, alerta e recusa; preservar modelo-base, adapter final, resultados e dados. Não declarar generalização clínica ou cobertura universal. `scripts/run_contextual_experiment.py` preserva etapas/erros em `artifacts/experiments/<run_id>/experiment.json`; um resultado inferior deve ser explicitamente rejeitado.

**Evidência (05/10/2026):** base oficial restaurada em `llm/models/base/Qwen3.5-9B`; probe ROCm com backward/AdamW aprovado. Adapter `20261005T101013Z_qwen3.5_9b_qlora_rocm`: 120 passos, 960 exemplos, uma época, 2.014,7 segundos de treino; pesos salvos e recarga concluída. Validação: base 47/120, adapter 120/120; teste reservado: 46/120 e 120/120. Prova de formulações novas, preparada antes do treino: 15/32 e 25/32. Sete erros do adapter nessa prova: status de formulários e omissão do campo de alerta inativo. Ganho contextual comprovado, generalização parcial; não comprova conhecimento clínico. Resultados e revisão em `llm/artifacts/experiments/20261005T095351Z_contextual_v1/review.md`. Checkpoint duplicado removido após conferir hash idêntico aos pesos finais; sondagens históricas sem pesos/manifesto e caches enviados à lixeira.

### DAT-10 — Ampliar temas factuais com documentos oficiais traduzidos

**Status:** CONCLUÍDA

**Objetivo:** complementar a rodada contextual com temas factuais do subconjunto MedQuAD traduzido em `recogna-nlp/drbode_dataset`, revisão `5ce2ec87299adef39602d664209f79ed5914881a`. Inspeção remota de 05/10/2026: 121.744 linhas totais, 9.579 com URLs oficiais GHR/GARD e 3.153 documentos. O restante inclui conversas médico-paciente e não é aprovado por esta inspeção.

**Depende de:** FT-10 encerrada. Não começar um segundo treino concorrente.

**Aceite:** selecionar apenas informações educativas documentadas, preservar texto, contexto, URL, origem e licença; fontes/doenças e sinônimos separados entre splits, controles de duplicidade e completude dos trechos; distinguir apoio textual de validação clínica. Incluir perguntas educativas gerais e perguntas com documentos, além das oito atividades contextuais. Confrontar as traduções com o original MedQuAD fixo `577bd37b96c02d1833b2c9eed2de9f96964e96cb`; rejeitar frases incompletas, divergências numéricas e perda de qualificadores. Registrar que filtros automáticos e revisão semântica amostral por IA não validam clinicamente todo o corpus. Não usar MedPT bruto nem benchmarks externos no treino. Não usar os testes já observados para escolher hiperparâmetros.

**Evidência (05/10/2026):** `llm/scripts/prepare_factual_corpus.py` e `llm/data/factual_v2/`: 1.408/160/160 casos e prova adicional de 40. Treino: 1.024 questões factuais (512 definições, 256 genética, 256 herança) mais 384 casos contextuais com instruções variadas. Perguntas gerais sem trechos, perguntas com documentos/distratores e casos de fonte ausente são distintos. 596 grupos médicos no treino, 67 na validação, 69 no teste; grupos transitivos por sinônimos/CUIs separados inclusive para documentos usados como distratores. Proveniência, original inglês, tradução, ajustes lexicais, URL, licença e hashes conservados nos dados; snapshots completos não conservados. Revisão semântica por IA é amostral: 35 definições e 16 trechos de genética/herança; cinco documentos vetados. O corpus permanece concentrado em doenças raras/genética. Oito testes focados passaram. Auditoria com o tokenizador do treinador preservou todas as sequências/EOS em 1.408/160/160 casos; máximo 512 tokens e respostas-alvo de até 112 tokens. `llm/data/factual_v2/token_audit.json` registra a evidência. Rascunhos descartados via lixeira antes de qualquer treino.

### FT-11 — Treinar a ampliação factual e comparar conteúdo e formato

**Status:** CONCLUÍDA — seleção confirmada no teste principal; generalização parcial

**Objetivo:** continuar a adaptação da mesma base oficial com DAT-10, preservando o resultado contextual da FT-10. Parâmetros previstos antes da avaliação factual: uma época, LR 1e-5, máximo 512 tokens sem cortar respostas, QLoRA NF4/BF16. Validar a memória antes do treino e registrar a origem dos pesos continuados.

**Depende de:** DAT-10 e FT-10 concluídas.

**Aceite:** comparar base e adapter nos mesmos casos e prompt; avaliar separadamente JSON/campos/fontes e adequação semântica das respostas educativas contra referências documentadas. Revisão documental por IA com critérios explícitos, sem alegar validação clínica. Seleção registrada pela validação antes da geração/pontuação do teste reservado novo. Preservar ou melhorar adequação factual e as atividades contextuais, incluindo alertas e recusas. Conservar pesos finais, base, dados, manifestos e resultados; remover checkpoints duplicados e temporários somente depois das verificações necessárias.

**Evidência (05/10/2026):** `llm/artifacts/experiments/20261005T120254Z_factual_v2/`: critérios semânticos registrados antes das respostas. Probe com parent FT-10, contexto 512, backward/AdamW e pico de 12,86 GiB aprovado. Treino com 1.408 casos, uma época, LR 1e-5: 176 etapas, 3.541,4 s, perdas treino/validação 0,2470/0,0968; pesos finais recarregados. Validação factual documental por IA com rótulos ocultados: base 21/2/25, FT-10 15/6/27, FT-11 23/8/17 (adequado/parcial/inadequado, 48 casos); documentos/contexto 29/75/101 acertos em 112. Escolha FT-11 registrada antes do teste em `validation_selection.json`. Teste reservado: base 20/7/21, FT-11 22/15/11; documentos/contexto 29/112 e 100/112; alertas 6/8 e 8/8, recusas 8/8 em ambos. Todos os critérios operacionais confirmados em `test_confirmation.json`, sem demonstração estatística de não inferioridade. Prova adicional: base 6/1/5 e FT-11 5/4/3 em 12 factuais; regressão de um adequado apesar de menos inadequados. Documentos/contexto 9/28 e 24/28. A seleção não foi refeita usando essa prova; generalização parcial, erros factuais e falha de chaves nas FAQs permanecem. Revisão completa em `review.md`; `llm/artifacts/best_model.json` aponta para FT-11 sem duplicar pesos. Base, dois adaptadores finais, dados, proveniência, resultados e ambiente preservados; logs concluídos, checkpoints idênticos, caches e temporários enviados à lixeira. Hashes dos dados e do parent permaneceram intactos. Oito testes focados passaram; consolidação validou exatamente 48/48/12 casos sem exclusões, em validação/teste/prova. Sem alegar validação clínica ou cobertura universal. Uma primeira sondagem falhou por caminho relativo do parent, corrigido antes de qualquer treino.

### FT-12 — Continuar FT-11 com todo o conjunto de treino

**Status:** CONCLUÍDA — treino completo e comparativo; FT-11 mantida

**Objetivo:** atender à solicitação de 05/10/2026: registrar o ganho comprovado da FT-11 e continuar seus pesos com todos os 1.408 exemplos do conjunto atual, que já havia sido percorrido integralmente na rodada anterior. Uma época, LR 1e-5, contexto 512, QLoRA NF4/BF16, `max_steps=-1`. Os 160 casos de validação e 160 de teste não entram no treino.

**Depende de:** FT-11 concluída. O melhor modelo atual permanece FT-11 até avaliação do novo resultado.

**Aceite desta solicitação:** iniciar processo independente da sessão, registrar comando/PIDs/log, conferir carregamento, forward/backward e primeiros passos reais com perda finita; encerrar o acompanhamento inicial conforme pedido do usuário. Posteriormente, o usuário autorizou conferir a conclusão e comparar com o melhor modelo. Preservar os pesos FT-11 e FT-12, comparar os mesmos conjuntos e registrar método, resultados e decisão. Os conjuntos já observados não constituem novo teste independente; a recarga técnica não substitui avaliação de respostas.

**Execução (05/10/2026):** `llm/artifacts/experiments/20261005T165650Z_ft12_full_train/`: início conferido conforme escopo original; supervisor/treino históricos PIDs 10671/10672. Treino terminou às 18:04:04 UTC (15:04:04 Brasília), status `completed`, exit 0, 176 passos, todos os 1.408 casos, uma época. GPU Radeon RX 9060 XT 16 GB/ROCm, SDK `llm/.venv/bin/python` 3.14.4. Runtime 3.507,9673 s; train_loss 0,2045344, eval_loss 0,0900386; pesos e recarga técnica concluídos em `llm/artifacts/adapters/20261005T165900Z_qwen3.5_9b_qlora_rocm/`. Comando preservado em `run.sh`; parâmetros, métricas e hashes em `metadata.json` e manifesto do adaptador.

**Comparação autorizada (05/10/2026):** protocolo prévio e revisão em `llm/artifacts/experiments/20261005T203707Z_ft12_comparison/`. Comando de geração: `llm/.venv/bin/python -B llm/scripts/evaluate_qwen3_base_vs_adapter_rocm.py --adapter-only --adapter-dir <FT12/adapter> --model-dir llm/models/base/Qwen3.5-9B --test-file <split> --output-dir <candidate_split> --system-prompt-file llm/data/factual_v2/system_prompt.txt --limit 99999 --max-new-tokens 192`; argumentos exatos nos `request.json`. Greedy, thinking false, seed 42, mesmo prompt/IDs/hashes do controle FT-11. Pontuação com `score_contextual_results.py` e `score_semantic_review.py`; conferência final com `finalize_comparison.py`. Todos os 360 casos gerados e pontuados, sem exclusões; revisão factual completa de 48/48/12 perguntas por IA, sem validação clínica. FT-11 versus FT-12, adequado/parcial/inadequado: validação 23/8/17 versus 19/11/18; teste 22/15/11 versus 20/16/12; desafios 5/4/3 versus 6/4/2. Documentos/contexto: 101/112 versus 100/112; 100/112 versus 102/112; 24/28 em ambos. JSON, alertas e recusas preservados nos dois. Validação rejeitou FT-12; `best_model.json` permanece FT-11 e inalterado. Notas históricas do controle fixas; equivalências e divergências provisórias da revisão estão auditadas. O teste conhecido é diagnóstico comparativo, não nova prova independente de generalização. Resultado e limitações em `review.md` e `comparison.json`; anotação atualizada também na revisão histórica FT-09 indicada pelo usuário. Nenhum novo treino foi iniciado.

### DAT-11 — Curar temas gerais e reservar nova avaliação

**Status:** CONCLUÍDA

**Objetivo:** solicitação de 06/10/2026: preparar uma nova base com temas médicos gerais, fatos vinculados a fontes oficiais revisados documentalmente por IA, instruções variadas e tarefas contextuais com chaves explícitas. Manter os datasets anteriores intactos. Separar temas e cenários entre treino, validação e teste antes de executar modelos.

**Aceite:** origem, licença/atribuição, revisão, seed, contagens e hashes registrados; nenhuma decisão clínica individual ou dado pessoal real; referências factuais conferidas, ausência de duplicatas/vazamento entre grupos e respostas completas no contexto de treino. Teste novo congelado e não usado para ajustar o piloto. Revisão por IA não equivale a validação clínica. Não prometer cobertura universal nem ausência de memorização.

**Execução (06/10/2026):** `llm/scripts/prepare_general_factual_corpus.py` gerou `llm/data/factual_v3/`: 1.920 exemplos de treino (5.581.252 bytes), 80 de validação e 80 de teste. São 48 temas: 32/8/8, com 96/24/24 fatos distintos revisados individualmente por IA contra os resumos oficiais MedlinePlus coletados em 06/10/2026; não revisão clínica. Treino: 384 educativos gerais, 384 com documento, 128 com fonte ausente e 1.024 contextuais. Variações de instrução e apresentação não contam como fatos novos. Fontes de domínio público com atribuição NLM; snapshots/hashes em `sources.json`, referências em `curation.tsv`, revisão vinculada em `curation_review.json`, seed 2026100613 e manifesto em `data/splits/manifest.json`. SDK `llm/.venv/bin/python` 3.14.4. Checks de contrato JSON, IDs, hashes, grupos disjuntos e referências documentais passaram; máximo de 500/484/509 tokens no template real, sem truncar respostas. Um tema inicialmente proposto foi substituído por sobreposição de nome com a base anterior; conjuntos congelados antes de iniciar modelos. Fontes anteriores e melhor modelo intactos; conjuntos históricos deixam de servir como testes independentes para este currículo. Não há garantia de ausência dos fatos no pré-treino da base.

### FT-13 — Piloto curto com a base revisada

**Status:** CONCLUÍDA — piloto comparado; não expandido pelo critério prévio

**Objetivo:** após DAT-11, continuar FT-11 em piloto com parâmetros registrados antes da avaliação; comparar modelo-base, FT-11 e candidato nos mesmos casos novos de validação. Avaliação factual semântica separada de contrato JSON/documentos. Só autorizar a expansão pelo resultado de validação previamente definido; manter FT-11 se houver regressão ou equivalência.

**Aceite:** pesos, manifesto, métricas e recarga técnica; comparação completa, sem exclusão de falhas. Candidato deve melhorar adequados factuais sobre FT-11, não aumentar inadequados e preservar conteúdo contextual e status de alertas/recusas, além de comparar com a base. Seleção registrada antes de gerar o teste novo; não usar o teste para novos ajustes. Preservar o melhor modelo atual e registrar resultados honestos, inclusive falha do piloto.

**Execução (06/10/2026):** protocolo registrado antes de iniciar em `llm/artifacts/experiments/20261006T232619Z_ft13_pilot/protocol.json`; executor `llm/.venv/bin/python -B -u llm/artifacts/experiments/20261006T232619Z_ft13_pilot/run.py`. Piloto inicia FT-11, LR 5e-6, max_length 512, max_steps 48, batch 1/acumulação 8, seed 42; aproximadamente 384 exemplos consumidos. Mesmo prompt e geração greedy/thinking false/max_new_tokens 192 para base, controle FT-11 e candidato nos 80 casos de validação. O executor termina na geração da validação, aguardando revisão factual por IA; não inicia teste ou expansão automaticamente. Hashes de dados e do melhor modelo conferidos em cada etapa. O teste permanece congelado até decisão registrada pela validação.

**Evidência e decisão (06/10/2026, Brasília):** 48 passos concluídos, 384 exemplos (20% de uma época), runtime 991,96 s, perdas treino/validação 0,09048/0,16218; probe com pico de 12,86 GiB e recarga aprovada. Pesos em `llm/artifacts/adapters/20261006T232654Z_qwen3.5_9b_qlora_rocm/adapter`. Todos os 80 casos gerados e pontuados por base, FT-11 e piloto, sem exclusões; revisão documental por IA com rótulos ocultados, exatamente 24 perguntas factuais por modelo. Adequado/parcial/inadequado: base 22/1/1, FT-11 19/1/4, FT-13 21/1/2. Documentos/contexto: 14/56, 55/56 e 55/56; JSON parseável 76/80, 80/80 e 80/80; status de alertas/recusas 4/4 em todos. O candidato melhorou o controle, mas não cumpriu adequados >= base e inadequados <= base, critérios registrados previamente. `validation_selection.json` rejeitou a expansão; nenhum treino completo ou teste novo iniciado. FT-11 e `best_model.json` intactos; candidato preservado. `finalize.py` conferiu todos os casos, referências, parâmetros e hashes. Checkpoint-48 idêntico ao adaptador final enviado à lixeira (260.185.487 bytes), com `training_state.json`, pesos finais, logs e evidências preservados; registro em `cleanup.json`. Resultado e limites na [revisão FT-13](../llm/artifacts/experiments/20261006T232619Z_ft13_pilot/review.md). Revisão por IA, não clínica; oito temas de validação e 24 perguntas são amostra diagnóstica pequena, sem prova estatística de não inferioridade. O teste de 80 casos permanece congelado e sem respostas de modelos.

### RAG-00 — Planejar recuperação com o modelo selecionado

**Status:** CONCLUÍDA

**Objetivo:** solicitação de 07/10/2026: planejar corpus, recuperação, conexão ao FT-11, fontes verificáveis e avaliação do RAG a partir do código existente, sem executar ingestão, indexação ou integração nesta tarefa.

**Aceite:** plano distingue comportamento existente de implementação futura, estabelece sequência e dependências, registra proveniência e versionamento, separa recuperação de geração e prevê comparação com/sem RAG. O teste factual_v3 continua reservado; planejamento não comprova integração ou qualidade clínica.

**Evidência (07/10/2026):** [PLANO_RAG.md](PLANO_RAG.md) registra escopo inicial documental, fontes candidatas e versões, recuperação lexical versus semântica, orçamento de contexto, backend FT-11 real, cadeia LangChain, citações resolvidas e avaliação própria com quatro controles. Inspeção de `retrieval/ollama_adapter.py`, `application/assistant.py`, executor ROCm e `best_model.json` confirmou busca por palavras, resposta simulada na aplicação e ausência de integração comprovada do adapter. Documentação primária LangChain, MedlinePlus e portal PCDT consultada. Nenhuma ingestão, indexação, chamada ao modelo, alteração de dependência ou implementação foi executada; testes de software não se aplicam à mudança documental. Próxima etapa prevista: inventário e manifesto do corpus piloto; APP-00/APP-01 seguem pendentes.

### RAG-01 — Inventariar o corpus documental piloto

**Status:** CONCLUÍDA

**Objetivo:** iniciar o RAG pela próxima tarefa delimitada no PLANO_RAG: inventário e manifesto dos documentos, assuntos e critérios da avaliação. Criar fontes sintéticas novas e registrar três documentos brasileiros oficiais candidatos, distinguindo conteúdo disponível de fonte ainda pendente de ingestão/revisão.

**Depende de:** RAG-00 e ORG-05 concluídas.

**Aceite:** documentos/versões identificáveis, seções rastreáveis e hashes de conteúdo disponível; candidatos externos com origem e pendências explícitas; critérios de avaliação registrados separadamente do corpus; geração/validação reproduzível sem rede ou modelo. Recuperação, embeddings e integração ficam para tarefas posteriores.

**Evidência (10/10/2026):** `llm/data/rag_pilot_v1/{catalog,manifest,evaluation_spec}.json` registra dez documentos/versões: sete sintéticos próprios, dos quais seis vigentes no piloto e uma edição substituída, além de três PCDTs externos candidatos (HAS, DM2, dislipidemia). Quatorze seções com IDs por documento/versão/seção, texto preservado e SHA-256; doze elegíveis. Fontes externas sem texto admitido, hash de conteúdo nulo e pendências de PDF, edição/vigência, termos e revisão. Páginas oficiais e recomendação DM2/2026 consultadas; aviso CC BY-ND 3.0 registrado sem presumir autorização específica para os PDFs. Nenhum documento clínico foi extraído ou admitido. Especificação separada registra 40+40 perguntas futuras, rubrica e comparação das quatro condições; perguntas/gabaritos ainda não criados. `llm/src/rag/prepare_manifest.py` gera manifesto determinístico, recusa sobrescrita e verifica divergências com --check. SDK PyCharm Python 3.14.4, `llm/.venv/bin/python`; geração: `-B src/rag/prepare_manifest.py` (em llm); testes: `-B -m pytest tests/test_rag_manifest.py -q -p no:cacheprovider`: oito aprovados. Verificados exclusão de fontes pendentes/versão antiga, hashes, estabilidade de IDs, duplicações, licença pendente, separação dos critérios e recusa de alteração silenciosa. Documentação/ignore atualizados; API, dependências, modelos e dados de treino preservados. Esta tarefa conclui o inventário, não toda a etapa de corpus/avaliação, busca ou integração.

### RAG-02 — Admitir fontes oficiais e preparar avaliação reservada

**Status:** CONCLUÍDA

**Objetivo:** completar o corpus do piloto após conferência documental das fontes candidatas e criar as perguntas de avaliação conforme a especificação RAG-01.

**Depende de:** RAG-01 concluída.

**Aceite:** snapshots originais com hashes e proveniência, edição/vigência/termos conferidos, texto e páginas/seções revisados antes da admissão; nova versão do corpus. Avaliação separada com 40 perguntas de validação e 40 de teste, evidências/gabaritos rastreáveis e categorias previamente fixadas. Não iniciar recuperação ou geração de respostas nesta tarefa.

**Evidência (10/10/2026):** `llm/data/rag_pilot_v2/` preserva v1 e registra dez documentos/versões, nove elegíveis, 17 seções/15 elegíveis. Três PDFs oficiais completos baixados (HAS 97 páginas, DM2 80, dislipidemia 31), SHA-256 e URLs em sources.json/SHA256SUMS; originais e extrações integrais locais/ignorados. `pdftotext -layout` 26.01.0; somente página 1 de cada PDF (texto das portarias) revisada documentalmente por IA e admitida sem normalização. Anexos clínicos/tabelas não revisados e fora do corpus elegível. Atos: SECTICS/MS 49/2025, SCTIE/MS 13/2026 (revoga 7/2024), Conjunta 8/2019; publicação/edição conferidas no catálogo Conitec/MS, anexo dislipidemia alterado em 2022. Termos da admissão delimitados ao texto de atos oficiais (Lei 9.610/1998, art. 8º, IV), atribuição preservada; sem estender decisão aos anexos. Instruções de download/verificação no README do corpus. `evaluation/{validation,test}.jsonl`: 40+40 perguntas distintas, 24 respondíveis/8 ausentes/4 conflitos/4 ataques por split, referências literais e comportamento esperado. Spec fixa hashes antes de recuperação/geração; fixtures adversariais somente na avaliação, separados do índice. Splits compartilham conceitos/corpus; sem alegar holdout temático, validação clínica ou revisão humana. SDK PyCharm Python 3.14.4, `llm/.venv/bin/python`; em llm: `-B -m pytest tests/test_rag_manifest.py tests/test_rag_corpus.py -q -p no:cacheprovider`: 11 aprovados. `-B src/rag/prepare_manifest.py --catalog data/rag_pilot_v2/catalog.json --output data/rag_pilot_v2/manifest.json --check` aprovado; SHA256SUMS dos três originais aprovado. Testes conferem manifesto, hashes de splits/referências e arquivos locais presentes, preservação dos sete documentos sintéticos v1, categorias, unicidade, exclusão de fixtures do corpus e ausência de cópia literal de perguntas nos JSONL anteriores. Nenhum índice, embedding, chamada de modelo, alteração da API/dependências ou novo treino.

### RAG-03 — Comparar recuperação lexical e semântica

**Status:** CONCLUÍDA

**Objetivo:** comparar a recuperação isolada no corpus rag_pilot_v2, com IDs rastreáveis e filtros de vigência, antes de chamar a LLM.

**Depende de:** RAG-02 concluída.

**Aceite:** indexar somente conteúdo elegível do manifesto, excluir avaliação/fixtures, comparar top-3 lexical/semântico na validação, registrar configuração, hashes, latência e memória; congelar a configuração antes de eventual teste reservado. Não integrar FT-11 nem gerar respostas nesta tarefa.

**Evidência (11/10/2026):** `llm/src/rag/compare_retrieval.py` compara BM25 k1=1,5/b=0,75 com multilingual-E5-small revisão `614241f622f53c4eeff9890bdc4f31cfecc418b3`/MIT, usando transformers/torch existentes, CPU float32/quatro threads, prefixos query/passage, média mascarada/L2. Corpus v2 validado contra spec; 15 seções elegíveis divididas em 21 chunks de alvo 192 tokens/sobreposição 32, limites de palavras, offsets/hash preservados, metadados iguais para ambos. Sem QAs/fixtures/versão antiga no índice e sem leitura de test.jsonl no runner. Execução final `llm/artifacts/experiments/20261011T212901816585Z_rag03/`: seção esperada top-3 24/24 para ambos, porém revisão de spans encontrou evidência efetiva BM25 24/24/E5 23/24 (E5 perde cabeçalho necessário em validation-19). Seleção final BM25 em frozen_config.json, hash `36ef967ca411cb8fcd55b3939de3bad91a5ea0b26aa4131a4dd2ddd58e639e99`; selected_config.json/summary.json automáticos preservados como candidato anterior à revisão. Medianas 0,040/416,011 ms; p95 0,056/532,270 ms; pico RSS VmHWM workers separados 23,23/1.646,20 MiB; zero erros/truncamentos. Preliminar `20261011T212752609941Z_rag03` preservada e descartada para memória devido a pico pré-exec herdado em getrusage; comparação repetida após correção, ranking igual. SDK PyCharm Python 3.14.4; em llm: `-B src/rag/compare_retrieval.py`; `-B -m pytest tests/test_rag_retrieval.py tests/test_rag_manifest.py tests/test_rag_corpus.py -q -p no:cacheprovider`: 14 aprovados. Integridade de chunks/embeddings/script/modelo/configuração conferida. Resultados brutos das 40 perguntas, ambiente, índices e revisão registrados; pesos E5/embeddings binários locais. Sem limiar de abstenção, avaliação de comportamento da LLM ou segurança/citações da resposta; sem GPU/FT-11/API/novas dependências. Teste reservado não executado. Próxima etapa: acesso comprovado ao FT-11 antes da cadeia documental.

### ORG-05 — Limpar LLM e separar fine-tuning de RAG

**Status:** CONCLUÍDA

**Objetivo:** solicitação de 10/10/2026: resumir a árvore LLM, preservar treino/modelo selecionado e resultados comparativos; separar código de fine-tuning e RAG em `llm/src/`, antes da implementação do RAG.

**Aceite:** comandos e imports ajustados, validação focada sem novo treino, integridade dos pesos/dados preservados e registro da limpeza recuperável.

**Evidência (10/10/2026):** código LLM separado em `llm/src/finetuning/{data,training,evaluation,scripts}` e `llm/src/rag/` reservado, sem implementar RAG. Imports, caminhos calculados, subprocessos, testes, exceções Git e comandos do README ajustados. Limpeza recuperável via `gio trash`, 212,4 MiB retirados da árvore ativa; itens em `llm/artifacts/cleanup_20261010.json`. Pesos rejeitados FT-12/FT-13, checkpoints dispensáveis, treino antigo e caches removidos. FT-11, parent FT-10, base 9B, corpora factual_v2/contextual_v1, ambiente, fonte sintética, dados curados da API e resultados comparativos preservados. SHA-256 dos arquivos remanescentes de dados/artefatos conferido contra inventário anterior, dos splits atuais contra manifestos de treino e dos pesos FT-11 contra best_model.json. SDK PyCharm: Python 3.14.4 em `llm/.venv/bin/python`. Comando: `.venv/bin/python -B -m pytest tests -q -p no:cacheprovider` (em `llm/`): 23 testes aprovados. Sintaxe de todos os módulos e seis CLIs `--help` aprovadas, sem rede, carga de modelo ou novo treino. Comandos históricos nos registros permanecem como evidência da época, não instruções atuais.

### ORG-06 — Versionar a reprodução do treinamento

**Status:** CONCLUÍDA

**Objetivo:** solicitação de 10/10/2026: commitar os arquivos necessários ao treino e documentar o download dos arquivos grandes.

**Aceite:** clone com código, dependências, corpora FT-10/FT-11 e metadados; instruções de download da revisão da base e geração dos adapters sem presumir URL inexistente; validação sem novo treinamento e commit restrito ao escopo.

**Evidência (10/10/2026):** README LLM documenta ambiente Linux/Python 3.14/ROCm, `hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir models/base/Qwen3.5-9B`, pré-check, treino FT-10 e continuação FT-11 usando o novo run_id, além de geração na validação. CLI oficial e página do modelo consultadas; adapters locais não têm URL pública registrada, portanto são gerados pelo treino ou copiados de execução existente. Corpora pequenos contextual_v1/factual_v2, prompts, manifestos, proveniência/atribuições, código, dependências e metadados FT-10/FT-11 incluídos no conjunto para commit. Pesos, modelos-base, ambientes, avaliações históricas, data_audits, logs/PIDs e executores históricos ignorados. Exportação de índice Git isolado validada com o SDK `llm/.venv/bin/python` 3.14.4: 23 testes, sintaxe, seis CLIs --help, contagens e SHA-256 dos seis splits contra manifestos históricos aprovados; chamada de download na revisão fixa conferida com mock. Nenhum download de pesos ou novo treino executado. Maior arquivo LLM selecionado: train.jsonl factual_v2, 3.405.766 bytes. Commit limitado ao treino, sua organização e documentação; alterações locais da API/IDE e outros itens fora do escopo preservados.

### RAG-04 — Comprovar backend local do FT-11

**Status:** CONCLUÍDA

**Objetivo:** próxima etapa autorizada em 11/10/2026: carregar base + pesos FT-11 selecionados e comprovar geração real antes da cadeia documental.

**Depende de:** RAG-03 concluída e FT-11 selecionado.

**Aceite:** resolver seleção em best_model.json, conferir hash dos pesos/configuração/manifestos, reutilizar carregamento ROCm existente, registrar identidade/base/revisão e recursos; casos sintéticos de carga/geração em processo novo. Ausência/corrupção gera erro explícito, sem substituição por outro modelo. Não implementar LangChain/API ou avaliar splits RAG nesta tarefa. APP-00 permanece separada por tratar política de modos da API.

**Evidência (11/10/2026):** vitória BM25 anotada no histórico. `llm/src/rag/ft11_backend.py` resolve best_model.json e valida elegibilidade, hash FT-11, manifesto/revisão, configuração LoRA, shards e prompt treinado; registra hashes da base local sem alegar conferência remota da revisão. Carregamento reutilizado de `finetuning/scripts/run_qwen3_adapter_rocm.py` via função extraída load_adapter, com ROCm/NF4/BF16/eager e arquivos locais. Instância reutilizável e geração por prompt, orçamento total 512, sem corte/fallback; erros explícitos e error.json na CLI. Execução real em processo novo `llm/artifacts/experiments/20261011T214416417521Z_rag04/`: base Qwen3.5-9B + adapter FT-11 ativo, hash de pesos `6d050564ef3979d75066aabb7eec4eeedc78488fe52491d5ee9012db1478946b`. Carga 58,58 s (exclui hashes prévios), pico VRAM alocada/reservada 7,40/7,45 GiB na RX 9060 XT. Dois probes sintéticos novos aprovados: JSON completo, dados/fonte exatos, status informativo/pendencias [], validação_humana true; 243/53 e 237/48 tokens, gerações 25,54/6,38 s, sem atingir limite. Dados do segundo não incluem os do primeiro, sem generalizar isolamento clínico. SDK Python 3.14.4; em llm: `-B src/rag/ft11_backend.py`; `-B -m pytest tests/test_ft11_backend.py -q -p no:cacheprovider`: sete aprovados; CLI de recarga --help aprovada. identity/request/results/summary/verification/review preservam hashes, versões, respostas e recursos. Sem índices/splits RAG consultados, LangChain/API, novas dependências ou treino. Próxima tarefa RAG-05: cadeia documental, sem declarar APP-00/APP-01 concluídas.

### RAG-05 — Integrar cadeia documental BM25 + FT-11

**Status:** CONCLUÍDA

**Objetivo:** primeira parte documental da APP-01: cadeia LangChain que usa a configuração BM25 congelada, monta contexto rastreável e chama o backend FT-11 comprovado.

**Depende de:** RAG-03 e RAG-04 concluídas.

**Aceite:** fonte citada resolve para documento/versão/seção/trecho efetivamente fornecido; contexto respeita orçamento sem cortar evidência silenciosamente; ausência de suporte e contrato/fonte inválidos geram resultado explícito. Validar fluxo com mocks e casos documentais novos; não consumir teste reservado antes de congelar a cadeia. API/prontuário/fluxo completo continuam nas APP-00/APP-01/APP-02.

**Evidência (11/10/2026):** `llm/src/rag/document_chain.py` executa RunnableSequence LCEL (recuperação/orçamento → backend FT-11 → contrato/citação/resolução), usando BM25 congelado e validando hashes do índice/corpus, texto/offsets e elegibilidade. Nova dependência direta necessária langchain-core 1.4.0 instalada com transitivos no SDK LLM, sem atualizar pacotes existentes. Contexto com chunks inteiros, top-3 e aliases R1/R2/R3, orçamento total 512/reserva 128, tokenizer input_ids real; omissões explícitas, sem truncamento. Resposta inicial extrativa `dados.trecho`; cinco chaves estritas, fonte fornecida e trecho literal na fonte (normalização apenas de espaços); alias resolve para chunk/documento/título/versão/URL/seção/página/texto. Falhas de contrato/fonte/trecho e lacunas explícitas, revisão profissional true; sem reparo de JSON. Limite lexical de prescrição/diagnóstico antes da geração e tracing remoto desativado, sem alegar SAFE-01. Execução final `llm/artifacts/experiments/20261011T215906287410Z_rag05/`: dois casos novos documentais com FT-11 real retornaram citações resolvidas, entradas 363/375 tokens, gerações 21,70/6,66 s; VRAM alocada pico 7,42 GiB. Três casos novos encerraram sem geração: context_budget_exceeded/no_evidence/blocked_request. Primeiro positivo retorna fragmento curto com núcleo solicitado, sem generalizar adequação semântica. Inicial `20261011T215705006500Z_rag05` preservada: omissão de pendencias em caso 1 rejeitada; prompt reforçado, mesmos casos repetidos sem relaxar validador. `chain_config.json` congelado, hash `6ef570406a4bddd10b755bc53cad28e7aa34c91e66a7b8b05c10b6beb90c3af8`; requests/respostas/identidade/hashes/revisão preservados. SDK Python 3.14.4; em llm: `-B src/rag/document_chain.py`; `-B -m pytest tests/test_document_chain.py tests/test_rag_retrieval.py tests/test_ft11_backend.py -q -p no:cacheprovider`: 28 aprovados (0,20 s). Warnings internos de depreciação LangChain/Python, sem falha ou correção fora do escopo. Sem splits RAG consultados, treino, API, prontuário ou LangGraph. Próxima tarefa RAG-06: comparação na validação; teste reservado permanece sem avaliação.

### RAG-06 — Avaliar a cadeia documental na validação

**Status:** CONCLUÍDA

**Objetivo:** avaliar a cadeia congelada nas 40 perguntas de validação, incluindo ausência, conflito e documentos adversariais, com controles base/FT-11 com/sem recuperação.

**Depende de:** RAG-05 concluída.

**Aceite:** registrar as quatro condições, prompts/contextos/IDs/identidade/configuração e erros sem descartes; revisar adequação e suporte sem confundir quote literal com pertinência. Usar somente validação para decisões; preservar critérios prévios, documentar limitações do modo extrativo e congelar decisão antes de eventual teste reservado. Não alterar API/prontuários ou iniciar treino nesta tarefa.

**Evidência (11/10/2026):** `llm/src/rag/evaluate_validation.py` conferiu módulos/pacotes/corpus/validação/cadeia congelados e executou quatro condições, 40 casos cada, em `llm/artifacts/experiments/20261011T225210885223Z_rag06_validation/`. Um backend quantizado comum, PEFT disable_adapter nos controles base, estado enabled false/true verificado e identidade efetiva por condição; modelo/treino/cadeia/API preservados. Mesmo sistema/pergunta/instrução e orçamento 512/128; sem RAG zero documentos com geração real; desafios com fixtures fora do índice/gabaritos fora do prompt, orçamento preservado e exposição registrada. 160 casos/revisões sem descartes. Adequadas respondíveis: base sem RAG 0/24, FT-11 sem RAG 0/24, base+RAG 12/24 (uma parcial), FT-11+RAG 10/24 (duas parciais). Zero sem RAG é limitação esperada do contrato extrativo com fonte fornecida, sem inferir zero conhecimento. Ausência correta 5/8, 1/8, 0/8 e 4/8, respectivamente. Com RAG, conflitos 0/4 e ataques base 0/4/FT-11 2/4; quatro conflitos e três ataques expostos, um ataque inconclusivo por omissão. BM25 top-3 manteve spans esperados 24/24, mas somente 15/24 chegaram ao prompt. Citações resolvidas base+RAG 15/FT-11+RAG 20; duas/cinco semanticamente inadequadas apesar de presença literal. Caso 29 confunde exemplo 4,7 mmol/L com faixa de referência; FT-11 caso 33 escolhe fila de edição sem vigência; ataque 40 contamina bruto (rejeitado), sem afirmar execução externa/aprovação. Medianas de geração 4,40/6,09/5,12/6,61 s, pico VRAM 7,41–7,42 GiB; saídas no limite 1/0/4/0. Identidade residente do backend distinguida de adapter efetivo; memória base inclui adapter inativo. `protocol`, resultados/identidades, `documentary_review`, `reviewed_summary`, `verification` e `decision.json` preservados; decisão congelada **não avançar ao teste**. SDK Python 3.14.4; comandos em llm: `-B -m pytest tests/test_rag_validation.py -q -p no:cacheprovider`: quatro aprovados (0,12 s); `-B src/rag/evaluate_validation.py` concluiu 160 registros. Comparação/revisão concluídas, critérios de qualidade não aprovados; sem clínica/revisão humana/treino/API/teste reservado. Próxima tarefa RAG-07: corrigir pipeline e reavaliar na validação, sem mudar critérios para fazer passar.

### RAG-07 — Corrigir evidência, ausência e conflitos no piloto

**Status:** CONCLUÍDA

**Objetivo:** corrigir as falhas observadas na RAG-06 antes de executar o teste reservado, preservando a versão/decisão atuais.

**Depende de:** RAG-06 concluída.

**Aceite:** nova versão rastreável da configuração/índice quando necessária; evidência pertinente entregue por orçamento medido, respostas ausentes sem citação irrelevante e conflitos sem escolher edição de vigência desconhecida. Reavaliar as quatro condições somente na validação, mantendo rubrica e falhas; registrar exposição dos ataques sem aprovar erros/casos omitidos. Não consumir teste, alterar seleção/pesos, API ou iniciar treino nesta tarefa.

**Evidência (11/10/2026):** cadeia v2 em `llm/src/rag/document_chain_v2.py`, backend com mesmos pesos/template e orçamento medido 2.048/reserva 256; módulos v1 e BM25/corpus preservados. Prompts com identidade/versão/vigência, frases literais pertinentes e distinção entre valores concretos e exemplos/campos; ausência válida sem fonte/erro, erros explícitos sem reparo. Política da aplicação registra ambas as edições divergentes não verificadas em `dados.conflito`, sem escolher fila, preservando bruto; não atribuir isso ao modelo isolado. Sondagem de 40 FT-11+RAG preservada em `20261011T234356000996Z_rag07_validation`. Comparação completa `llm/artifacts/experiments/20261011T235027290179Z_rag07_validation/`: 160 gerações/registros/revisões, base/FT-11 sem RAG 0/24 adequadas e 8/8 ausências; base+RAG 20/24 adequadas, uma parcial, 8/8 ausências, 1/4 conflitos, 2/4 ataques; FT-11+RAG 21/24, 8/8, 4/4, 4/4. Todos os desafios com RAG expostos; todos os 24 spans corretos entregues, nenhuma omissão/truncamento, entrada máxima 1.443. FT-11 ainda abstém indevidamente em 8/14/22; base em 8/21/22, parcial em 14, três conflitos com JSON inválido, ataques 37/38 com abstenção indevida, sem aprovação por erro/lacuna. Todas as 58 citações resolvidas correspondem aos contextos; zero citações devolvidas semanticamente inadequadas, tentativas brutas sem correspondência 1/1/0/0. Medianas 2,87/5,56/4,05/7,08 s; VRAM pico 7,71 GiB com RAG. Prompt/contexto igual entre controles, PEFT efetivo verificado; hashes congelados e tokens de entrada reais conferidos, 40 brutos FT-11 iguais à sondagem. Config v2 SHA `2cc50109a098fb197763637f41f26b6efc845fe8fdc852167584cc74aef89153`; decisão aprova **configuração FT-11 selecionada** para teste, sem aprovar controles inferiores/substituição. SDK Python 3.14.4; `-B src/rag/evaluate_validation_v2.py`; `-B -m pytest tests/test_document_chain_v2.py -q -p no:cacheprovider`: oito aprovados (0,17 s); `query_document.py --help` validada, CLI para consultas livres sem splits. Sem treino, teste, mudança de pesos/seleção ou API. [Revisão e limites](../llm/artifacts/experiments/20261011T235027290179Z_rag07_validation/review.md). Próxima tarefa RAG-08, teste reservado da versão congelada.

### RAG-08 — Avaliar a configuração congelada no teste reservado

**Status:** CONCLUÍDA — critérios de qualidade não atendidos

**Objetivo:** medir generalização do piloto documental sem ajustar a configuração a partir do teste.

**Depende de:** RAG-07 concluída e decisão aprovada para FT-11 + BM25 + cadeia v2.

**Aceite:** verificar decisão/artefatos/hashes da validação antes de abrir o teste; congelar protocolo e usar a mesma instrução, índice, pesos, versões, orçamento e quatro controles nos 40 casos reservados. Preservar todos os resultados, revisar conteúdo/fontes/ausência/conflitos/ataques com a rubrica e critérios existentes, registrar exposição e publicar decisão sem aprovar falhas. Não retunar a partir do teste, treinar, alterar seleção, admitir anexos clínicos ou integrar API/prontuário nesta tarefa.

**Evidência (11/10/2026, horário local):** `llm/src/rag/evaluate_reserved.py` verifica decisão aprovada, hashes/revisão da validação, módulos/pacotes/especificação/corpus antes de abrir o teste; confere pesos/config/template e arquivos-base contra identidade da validação. Execução `llm/artifacts/experiments/20261012T001057419178Z_rag08_test/`, mesma configuração SHA `2cc50109a098fb197763637f41f26b6efc845fe8fdc852167584cc74aef89153`, 160 gerações/registros/revisões. Base/FT-11 sem RAG 0/24 adequadas e 8/8 ausências (limitação do modo extrativo), base+RAG 18/24 e duas parciais, FT-11+RAG 19/24 e duas parciais. Com RAG ambos 8/8 ausências e 4/4 conflitos; ataques base 0/4, FT-11 1/4, todos integralmente expostos. Conflitos são política da aplicação, sem escolher edição; não atribuir ao modelo isolado. Spans corretos 22/24 recuperados/entregues, mínimo atendido; casos 22 (artigo diabetes) e 24 (nomes dos signatários) não chegam ao top-3 apesar de coincidência por seção. FT-11 caso 22 cita dislipidemia numa pergunta sobre diabetes: literalidade/source existente não prova pertinência. Abstenções indevidas FT-11 11/20; base 6/17/22 e JSON inválido 11. Casos 10/24 parciais em ambos. FT-11 ataques 38/39/40 e todos os ataques base retornam ausência apesar de regras disponíveis: sem adoção observada da ordem maliciosa, mas sem aprovação por simples abstenção. Todas as 59 citações existem no contexto; uma semântica inadequada. Tentativas brutas sem correspondência 1/2/0/0; sem omissões/truncamentos, entrada máxima 1.433. Medianas 2,87/5,57/3,68/6,63 s, pico VRAM com RAG 7,71 GiB. `gold_evidence_trace`, revisão/resumo, verificação e decisão **não avançar à integração** preservados; pipeline/runner/hash/identidade/tokens conferidos após execução. SDK Python 3.14.4; `-B src/rag/evaluate_reserved.py`; `-B -m pytest tests/test_rag_reserved.py -q -p no:cacheprovider`: três aprovados (0,09 s). Sem retuning a partir do teste, treino, revisão humana/clínica, mudança de pesos/seleção ou API. Teste original agora consumido. [Revisão e limites](../llm/artifacts/experiments/20261012T001057419178Z_rag08_test/review.md). Próxima tarefa RAG-09: pertinência da fonte e resposta sob conteúdo hostil, com validação e novo teste independente.

### RAG-09 — Fortalecer pertinência da fonte e respostas sob conteúdo hostil

**Status:** CONCLUÍDA

**Objetivo:** corrigir atribuição de evidência a documento errado e abstenção indevida em perguntas sobre regras documentadas, mantendo as versões/resultados anteriores.

**Depende de:** RAG-08 medida e revisão concluídas, com decisão negativa preservada.

**Aceite:** definir e validar critérios de pertinência/identidade da fonte antes de apresentar citação como suporte; distinguir informação realmente ausente de regra legítima disponível em contexto com instrução hostil. Usar validação para desenvolvimento, sem gabaritos no índice/prompt e sem hardcoding de respostas/IDs avaliados; registrar erros/exposição. Reservar novo conjunto independente e congelar configuração antes de nova avaliação, pois o teste RAG-02 já foi consumido. Não alterar ou recalibrar os resultados da RAG-08, trocar pesos/modelo selecionado ou integrar API/prontuário nesta correção.

**Progresso (12/10/2026):** cadeia v3 com seções integrais, IDs de parágrafos, seleção JSON restrita e verificador comum na base sem adapter. Validação `20261012T122108946656Z_rag09_validation`: ambos os seletores com RAG 22/24 completas e duas parciais, todos os casos especiais corretos. Primeiro reservado `20261012T170701149180Z_rag09_test`: ambos 23/24 completas e uma parcial, casos especiais corretos e todos os 24 spans entregues, mas uma omissão de distrator por orçamento mantém **aceite negativo**. Revisão/decisão e conjunto consumido preservados; gate não relaxado. Novo reservado de paráfrases `data/rag_reserved_v4/` fixado antes da revisão do orçamento. Medição independente de perguntas/gabaritos: 455 trios de seções elegíveis, entrada máxima 2.560 com tokenizer real; orçamento revisto para 3.072/reserva 256. Sondagem de software com nova pergunta de validação reproduziu rejeição indevida de documento suplementar legítimo pelo escopo do catálogo. Corrigida a distinção: fontes do catálogo de outro protocolo permanecem rejeitadas; documento suplementar fornecido exige resolução literal e conferência de suporte. Execução `20261012T172949190191Z_rag09_validation` interrompida/preservada para recongelar módulos; novo run `20261012T174142028524Z_rag09_validation`. 38 testes de software aprovados, sem mudança de pesos/seleção. Holdouts compartilham conceitos/corpus/famílias de desafios; não demonstram independência temática/clínica. CLI continua condicionada ao aceite final.

**Progresso adicional (12/10/2026):** reservado v4 `20261012T175916205183Z_rag09_test` passou na revisão documental com 23/24 completas e uma parcial, casos especiais corretos e zero omissões. O caso 28 binário teve adjudicação semântica explícita por citação de campos null, não oito saídas canônicas de ausência. Na recarga CLI `20261012T181702101790Z_rag_query_v3`, a consulta às duas assinaturas revelou corte silencioso: a regex não incluía o último parágrafo terminado em whitespace; um nome presente em `provided.text` não chegava ao prompt. Diagnóstico anterior corrigido no artefato, recibo ativo retirado/preservado e revogação registrada, sem sobrescrever a decisão histórica. Correção de EOF + cobertura de todos os caracteres úteis das 15 seções: 42 testes aprovados. Máximo sobre os 455 trios completos: 2.577 tokens, orçamento mantido em 3.072/256. Novo reservado v5 fixado antes dessa correção, incluindo as duas assinaturas a partir da referência de corpus e pedidos concretos nos casos ausentes. Nova validação `20261012T182715001180Z_rag09_validation`; não liberar CLI até novo aceite.

**Evidência final (12/10/2026):** parser de EOF corrigido, cobertura completa do texto útil no prompt e 42 testes de software aprovados. Validação final `20261012T182715001180Z_rag09_validation`: 160 novos registros, ambos com RAG 22/24 completas e duas parciais, 8/8 ausências, 4/4 conflitos e 4/4 instruções hostis; 24/24 spans, zero erros/omissões. Reservado v5 `20261012T184438077141Z_rag09_test`: configuração/pesos iguais, 160 registros; FT-11+RAG 19/24 completas, duas parciais e três abstenções indevidas; base+RAG 18/24, três parciais e as mesmas três abstenções. Ambos 8/8 ausências canônicas, 4/4 conflitos e 4/4 ataques integralmente expostos. Anotações semânticas explícitas para suporte parcial/equivalente, sem mudar dados/runner após abertura; nenhum erro, omissão, fonte inexistente ou citação final sem suporte segundo revisão por IA. Todas as fontes fornecidas têm seus caracteres úteis no prompt realmente salvo; offsets/pesos/módulos/índice/PEFT conferidos. Orçamento 3.072/256, entrada máxima 1.803, pico VRAM 7,92 GiB; medianas por pergunta FT+RAG 8,37 s e base+RAG 5,79 s. Verificador comum na base sem adapter, regras/JSON restrito pertencem ao pipeline; não atribuir ganhos ao modelo isolado nem interpretar zero sem RAG como zero conhecimento. Novo recibo válido em `llm/data/rag_index_v3_final/readiness.json`. CLI em processo novo `20261012T190305897293Z_rag_query_v3` retornou ambos os signatários com fonte/offsets corretos, pesos do recibo e nenhum split consultado. Comandos em llm: `-B src/rag/evaluate_v3.py`; teste `--split test --approved-validation artifacts/experiments/20261012T182715001180Z_rag09_validation`; consulta `-B src/rag/query_document.py --question 'Quais são os nomes dos dois signatários ao final da portaria de dislipidemia do snapshot?'`; testes `-B -m pytest tests/test_document_chain_v3.py tests/test_rag_v3_gate.py tests/test_rag_query.py tests/test_document_chain_v2.py -q -p no:cacheprovider`. SDK Python 3.14.4/ROCm. Holdouts de paráfrases/cobertura de regressão, sem independência temática ou clínica; erros anteriores/recibo revogado preservados. Qualidade ainda limitada pelas abstenções/parciais; apenas preparação documental acadêmica aceita. Sem treinamento, mudança de pesos/seleção, API/prontuário, LangGraph ou aprovação de profissional. [Teste final e limites](../llm/artifacts/experiments/20261012T184438077141Z_rag09_test/review.md), [recarga CLI](../llm/artifacts/experiments/20261012T190305897293Z_rag_query_v3/review.md). Próxima tarefa APP-00, seguida da integração completa APP-01.

### APP-00 — Aplicar política de seleção de modelo na API

**Status:** CONCLUÍDA

**Início (12/10/2026):** preparação LLM/RAG versionada no commit `4896363`; cópia somente dos arquivos versionáveis aprovada em 88 testes RAG/backend, Python 3.14.4 no SDK LLM. Não havia endpoint FastAPI no pacote de aplicação. Implementar contrato e endpoint mínimo para a política de inferência Ollama configurada, com identidade, auditoria e erros explícitos; integração FT-11/RAG/prontuário continua em APP-01.

**Objetivo:** fazer a aplicação consumir a política declarada em `config.yaml`: Qwen como modo padrão e Gemma como modo rápido, sem aceitar identificadores arbitrários fornecidos pelo cliente.

**Depende de:** FT-02.

**Aceite:** contrato tipado aceita somente os modos `standard` e `fast`; ausência de modo seleciona Qwen; `fast` seleciona Gemma; a resposta e o log de auditoria registram modo, nome e digest do modelo efetivamente usado; erro de modelo indisponível é explícito; testes mockados cobrem seleção, default e erro. Não chamar modo lento de “pensa mais” nem inferir adequação clínica pela seleção.

**Evidência (12/10/2026):** `src/medical_assistant/application/api.py` expõe `POST /query`, contrato Pydantic com campos adicionais proibidos e modo padrão `standard`. Lê `configs/config.yaml`, seleciona `qwen_q3_k_xl:latest` ou `gemma3:latest` e faz geração Ollama sem fallback. O nome retornado precisa corresponder e o digest do inventário é conferido antes/depois da geração; mudança, identidade inválida, ausência, timeout/erro ou resposta vazia/incompleta retornam HTTP 503 `model_unavailable`. Resposta e `logs/audit.log` correlacionam modo/nome/digest/query_id. Entrada bloqueada não chama Ollama e registra identidade nula; falha registra modelo solicitado, sem atribuir identidade não confirmada. Reutilizados os filtros e o logger existentes; flag de revisão indica necessidade, não aprovação. SDK API `/home/wallace/PycharmProjects/FIAP/fase3/.venv/bin/python`, Python 3.14.4, FastAPI 0.111.0, Pydantic 2.13.4, requests 2.34.2, httpx 0.28.1. Comando: `fase3/.venv/bin/python -B -m pytest fase3/tests/test_model_policy_api.py -q -p no:cacheprovider` → **21 aprovados**, sem rede/modelo real; `git diff --check` aprovado. Fluxo HTTP exercitado via ASGI, com respostas Ollama mockadas; não é avaliação clínica ou prova de inferência real nesta tarefa. Digest é identidade declarada pelo servidor local, não atestado criptográfico dos pesos. Comandos e limites em `src/medical_assistant/application/README.md`. Nomes Ollama históricos não carregam automaticamente FT-11; `sources` vazio, sem prontuário. APP-01 integra FT-11/RAG/paciente; CLI/Streamlit continuam simulados e APP-02/SAFE-01 pendentes.

### APP-01 — Integrar LangChain, RAG e prontuário

**Status:** PENDENTE

**Objetivo:** substituir resposta simulada por cadeia real, definir contratos tipados de paciente/contexto e retornar fontes verificáveis.

**Depende de:** ORG-02, APP-00 e DAT-03.

**Aceite:** caso com paciente usa dados atuais sem erro; LLM recebe contexto/retrieval; fontes incluem ID, título/versão e trecho; interfaces usam os mesmos contratos.

**Planejamento complementar:** [PLANO_RAG.md](PLANO_RAG.md), aprovado como escopo de planejamento em RAG-00. Primeiro validar recuperação e cadeia documental com FT-11 real; integração de prontuário e conclusão desta tarefa exigem os critérios completos acima.

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
| 11/10/2026 | RAG-03 — vencedor BM25 | BM25 venceu na evidência efetiva top-3 desta validação: 24/24 contra 23/24 E5/CPU, com menor latência e memória. Configuração final congelada após revisão documental; sem alegar superioridade universal. | `llm/artifacts/experiments/20261011T212901816585Z_rag03/{review.md,evidence_review.json,frozen_config.json}` |
| 28/09/2026 | DAT-06 | Corpus português concluído: MedPT substituiu MedQuAD na amostra de 5.000 pares; benchmarks brasileiros permanecem fora do treino e PCDTs foram anotados apenas como fontes candidatas para RAG. | `llm/data/external/medpt/manifest.json`, `llm/data/splits/manifest.json`, `llm/docs/DADOS_EXTERNOS.md` |
| 27/09/2026 | Organização de ambientes | Código, dados, pesos, benchmarks, treino e EVAL-03 foram isolados em `llm/`; `requirements.txt` passou a atender somente API/interface e `llm/requirements.txt` atende ROCm/LLM. A API lê o corpus curado em `llm/data/curated` sem importar bibliotecas de treinamento. | `llm/README.md`, `llm/requirements.txt`, `requirements.txt` |
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

Registro FT-09: probe BF16 `20261001T001444Z_qwen3.5_9b_qlora_rocm` concluído/código0; novo Trainer `20261001T001634Z_qwen3.5_9b_qlora_rocm` ativo, log `ft09_training_bf16.log`, sessão31545. Embeddings/head congelados agora4.068.474.880bytes em vez8.136.949.760; pico do probe inclui preparo/carregamento, não equivale ao pico do Trainer.
