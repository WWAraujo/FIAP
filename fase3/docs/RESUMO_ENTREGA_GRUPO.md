# Apresentação da entrega — Fase 3

Consolidação em 16/10/2026, branch `feature/fase-3`. Este resumo reúne o trabalho acumulado da aplicação, integração e organização da entrega. O fine-tuning FT-10/FT-11 já havia sido realizado; a integração recente utiliza esses pesos, sem treinamento novo.

## O que o projeto entrega

Um assistente acadêmico para médicos/equipe assistencial, com interface web, prontuários sintéticos em SQLite, modelo Qwen3.5-9B adaptado com QLoRA (FT-11), recuperação de documentos, cálculo de IMC, workflow LangGraph e revisão humana registrada. O sistema apresenta evidências da resposta e grava eventos para auditoria.

O fluxo principal é: selecionar paciente → ler contexto atual do banco → verificar a solicitação → escolher a rota → consultar registros/documentos ou calcular IMC → validar resposta/fontes → apresentar rascunho → registrar decisão explícita de revisão quando aplicável.

## Organização dos arquivos

Todos os caminhos abaixo são relativos a `fase3/`, exceto `.gitignore` e `AGENTS.md` da raiz FIAP.

```text
fase3/
├── app.py                           entrada da interface
├── .streamlit/config.toml           apresentação e execução local
├── requirements*.txt / pyproject.toml
├── configs/config.yaml             configuração da rota legada Ollama
├── src/medical_assistant/
│   ├── application/                 API, assistente, contratos e LangGraph
│   ├── clinical/                    SQLite, contexto e pacientes sintéticos
│   ├── tools/                       catálogo e cálculo de IMC
│   ├── retrieval/                   cliente HTTP do FT-11
│   ├── safety/                      bloqueios, auditoria e revisão
│   └── interfaces/                  Streamlit e CLI
├── tests/                           testes da aplicação
├── llm/
│   ├── src/finetuning/              preparação, treino e avaliação
│   ├── src/rag/                     modelo, recuperação e cadeias LCEL
│   ├── data/                        corpora e índices permitidos
│   ├── artifacts/                   metadados e evidências selecionadas
│   ├── tests/                       testes do módulo LLM
│   └── requirements.txt / pyproject.toml
└── docs/                            relatório, validação e apresentação
```

### Interface e aplicação

| Arquivo | Responsabilidade |
|---|---|
| `app.py` | Entrada para iniciar o site com Streamlit. |
| `interfaces/streamlit_app.py` em `src/medical_assistant/` | Busca/seleção de paciente, chat, prontuário, triagem, fontes, auditoria e formulário de revisão. Limpa conversa/workflow ao trocar paciente. Visual atualizado e sem botão de iniciar sistema. |
| `interfaces/cli.py` | Interface de linha de comando mantida compatível com o assistente. |
| `application/assistant.py` | Coordena a pergunta: valida entrada, aplica guardas, reconhece rotas, chama o backend, reúne fontes e registra rascunho/auditoria. |
| `application/contracts.py` | Define estruturas tipadas de pergunta, contexto, resposta e fonte. Confere citações literais/offsets e vínculo entre resposta e fonte fornecida. |
| `application/api.py` | Disponibiliza `/assistant/query`, `/workflows`, `/reviews/{id}` e `/reviews/{id}/decision`. Mantém `/query` como legado Ollama independente do chat FT-11. |
| `retrieval/ft11_client.py` | Envia pergunta/contexto por HTTP ao runtime FT-11 e valida a resposta; indisponibilidade é erro explícito. |

Os caminhos abreviados dessa tabela partem de `src/medical_assistant/`.

### Prontuário e tool de IMC

| Arquivo | Responsabilidade |
|---|---|
| `clinical/patient_database.py` | Cria/acessa o SQLite e consulta dados vinculados ao ID do paciente. |
| `clinical/seed_patients.py` | Insere dez perfis sintéticos variados, preservando registros existentes e reconhecendo os IDs renomeados conhecidos. Banco vazio recebe também o perfil inicial da aplicação. |
| `clinical/patient_queries.py` | Trata perguntas de alergia/IMC, utiliza dados cadastrados e constrói respostas com fontes. |
| `clinical/patient_evidence.py` | Reconhece assunto por regras e seleciona medicamentos, exames, consultas, sinais, histórico ou idade do contexto atual. Filtra, por exemplo, exames vencidos e próxima consulta. Monta trechos/IDs de evidência. |
| `tools/__init__.py` | Expõe o catálogo e a execução das ferramentas disponíveis. |
| `tools/bmi.py` | Implementa a tool `calcular_imc`, com validação dos valores e cálculo determinístico peso/altura². |

Dados do paciente ficam em `data/prontuarios.db`, local. Não são documentos vetorizados nem informações gravadas nos pesos da LLM. O sistema consulta o banco novamente a cada pergunta. A base local customizada tem 11 pacientes; nomes/IDs e datas podem diferir de uma base recém-gerada.

### Modelo, RAG e LangChain

| Arquivo/pasta | Responsabilidade |
|---|---|
| `llm/src/rag/runtime_server.py` | Processo persistente que carrega FT-11, verifica identidade/recibo e atende consultas na porta 8001. Escolhe cadeia factual de paciente ou documental. |
| `llm/src/rag/ft11_backend.py` | Carregamento local da base/adapter, identidade e geração; versões posteriores reutilizam esse backend. |
| `llm/src/rag/document_chain.py`, `document_chain_v2.py`, `document_chain_v3.py` | Versões preservadas das cadeias documentais LCEL. O runtime atual utiliza v3: BM25, seleção FT-11, verificador-base e resolução literal de fontes. |
| `llm/src/rag/patient_facts_chain.py` | Cadeia LCEL factual: prepara prompt com registros fornecidos, consulta FT-11 e valida escolha entre evidência e ausência. Valores apresentados são resolvidos a partir dos registros, sem permitir novos valores livres. |
| `llm/src/rag/query_document.py` | Consulta documental pela CLI e auditoria por execução; não é o ponto de entrada do site. |
| `llm/src/rag/prepare_manifest.py`, `compare_retrieval.py` | Preparação/conferência do corpus e comparação de recuperação BM25/E5. BM25 foi escolhido; E5 não é dependência do chat atual. |
| `llm/src/rag/evaluate_*.py`, `review_v3.py`, `prepare_reserved_v*.py` | Avaliação, revisão e preparação dos conjuntos versionados. Testes reservados consumidos não devem ser reabertos como novos testes independentes. |
| `llm/src/rag/evaluate_tool_selection.py`, `smoke_test_tools.py` | Ensaios separados de seleção de ferramentas e três rotas. Não integram um agente autônomo ao chat. |
| `llm/data/rag_pilot_v2/`, `rag_index_v3_final/` | Corpus admitido, manifestos, avaliação e índice/recibo documental. |

A recuperação atual é lexical **BM25**, não uma base vetorial. O piloto possui 15 seções elegíveis de nove documentos. As fontes oficiais admitidas contêm páginas de portaria, sem anexos clínicos completos. Adicionar protocolos exige nova versão/admissão/avaliação, sem sobrescrever o índice aceito.

### LangGraph e workflow

O grafo está em **`src/medical_assistant/application/workflow.py`**, classe `MedicalWorkflowGraph`, método `_build_graph()`. Compila um `StateGraph` com avaliação inicial, classificação demonstrativa, seleção documental, exames, propostas e conclusão. Erros interrompem o fluxo; cada execução/nó trabalha com estado isolado.

**`application/workflow_service.py`** conecta esse grafo à aplicação: lê prontuário/exames, consulta documentos, executa o grafo, agrega fontes, grava auditoria e cria revisão pendente. A UI/API usa esse serviço. As pendências refletem exames cadastrados; ausência documental exige completar evidência. As regras de prioridade são demonstrativas, sem validação clínica.

### Segurança, revisão e auditoria

| Arquivo | Responsabilidade |
|---|---|
| `safety/guardrails.py` | Aplica bloqueios lexicais a pedidos de prescrição/diagnóstico e instruções indevidas, antes da inferência e na resposta. |
| `safety/audit.py` | Registra consultas, respostas completas, fontes, modelo e eventos correlacionados por IDs. |
| `safety/reviews.py` | Persiste snapshot/hash, paciente, consulta e revisão em `data/reviews.db`. Exige identificação/decisão/confirmação explícitas, verifica integridade e impede sobrescrever uma decisão. |

A aprovação é uma ação da pessoa no formulário/API; o modelo não aprova respostas. `validacao_humana: true` indica necessidade de revisão. A identidade é autodeclarada: não há autenticação profissional ou validação de CRM. Os ensaios de aprovação usaram bases temporárias; os rascunhos da integração real permaneceram pendentes.

## Fine-tuning e evidências acadêmicas

A preparação está em `llm/src/finetuning/data/` e nos scripts `prepare_contextual_corpus.py`/`prepare_factual_corpus.py`. O treinamento é executado por `llm/src/finetuning/scripts/train_qwen3_8b_rocm.py`; a comparação base/adapter, por `evaluate_qwen3_base_vs_adapter_rocm.py`. Os comandos completos estão em [README LLM](../llm/README.md).

FT-10 usa corpus sintético `llm/data/contextual_v1/` (960/120/120). FT-11 continua FT-10 com `factual_v2/` (1.408/160/160 e desafio de 40), incluindo protocolos, FAQs, modelos de laudos/receitas e procedimentos, além de perguntas públicas traduzidas/curadas. Proveniência e atribuições acompanham o corpus.

FT-11 utiliza Qwen3.5-9B, QLoRA NF4/BF16, uma época, LR 1e-5, comprimento 512 e LoRA r=8. No teste principal: 100/112 acertos documentais/contextuais contra 29/112 da base; 22/48 factuais adequados contra 20/48, ainda com 11 inadequados. No desafio adicional houve regressão de adequados (5/12 contra 6/12). São resultados limitados, sem validação clínica ou superioridade universal.

A seleção fica em `llm/artifacts/best_model.json`. Manifestos/configurações FT-10/FT-11 e sete execuções completas selecionadas permanecem no Git, junto de revisões históricas, incluindo reprovações. A relação exata está em `llm/artifacts/versioning_policy.json` e no README dos artefatos. Os pesos ficam locais.

## Testes e validação

Foram aprovados **82 testes de aplicação/UI** e **124 testes do módulo LLM**. Incluem contratos, cálculo, banco, fontes, grafo, concorrência, falhas, revisão, bloqueios e formulário. Os mesmos testes passaram em cópia contendo somente os arquivos selecionados para a entrega, usando os SDKs existentes. Isso não representa instalação limpa em outra máquina nem validação clínica.

A integração `20261016T101610484760Z_delivery_live` passou em **6/6 verificações com FT-11 real**: medicamentos, exame vencido, consulta futura, portaria, bloqueio e workflow com fontes. Resumo versionado em `llm/artifacts/experiments/`; respostas operacionais permanecem locais.

Testes novos/ajustados relevantes: `tests/test_workflow_graph.py`, `test_review_and_context.py`, `test_streamlit_app.py`, `test_ft11_integration.py`, `test_patient_allergies.py`, `test_patient_bmi.py`, `test_bmi_tool.py`, `test_seed_patients.py`, `tests/conftest.py` e `llm/tests/test_patient_facts_chain.py`/`test_app_runtime.py`. As demais regressões documentais também passaram.

## O que entra no commit

Código Python da aplicação e pipeline de treino/RAG, configuração/dependências, testes, corpora permitidos, manifestos/índices, evidências acadêmicas selecionadas e documentação. `.gitignore` utiliza seleção explícita para a Fase 3.

Arquivos operacionais e sondagens dispensáveis saem do controle do Git conforme a limpeza autorizada anteriormente; os existentes são preservados localmente. Por isso a contagem de exclusões no commit é grande. A retirada do índice não apaga o histórico dos commits anteriores. Docker e arquivos de demonstração/adapters antigos sem uso foram removidos em tarefas anteriores; o início atual é por processos Python locais.

Ficam fora: pesos-base/adapters binários, SDKs `.venv`, caches, SQLite, logs, respostas de consultas com prontuário, PDFs integrais/extrações locais e detalhes de experimentos não selecionados. Alterações pessoais do PyCharm em `.idea/` não fazem parte da entrega. Nenhum push é realizado por este pedido de commit.

## Documentação para dividir com o grupo

| Documento | Uso |
|---|---|
| [README principal](../README.md) | Instalação, processos, banco, comandos e limitações. |
| [Relatório técnico](RELATORIO_TECNICO.md) | Arquitetura/diagramas, fine-tuning, avaliação e análise completa. |
| [Guia de validação](GUIA_VALIDACAO.md) | Perguntas e resultados esperados para demonstração. |
| [Roteiro de vídeo](ROTEIRO_VIDEO.md) | Apresentação de até 15 minutos. |
| [Plano de execução](PLANO_EXECUCAO.md) | Critérios, histórico e evidências por tarefa. |
| [Revisão documental](REVISAO_DOCUMENTACAO.md) | Inventário dos Markdown ativos/históricos/gerados. |
| [AGENTS da fase](../AGENTS.md) | Instruções de manutenção do código e de cada documento. |

## Como demonstrar

Na raiz FIAP, iniciar o modelo e aguardar prontidão:

```bash
fase3/llm/.venv/bin/python -B fase3/llm/src/rag/runtime_server.py
```

Em outro terminal:

```bash
cd fase3
.venv/bin/python -B -m streamlit run app.py
```

Abrir `http://localhost:8501`. Mostrar paciente, alergia, IMC, medicamento, exame vencido, fonte documental, workflow e revisão explícita, seguindo o guia. A API FastAPI é opcional ao site.

Em outro computador, o clone sozinho não carrega a LLM: é necessário preparar os dois SDKs e copiar os pesos da execução aceita ou reproduzir treino/avaliação, respeitando caminhos/identidade e recibos. Não alterar hashes para contornar a conferência. Instalação/configuração detalhadas nos READMEs.

## Limites e pendências para a apresentação

O chat usa regras para selecionar rotas; o catálogo contém apenas IMC, sem agente autônomo escolhendo tools. Guardas e citações não garantem segurança universal. A base documental é pequena; o modelo pode errar ou se abster. Não há prescrição, diagnóstico definitivo, autenticação profissional ou validação clínica comprovada.

O desenvolvimento identificado está implementado e testado no escopo descrito. Ainda é necessário gravar/apresentar o vídeo e organizar o envio institucional. Uma decisão simulada ou um teste automatizado não substitui avaliação de profissional de saúde.
