# Fase 3 — Assistente médico acadêmico

Protótipo para médicos/equipe assistencial, com prontuários sintéticos, FT-11 (Qwen3.5-9B com QLoRA), LangChain/LCEL, LangGraph, fontes e revisão registrada. Não foi validado clinicamente e não tem autenticação profissional. O sistema não emite prescrição ou diagnóstico definitivo.

## Iniciar no PyCharm

Há dois ambientes independentes. Se os SDKs já estão configurados, não os recrie.

| Processo | Interpretador | Configuração |
|---|---|---|
| FT-11/RAG | `fase3/llm/.venv/bin/python` | Script `fase3/llm/src/rag/runtime_server.py` |
| Interface | `fase3/.venv/bin/python` | Módulo `streamlit`, parâmetros `run app.py`, diretório de trabalho `fase3` |

Na raiz FIAP, inicie primeiro o modelo e aguarde a prontidão:

```bash
fase3/llm/.venv/bin/python -B fase3/llm/src/rag/runtime_server.py
```

Mensagem esperada: `FT-11/RAG pronto em http://127.0.0.1:8001`. Em outro terminal:

```bash
cd fase3
.venv/bin/python -B -m streamlit run app.py
```

Abra **http://localhost:8501**. Não há botão de inicialização. Selecione um paciente e use Atendimento, Prontuário, Triagem e Auditoria. O processo do modelo mantém uma cópia carregada; consultas são serializadas porque o verificador documental alterna o adapter.

## Preparar outro ambiente

A instalação validada usa Linux, Python 3.14.4 e GPU AMD RX 9060 XT de 16 GiB com ROCm. `llm/requirements.txt` é específico desse ambiente. Em um clone, prepare cada módulo separadamente:

```bash
cd fase3
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -e .
cd llm
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
```

Esses comandos não instalam o driver da GPU. A base Qwen e os adapters FT-10/FT-11 são arquivos grandes locais e não estão no Git. Obtenção, reprodução e caminhos estão no [README LLM](llm/README.md). O adapter do projeto não possui URL pública registrada: é preciso copiá-lo de uma execução existente ou reproduzir os treinos e avaliar a nova seleção. Nova execução não herda o recibo do modelo histórico.

`best_model.json`, a configuração do adapter e o manifesto contêm caminhos absolutos da execução original. Preserve os caminhos canônicos ao transferir essa execução; uma mudança de identidade/configuração exige verificação e registro próprios, sem editar hashes para contornar o recibo. O runtime confere versões, módulos, índice, configuração e pesos antes de atender. Sem esses arquivos locais, o clone não executa inferência. Os testes de software usam mocks e podem ser executados sem carregar os pesos.

## Comportamento disponível

- Prontuário: SQLite relido a cada consulta. Alergias e IMC têm rotas determinísticas com fontes.
- Fatos do paciente: medicamentos, exames, consultas futuras, idade e sinais vitais são recuperados do contexto e conferidos pelo FT-11 na cadeia LCEL factual. A resposta copia registros fornecidos ou declara ausência, sem inferir tratamento.
- Documentos: BM25, seleção documental FT-11, verificador-base e fontes com título, versão, trecho e offsets. O corpus oficial admitido contém páginas de portaria, sem anexos clínicos; não supor cobertura completa dos PCDTs.
- Triagem: grafo LangGraph com estado isolado, consulta ao prontuário/documentos, exames reais pendentes/vencidos, alertas e propostas rastreáveis. Regras de prioridade são demonstrativas. Sem documento pertinente, o sistema solicita completar evidências.
- Revisão: cada rascunho elegível tem snapshot/hash e estado pendente. O usuário informa responsável, decisão e confirmação explícita. A decisão é única e vinculada ao paciente. A identidade é declarada, não autenticação profissional; testes simulados não são aprovação clínica.

O catálogo `tools` contém somente `calcular_imc`. O chat ainda usa regras para reconhecer rotas; não há agente escolhendo tools automaticamente. Não há rota geral de conhecimento livre sem documentos. Bloqueios lexicais e conferência de fontes reduzem erros, mas não garantem segurança universal.

## Banco e registros locais

`data/prontuarios.db` contém os prontuários; `data/reviews.db`, snapshots/decisões. `logs/audit.log` grava respostas completas, consultas, fontes, modelo, workflow e revisão correlacionados. Resultados do runtime ficam em `llm/artifacts/experiments/`. Esses dados não entram no Git.

Um banco vazio recebe um paciente sintético inicial. Para gerar os dez perfis de simulação, no SDK API a partir de `fase3`:

```bash
.venv/bin/python -B -m medical_assistant.clinical.seed_patients
```

O seed conserva registros existentes, incluindo os IDs renomeados conhecidos `paciente_*`; um banco novo usa os IDs sintéticos originais. Não usar dados reais de pacientes. Para outra máquina, transfira a base de demonstração separadamente ou recrie os exemplos.

## API e testes

A API é opcional ao site. Na raiz FIAP:

```bash
fase3/.venv/bin/python -B -m uvicorn medical_assistant.application.api:app --app-dir fase3/src --host 127.0.0.1 --port 8000
fase3/.venv/bin/python -B -m pytest fase3/tests -q -p no:cacheprovider
fase3/llm/.venv/bin/python -B -m pytest fase3/llm/tests -q -p no:cacheprovider
```

Rotas: `/assistant/query`, `/workflows`, `/reviews/{id}` e `/reviews/{id}/decision`. A rota `/query` é o legado Ollama, independente do chat FT-11. Contratos em [application/README.md](src/medical_assistant/application/README.md). `FT11_RUNTIME_URL` altera o destino padrão localhost:8001; o servidor aceita `--port`. `REVIEW_DB_PATH` permite isolar a persistência de revisão em testes.

## Entrega e documentação

O Git inclui código da aplicação, treino/preparação, testes, corpora sintéticos/curados permitidos, metadados e evidências selecionadas. Pesos, SDKs, banco, logs e sondagens operacionais permanecem locais. Não é necessário versionar milhares de execuções.

- [Proposta original](PROPOSTA.md).
- [Relatório técnico e diagramas](docs/RELATORIO_TECNICO.md).
- [Resumo da entrega para o grupo](docs/RESUMO_ENTREGA_GRUPO.md).
- [Guia de validação](docs/GUIA_VALIDACAO.md).
- [Roteiro do vídeo de até 15 minutos](docs/ROTEIRO_VIDEO.md).
- [Plano e evidências de execução](docs/PLANO_EXECUCAO.md).
- [Índice da revisão Markdown](docs/REVISAO_DOCUMENTACAO.md).
- [Instruções de manutenção](AGENTS.md).

Desenvolvimento do fluxo acadêmico implementado e validado no escopo registrado no relatório. Gravação/apresentação do vídeo e validação clínica por profissionais não são substituídas por código nem por decisões simuladas em testes.
