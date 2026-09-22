# Fase 3 — Assistente médico com LLM

Projeto acadêmico do Tech Challenge FIAP para construir um assistente de apoio a profissionais de saúde, personalizado com dados sintéticos do hospital e integrado a prontuários e fluxos de decisão.

**Status em 20/09/2026: protótipo parcial.** Há componentes reutilizáveis, mas fine-tuning, integração real com LangChain/LangGraph e validação ponta a ponta ainda não estão concluídos. O projeto não é validado para uso clínico.

## Requisitos da entrega

A referência são as cinco primeiras páginas da [PROPOSTA.md](PROPOSTA.md). As autoavaliações posteriores não alteram o enunciado.

1. Fine-tuning de uma LLM com protocolos, FAQs, laudos, receitas e procedimentos, após preprocessing, anonimização e curadoria.
2. Pipeline LangChain que integre a LLM personalizada, prontuários estruturados e contexto atualizado do paciente.
3. Fluxos LangGraph, limites de atuação, validação humana, fontes verificáveis e auditoria.
4. Código modular, dados sintéticos/anonimizados, relatório com diagrama e avaliação, e vídeo de até 15 minutos.

RAG recupera contexto em tempo de resposta; ele complementa, mas não substitui, o fine-tuning exigido.

## Organização do trabalho

O [plano de execução](docs/PLANO_EXECUCAO.md) é a fonte de verdade para tarefas, dependências, critérios de aceite e evidências.

| Área | Responsabilidade | Situação atual |
|---|---|---|
| [src/medical_assistant/evaluation/](src/medical_assistant/evaluation/README.md) | Comparar candidatos e depois base versus adapter treinado | Benchmark de inferência existente; não escolhe sozinho o modelo para treino |
| `src/medical_assistant/data/` | Gerar, anonimizar, curar e dividir dados | Pipeline existe, mas precisa ser corrigido e validado |
| `src/medical_assistant/training/` | Treinar, salvar, recarregar e avaliar adapter | Só há configuração; não há treinamento real |
| `src/medical_assistant/retrieval/` | Recuperar protocolos com fontes identificáveis | Adaptador RAG existente; ranking e fontes precisam evoluir |
| `src/medical_assistant/clinical/` e `application/` | Prontuário, assistente e workflow | SQLite e regras determinísticas existem; integração está incompleta |
| `tests/` | Testar comportamento e integração | Diretório sem testes |
| `artifacts/`, `models/`, `logs/` | Evidências de execuções e artefatos | Resultados precisam ser preservados e rastreáveis |

Não há motivo para transformar essas áreas em serviços independentes. São pacotes internos da mesma aplicação: avaliação compara modelos e testes validam software.

## Estrutura atual e planejada

```text
fase3/
├── data/
│   ├── raw/                        # Fonte sintética mínima, versionada
│   ├── curated/                    # Dados anonimizados/curados, gerados localmente
│   └── splits/                     # Splits gerados localmente
├── artifacts/                      # Resultados de execuções
│   ├── benchmarks/
│   └── adapters/
├── models/                         # Modelos locais e futuros adapters
├── src/
│   └── medical_assistant/
│       ├── data/                   # Geração e preprocessing
│       ├── evaluation/             # Benchmark e casos
│       ├── training/               # Configuração e futuro executor de treino
│       ├── retrieval/              # RAG e fontes
│       ├── clinical/               # Prontuários
│       ├── application/            # Assistente e workflow
│       ├── safety/                 # Guardrails futuros
│       └── interfaces/             # CLI e Streamlit
├── tests/                          # A implementar
├── scripts/                        # Demonstrações e automações manuais
├── docs/PLANO_EXECUCAO.md          # Controle de tarefas e evidências
└── PROPOSTA.md                     # Enunciado
```

A separação física foi concluída em **ORG-02**. O executor de fine-tuning continua pendente em **FT-01**.

## Inventário honesto

| Área | Existe | Falta |
|---|---|---|
| Dados | Fonte sintética versionada, curadoria com metadados e testes de PII | Separar treino/validação/teste sem vazamento e com cobertura por tipo |
| Benchmark | CLI Ollama, 7 casos sintéticos, CSV/JSON e resumo | Critérios confiáveis, ambiente/versionamento, revisão humana e decisão de treinabilidade |
| Fine-tuning | Campos de configuração de LoRA | Treino, checkpoint/adapter, recarga e comparação com baseline |
| RAG | Busca lexical e cliente Ollama | Recuperação com ranking, fontes por ID/título/versão e separação do treinamento |
| Prontuário | SQLite com seis tabelas e consultas | Contratos coerentes com assistente/workflow |
| Assistente | Memória local, respostas simuladas e regras de triagem | LLM real via LangChain, LangGraph real e estado isolado por execução |
| Segurança/auditoria | Filtros lexicais e logs JSON | Cobertura do caminho real, aprovação humana verificável e fontes rastreáveis |
| Interfaces | Terminal, Streamlit e Docker | Correções de contratos e demonstração ponta a ponta |

Não trate as métricas históricas de “100% de segurança” como evidência: o avaliador atual aprova respostas vazias em vários cenários. Também não há resultados brutos auditáveis no checkout atual.

## Seleção do modelo para fine-tuning

O benchmark mede inferência; ele não prova que um modelo é o melhor candidato para treino. A decisão precisa combinar:

1. Qualidade e segurança das respostas, avaliadas por casos reservados e revisão humana.
2. Licença, origem dos pesos, tokenizer/template e compatibilidade com PEFT/LoRA.
3. Memória, tempo e hardware disponíveis para executar um ensaio de treino e recarga.
4. Comparação base versus adapter sob as mesmas condições.

Mapeie cada alias Ollama ao modelo-base e à quantização. A decisão final vem após um ensaio curto de treinamento, não da menor latência de um GGUF.

## Comandos disponíveis hoje

Os exemplos partem da raiz `FIAP/` e usam a `.venv/` existente.

```powershell
# Recriar a fonte sintética versionada
& .\.venv\Scripts\python.exe -B fase3/src/medical_assistant/data/generate_synthetic.py

# Gerar a versão curada local
& .\.venv\Scripts\python.exe -B fase3/src/medical_assistant/data/preprocessing.py

# Criar os JSONL reprodutíveis de treino, validação e teste
& .\.venv\Scripts\python.exe -B fase3/src/medical_assistant/data/split_dataset.py --seed 42

# Para misturar o corpus externo com os exemplos sintéticos em português,
# consulte docs/DADOS_EXTERNOS.md. Requer rede e gera dados ignorados pelo Git.

# Validar anonimização e curadoria sem chamar modelos
& .\.venv\Scripts\python.exe -m pytest fase3/tests/test_preprocessing.py

# Consultar as opções do benchmark sem executar modelos
& .\.venv\Scripts\python.exe -B fase3/src/medical_assistant/evaluation/benchmark_ollama.py --help

# Consultar o resumo sem executar modelos
& .\.venv\Scripts\python.exe -B fase3/src/medical_assistant/evaluation/evaluate_results.py --help
```

Para executar o benchmark, siga o [README da avaliação](src/medical_assistant/evaluation/README.md). O script pode baixar modelos ausentes; `--no-auto-setup` apenas desativa instalação/inicialização automática do Ollama.

```powershell
& .\.venv\Scripts\python.exe fase3/src/medical_assistant/interfaces/cli.py
& .\.venv\Scripts\python.exe -m streamlit run fase3/src/medical_assistant/interfaces/streamlit_app.py
```

As interfaces são pontos de entrada do protótipo, não demonstração de funcionamento completo. Ainda não há comando funcional de fine-tuning; `scripts/demo.py` e `retrieval/ollama_adapter.py` não treinam o modelo.

## Ambiente reproduzível

O ambiente validado pela tarefa ORG-03 usa Python 3.12, as versões fixadas em
`requirements.txt` e Ollama instalado no host. Para recriá-lo sem alterar a
`.venv` principal:

```powershell
py -3.12 -m venv fase3/.venv-org03
& fase3/.venv-org03/Scripts/python.exe -m pip install --upgrade pip
& fase3/.venv-org03/Scripts/python.exe -m pip install -r fase3/requirements.txt
& fase3/.venv-org03/Scripts/python.exe fase3/scripts/verify_environment.py `
  --output fase3/artifacts/environment/org03_YYYYMMDD.json
```

No ambiente validado, a AMD Radeon RX 9060 XT foi detectada pelo Windows, mas
o PyTorch é `+cpu` e não expõe CUDA/HIP/ROCm. Portanto, o fine-tuning local
permanece CPU-only até que a compatibilidade de aceleração seja validada em um
ambiente suportado. Consulte o artefato de ambiente antes de planejar FT-01.

## Documentação

- [Plano de execução](docs/PLANO_EXECUCAO.md): tarefas e evidências.
- [Dados externos](docs/DADOS_EXTERNOS.md): reprodução, licença e limites do MedQuAD.
- [Guia da fase](AGENTS.md): regras de execução e atualização do plano.
- [Avaliação de modelos](src/medical_assistant/evaluation/README.md): operação e limites do benchmark.
- [Treinamento](src/medical_assistant/training/README.md): escopo e artefatos esperados do fine-tuning.
- [Relatório técnico](docs/RELATORIO_TECNICO.md) e [benchmark histórico](docs/BENCHMARK_FINAL.md): documentos a corrigir em **ENT-01**; não usá-los como aceite.

O `.gitignore` atual exclui `data/`, portanto clone do repositório não recebe os dados sintéticos nem o gerador. A tarefa **DAT-01** resolve isso sem versionar pesos grandes ou logs de pacientes.
