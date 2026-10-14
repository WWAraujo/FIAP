# Fase 3 — Assistente médico com LLM

Projeto acadêmico da FIAP para treinar uma LLM com dados sintéticos/anonimizados e integrá-la a um assistente com fontes, revisão humana e auditoria. Não é validado para uso clínico.

## Dois ambientes independentes

`llm/` concentra tarefas que exigem GPU, muitos dados ou ferramentas de avaliação: preparação do corpus, benchmark, QLoRA, adapters e comparação base versus adapter (EVAL-03). `requirements.txt` contém apenas as dependências da aplicação atual; treino e avaliação mantêm seu ambiente em `llm/requirements.txt`.

```text
fase3/
├── llm/
│   ├── data/                 # fonte sintética, corpus curado, MedQuAD e splits
│   ├── artifacts/            # benchmarks, ambiente, checkpoints e adapters
│   ├── models/               # pesos-base locais
│   ├── src/finetuning/       # preparação, treino e avaliação
│   ├── src/rag/              # corpus, busca, cadeia documental local e avaliação
│   ├── tests/
│   └── requirements.txt      # ambiente ROCm/LLM
├── src/medical_assistant/    # aplicação, API, interfaces, segurança e RAG em runtime
├── app.py                   # entrada Streamlit
├── requirements.txt         # API e interface
├── requirements-dev.txt     # testes
└── docs/PLANO_EXECUCAO.md    # tarefas, critérios e evidências
```

O corpus RAG atual está em `llm/data/rag_pilot_v2/`, preservando o inventário v1, com geração/validação do manifesto em `llm/src/rag/` e avaliação reservada em `evaluation/`. Quando a API consultar fontes em tempo de resposta, o código de recuperação permanece em `src/medical_assistant/retrieval/`; assim ela não depende das bibliotecas de treinamento.

## Executar a aplicação

Com Docker Engine/Compose ativos, partindo da raiz:

```bash
cd fase3
docker compose up --build -d
```

Acesse http://localhost:8501 e clique em **Inicializar Sistema**. Para acompanhar logs e encerrar:

```bash
docker compose logs -f medical-assistant
docker compose down
```

O Docker instala dependências, configura o diretório de trabalho e inicia o Streamlit. A porta é publicada apenas em localhost. Dados e logs persistem em `fase3/data/` e `fase3/logs/`; pesos, ambientes virtuais e artefatos de treinamento não entram no contexto de build. A instalação editável dentro da imagem mantém os caminhos de recursos relativos a `/app/src` usados pelo código atual.

**Limite atual:** chat com respostas simuladas e workflow sequencial, sem integração FT-11/RAG/Ollama pela interface. O banco cria dados sintéticos de exemplo na primeira execução. A base opcional do protótipo é `data/curated/dataset_medico_anonimizado.json`; sua ausência não impede abrir a interface, mas deixa a busca local vazia. A API é uma entrada separada, descrita abaixo. Este Compose ainda não entrega o assistente integrado.

## Desenvolvimento sem Docker

No Ubuntu, a partir de `fase3`:

```bash
python3 -m venv .venv-api
.venv-api/bin/python -m pip install -r requirements-dev.txt
.venv-api/bin/python -m pip install -e .
.venv-api/bin/python -m streamlit run app.py
.venv-api/bin/python -m pytest tests -q
```

No Windows, use o executável equivalente do ambiente existente: `..\\.venv\\Scripts\\python.exe`, partindo de `fase3`. O arquivo `app.py` é a entrada do Streamlit, não um launcher Python independente. Configuração local em `.streamlit/config.toml`; o Docker sobrescreve o endereço para escutar dentro do contêiner.

APP-00 implementa `POST /query` em FastAPI com modos `standard`/Qwen e `fast`/Gemma conforme `configs/config.yaml`, identidade do modelo, auditoria e erro explícito sem fallback. **21 testes mockados aprovados**. [Execução da API e limites](src/medical_assistant/application/README.md). A API ainda não carrega FT-11 nem consulta RAG/prontuário; essa integração é APP-01. CLI/Streamlit mantêm respostas simuladas.

## Ambiente LLM local ROCm

```bash
cd fase3/llm
sudo apt install python3.14-venv python3.14-dev
python3.14 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python src/finetuning/scripts/check_rocm_before_training.py
```

Em 05/10/2026, a FT-11 treinou e recarregou um adaptador do `Qwen/Qwen3.5-9B` com 1.408 exemplos. No teste reservado, houve 22 respostas factuais adequadas e 11 inadequadas em 48, contra 20 e 21 do base; documentos/contexto tiveram 100/112 acertos completos contra 29/112. A seleção está em `llm/artifacts/best_model.json`. Na prova adicional, houve regressão de 6 para 5 adequados em 12 factuais, apesar de menos erros e ganho contextual; generalização parcial. Há erros e não há validação clínica. Consulte a [revisão da FT-11](llm/artifacts/experiments/20261005T120254Z_factual_v2/review.md) e os comandos atuais em [llm/README.md](llm/README.md).

Em 06/10/2026, `llm/data/factual_v3/` foi preparada com 1.920 exemplos de treino e 80/80 de validação/teste, referências MedlinePlus revisadas documentalmente por IA e temas separados. O piloto FT-13 concluiu 48 passos a partir de FT-11: na nova validação, 21/24 factuais adequadas contra 19/24 do controle e 22/24 da base, mantendo 55/56 acertos documentais/contextuais do controle. Melhorou FT-11 neste conjunto, mas não cumpriu todos os critérios para expandir; treino completo não iniciado e teste novo sem geração. Na limpeza de 10/10/2026, os pesos rejeitados FT-12/FT-13 foram removidos; seus resultados e entradas de avaliação permanecem, junto com FT-11 e seu antecessor FT-10. FT-11 permanece selecionado. Consulte a [revisão FT-13](llm/artifacts/experiments/20261006T232619Z_ft13_pilot/review.md); o resultado não é validação clínica.

## Próximas tarefas

Consulte o [plano de execução](docs/PLANO_EXECUCAO.md). A comparação local do adaptador não comprova integração à aplicação; essa integração e sua validação permanecem pendentes.

O [planejamento do RAG](docs/PLANO_RAG.md) registra BM25 vencedor e **preparação documental RAG-09 concluída em 12/10/2026**. A cadeia local LCEL usa FT-11 para selecionar evidência, verificador comum na base sem adapter, contexto completo e fontes/offsets verificáveis. Validação final: 22/24 completas, duas parciais e todos os 16 casos especiais corretos em ambas as condições com RAG. No [teste final de 160 registros](llm/artifacts/experiments/20261012T184438077141Z_rag09_test/review.md), FT-11+RAG teve 19/24 completas, duas parciais e três abstenções indevidas; base+RAG 18/24, três parciais e as mesmas abstenções. Ambos 8/8 ausências, 4/4 conflitos e 4/4 instruções hostis; 24/24 spans e todo o texto útil entregues ao prompt, zero erros/omissões. Orçamento 3.072/reserva 256, pico VRAM 7,92 GiB, 42 testes de software aprovados. [Recarga pela CLI](llm/artifacts/experiments/20261012T190305897293Z_rag_query_v3/review.md) retorna ambas as assinaturas com fonte correta, sem ler perguntas/gabaritos. Comandos e recibo de aceite no [README RAG](llm/src/rag/README.md). Falhas e versões anteriores preservadas; conjuntos reservados de paráfrases compartilham corpus/conceitos, sem independência temática ou clínica. A qualidade permanece limitada; sem treino novo ou alteração dos pesos selecionados. APP-00 concluída; próxima tarefa APP-01 para integrar FT-11/RAG/prontuário; LangGraph/revisão humana no fluxo real continuam pendentes.
