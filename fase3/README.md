# Fase 3 — Assistente médico com LLM

Projeto acadêmico da FIAP para treinar uma LLM com dados sintéticos/anonimizados e integrá-la a um assistente com fontes, revisão humana e auditoria. Não é validado para uso clínico.

## Dois ambientes independentes

`llm/` concentra tarefas que exigem GPU, muitos dados ou ferramentas de avaliação: preparação do corpus, benchmark, QLoRA, adapters e comparação base versus adapter (EVAL-03). A API em `src/` não instala PyTorch, ROCm, PEFT ou datasets.

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
├── requirements.txt          # ambiente leve da API
└── docs/PLANO_EXECUCAO.md    # tarefas, critérios e evidências
```

O corpus RAG atual está em `llm/data/rag_pilot_v2/`, preservando o inventário v1, com geração/validação do manifesto em `llm/src/rag/` e avaliação reservada em `evaluation/`. Quando a API consultar fontes em tempo de resposta, o código de recuperação permanece em `src/medical_assistant/retrieval/`; assim ela não depende das bibliotecas de treinamento.

## Ambiente da API

```bash
cd fase3
python3 -m venv .venv-api
.venv-api/bin/python -m pip install -r requirements.txt
```

O Docker usa este ambiente. Ele monta somente `llm/data/curated` como leitura para o protótipo consumir o corpus curado.

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

O [planejamento do RAG](docs/PLANO_RAG.md) registra BM25 vencedor e **preparação documental RAG-09 concluída em 12/10/2026**. A cadeia local LCEL usa FT-11 para selecionar evidência, verificador comum na base sem adapter, contexto completo e fontes/offsets verificáveis. Validação final: 22/24 completas, duas parciais e todos os 16 casos especiais corretos em ambas as condições com RAG. No [teste final de 160 registros](llm/artifacts/experiments/20261012T184438077141Z_rag09_test/review.md), FT-11+RAG teve 19/24 completas, duas parciais e três abstenções indevidas; base+RAG 18/24, três parciais e as mesmas abstenções. Ambos 8/8 ausências, 4/4 conflitos e 4/4 instruções hostis; 24/24 spans e todo o texto útil entregues ao prompt, zero erros/omissões. Orçamento 3.072/reserva 256, pico VRAM 7,92 GiB, 42 testes de software aprovados. [Recarga pela CLI](llm/artifacts/experiments/20261012T190305897293Z_rag_query_v3/review.md) retorna ambas as assinaturas com fonte correta, sem ler perguntas/gabaritos. Comandos e recibo de aceite no [README RAG](llm/src/rag/README.md). Falhas e versões anteriores preservadas; conjuntos reservados de paráfrases compartilham corpus/conceitos, sem independência temática ou clínica. A qualidade permanece limitada; sem treino novo ou alteração dos pesos selecionados. Próxima tarefa APP-00 e integração APP-01 com API/prontuário; LangGraph/revisão humana no fluxo real continuam pendentes.
