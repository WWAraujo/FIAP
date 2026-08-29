# API de Predição de Hipertensão

API FastAPI que prevê risco de hipertensão usando modelo Random Forest otimizado com interpretações em linguagem natural via LLM.

**Modelo:** Random Forest (50 árvores, otimizado por algoritmo genético)
**LLM:** Google Gemini ou Ollama local (escolha por variável de ambiente ou por requisição)
**Framework:** FastAPI + Uvicorn

## Requisitos

- Python 3.9+
- pip
- (Opcional) Docker + Docker Compose

## Instalação

```bash
# Instalar dependências
pip install -r requirements.txt
```

## Configuração

### Variáveis de Ambiente

Crie arquivo `.env`:

```env
LOG_LEVEL=INFO

# Provedor usado quando a requisição não escolhe um explicitamente
LLM_PROVIDER=google

# Google Gemini
GOOGLE_API_KEY=sua_chave_aqui
GOOGLE_MODEL=gemini-3.7-flash

# Ollama local (não precisa de chave)
OLLAMA_URL=http://host.docker.internal:11434
OLLAMA_MODEL=qwen3.5

TIMEOUT_LLM=300
NOME_ARQUIVO_MODELO=modelo_genetico_vencedor.joblib
```

> `host.docker.internal` só funciona porque a API roda em Docker — é o
> endereço que o container usa para alcançar o Ollama rodando no host.
> Se rodar a API fora do Docker, troque por `http://localhost:11434`.

### Google Gemini (LLM)

1. Acesse: https://aistudio.google.com/app/apikey
2. Clique "Create API Key"
3. Copie a chave no `.env`

### Ollama (LLM local, sem custo)

1. Instale o [Ollama](https://ollama.com) e baixe um modelo, ex.:
   `ollama pull qwen3.5`
2. Confirme que está rodando: `ollama list`
3. Defina `LLM_PROVIDER=ollama` no `.env` (ou escolha por requisição,
   veja `POST /prever` abaixo)

## Como Rodar

### Desenvolvimento (local)

```bash
cd api
docker-compose up
```

## Como rodar api localmente com apenas 1 instancia e refazendo o build após uma alteração

```bash
cd api
docker compose up -d --build --force-recreate --scale api=1  
```

API roda na porta: http://localhost:8000


## Parar e remover os containers
```bash
cd api
docker compose down
```

## Endpoints Principais

### `POST /prever`
Faz predição de hipertensão

**Query params (opcionais):** `provedor` (`google` ou `ollama`) e `modelo`
— escolhem o LLM usado nessa chamada, sobrescrevendo o padrão do `.env`.
Ex.: `POST /prever?provedor=ollama&modelo=qwen3.5`

**Response:**
```json
{
  "classe_prevista": 1,
  "descricao": "Com indicativo de hipertensão",
  "probabilidade_hipertensao": 0.75,
  "probabilidade_percentual": 75.0,
  "interpretacao_llm": "Com base nos dados...",
  "duracao_llm_segundos": 2.3,
  "llm_habilitado": true,
  "instance_id": "..."
}
```
`llm_habilitado` reflete se **essa chamada** gerou interpretação
(`interpretacao_llm` não veio `null`) — não se o Google especificamente
está configurado, já que o provedor pode ser o Ollama.

### `GET /`
Interface web (formulário)

### `GET /health`
Status da API

### `GET /metadata`
Metadados do modelo

### `GET /variaveis`
Lista variáveis esperadas

### `GET /metrics`
Métricas Prometheus

## Testes

```bash
pip install python-dotenv
python scripts/testar_llm_integration.py
```

O script importa `gerar_interpretacao_llm` diretamente de
`src/app/llm_interpreter.py` (não precisa da API rodando) e testa 3
cenários (alto risco, baixo risco, intermediário) contra o provedor
configurado em `LLM_PROVIDER` no `.env`.

## Estrutura

```
api/
├── src/app/
│   ├── api_modelo.py               # Cria a FastAPI e registra os routers
│   ├── config.py                   # Configurações e caminhos
│   ├── logging_config.py           # Logging estruturado (JSON)
│   ├── metrics.py                  # Métricas Prometheus
│   ├── model_loader.py             # Carregamento do modelo
│   ├── middleware.py                # Observabilidade das requisições
│   ├── llm_interpreter.py          # Google Gemini + Ollama
│   ├── formulario.html             # Interface web
│   ├── routers/
│   │   ├── health.py               # GET /health, GET /metrics
│   │   ├── info.py                 # GET /, GET /metadata, GET /variaveis
│   │   └── predicao.py             # POST /prever
│   └── __init__.py
├── modelo_api/
│   ├── modelo_genetico_vencedor.joblib
│   ├── metadata_modelo_api.json
│   └── exemplo_entrada_api.json
├── scripts/
│   ├── testar_llm_integration.py
│   └── teste_escalabilidade.sh
├── nginx/                          # Load balancer (config do docker-compose)
├── autoscaler/                     # Autoscaling baseado em CPU
├── monitoring/                     # Prometheus + Grafana (dashboards)
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── .env.example
└── .env (não versionar)
```

## Fluxo de Predição

```
1. Formulário HTML
   ↓
2. POST /prever
   ↓
3. Validação de variáveis
   ↓
4. Random Forest prediz (~50ms)
   ↓
5. LLM interpreta — Google Gemini (~2-5s) ou Ollama local (varia com a máquina)
   ↓
6. JSON com resultado
   ↓
7. Card HTML exibe resultado
```

## Logging

Logs em JSON estruturado, pronto para observabilidade (ELK, Loki, etc.)

Níveis: DEBUG, INFO, WARNING, ERROR

## Troubleshooting

| Problema | Solução |
|----------|---------|
| Chave API do Google inválida | Regenere em https://aistudio.google.com/app/apikey |
| `LLM_PROVIDER` não chega ao container | Confira se está listado em `environment:` no `docker-compose.yml` (não basta estar só no `.env`) e recrie com `--force-recreate` |
| Ollama não responde | Confirme que está rodando (`ollama list`) e que `OLLAMA_URL` usa `host.docker.internal` (não `localhost`) quando a API roda em Docker |
| Variáveis ausentes | Use `GET /variaveis` para ver lista completa |
| LLM timeout | Aumente `TIMEOUT_LLM` no `.env` |
| Modelo não encontrado | Verifique `api/modelo_api/modelo_genetico_vencedor.joblib` |

## Documentação Interativa

```
http://localhost:8000/docs          # Swagger UI
http://localhost:8000/redoc         # ReDoc
```