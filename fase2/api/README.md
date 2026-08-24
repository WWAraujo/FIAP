# API de Predição de Hipertensão

API FastAPI que prevê risco de hipertensão usando modelo Random Forest otimizado com interpretações em linguagem natural via Google Gemini.

**Modelo:** Random Forest (50 árvores, otimizado por algoritmo genético)  
**LLM:** Google Gemini (gratuito)  
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
GOOGLE_API_KEY=sua_chave_aqui
GOOGLE_MODEL=gemini-3.7-flash
TIMEOUT_LLM=20
NOME_ARQUIVO_MODELO=modelo_genetico_vencedor.joblib
```

### Google Gemini (LLM)

1. Acesse: https://aistudio.google.com/app/apikey
2. Clique "Create API Key"
3. Copie a chave no `.env`

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

## Endpoints Principais

### `POST /prever`
Faz predição de hipertensão

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
# Testar integração com Google Gemini
pip install python-dotenv
python api/scripts/testar_llm_integration.py
```

Testa:
- ✅ Conexão com Gemini
- ✅ Geração de 3 interpretações
- ✅ Tempo de processamento

## Estrutura

```
api/
├── src/app/
│   ├── api_modelo.py              # Rotas + Modelo + LLM
│   ├── llm_interpreter.py         # Google Gemini
│   ├── logging_config.py          # Logging
│   ├── formulario.html            # Interface web
│   └── __init__.py
├── modelo_api/
│   ├── modelo_genetico_vencedor.joblib
│   └── metadata_modelo_api.json
├── scripts/
│   └── testar_llm_integration.py
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
5. Google Gemini interpreta (~2-5s)
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
| Chave API inválida | Regenere em https://aistudio.google.com/app/apikey |
| Variáveis ausentes | Use `GET /variaveis` para ver lista completa |
| LLM timeout | Aumente `TIMEOUT_LLM` no `.env` |
| Modelo não encontrado | Verifique `api/modelo_api/modelo_genetico_vencedor.joblib` |

## Documentação Interativa

```
http://localhost:8000/docs          # Swagger UI
http://localhost:8000/redoc         # ReDoc
```
