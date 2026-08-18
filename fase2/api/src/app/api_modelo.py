import os
import json
import logging
import socket
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_ARQUIVO_ATUAL = os.path.dirname(os.path.abspath(__file__))

# api_modelo.py fica em fase2/api/src/app/, enquanto os artefatos do
# modelo ficam em fase2/api/modelo_api/ (irmão de src/).
PASTA_PROJETO = os.path.dirname(os.path.dirname(PASTA_ARQUIVO_ATUAL))

PASTA_MODELO_API = os.path.join(PASTA_PROJETO, "modelo_api")

NOME_ARQUIVO_MODELO = os.environ.get(
    "NOME_ARQUIVO_MODELO",
    "modelo_genetico_vencedor.joblib"
)

CAMINHO_MODELO = os.path.join(
    PASTA_MODELO_API,
    NOME_ARQUIVO_MODELO
)

CAMINHO_METADATA = os.path.join(
    PASTA_MODELO_API,
    "metadata_modelo_api.json"
)

VERSAO_API = "2.0.0"

# Identifica o container/instância que atendeu a requisição.
# Essencial para rastrear comportamento quando a API roda escalada
# horizontalmente atrás do load balancer (várias réplicas simultâneas).
INSTANCE_ID = os.environ.get("HOSTNAME", socket.gethostname())

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


# ============================================================
# LOGGING ESTRUTURADO (JSON)
# ============================================================
# Logs em JSON facilitam a coleta/parsing por ferramentas de
# observabilidade (Loki, ELK, CloudWatch, etc.) quando a API roda
# em múltiplos containers e os logs precisam ser correlacionados.

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "instance_id": INSTANCE_ID,
        }

        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields:
            payload.update(extra_fields)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def configurar_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger("api_modelo")
    logger.setLevel(LOG_LEVEL)
    logger.handlers = [handler]
    logger.propagate = False

    return logger


logger = configurar_logging()


def log_evento(mensagem: str, nivel: str = "info", **campos: Any) -> None:
    log_fn = getattr(logger, nivel, logger.info)
    log_fn(mensagem, extra={"extra_fields": campos})


# ============================================================
# MÉTRICAS PROMETHEUS
# ============================================================

HTTP_REQUESTS_TOTAL = Counter(
    "api_http_requests_total",
    "Total de requisições HTTP recebidas pela API",
    ["method", "path", "status_code"]
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "api_http_request_duration_seconds",
    "Duração das requisições HTTP em segundos",
    ["method", "path"]
)

PREDICOES_TOTAL = Counter(
    "api_predicoes_total",
    "Total de predições realizadas pelo modelo",
    ["classe_prevista"]
)

PREDICAO_PROBABILIDADE = Histogram(
    "api_predicao_probabilidade",
    "Distribuição das probabilidades de hipertensão previstas",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

PREDICOES_ERRO_TOTAL = Counter(
    "api_predicoes_erro_total",
    "Total de erros ao executar predição no modelo"
)

MODELO_INFO = Gauge(
    "api_modelo_info",
    "Informações estáticas do modelo carregado (valor sempre 1)",
    ["nome_modelo", "algoritmo", "versao_api", "instance_id"]
)


# ============================================================
# CARREGAMENTO DO MODELO E METADATA
# ============================================================

if not os.path.exists(CAMINHO_MODELO):
    raise FileNotFoundError(f"Modelo não encontrado em: {CAMINHO_MODELO}")

if not os.path.exists(CAMINHO_METADATA):
    raise FileNotFoundError(f"Metadata não encontrada em: {CAMINHO_METADATA}")

log_evento("Carregando modelo", caminho_modelo=CAMINHO_MODELO)

TEMPO_INICIO_CARREGAMENTO = time.time()
modelo = joblib.load(CAMINHO_MODELO)
DURACAO_CARREGAMENTO = round(time.time() - TEMPO_INICIO_CARREGAMENTO, 3)

with open(CAMINHO_METADATA, "r", encoding="utf-8") as arquivo:
    metadata = json.load(arquivo)

VARIAVEIS_ENTRADA = metadata["variaveis_entrada"]
VARIAVEIS_NUMERICAS = metadata.get("variaveis_numericas", [])
VARIAVEIS_CATEGORICAS = metadata.get("variaveis_categoricas", [])
THRESHOLD = metadata.get("threshold_classificacao", 0.5)
NOME_MODELO = metadata.get("nome_modelo", "modelo_desconhecido")
ALGORITMO_MODELO = metadata.get("algoritmo", "RandomForestClassifier")

APP_START_TIME = time.time()

MODELO_INFO.labels(
    nome_modelo=NOME_MODELO,
    algoritmo=ALGORITMO_MODELO,
    versao_api=VERSAO_API,
    instance_id=INSTANCE_ID
).set(1)

log_evento(
    "Modelo carregado com sucesso",
    nome_modelo=NOME_MODELO,
    algoritmo=ALGORITMO_MODELO,
    duracao_carregamento_segundos=DURACAO_CARREGAMENTO,
    total_variaveis_entrada=len(VARIAVEIS_ENTRADA)
)


# ============================================================
# CRIAÇÃO DA API
# ============================================================

app = FastAPI(
    title="API Modelo de Hipertensão",
    description="API para predição de hipertensão com modelo Random Forest otimizado por algoritmo genético",
    version=VERSAO_API
)


# Permite chamadas a partir do formulário HTML local ou hospedado
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MIDDLEWARE DE LOGGING E MÉTRICAS
# ============================================================

@app.middleware("http")
async def middleware_observabilidade(request: Request, call_next):
    request_id = str(uuid.uuid4())
    inicio = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception as erro:
        duracao = round(time.perf_counter() - inicio, 4)

        log_evento(
            "Falha não tratada ao processar requisição",
            nivel="error",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            duracao_segundos=duracao,
            erro=str(erro)
        )

        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            path=request.url.path,
            status_code="500"
        ).inc()

        raise

    duracao = round(time.perf_counter() - inicio, 4)

    HTTP_REQUESTS_TOTAL.labels(
        method=request.method,
        path=request.url.path,
        status_code=str(response.status_code)
    ).inc()

    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=request.method,
        path=request.url.path
    ).observe(duracao)

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Instance-ID"] = INSTANCE_ID

    log_evento(
        "Requisição processada",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duracao_segundos=duracao,
        client_ip=request.client.host if request.client else None
    )

    return response


# ============================================================
# ROTAS
# ============================================================

# Rota para abrir o formulário HTML
@app.get("/", response_class=HTMLResponse)
def abrir_formulario():
    # Caminho para o seu arquivo html
    caminho_html = os.path.join(os.path.dirname(__file__), "formulario.html")

    with open(caminho_html, "r", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content, status_code=200)

""" @app.get("/")
def home():
    return {
        "mensagem": "API do modelo de hipertensão ativa.",
        "modelo": metadata.get("algoritmo", "RandomForestClassifier"),
        "variaveis_esperadas": len(VARIAVEIS_ENTRADA),
        "endpoint_predicao": "/prever",
        "documentacao": "/docs"
    } """


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "modelo_carregado": True,
        "metadata_carregada": True,
        "nome_modelo": NOME_MODELO,
        "versao_api": VERSAO_API,
        "instance_id": INSTANCE_ID,
        "uptime_segundos": round(time.time() - APP_START_TIME, 1)
    }


@app.get("/metrics")
def metricas_prometheus():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/metadata")
def obter_metadata():
    return metadata


@app.get("/variaveis")
def listar_variaveis():
    return {
        "total": len(VARIAVEIS_ENTRADA),
        "variaveis_entrada": VARIAVEIS_ENTRADA,
        "variaveis_numericas": VARIAVEIS_NUMERICAS,
        "variaveis_categoricas": VARIAVEIS_CATEGORICAS
    }


@app.post("/prever")
def prever_hipertensao(dados: Dict[str, Any]):
    """
    Recebe um JSON com as variáveis esperadas pelo modelo
    e retorna a classificação prevista.
    """

    # --------------------------------------------------------
    # Validação de variáveis obrigatórias
    # --------------------------------------------------------

    variaveis_ausentes = [
        variavel for variavel in VARIAVEIS_ENTRADA
        if variavel not in dados
    ]

    if variaveis_ausentes:
        raise HTTPException(
            status_code=400,
            detail={
                "erro": "Variáveis obrigatórias ausentes.",
                "variaveis_ausentes": variaveis_ausentes
            }
        )

    # --------------------------------------------------------
    # Montagem do DataFrame na ordem correta
    # --------------------------------------------------------

    entrada_modelo = {
        variavel: dados.get(variavel)
        for variavel in VARIAVEIS_ENTRADA
    }

    df_entrada = pd.DataFrame([entrada_modelo])

    # --------------------------------------------------------
    # Conversão de numéricos quando possível
    # --------------------------------------------------------

    for coluna in VARIAVEIS_NUMERICAS:
        if coluna in df_entrada.columns:
            df_entrada[coluna] = pd.to_numeric(
                df_entrada[coluna],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Predição
    # --------------------------------------------------------

    try:
        probabilidade = float(modelo.predict_proba(df_entrada)[0][1])
        classe_prevista = int(probabilidade >= THRESHOLD)

    except Exception as erro:
        PREDICOES_ERRO_TOTAL.inc()

        log_evento(
            "Erro ao executar predição no modelo",
            nivel="error",
            erro=str(erro)
        )

        raise HTTPException(
            status_code=500,
            detail={
                "erro": "Erro ao executar predição no modelo.",
                "detalhe": str(erro)
            }
        )

    PREDICOES_TOTAL.labels(classe_prevista=str(classe_prevista)).inc()
    PREDICAO_PROBABILIDADE.observe(probabilidade)

    if classe_prevista == 1:
        descricao = "Com indicativo de hipertensão"
    else:
        descricao = "Sem indicativo de hipertensão"

    return {
        "classe_prevista": classe_prevista,
        "descricao": descricao,
        "probabilidade_hipertensao": round(probabilidade, 4),
        "probabilidade_percentual": round(probabilidade * 100, 2),
        "threshold_utilizado": THRESHOLD,
        "modelo": metadata.get("algoritmo", "RandomForestClassifier"),
        "nome_modelo": NOME_MODELO,
        "instance_id": INSTANCE_ID,
        "variaveis_recebidas": entrada_modelo
    }
