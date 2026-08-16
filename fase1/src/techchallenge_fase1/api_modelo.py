import os
import json
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_PROJETO = os.path.dirname(os.path.abspath(__file__))

PASTA_MODELO_API = os.path.join(PASTA_PROJETO, "modelo_api")

CAMINHO_MODELO = os.path.join(
    PASTA_MODELO_API,
    "modelo_hipertensao_api.joblib"
)

CAMINHO_METADATA = os.path.join(
    PASTA_MODELO_API,
    "metadata_modelo_api.json"
)


# ============================================================
# CARREGAMENTO DO MODELO E METADATA
# ============================================================

if not os.path.exists(CAMINHO_MODELO):
    raise FileNotFoundError(f"Modelo não encontrado em: {CAMINHO_MODELO}")

if not os.path.exists(CAMINHO_METADATA):
    raise FileNotFoundError(f"Metadata não encontrada em: {CAMINHO_METADATA}")

modelo = joblib.load(CAMINHO_MODELO)

with open(CAMINHO_METADATA, "r", encoding="utf-8") as arquivo:
    metadata = json.load(arquivo)

VARIAVEIS_ENTRADA = metadata["variaveis_entrada"]
VARIAVEIS_NUMERICAS = metadata.get("variaveis_numericas", [])
VARIAVEIS_CATEGORICAS = metadata.get("variaveis_categoricas", [])
THRESHOLD = metadata.get("threshold_classificacao", 0.5)


# ============================================================
# CRIAÇÃO DA API
# ============================================================

app = FastAPI(
    title="API Modelo de Hipertensão",
    description="API para predição de hipertensão com modelo Random Forest",
    version="1.0.0"
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
        "metadata_carregada": True
    }


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
        raise HTTPException(
            status_code=500,
            detail={
                "erro": "Erro ao executar predição no modelo.",
                "detalhe": str(erro)
            }
        )

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
        "variaveis_recebidas": entrada_modelo
    }