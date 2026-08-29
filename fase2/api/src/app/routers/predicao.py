import time
from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from ..config import INSTANCE_ID
from ..logging_config import log_evento
from ..metrics import PREDICAO_PROBABILIDADE, PREDICOES_ERRO_TOTAL, PREDICOES_TOTAL

try:
    from ..llm_interpreter import (
        gerar_interpretacao_llm,
        VARIAVEIS_NOMES_CLINICOS,
        LLM_HABILITADO,
    )
except ImportError:
    LLM_HABILITADO = False
    gerar_interpretacao_llm = None
    VARIAVEIS_NOMES_CLINICOS = {}

router = APIRouter()


def _montar_dataframe_entrada(dados: Dict[str, Any], modelo_info):
    variaveis_ausentes = [
        variavel for variavel in modelo_info.variaveis_entrada
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

    entrada_modelo = {
        variavel: dados.get(variavel)
        for variavel in modelo_info.variaveis_entrada
    }
    df_entrada = pd.DataFrame([entrada_modelo])

    for coluna in modelo_info.variaveis_numericas:
        if coluna in df_entrada.columns:
            df_entrada[coluna] = pd.to_numeric(df_entrada[coluna], errors="coerce")

    return df_entrada, entrada_modelo


def _gerar_interpretacao(probabilidade: float, classe_prevista: int, entrada_modelo: dict):
    if not (LLM_HABILITADO and gerar_interpretacao_llm):
        return None, 0

    inicio_llm = time.perf_counter()
    interpretacao = gerar_interpretacao_llm(
        probabilidade=probabilidade,
        classe_prevista=classe_prevista,
        variaveis_entrada=entrada_modelo,
        variaveis_nomes_clinicos=VARIAVEIS_NOMES_CLINICOS
    )
    duracao_llm = round(time.perf_counter() - inicio_llm, 3)

    log_evento(
        "Interpretação LLM gerada",
        duracao_segundos=duracao_llm,
        tem_interpretacao=bool(interpretacao)
    )
    return interpretacao, duracao_llm


@router.post("/prever")
def prever_hipertensao(dados: Dict[str, Any], request: Request):
    """
    Recebe um JSON com as variáveis esperadas pelo modelo
    e retorna a classificação prevista com interpretação LLM.
    """
    modelo_info = request.app.state.modelo_carregado
    df_entrada, entrada_modelo = _montar_dataframe_entrada(dados, modelo_info)

    try:
        probabilidade = float(modelo_info.modelo.predict_proba(df_entrada)[0][1])
        classe_prevista = int(probabilidade >= modelo_info.threshold)
    except Exception as erro:
        PREDICOES_ERRO_TOTAL.inc()
        log_evento("Erro ao executar predição no modelo", nivel="error", erro=str(erro))
        raise HTTPException(
            status_code=500,
            detail={"erro": "Erro ao executar predição no modelo.", "detalhe": str(erro)}
        )

    PREDICOES_TOTAL.labels(classe_prevista=str(classe_prevista)).inc()
    PREDICAO_PROBABILIDADE.observe(probabilidade)

    descricao = "Com indicativo de hipertensão" if classe_prevista == 1 else "Sem indicativo de hipertensão"

    interpretacao_llm, duracao_llm = _gerar_interpretacao(probabilidade, classe_prevista, entrada_modelo)

    return {
        "classe_prevista": classe_prevista,
        "descricao": descricao,
        "probabilidade_hipertensao": round(probabilidade, 4),
        "probabilidade_percentual": round(probabilidade * 100, 2),
        "threshold_utilizado": modelo_info.threshold,
        "modelo": modelo_info.algoritmo,
        "nome_modelo": modelo_info.nome_modelo,
        "instance_id": INSTANCE_ID,
        "variaveis_recebidas": entrada_modelo,
        "interpretacao_llm": interpretacao_llm,
        "duracao_llm_segundos": duracao_llm,
        "llm_habilitado": LLM_HABILITADO
    }