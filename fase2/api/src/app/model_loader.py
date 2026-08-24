import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import joblib

from .config import CAMINHO_METADATA, CAMINHO_MODELO, INSTANCE_ID, VERSAO_API
from .logging_config import log_evento
from .metrics import MODELO_INFO


@dataclass
class ModeloCarregado:
    modelo: Any
    metadata: Dict[str, Any]
    variaveis_entrada: List[str]
    variaveis_numericas: List[str]
    variaveis_categoricas: List[str]
    threshold: float
    nome_modelo: str
    algoritmo: str
    duracao_carregamento_segundos: float


def carregar_modelo() -> ModeloCarregado:
    """Carrega o modelo treinado e sua metadata do disco.

    Levanta FileNotFoundError se os artefatos não existirem — assim o
    processo falha rápido no startup em vez de quebrar silenciosamente
    na primeira requisição de /prever.
    """
    if not os.path.exists(CAMINHO_MODELO):
        raise FileNotFoundError(f"Modelo não encontrado em: {CAMINHO_MODELO}")

    if not os.path.exists(CAMINHO_METADATA):
        raise FileNotFoundError(f"Metadata não encontrada em: {CAMINHO_METADATA}")

    log_evento("Carregando modelo", caminho_modelo=CAMINHO_MODELO)

    inicio = time.time()
    modelo = joblib.load(CAMINHO_MODELO)
    duracao = round(time.time() - inicio, 3)

    with open(CAMINHO_METADATA, "r", encoding="utf-8") as arquivo:
        metadata = json.load(arquivo)

    nome_modelo = metadata.get("nome_modelo", "modelo_desconhecido")
    algoritmo = metadata.get("algoritmo", "RandomForestClassifier")

    MODELO_INFO.labels(
        nome_modelo=nome_modelo,
        algoritmo=algoritmo,
        versao_api=VERSAO_API,
        instance_id=INSTANCE_ID
    ).set(1)

    log_evento(
        "Modelo carregado com sucesso",
        nome_modelo=nome_modelo,
        algoritmo=algoritmo,
        duracao_carregamento_segundos=duracao,
        total_variaveis_entrada=len(metadata["variaveis_entrada"])
    )

    return ModeloCarregado(
        modelo=modelo,
        metadata=metadata,
        variaveis_entrada=metadata["variaveis_entrada"],
        variaveis_numericas=metadata.get("variaveis_numericas", []),
        variaveis_categoricas=metadata.get("variaveis_categoricas", []),
        threshold=metadata.get("threshold_classificacao", 0.5),
        nome_modelo=nome_modelo,
        algoritmo=algoritmo,
        duracao_carregamento_segundos=duracao,
    )