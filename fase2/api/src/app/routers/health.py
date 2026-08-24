import time

from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..config import INSTANCE_ID, VERSAO_API

router = APIRouter()

APP_START_TIME = time.time()


@router.get("/health")
def health_check(request: Request):
    modelo_info = request.app.state.modelo_carregado
    return {
        "status": "ok",
        "modelo_carregado": True,
        "metadata_carregada": True,
        "nome_modelo": modelo_info.nome_modelo,
        "versao_api": VERSAO_API,
        "instance_id": INSTANCE_ID,
        "uptime_segundos": round(time.time() - APP_START_TIME, 1)
    }


@router.get("/metrics")
def metricas_prometheus():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )