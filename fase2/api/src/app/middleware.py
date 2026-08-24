import time
import uuid

from fastapi import Request

from .config import INSTANCE_ID
from .logging_config import log_evento
from .metrics import HTTP_REQUEST_DURATION_SECONDS, HTTP_REQUESTS_TOTAL

ROTAS_SEM_LOG = {"/health", "/metrics"}

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

    if request.url.path not in ROTAS_SEM_LOG:
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