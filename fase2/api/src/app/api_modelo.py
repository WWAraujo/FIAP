from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import VERSAO_API
from .model_loader import carregar_modelo
from .middleware import middleware_observabilidade
from .routers import health, info, predicao


def criar_app() -> FastAPI:
    app = FastAPI(
        title="API Modelo de Hipertensão",
        description="API para predição de hipertensão com modelo Random Forest otimizado por algoritmo genético",
        version=VERSAO_API
    )

    # Carrega o modelo uma vez e disponibiliza para todos os routers via
    # app.state, em vez de globals no nível do módulo (facilita testes
    # com um modelo mockado, sem precisar de um .joblib real no disco).
    app.state.modelo_carregado = carregar_modelo()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.middleware("http")(middleware_observabilidade)

    app.include_router(info.router)
    app.include_router(health.router)
    app.include_router(predicao.router)

    return app


app = criar_app()