import os

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def abrir_formulario():
    caminho_html = os.path.join(os.path.dirname(os.path.dirname(__file__)), "formulario.html")
    with open(caminho_html, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


@router.get("/metadata")
def obter_metadata(request: Request):
    return request.app.state.modelo_carregado.metadata


@router.get("/variaveis")
def listar_variaveis(request: Request):
    modelo_info = request.app.state.modelo_carregado
    return {
        "total": len(modelo_info.variaveis_entrada),
        "variaveis_entrada": modelo_info.variaveis_entrada,
        "variaveis_numericas": modelo_info.variaveis_numericas,
        "variaveis_categoricas": modelo_info.variaveis_categoricas
    }