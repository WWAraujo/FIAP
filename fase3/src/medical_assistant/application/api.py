"""API de inferência com a política de modelos da APP-00."""

import os
import re
from pathlib import Path
from typing import Literal

import requests
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from medical_assistant.application.assistant import AuditLogger, SecuritySystem

Mode = Literal["standard", "fast"]
CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs/config.yaml"
SYSTEM_PROMPT = (
    "Você é um assistente médico acadêmico. Responda em português. "
    "Não prescreva nem dê diagnóstico definitivo. Não invente fontes ou dados de pacientes. "
    "Informe limitações e a necessidade de validação por um profissional."
)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1)
    mode: Mode = "standard"


class QueryResponse(BaseModel):
    response: str
    query_id: str
    mode: Mode
    model_name: str | None
    model_digest: str | None
    security_blocked: bool
    requires_validation: bool = True
    sources: list[str] = Field(default_factory=list)


class ModelSettings(BaseModel):
    base_model: str = Field(min_length=1)
    alternative_model: str = Field(min_length=1)
    default_mode: Literal["standard"] = "standard"
    fast_mode: Literal["fast"] = "fast"
    ollama_url: str = Field(min_length=1)

    def select(self, mode: Mode) -> str:
        return self.alternative_model if mode == "fast" else self.base_model


class ModelUnavailable(RuntimeError):
    pass


class OllamaInference:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def _digest(self, model_name: str) -> str:
        response = requests.get(f"{self.url}/api/tags", timeout=10)
        response.raise_for_status()
        for model in response.json()["models"]:
            if model["name"] == model_name:
                digest = model["digest"]
                if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
                    return digest
                raise ModelUnavailable(f"Digest ausente ou inválido para {model_name}.")
        raise ModelUnavailable(f"Modelo configurado indisponível: {model_name}.")

    def generate(self, model_name: str, query: str) -> tuple[str, str]:
        try:
            digest = self._digest(model_name)
            response = requests.post(
                f"{self.url}/api/generate",
                json={"model": model_name, "prompt": query, "system": SYSTEM_PROMPT, "stream": False},
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            if (result.get("model") != model_name or result.get("done") is not True
                    or not isinstance(result.get("response"), str) or not result["response"].strip()):
                raise ModelUnavailable(f"Resposta incompleta ou identidade divergente: {model_name}.")
            if self._digest(model_name) != digest:
                raise ModelUnavailable(f"Modelo alterado durante a geração: {model_name}.")
            return result["response"], digest
        except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ModelUnavailable(f"Falha de inferência ou identidade do modelo {model_name}.") from exc


def create_app(config_path: Path = CONFIG_PATH) -> FastAPI:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = ModelSettings.model_validate(config["model"])
    backend = OllamaInference(os.environ.get("OLLAMA_BASE_URL", settings.ollama_url))
    audit = AuditLogger()
    security = SecuritySystem()
    application = FastAPI(title="FIAP — inferência acadêmica", version="0.1.0")

    @application.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        query_id = audit.log_query(request.query)
        usage = {"mode": request.mode, "model_name": None, "model_digest": None}
        guard = security.check_query(request.query)
        if not guard["allowed"]:
            audit.log_response(query_id, guard["message"], [], True, model_usage=usage)
            return QueryResponse(response=guard["message"], query_id=query_id,
                                 security_blocked=True, **usage)
        selected = settings.select(request.mode)
        try:
            text, digest = backend.generate(selected, request.query)
        except ModelUnavailable as exc:
            audit.log_response(query_id, "", [], True,
                               model_usage={**usage, "requested_model": selected}, error=str(exc))
            raise HTTPException(status_code=503, detail={"code": "model_unavailable",
                                "query_id": query_id, "mode": request.mode,
                                "requested_model": selected, "message": str(exc)}) from exc
        usage.update(model_name=selected, model_digest=digest)
        guard = security.check_response(text)
        if not guard["allowed"]:
            text = guard["message"]
        audit.log_response(query_id, text, [], True, model_usage=usage)
        return QueryResponse(response=text, query_id=query_id,
                             security_blocked=not guard["allowed"], **usage)

    return application


app = create_app()
