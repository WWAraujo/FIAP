"""API de inferência com a política de modelos da APP-00."""

import os
import re
from pathlib import Path
from typing import Literal

import requests
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from medical_assistant.safety.audit import AuditLogger
from medical_assistant.safety.guardrails import SecuritySystem
from medical_assistant.application.assistant import MedicalAssistantLangChain
from medical_assistant.application.contracts import AssistantResponse, PatientContext
from medical_assistant.clinical.patient_database import PatientDatabase
from medical_assistant.retrieval.ft11_client import InferenceUnavailable
from medical_assistant.application.workflow_service import start_workflow
from medical_assistant.safety.reviews import ReviewStore

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


class AssistantQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=8000)
    patient_id: str | None = Field(default=None, min_length=1)


class ModelSettings(BaseModel):
    base_model: str = Field(min_length=1)
    alternative_model: str = Field(min_length=1)
    default_mode: Literal["standard"] = "standard"
    fast_mode: Literal["fast"] = "fast"
    ollama_url: str = Field(min_length=1)

    def select(self, mode: Mode) -> str:
        return self.alternative_model if mode == "fast" else self.base_model


class WorkflowRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    patient_id: str = Field(min_length=1)
    query: str = Field(min_length=1,max_length=8000)


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    patient_id: str | None
    reviewer: str = Field(min_length=1,max_length=120)
    decision: Literal['approved','rejected']
    notes: str = Field(default='',max_length=2000)
    confirmed_human: Literal[True]


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

    @application.post('/workflows')
    def workflow(request: WorkflowRequest):
        try:
            return start_workflow(request.patient_id,request.query)
        except ValueError as exc:
            raise HTTPException(status_code=400,detail=str(exc)) from exc
        except InferenceUnavailable as exc:
            raise HTTPException(status_code=503,detail=str(exc)) from exc

    @application.get('/reviews/{review_id}')
    def review(review_id: str):
        try:
            return ReviewStore().get(review_id)
        except ValueError as exc:
            raise HTTPException(status_code=404,detail=str(exc)) from exc

    @application.post('/reviews/{review_id}/decision')
    def decide(review_id: str, request: ReviewDecision):
        try:
            return ReviewStore().decide(review_id,**request.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=409,detail=str(exc)) from exc

    @application.post("/assistant/query", response_model=AssistantResponse)
    def assistant_query(request: AssistantQueryRequest) -> AssistantResponse:
        context = None
        if request.patient_id is not None:
            db = PatientDatabase()
            try:
                context = PatientContext.from_database(db, request.patient_id)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            finally:
                db.close()
        try:
            result = MedicalAssistantLangChain().process_query(request.query, context)
        except InferenceUnavailable as exc:
            raise HTTPException(status_code=503, detail={"code": "model_unavailable", "message": str(exc)}) from exc
        return AssistantResponse.model_validate(result)

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
