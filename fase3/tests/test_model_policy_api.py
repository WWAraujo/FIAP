import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from medical_assistant.application import api

DIGEST = "a" * 64
QWEN = "qwen_q3_k_xl:latest"
GEMMA = "gemma3:latest"


def reply(payload):
    response = Mock()
    response.json.return_value = payload
    return response


@pytest.fixture
def endpoint(monkeypatch, tmp_path):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    def init_audit(self):
        self.log_dir = tmp_path
        self.audit_file = tmp_path / "audit.log"

    monkeypatch.setattr(api.AuditLogger, "__init__", init_audit)
    tags = {"models": [{"name": name, "digest": DIGEST} for name in (QWEN, GEMMA)]}
    get = Mock(return_value=reply(tags))
    post = Mock(side_effect=lambda *a, **kw: reply(
        {"model": kw["json"]["model"], "response": "Revisão profissional necessária.", "done": True}))
    monkeypatch.setattr(api.requests, "get", get)
    monkeypatch.setattr(api.requests, "post", post)
    app = api.create_app()

    def send(payload):
        async def run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://test") as client:
                return await client.post("/query", json=payload)
        return asyncio.run(run())

    def logs():
        file = tmp_path / "audit.log"
        return [json.loads(line) for line in file.read_text().splitlines()] if file.exists() else []

    return send, get, post, logs


@pytest.mark.parametrize("mode,expected", [(None, QWEN), ("standard", QWEN), ("fast", GEMMA)])
def test_selection_and_actual_identity(endpoint, mode, expected):
    send, get, post, logs = endpoint
    payload = {"query": "Quais são as limitações do assistente?"}
    if mode is not None:
        payload["mode"] = mode
    response = send(payload)
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == (mode or "standard")
    assert data["model_name"] == expected and data["model_digest"] == DIGEST
    assert data["response"] == "Revisão profissional necessária."
    assert data["requires_validation"] and data["sources"] == []
    assert post.call_args.kwargs["json"]["model"] == expected
    assert post.call_args.kwargs["json"]["stream"] is False
    assert get.call_count == 2
    audit = logs()[-1]
    for field in ("mode", "model_name", "model_digest", "query_id"):
        assert audit[field] == data[field]


@pytest.mark.parametrize("payload", [
    {"query": "Consulta", "mode": "slow"},
    {"query": "Consulta", "mode": None},
    {"query": "Consulta", "model": "outro:latest"},
    {"query": "Consulta", "model_name": "outro:latest"},
    {"query": "Consulta", "patient_context": {"id": 1}},
    {"query": "  "},
])
def test_invalid_contract_never_calls_model(endpoint, payload):
    send, get, post, logs = endpoint
    assert send(payload).status_code == 422
    get.assert_not_called()
    post.assert_not_called()
    assert logs() == []


@pytest.mark.parametrize("mode", ["standard", "fast"])
def test_absent_model_has_explicit_error_without_fallback(endpoint, mode):
    send, get, post, logs = endpoint
    get.return_value = reply({"models": []})
    response = send({"query": "Consulta documental", "mode": mode})
    assert response.status_code == 503
    error = response.json()["detail"]
    assert error["code"] == "model_unavailable"
    assert error["requested_model"] == (GEMMA if mode == "fast" else QWEN)
    audit = logs()[-1]
    assert audit["requested_model"] == error["requested_model"]
    assert audit["query_id"] == error["query_id"]
    assert audit["mode"] == mode and audit["error"]
    assert audit["model_name"] is None and audit["model_digest"] is None
    post.assert_not_called()


def test_server_timeout_is_explicit(endpoint):
    send, get, post, logs = endpoint
    get.side_effect = requests.Timeout("servidor indisponível")
    assert send({"query": "Consulta documental"}).status_code == 503
    assert logs()[-1]["error"]
    post.assert_not_called()


@pytest.mark.parametrize("result", [
    {"model": GEMMA, "response": "Texto", "done": True},
    {"model": QWEN, "response": "", "done": True},
    {"model": QWEN, "response": "Texto", "done": False},
])
def test_invalid_generation_is_not_reported_as_success(endpoint, result):
    send, get, post, logs = endpoint
    post.side_effect = None
    post.return_value = reply(result)
    assert send({"query": "Consulta documental"}).status_code == 503
    assert logs()[-1]["error"]


def test_model_disappears_at_generation(endpoint):
    send, get, post, logs = endpoint
    failed = reply({"error": "model not found"})
    failed.raise_for_status.side_effect = requests.HTTPError("404")
    post.side_effect = None
    post.return_value = failed
    assert send({"query": "Consulta documental"}).status_code == 503
    assert post.call_count == 1 and logs()[-1]["error"]


@pytest.mark.parametrize("digest", [None, "unknown"])
def test_unknown_digest_blocks_generation(endpoint, digest):
    send, get, post, logs = endpoint
    get.return_value = reply({"models": [{"name": QWEN, "digest": digest}]})
    assert send({"query": "Consulta documental"}).status_code == 503
    post.assert_not_called()


def test_changing_model_digest_rejects_response(endpoint):
    send, get, post, logs = endpoint
    get.side_effect = [reply({"models": [{"name": QWEN, "digest": DIGEST}]}),
                       reply({"models": [{"name": QWEN, "digest": "b" * 64}]})]
    assert send({"query": "Consulta documental"}).status_code == 503
    assert "alterado" in logs()[-1]["error"]


def test_input_block_has_no_model_identity(endpoint):
    send, get, post, logs = endpoint
    response = send({"query": "Quero prescrever medicamento", "mode": "fast"})
    assert response.status_code == 200 and response.json()["security_blocked"]
    assert response.json()["model_name"] is None and response.json()["model_digest"] is None
    assert logs()[-1]["mode"] == "fast" and logs()[-1]["model_name"] is None
    get.assert_not_called()
    post.assert_not_called()


def test_response_block_preserves_actual_model_identity(endpoint):
    send, get, post, logs = endpoint
    post.side_effect = None
    post.return_value = reply({"model": QWEN, "response": "Vou prescrever medicamento", "done": True})
    response = send({"query": "Consulta documental"})
    assert response.status_code == 200
    assert response.json()["security_blocked"]
    assert response.json()["model_name"] == QWEN and response.json()["model_digest"] == DIGEST
    assert "Vou prescrever" not in response.json()["response"]
    assert logs()[-1]["model_digest"] == DIGEST
