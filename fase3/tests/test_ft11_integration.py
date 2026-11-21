import asyncio
import json
from unittest.mock import Mock

import httpx
import pytest
import requests

from medical_assistant.application import api
from medical_assistant.application.assistant import MedicalAssistantLangChain
from medical_assistant.application.contracts import PatientContext, RuntimeResponse
from medical_assistant.clinical.patient_database import PatientDatabase
from medical_assistant.retrieval.ft11_client import FT11Client, InferenceUnavailable
from medical_assistant.safety.audit import AuditLogger


@pytest.fixture
def runtime_result():
    source = {"id": "DOC@v1#s", "document_id": "DOC", "title": "Documento de teste", "version": "v1",
              "page": 1, "section_id": "DOC@v1#s", "section": "Identificação", "text": "Texto verificável",
              "quoted_text": "Texto verificável", "char_start": 0, "char_end": 17, "evidence_id": "R1.P1"}
    return RuntimeResponse(answer={"status": "informativo", "dados": {"trecho": source["quoted_text"]},
        "pendencias": [], "fonte": source["id"], "validacao_humana": True}, sources=[source],
        model_usage={"model_id": "FT-11-test", "adapter_weights_sha256": "a" * 64},
        patient_context_used=True, run_id="runtime-test")


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    def init(self):
        self.audit_file = tmp_path / "audit.log"
    monkeypatch.setattr(AuditLogger, "__init__", init)
    original = PatientDatabase.__init__
    monkeypatch.setattr(PatientDatabase, "__init__", lambda self: original(self, tmp_path / "patients.db"))
    return tmp_path


def test_current_patient_and_sources_reach_model_api_and_audit(isolated, monkeypatch, runtime_result):
    invoke = Mock(return_value=runtime_result)
    monkeypatch.setattr(FT11Client, "invoke", invoke)
    app = api.create_app()

    def send(payload):
        async def run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                return await client.post("/assistant/query", json=payload)
        return asyncio.run(run())

    payload = {"query": "Quem assina o documento?", "patient_id": "patient_001"}
    assert send(payload).status_code == 200
    db = PatientDatabase()
    try:
        db.add_vital_signs("patient_001", {"blood_pressure": "123/79", "heart_rate": 67})
    finally:
        db.close()
    response = send(payload)
    assert response.status_code == 200
    request = invoke.call_args.args[0]
    assert request.query == payload["query"]
    assert request.patient_context.patient_id == "patient_001"
    assert request.patient_context.sinais_vitais[0]["blood_pressure"] == "123/79"
    assert response.json()["sources"][0]["version"] == "v1"
    assert response.json()["sources"][0]["quoted_text"] == "Texto verificável"
    assert send({"query": payload["query"]}).status_code == 200
    assert invoke.call_args.args[0].patient_context is None
    count = invoke.call_count
    assert send({**payload, "patient_id": "missing"}).status_code == 404
    assert invoke.call_count == count
    assert send({**payload, "patient_context": {}}).status_code == 422
    logs = [json.loads(line) for line in (isolated / "audit.log").read_text().splitlines()]
    assert logs[3]["sources"][0]["id"] == "DOC@v1#s"
    assert logs[3]["adapter_weights_sha256"] == "a" * 64
    assert logs[3]["query_id"] == response.json()["query_id"]


def test_blocked_query_never_reaches_runtime(isolated, monkeypatch):
    invoke = Mock()
    monkeypatch.setattr(FT11Client, "invoke", invoke)
    result = MedicalAssistantLangChain().process_query("Quero prescrever medicamento")
    assert result["security_blocked"] and result["requires_validation"]
    assert result["model_usage"] is None
    invoke.assert_not_called()


def test_unavailable_runtime_does_not_simulate_response(isolated, monkeypatch):
    monkeypatch.setattr(FT11Client, "invoke", Mock(side_effect=InferenceUnavailable("offline")))
    with pytest.raises(InferenceUnavailable):
        MedicalAssistantLangChain().process_query("oi")
    logs = [json.loads(line) for line in (isolated / "audit.log").read_text().splitlines()]
    assert logs[-1]["error"] == "offline" and logs[-1]["response"] == ""


@pytest.mark.parametrize("failure", [requests.Timeout("timeout"), ValueError("invalid JSON")])
def test_client_rejects_transport_or_contract_errors(monkeypatch, failure):
    monkeypatch.setattr(requests, "post", Mock(side_effect=failure))
    from medical_assistant.application.contracts import AssistantRequest
    with pytest.raises(InferenceUnavailable):
        FT11Client().invoke(AssistantRequest(query="Pergunta"))


def test_source_text_and_offsets_survive_contract_without_whitespace_changes(runtime_result):
    payload = runtime_result.model_dump()
    source = payload["sources"][0]
    source.update(text="   Texto verificável\n", char_start=3, char_end=20)
    result = RuntimeResponse.model_validate(payload)
    assert result.sources[0].text == source["text"]
    assert result.sources[0].text[source["char_start"]:source["char_end"]] == source["quoted_text"]


def test_streamlit_preserves_source_details_on_rerun(isolated, monkeypatch, runtime_result):
    from streamlit.testing.v1 import AppTest
    from medical_assistant.interfaces import streamlit_app as ui
    monkeypatch.setattr(FT11Client, "invoke", Mock(return_value=runtime_result))
    monkeypatch.setattr(ui, "PROJECT_DIR", isolated)
    (isolated / "logs").mkdir()
    app = AppTest.from_string('''
import streamlit as st
from medical_assistant.interfaces import streamlit_app as ui
ui.configure_page()
ui.initialize_session()
db = ui.PatientDatabase()
try:
    st.session_state.selected_patient = 'patient_001'
    ui.initialize_system()
    ui.display_chat_interface(db)
finally:
    db.close()
''').run()
    app.chat_input[0].set_value("Quem assina o documento?").run()
    app.run()
    assert not app.exception and not app.error
    assert app.expander[0].label == "Fonte: Documento de teste — v1"
    assert app.session_state.messages[-1]["result"]["sources"][0]["id"] == "DOC@v1#s"


def test_switching_patient_clears_chat(isolated):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    db = PatientDatabase()
    try:
        db.conn.execute("INSERT INTO patients (id, name, age, gender) VALUES (?, ?, ?, ?)",
                        ("patient_002", "Paciente sintético 2", 35, "F"))
        db.conn.commit()
    finally:
        db.close()
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    app.session_state.messages = [{"role": "assistant", "content": "Paciente anterior"}]
    app.selectbox[0].select(app.selectbox[0].options[1]).run()
    assert not app.exception
    assert app.session_state.messages == []


def test_cli_uses_shared_patient_and_response_contract(isolated, monkeypatch, runtime_result):
    from medical_assistant.interfaces.cli import MedicalAssistantInterface
    invoke = Mock(return_value=runtime_result)
    monkeypatch.setattr(FT11Client, "invoke", invoke)
    cli = MedicalAssistantInterface()
    try:
        result = cli.process_medical_query("Pergunta documental", "patient_001")
        assert result["langchain_response"]["sources"][0]["id"] == "DOC@v1#s"
        assert invoke.call_args.args[0].patient_context.patient_id == "patient_001"
    finally:
        cli.patient_db.close()
