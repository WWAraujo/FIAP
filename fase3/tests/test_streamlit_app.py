import importlib
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
from medical_assistant.clinical.patient_database import PatientDatabase
from medical_assistant.safety.audit import AuditLogger
from medical_assistant.application.assistant import MedicalAssistantLangChain

APP = ROOT / "app.py"


@pytest.fixture
def isolated_app(monkeypatch, tmp_path):
    original_init = PatientDatabase.__init__
    monkeypatch.setattr(PatientDatabase, "__init__",
                        lambda self: original_init(self, tmp_path / "patients.db"))

    def init_audit(self):
        self.log_dir = tmp_path
        self.audit_file = tmp_path / "audit.log"

    monkeypatch.setattr(AuditLogger, "__init__", init_audit)
    from medical_assistant.interfaces import streamlit_app
    monkeypatch.setattr(streamlit_app, "PROJECT_DIR", tmp_path)
    (tmp_path / "logs").mkdir()
    return tmp_path


def page(name, tmp_path):
    return AppTest.from_string(f'''
import streamlit as st
from medical_assistant.interfaces import streamlit_app as ui
ui.configure_page()
ui.initialize_session()
db = ui.PatientDatabase()
st.session_state.selected_patient = 'patient_001'
try:
    ui.initialize_system()
    ui.{name}(db)
finally:
    db.close()
''').run()


def test_initialization_and_reruns(isolated_app):
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    app.button[0].click().run()
    app.run()
    assert not app.exception
    assert not app.error
    assert app.session_state.assistant is not None


def test_import_does_not_initialize_ui_or_database(monkeypatch):
    from medical_assistant.interfaces import streamlit_app

    def unexpected(*args, **kwargs):
        raise AssertionError("Import must not initialize resources")

    monkeypatch.setattr(PatientDatabase, "__init__", unexpected)
    monkeypatch.setattr(streamlit_app.st, "set_page_config", unexpected)
    monkeypatch.setattr(streamlit_app.st, "markdown", unexpected)
    importlib.reload(streamlit_app)


def test_chat_response(isolated_app):
    app = page("display_chat_interface", isolated_app)
    app.chat_input[0].set_value("Quais são os protocolos internos?").run()
    assert not app.exception
    assert not app.error
    messages = app.session_state.messages
    assert messages[-1]["role"] == "assistant"
    assert isinstance(messages[-1]["content"], str)
    assert messages[-1]["content"]
    assert len(app.chat_message) == 2


def test_chat_keeps_patient_history_separate_from_question(isolated_app, monkeypatch):
    monkeypatch.setattr(MedicalAssistantLangChain, "_load_medical_knowledge", lambda self: {
        "protocols": [{"titulo": "Hipertensão", "conteudo": "Protocolo de hipertensão"}],
        "faqs": [], "procedures": [],
    })
    received = []
    original = MedicalAssistantLangChain._simulate_llm_response

    def generate(self, system_prompt, user_query):
        received.append((system_prompt, user_query))
        return original(self, system_prompt, user_query)

    monkeypatch.setattr(MedicalAssistantLangChain, "_simulate_llm_response", generate)
    app = page("display_chat_interface", isolated_app)
    questions = ["estou com dor na barriga o que devo fazer?", "Eu falei que estou com dor na barriga"]
    for question in questions:
        app.chat_input[0].set_value(question).run()
        assert not app.exception
        assert not app.error
        assert received[-1][1] == question
        assert "hipertensão" in received[-1][0].lower()
        answer = app.session_state.messages[-1]["content"]
        assert answer
        assert "hipertensão" not in answer.lower()

    queries = [json.loads(line) for line in (isolated_app / "audit.log").read_text(
        encoding="utf-8").splitlines() if json.loads(line)["type"] == "query"]
    assert [entry["query"] for entry in queries] == questions
    assert "Hipertensão" in queries[0]["patient_context"]["historico"]
    assert queries[0]["patient_context"]["queixa"] == questions[0]

    app.chat_input[0].set_value("Quais são os sintomas de hipertensão?").run()
    assert not app.exception
    assert "a hipertensão é frequentemente assintomática" in app.session_state.messages[-1]["content"]


def test_workflow_uses_patient_data_and_does_not_accumulate(isolated_app):
    app = page("display_workflow_interface", isolated_app)
    app.text_area[0].set_value("pressão alta").run()
    app.button[0].click().run()
    assert not app.exception
    assert [e.value for e in app.error] == ["INTERVENÇÃO HUMANA NECESSÁRIA"]
    metrics = {m.label: m.value for m in app.metric}
    assert metrics == {"Prioridade": "alta", "Alertas": "2", "Recomendações": "2"}
    pending = next(item.value for item in app.warning if "pending_exams" in item.value)
    assert "Exames de sangue" in pending
    assert "Eletrocardiograma" not in pending
    app.button[0].click().run()
    assert not app.exception
    assert [e.value for e in app.error] == ["INTERVENÇÃO HUMANA NECESSÁRIA"]
    assert {m.label: m.value for m in app.metric} == metrics
