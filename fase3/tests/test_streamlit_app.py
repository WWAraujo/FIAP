import importlib
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
from medical_assistant.clinical.patient_database import PatientDatabase
from medical_assistant.safety.audit import AuditLogger
from medical_assistant.application.assistant import MedicalAssistantLangChain
from medical_assistant.application.contracts import RuntimeResponse
from medical_assistant.retrieval.ft11_client import FT11Client

APP = ROOT / "app.py"


@pytest.fixture
def isolated_app(monkeypatch, tmp_path):
    monkeypatch.setattr(FT11Client, "invoke", lambda self, request: RuntimeResponse(
        answer={"status": "pendente", "dados": {}, "pendencias": ["Sem evidência documental."],
                "fonte": None, "validacao_humana": True}, sources=[],
        model_usage={"model_id": "FT-11-test"}, patient_context_used=request.patient_context is not None,
        run_id="test-run"))
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
    assistant = app.session_state.assistant
    app.run()
    assert not app.exception
    assert not app.error
    assert app.session_state.assistant is not None
    assert app.session_state.assistant is assistant
    assert not any("Inicializar Sistema" in button.label for button in app.button)


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
    received = []
    original = FT11Client.invoke

    def generate(self, request):
        received.append(request)
        return original(self, request)

    monkeypatch.setattr(FT11Client, "invoke", generate)
    app = page("display_chat_interface", isolated_app)
    questions = ["estou com dor na barriga o que devo fazer?", "Eu falei que estou com dor na barriga"]
    for question in questions:
        app.chat_input[0].set_value(question).run()
        assert not app.exception
        assert not app.error
        assert received[-1].query == question
        assert "hipertensão" in json.dumps(received[-1].patient_context.historico, ensure_ascii=False).lower()
        answer = app.session_state.messages[-1]["content"]
        assert answer
        assert "hipertensão" not in answer.lower()

    queries = [json.loads(line) for line in (isolated_app / "audit.log").read_text(
        encoding="utf-8").splitlines() if json.loads(line)["type"] == "query"]
    assert [entry["query"] for entry in queries] == questions
    assert "Hipertensão" in json.dumps(queries[0]["patient_context"]["historico"], ensure_ascii=False)
    assert queries[0]["patient_context"]["patient_id"] == "patient_001"

    app.chat_input[0].set_value("Quais são os sintomas de hipertensão?").run()
    assert not app.exception
    assert received[-1].query == "Quais são os sintomas de hipertensão?"
    assert "Sem evidência documental." in app.session_state.messages[-1]["content"]


def test_workflow_uses_patient_data_and_does_not_accumulate(isolated_app):
    db = PatientDatabase()
    try:
        with db.conn:
            db.conn.execute("UPDATE exams SET status='Vencido' WHERE exam_type='Hemoglobina Glicada'")
    finally:
        db.close()
    app = page("display_workflow_interface", isolated_app)
    app.text_area[0].set_value("pressão alta").run()
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert any('Aguardando revisão' in e.value for e in app.info)
    metrics = {m.label: m.value for m in app.metric}
    assert metrics == {"Prioridade": "alta", "Alertas": "2", "Recomendações": "2"}
    pending = next(item.value for item in app.warning if "pending_exams" in item.value)
    assert "Hemoglobina Glicada" in pending
    assert "Eletrocardiograma" not in pending
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert {m.label: m.value for m in app.metric} == metrics


def test_workspace_search_and_patient_chat(isolated_app):
    from medical_assistant.clinical.seed_patients import seed_patients
    db = PatientDatabase()
    try:
        seed_patients(db)
    finally:
        db.close()
    app = AppTest.from_file(str(APP)).run()
    app.text_input[0].set_value("Bruno").run()
    assert not app.exception
    assert app.session_state.selected_patient == "sim_002"
    assert len(app.selectbox[0].options) == 1
    next(button for button in app.button if button.label == "Esse paciente tem alguma alergia?").click().run()
    assert not app.exception
    assert "Alergia a penicilina" in app.session_state.messages[-1]["content"]
    assert app.session_state.messages[-1]["result"]["sources"][0]["source_type"] == "prontuario"
    app.text_input[0].set_value("Paciente inexistente xyz").run()
    assert not app.exception
    assert app.session_state.selected_patient is None
    assert not app.session_state.messages


def test_explicit_human_review_is_persisted_from_form(isolated_app):
    from medical_assistant.safety.reviews import ReviewStore
    app = page('display_chat_interface',isolated_app)
    app.chat_input[0].set_value('Qual o IMC desse paciente?').run()
    assert not app.exception
    review_id = app.session_state.messages[-1]['result']['review_id']
    app.text_input[0].set_value('Revisor de teste')
    app.radio[0].set_value('Aprovar')
    app.checkbox[0].set_value(True)
    next(b for b in app.button if b.label=='Registrar decisão').click().run()
    assert not app.exception
    record=ReviewStore().get(review_id)
    assert record['status']=='approved' and record['reviewer']=='Revisor de teste'
    assert any('Aprovado pelo responsável declarado' in i.value for i in app.info)
