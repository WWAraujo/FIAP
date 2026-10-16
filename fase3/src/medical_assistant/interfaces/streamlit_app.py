"""
Interface Web Streamlit para Assistente Médico - Fase 3 Tech Challenge
Interface gráfica para equipe médica com chat acoplado ao prontuário do paciente
"""

import streamlit as st
import json
import logging
from pathlib import Path
from datetime import date, datetime
from html import escape
from medical_assistant.application.assistant import MedicalAssistantLangChain
from medical_assistant.clinical.patient_database import PatientDatabase
from medical_assistant.application.workflow import MedicalWorkflowGraph
from medical_assistant.application.contracts import PatientContext
from medical_assistant.application.workflow_service import start_workflow
from medical_assistant.safety.reviews import ReviewStore

PROJECT_DIR = Path(__file__).resolve().parents[3]

def configure_page():
    st.set_page_config(
        page_title="Assistente Médico - Fase 3 Tech Challenge",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': None
        }
    )
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

PAGE_CSS = """
<style>
.stApp { background: var(--background-color); }
.stApp [data-testid="stVerticalBlock"] { gap: .7rem; }
[data-testid="stAppDeployButton"] { display:none; }
.stMainBlockContainer { max-width: 1800px; padding: 3.75rem 1.4rem 2rem; }
section[data-testid="stSidebar"] { min-width: 170px; max-width: 190px; }
section[data-testid="stSidebar"] > div { background: var(--secondary-background-color); }
h1 { font-size: 1.55rem !important; }
h2 { font-size: 1.2rem !important; }
h3 { font-size: 1rem !important; margin: 0 0 .5rem !important; }
p, li, label { font-size: .9rem; }
div[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: 12px; }
div[data-testid="stExpander"] { border-radius: 9px; }
div[data-testid="stChatMessage"] { border: 1px solid #71869b40; border-radius: 12px; padding: .9rem; }
div[data-testid="stChatInput"] { border-radius: 10px; }
.stButton > button { border-radius: 8px; }
.aegis-header { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap;
    gap:1rem; border-bottom:1px solid #71869b55; padding:.2rem 0 1rem; margin-bottom:1.3rem; }
.aegis-brand { font-size:1.35rem; font-weight:700; letter-spacing:-.04em; }
.aegis-brand b { color:#7fcac8; font-size:1.8rem; margin-right:.65rem; }
.aegis-subtitle { font-size:.76rem; opacity:.7; letter-spacing:.02em; margin-top:.2rem; }
.simulation-badge { color:#ffc5bd; background:#bd6f6222; border:1px solid #df98866b;
    padding:.4rem .7rem; border-radius:6px; font-size:.7rem; letter-spacing:.06em; }
.session-label { font-size:.8rem; color:var(--text-color); opacity:.8; }
.patient-card { border:1px solid #71869b55; border-radius:10px; padding:1rem;
    background:var(--secondary-background-color); margin:.4rem 0 1rem; }
.patient-avatar { display:inline-flex; align-items:center; justify-content:center;
    width:42px; height:42px; border-radius:50%; background:#537e94; color:white;
    font-size:1.05rem; font-weight:600; float:left; margin-right:.8rem; }
.patient-name { font-size:.95rem; font-weight:650; }
.patient-meta { font-size:.76rem; opacity:.75; margin-top:.3rem; }
.summary-label { font-size:.79rem; font-weight:650; margin:1rem 0 .35rem; }
.summary-item { background:var(--secondary-background-color); border:1px solid #71869b40;
    padding:.65rem .75rem; border-radius:8px; font-size:.85rem; margin:.3rem 0; }
.vitals-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:.75rem; padding:.5rem 0; }
.vitals-grid span { font-size:.76rem; opacity:.7; display:block; margin-bottom:.15rem; }
.vitals-grid b { font-size:.95rem; font-weight:600; }
.section-kicker { color:#85b7c8; font-size:.72rem; font-weight:650; letter-spacing:.12em; margin-bottom:.35rem; }
@media(max-width:1100px) {
    .vitals-grid { grid-template-columns:repeat(2,1fr); }
    .stMainBlockContainer { padding:3.75rem 1rem 1rem; }
}
@media(max-width:760px) {
    .aegis-header { align-items:flex-start; }
    div[data-testid="stHorizontalBlock"] { flex-wrap:wrap; }
    div[data-testid="stColumn"] { min-width: min(100%, 330px); flex:1 1 100%; }
}
</style>
"""


def html_text(value):
    return escape(str(value if value is not None and value != "" else "Não informado"))


def summary_item(value):
    st.markdown(f'<div class="summary-item">{html_text(value)}</div>', unsafe_allow_html=True)


def initialize_session():
    st.session_state.setdefault('assistant', None)
    st.session_state.setdefault('messages', [])
    st.session_state.setdefault('selected_patient', None)


def initialize_system():
    """Inicializa o sistema do assistente médico"""
    try:
        if st.session_state.assistant is None:
            st.session_state.assistant = MedicalAssistantLangChain()
        return True
    except Exception as e:
        st.error(f"❌ Erro ao inicializar sistema: {e}")
        return False

def get_patient_list(db):
    """Retorna lista de pacientes do banco de dados"""
    try:
        patients = db.search_patients("")
        return patients
    except Exception as e:
        st.error(f"❌ Erro ao buscar pacientes: {e}")
        return []

def get_patient_context(db, patient_id):
    """Retorna contexto completo do paciente"""
    try:
        return PatientContext.from_database(db, patient_id).model_dump()
    except Exception as e:
        st.error(f"❌ Erro ao buscar contexto do paciente: {e}")
        return None

def display_patient_info(db, patient_id):
    summary = db.get_patient_summary(patient_id)
    if "error" in summary:
        st.error(summary["error"])
        return
    with st.container(border=True):
        st.subheader("Sinais vitais")
        vitals = summary["recent_vital_signs"]
        if vitals:
            v = vitals[0]
            st.caption(f"Registro: {v['recorded_at']}")
            fields = [("Pressão arterial", v['blood_pressure'], "mmHg"),
                      ("Frequência cardíaca", v['heart_rate'], "bpm"),
                      ("Temperatura", v['temperature'], "°C"),
                      ("Saturação O₂", v['oxygen_saturation'], "%"),
                      ("Peso", v['weight'], "kg"), ("Altura", v['height'], "m")]
            tiles = "".join(f"<div><span>{html_text(label)}</span><b>{html_text(value)} "
                            f"{unit if value is not None else ''}</b></div>" for label, value, unit in fields)
            st.markdown(f'<div class="vitals-grid">{tiles}</div>', unsafe_allow_html=True)
            if len(vitals) > 1:
                with st.expander("Medições anteriores"):
                    st.dataframe(vitals[1:], hide_index=True, width="stretch")
        else:
            st.caption("Nenhuma medição registrada.")

    with st.container(border=True):
        st.subheader("Medicamentos ativos")
        medications = summary["current_medications"]
        if medications:
            st.dataframe([{"Medicamento": m["medication_name"], "Dose registrada": m["dosage"],
                           "Frequência": m["frequency"], "Início": m["start_date"],
                           "Profissional": m["prescribed_by"]} for m in medications],
                         hide_index=True, width="stretch")
        else:
            st.caption("Nenhum medicamento ativo registrado.")

    with st.container(border=True):
        st.subheader("Exames")
        exams = summary["recent_exams"]
        if not exams:
            st.caption("Nenhum exame registrado.")
        for exam in exams:
            label = f"{exam['exam_type']} · {exam['status']} · {exam['exam_date']}"
            if exam["status"] in ("Pendente", "Vencido", "Atenção"):
                st.warning(label, icon="⚠️")
            else:
                st.info(label, icon="🔬")
            with st.expander(f"Resultado e observações · {exam['exam_type']}"):
                st.write(f"**Resultado:** {exam['result'] or 'Não informado'}")
                st.write(exam["doctor_notes"] or "Sem observações registradas.")

    with st.container(border=True):
        st.subheader("Consultas e Próximos Agendamentos")
        consultations = summary["recent_consultations"]
        if not consultations:
            st.caption("Nenhuma consulta registrada.")
        for consultation in consultations:
            with st.expander(f"{consultation['consultation_date']} · {consultation['doctor']}"):
                st.write(f"**Motivo:** {consultation['chief_complaint']}")
                st.write(f"**Diagnóstico registrado:** {consultation['diagnosis'] or 'Não informado'}")
                st.write(f"**Tratamento registrado:** {consultation['treatment'] or 'Não informado'}")
                st.write(f"**Retorno:** {consultation['follow_up'] or 'Não informado'}")

    with st.expander("Histórico e dados cadastrais"):
        for item in summary["medical_history"]:
            st.write(f"**{item['condition']}** · {item['status']} · {item['diagnosis_date']}")
            st.caption(item["notes"] or "Sem observações.")
        p = summary["patient_info"]
        st.write(f"**Nascimento:** {p['birth_date']} · **Contato:** {p['contact']}")
        st.write(f"**Endereço:** {p['address']}")


def display_patient_summary(db, patient_id):
    summary = db.get_patient_summary(patient_id)
    if "error" in summary:
        st.error(summary["error"])
        return
    p = summary["patient_info"]
    initials = "".join(word[0] for word in p["name"].split(" — ")[0].split()[:2])
    st.markdown(f'<div class="patient-card"><div class="patient-avatar">{html_text(initials)}</div>'
                f'<div class="patient-name">{html_text(p["name"])}, {html_text(p["age"])} anos</div>'
                f'<div class="patient-meta">{html_text(p["id"])} · {html_text(p["gender"])}</div></div>',
                unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("Resumo do paciente")
        st.markdown('<div class="summary-label">Condições e histórico</div>', unsafe_allow_html=True)
        allergies = []
        conditions = []
        for item in summary["medical_history"]:
            if item["condition"].lower().startswith(("alergia", "sem alergia")):
                allergies.append(item)
            else:
                conditions.append(item)
        for condition in conditions:
            summary_item(condition["condition"])
        if not conditions:
            st.caption("Nenhuma condição registrada.")
        st.markdown('<div class="summary-label">Alergias</div>', unsafe_allow_html=True)
        for allergy in allergies:
            text = f"{allergy['condition']} — {allergy['notes']}"
            if allergy["condition"].lower().startswith("alergia a "):
                st.error(text, icon="⚠️")
            else:
                st.info(text)
        if not allergies:
            st.caption("Sem registro de alergias; não confirma ausência.")
        st.markdown('<div class="summary-label">Medicamentos contínuos</div>', unsafe_allow_html=True)
        for medication in summary["current_medications"]:
            summary_item(f"{medication['medication_name']} · {medication['frequency']}")
        if not summary["current_medications"]:
            st.caption("Nenhum medicamento ativo registrado.")
        st.markdown('<div class="summary-label">Pendências e próxima consulta</div>', unsafe_allow_html=True)
        future = [c for c in summary["recent_consultations"]
                  if c["consultation_date"][:10] >= date.today().isoformat()]
        if future:
            appointment = min(future, key=lambda c: c["consultation_date"])
            summary_item(f"Próxima consulta: {appointment['consultation_date']} · {appointment['doctor']}")
        else:
            st.caption("Nenhuma consulta futura registrada.")
        for exam in summary["recent_exams"]:
            if exam["status"] in ("Pendente", "Vencido"):
                st.warning(f"{exam['exam_type']} · {exam['status']} · {exam['exam_date']}")
        for consultation in summary["recent_consultations"]:
            if "reagendamento pendente" in (consultation["follow_up"] or "").lower():
                st.warning(consultation["follow_up"])
        st.caption(f"Cadastro atualizado em: {p['updated_at'] or 'Não informado'}")


def select_patient(db):
    query = st.text_input("Buscar paciente", placeholder="Buscar por nome ou ID", key="patient_search")
    patients = db.search_patients(query.strip())
    options = {p['id']: f"{p['id']} - {p['name']}" for p in patients}
    if not options:
        st.info("Nenhum paciente encontrado para esta busca.")
        st.session_state.selected_patient = None
        st.session_state.messages = []
        st.session_state.pop('workflow_result', None)
        return
    current = st.session_state.selected_patient
    ids = list(options)
    selected = st.selectbox("Paciente em atendimento", list(options.values()),
                            index=ids.index(current) if current in ids else 0)
    patient_id = ids[list(options.values()).index(selected)]
    if current != patient_id:
        st.session_state.messages = []
        st.session_state.pop('workflow_result', None)
        if st.session_state.assistant is not None:
            st.session_state.assistant.memory.clear()
    st.session_state.selected_patient = patient_id


def display_chat_interface(db):
    """Exibe interface de chat do assistente médico"""
    st.subheader("Assistente médico")
    st.caption("Consulte o prontuário e os documentos de referência.")
    if st.session_state.assistant is None:
        st.info("O assistente não pôde ser inicializado. Atualize a página para tentar novamente.")
        return

    conversation = st.container()
    # Exibir histórico de mensagens
    for message in st.session_state.messages:
        with conversation.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("result"):
                display_response_details(message["result"])

    prompt = None
    suggestions = st.empty()
    if not st.session_state.messages:
        with suggestions.container():
            if st.button("Esse paciente tem alguma alergia?", key="suggest_allergy", width="stretch"):
                prompt = "Esse paciente tem alguma alergia?"
    typed_prompt = st.chat_input("Pergunte sobre o paciente ou um documento…")
    if typed_prompt:
        prompt = typed_prompt
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with conversation.chat_message("user"):
            st.markdown(prompt)

        # Adicionar contexto do paciente se selecionado
        context = None
        if st.session_state.selected_patient:
            context = get_patient_context(db, st.session_state.selected_patient)
            if context is None:
                st.error("Não foi possível obter o prontuário atual. A consulta não foi enviada.")
                return

        # Processar com o assistente
        with conversation.chat_message("assistant"):
            with st.spinner("Processando sua solicitação..."):
                try:
                    result = st.session_state.assistant.process_query(prompt, patient_context=context)
                    response = result["response"]
                    st.markdown(response)
                    display_response_details(result)
                    st.session_state.messages.append({"role": "assistant", "content": response, "result": result})

                    suggestions.empty()

                except Exception as e:
                    error_msg = f"❌ Erro ao processar sua solicitação: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


def display_response_details(result):
    st.caption("Resposta sujeita à revisão profissional. Consulte o registro de revisão abaixo.")
    for source in result.get("sources", []):
        with st.expander(f"Fonte: {source['title']} — {source['version']}"):
            st.write(f"ID: {source['id']} | Seção: {source['section']}")
            if source.get("page") is not None:
                st.write(f"Página: {source['page']}")
            st.text(source["quoted_text"])
            if source.get("url"):
                st.link_button("Abrir documento", source["url"])
    if not result.get("sources"):
        st.caption("Nenhuma fonte documental retornada.")
    if result.get("model_usage"):
        st.caption(f"Modelo: {result['model_usage']['model_id']} | Consulta: {result['query_id']}")
    elif any(source.get("source_type") == "prontuario" for source in result.get("sources", [])):
        st.caption(f"Origem: prontuário SQLite | Consulta: {result['query_id']}")
    if result.get("patient_context_used"):
        st.caption("Dados atuais do paciente enviados ao modelo nesta consulta.")
    if result.get('review_id'):
        display_review(result['review_id'])


def display_review(review_id):
    record = ReviewStore().get(review_id)
    labels = {'pending':'Aguardando revisão', 'approved':'Aprovado pelo responsável declarado', 'rejected':'Rejeitado pelo responsável declarado'}
    st.info(labels[record['status']])
    if record['status'] != 'pending':
        st.caption(f"Responsável: {record['reviewer']} · {record['reviewed_at']}")
        st.write(record['notes'] or 'Sem observações adicionais.')
        return
    if not record['payload'].get('review_eligible'):
        st.caption('Sem evidências suficientes para aprovação. Complete dados/fontes antes de emitir nova resposta.')
        return
    with st.expander('Registrar revisão profissional'):
        st.caption('Protótipo acadêmico: a identificação é declarada, sem autenticação profissional.')
        with st.form('review_'+review_id):
            reviewer = st.text_input('Responsável pela revisão',max_chars=120)
            decision = st.radio('Decisão',['Selecione','Aprovar','Rejeitar'])
            notes = st.text_area('Observações da revisão',max_chars=2000)
            confirmed = st.checkbox('Revisei o conteúdo e as fontes deste rascunho')
            submitted = st.form_submit_button('Registrar decisão')
        if submitted:
            try:
                if decision == 'Selecione':
                    raise ValueError('Selecione uma decisão.')
                ReviewStore().decide(review_id,patient_id=st.session_state.selected_patient,
                    reviewer=reviewer,decision='approved' if decision=='Aprovar' else 'rejected',
                    notes=notes,confirmed_human=confirmed)
                st.rerun()
            except ValueError as error:
                st.error(str(error))

def display_workflow_interface(db):
    """Executa o grafo com prontuário, documentos e revisão do rascunho."""
    st.subheader('Workflow Automatizado de Triagem')
    st.caption('Triagem acadêmica por regras demonstrativas; resultados e referências exigem revisão profissional.')
    if not st.session_state.selected_patient:
        st.warning('Selecione um paciente para executar o workflow.')
        return
    query = st.text_area('Descrição do Caso Clínico',placeholder='Informe o objetivo da conferência do prontuário e dos documentos.',height=100)
    if st.button('🚀 Executar Workflow',type='primary'):
        st.session_state.pop('workflow_result',None)
        if not query.strip():
            st.error('Descreva o caso antes de executar.')
            return
        with st.spinner('Consultando prontuário, documentos e executando o grafo...'):
            try:
                result = start_workflow(st.session_state.selected_patient,query,assistant=st.session_state.assistant)
                st.session_state.workflow_result = result
            except (ValueError, RuntimeError) as error:
                st.error(str(error))
                return
    result = st.session_state.get('workflow_result')
    if not result:
        return
    if not result.get('completed'):
        st.error(result.get('error') or 'Workflow incompleto; requer revisão profissional.')
        return
    st.success('Análise concluída; rascunho sujeito à revisão profissional.')
    priority = next(d['priority'] for d in result['decisions'] if d['step']=='triage_classification')
    columns = st.columns(3)
    columns[0].metric('Prioridade',priority)
    columns[1].metric('Alertas',len(result['alerts']))
    columns[2].metric('Recomendações',len(result['recommendations']))
    for decision in result['decisions']:
        st.write(decision)
    for alert in result['alerts']:
        st.warning(str(alert))
    for recommendation in result['recommendations']:
        st.info(recommendation['description'])
    with st.expander('Fontes e rastreabilidade'):
        st.caption('Execução: '+result['query_id'])
        for source in result['sources']:
            st.write(source['title'])
            st.text(source['quoted_text'])
    if result.get('review_id'):
        display_review(result['review_id'])

def display_logs_interface():
    """Exibe interface de logs de auditoria"""
    st.subheader("📋 Logs de Auditoria")

    try:
        audit_log = PROJECT_DIR / "logs" / "audit.log"
        if audit_log.exists():
            with open(audit_log, "r", encoding="utf-8") as f:
                logs = f.readlines()

            if logs:
                # Mostrar últimos 10 logs
                recent_logs = logs[-10:]
                st.info(f"📊 Mostrando os últimos {len(recent_logs)} registros de {len(logs)} total")

                for log in recent_logs:
                    try:
                        log_data = json.loads(log.strip())
                        with st.expander(f"📅 {log_data['timestamp']} - {log_data['patient_id'] or 'Sem paciente'}"):
                            st.write(f"**Query:** {log_data['query']}")
                            st.write(f"**Response:** {log_data['response'][:200]}...")
                            st.caption(f"Contexto usado: {'Sim' if log_data['context_used'] else 'Não'}")
                    except:
                        st.code(log.strip())
            else:
                st.info("Nenhum log encontrado")
        else:
            st.info("Arquivo de log não encontrado")
    except Exception as e:
        st.error(f"❌ Erro ao ler logs: {e}")

def render_workspace(db):
    left, center, right = st.columns([1, 1.8, 1.3], gap="medium")
    with left:
        st.markdown('<div class="section-kicker">EM ATENDIMENTO</div>', unsafe_allow_html=True)
        select_patient(db)
        patient_id = st.session_state.selected_patient
        if patient_id:
            display_patient_summary(db, patient_id)
    if not patient_id:
        with center:
            st.info("Busque e selecione um paciente para abrir o prontuário e o chat.")
        return
    with center:
        st.subheader("Prontuário detalhado")
        display_patient_info(db, patient_id)
    with right:
        with st.container(border=True):
            patient = db.get_patient_info(patient_id)
            st.caption(f"Conversa atual: {patient['name']} · {patient_id}")
            display_chat_interface(db)


def render_home_page(db):
    render_workspace(db)


def render_chat_page(db):
    render_workspace(db)


def render_workflow_page(db):
    st.title("🔄 Workflow Automatizado de Triagem")
    if st.session_state.selected_patient:
        display_patient_info(db, st.session_state.selected_patient)
        st.markdown("---")
        display_workflow_interface(db)
    else:
        st.warning("⚠️ Selecione um paciente no sidebar para executar o workflow")

def render_logs_page():
    st.title("📋 Logs de Auditoria")
    display_logs_interface()

def render_full_data_page(db):
    st.title("📊 Dados Completos do Paciente")
    if st.session_state.selected_patient:
        display_patient_info(db, st.session_state.selected_patient)
    else:
        st.warning("⚠️ Selecione um paciente no sidebar para ver dados completos")

def render_app(db):
    active = st.session_state.assistant is not None
    st.markdown('<div class="aegis-header"><div><div class="aegis-brand"><b>Λ</b>Aegis IA</div>'
                '<div class="aegis-subtitle">Assistente médico · prontuário e fontes</div></div>'
                '<span class="simulation-badge">AMBIENTE DE SIMULAÇÃO ACADÊMICA</span>'
                f'<span class="session-label">{"Sessão do assistente ativa" if active else "Assistente não inicializado"}'
                ' · Equipe médica</span></div>', unsafe_allow_html=True)
    pages = [
        st.Page(lambda: render_home_page(db), url_path="home", title="Atendimento", icon=":material/home:", default=True),
        st.Page(lambda: render_workflow_page(db), url_path="workflow", title="Triagem", icon=":material/assignment:") ,
        st.Page(render_logs_page, title="Auditoria", icon=":material/history:"),
        st.Page(lambda: render_full_data_page(db), url_path="data", title="Prontuário", icon=":material/folder_shared:"),
    ]
    pg = st.navigation(pages, position="sidebar")
    with st.sidebar:
        st.markdown("### Λ Aegis IA")
        st.caption("Área do profissional")
        st.caption("Dados sintéticos · FIAP Fase 3")
        st.caption("As respostas exigem conferência profissional.")
    pg.run()


def main():
    logging.basicConfig(level=logging.INFO)
    configure_page()
    initialize_session()
    initialize_system()
    db = PatientDatabase()
    try:
        render_app(db)
    finally:
        db.close()
