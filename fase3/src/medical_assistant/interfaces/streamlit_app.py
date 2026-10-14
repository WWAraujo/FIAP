"""
Interface Web Streamlit para Assistente Médico - Fase 3 Tech Challenge
Interface gráfica para equipe médica com chat acoplado ao prontuário do paciente
"""

import streamlit as st
import json
import logging
from pathlib import Path
from datetime import datetime
from medical_assistant.application.assistant import MedicalAssistantLangChain
from medical_assistant.clinical.patient_database import PatientDatabase, PatientContextBuilder
from medical_assistant.application.workflow import MedicalWorkflowGraph

PROJECT_DIR = Path(__file__).resolve().parents[3]

def configure_page():
    st.set_page_config(
        page_title="Assistente Médico - Fase 3 Tech Challenge",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': None
        }
    )
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

# Custom CSS reativo ao tema do Streamlit
#
# O CSS anterior usava @media (prefers-color-scheme: light/dark), que segue
# a preferência do SISTEMA OPERACIONAL/navegador — não o toggle de tema que
# fica dentro do próprio app (menu "⋮" > Settings > Choose app theme). Por
# isso os dois prints ficavam quase iguais: o navegador do usuário está fixo
# num esquema (ex: dark) e só o bloco @media correspondente a ESSE esquema
# roda, não importa o que o usuário escolha no seletor do Streamlit.
#
# O Streamlit marca o tema realmente ativo com o atributo [data-theme="light"
# | "dark"] na raiz do app, e também expõe as cores computadas do tema como
# CSS custom properties (--primary-color, --background-color,
# --secondary-background-color, --text-color). Usar essas duas coisas
# elimina a necessidade de manter dois blocos de CSS hardcoded em paralelo.
PAGE_CSS = """
<style>
    /* Cards (métrica, expander, alert) usam a cor de fundo secundária que o
       Streamlit já calcula para o tema ativo — reage automaticamente. */
    div[data-testid="stMetric"],
    div[data-testid="stExpander"],
    .stAlert {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-color, rgba(128, 128, 128, 0.35));
        border-radius: 8px;
    }
    div[data-testid="stMetric"] {
        padding: 1rem;
    }
    div[data-testid="stMetric"] label {
        font-weight: 600;
    }

    /* No tema escuro, box-shadow não gera contraste sobre fundo já escuro —
       usa borda mais clara para dar profundidade em vez de sombra. */
    [data-theme="dark"] .block-container {
        box-shadow: none;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
    }
    [data-theme="light"] .block-container {
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        border-radius: 10px;
    }
    .main .block-container {
        padding-top: 2rem;
    }

    /* Sidebar: usa a cor de fundo secundária do tema (troca de verdade
       entre claro/escuro) em vez de navy/preto fixos. */
    section[data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
    }

    h1 { font-size: 2.5rem; margin-bottom: 1rem; font-weight: 600; }
    h2 { font-size: 1.8rem; margin-top: 1.5rem; margin-bottom: 0.8rem; font-weight: 600; }
    h3 { font-size: 1.3rem; margin-top: 1rem; margin-bottom: 0.5rem; font-weight: 600; }

    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }

    @media (max-width: 768px) {
        div[data-testid="column"] {
            flex-direction: column !important;
        }
    }
</style>
"""

def initialize_session():
    st.session_state.setdefault('assistant', None)
    st.session_state.setdefault('messages', [])
    st.session_state.setdefault('selected_patient', None)


def initialize_system():
    """Inicializa o sistema do assistente médico"""
    try:
        if st.session_state.assistant is None:
            st.session_state.assistant = MedicalAssistantLangChain()
            st.success("✅ Sistema do assistente médico inicializado com sucesso!")
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
        patient = db.get_patient_info(patient_id)
        if patient is None:
            return None
        context_builder = PatientContextBuilder(db)
        context = context_builder.build_context_for_assistant(patient_id)
        return {
            "idade": patient["age"],
            "genero": patient["gender"],
            "historico": context,
        }
    except Exception as e:
        st.error(f"❌ Erro ao buscar contexto do paciente: {e}")
        return None

def display_patient_info(db, patient_id):
    """Exibe informações do paciente selecionado"""
    patient_info = db.get_patient_info(patient_id)

    if patient_info:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("ID do Paciente", patient_info['id'])
            st.metric("Idade", f"{patient_info['age']} anos")

        with col2:
            st.metric("Gênero", patient_info['gender'])
            st.metric("Contato", patient_info['contact'])

        with col3:
            st.write(f"**Data de Nascimento:** {patient_info['birth_date']}")
            st.write(f"**Endereço:** {patient_info['address']}")

        # Histórico médico - Cards responsivos
        st.subheader("📋 Histórico Médico")
        medical_history = db.get_medical_history(patient_id)
        if medical_history:
            # Grid responsivo de cards
            cols = st.columns(min(len(medical_history), 3))
            for idx, condition in enumerate(medical_history):
                with cols[idx % 3]:
                    status_emoji = "✅" if condition['status'] == "Ativo" else "⚠️"
                    st.info(f"{status_emoji} **{condition['condition']}**\n\n{condition['status']} (desde {condition['diagnosis_date']})")
                    if condition['notes']:
                        st.caption(f"Notas: {condition['notes']}")
        else:
            st.info("Nenhum histórico médico registrado")

        # Sinais vitais recentes - Cards responsivos
        st.subheader("💓 Sinais Vitais Recentes")
        vital_signs = db.get_vital_signs(patient_id, limit=3)
        if vital_signs:
            # Grid responsivo de cards
            cols = st.columns(min(len(vital_signs), 3))
            for idx, vital in enumerate(vital_signs):
                with cols[idx % 3]:
                    with st.expander(f"📅 {vital['recorded_at'][:10]}"):
                        st.write(f"- **Pressão Arterial:** {vital['blood_pressure']}")
                        st.write(f"- **Frequência Cardíaca:** {vital['heart_rate']} bpm")
                        st.write(f"- **Temperatura:** {vital['temperature']}°C")
                        st.write(f"- **Saturação O2:** {vital['oxygen_saturation']}%")
                        st.write(f"- **Peso:** {vital['weight']} kg")
                        st.write(f"- **Altura:** {vital['height']} m")
        else:
            st.info("Nenhum registro de sinais vitais encontrado")

        # Medicamentos ativos - Cards responsivos
        st.subheader("💊 Medicamentos Ativos")
        medications = db.get_medications(patient_id, active_only=True)
        if medications:
            # Grid responsivo de cards
            cols = st.columns(min(len(medications), 3))
            for idx, med in enumerate(medications):
                with cols[idx % 3]:
                    st.success(f"💊 **{med['medication_name']}**\n\n{med['dosage']} ({med['frequency']})")
                    st.caption(f"Prescrito por: {med['prescribed_by']} | Início: {med['start_date']}")
        else:
            st.info("Nenhum medicamento ativo registrado")

        # Exames recentes - Cards responsivos
        st.subheader("🔬 Exames Recentes")
        exams = db.get_exams(patient_id, limit=5)
        if exams:
            # Grid responsivo de cards
            cols = st.columns(min(len(exams), 3))
            for idx, exam in enumerate(exams):
                with cols[idx % 3]:
                    is_normal = exam['status'] == "Normal"
                    status_emoji = "✅" if is_normal else "⚠️"
                    box = st.success if is_normal else st.warning
                    box(f"{status_emoji} **{exam['exam_type']}**\n\n({exam['exam_date']})")
                    st.write(f"Resultado: {exam['result']}")
                    st.caption(f"Status: {exam['status']} | Notas: {exam['doctor_notes']}")
        else:
            st.info("Nenhum exame registrado")

        # Consultas recentes - Cards responsivos
        st.subheader("👨‍⚕️ Consultas Recentes")
        consultations = db.get_consultations(patient_id, limit=3)
        if consultations:
            # Grid responsivo de cards
            cols = st.columns(min(len(consultations), 3))
            for idx, consultation in enumerate(consultations):
                with cols[idx % 3]:
                    with st.expander(f"📅 {consultation['consultation_date']} - {consultation['doctor']}"):
                        st.write(f"**Queixa Principal:** {consultation['chief_complaint']}")
                        st.write(f"**Diagnóstico:** {consultation['diagnosis']}")
                        st.write(f"**Tratamento:** {consultation['treatment']}")
                        st.write(f"**Follow-up:** {consultation['follow_up']}")
        else:
            st.info("Nenhuma consulta registrada")
    else:
        st.error("Paciente não encontrado")

def display_chat_interface(db):
    """Exibe interface de chat do assistente médico"""
    st.subheader("💬 Chat com Assistente Médico")
    st.caption("Protótipo com respostas simuladas; requer revisão profissional.")
    if st.session_state.assistant is None:
        st.info("Inicialize o sistema na barra lateral para usar o chat.")
        return

    # Exibir histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input do usuário
    if prompt := st.chat_input("Digite sua pergunta médica ou instruções clínicas:"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Adicionar contexto do paciente se selecionado
        context = None
        if st.session_state.selected_patient:
            context = get_patient_context(db, st.session_state.selected_patient)
            if context:
                context["queixa"] = prompt

        # Processar com o assistente
        with st.chat_message("assistant"):
            with st.spinner("Processando sua solicitação..."):
                try:
                    result = st.session_state.assistant.process_query(prompt, patient_context=context)
                    response = result["response"]
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                    # Logging da interação
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "patient_id": st.session_state.selected_patient,
                        "query": prompt,
                        "response": response,
                        "context_used": bool(context)
                    }

                    # Salvar no log de auditoria
                    with open(PROJECT_DIR / "logs" / "audit.log", "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

                except Exception as e:
                    error_msg = f"❌ Erro ao processar sua solicitação: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

def display_workflow_interface(db):
    """Exibe interface do workflow automatizado"""
    st.subheader("🔄 Workflow Automatizado de Triagem")

    if not st.session_state.selected_patient:
        st.warning("⚠️ Selecione um paciente primeiro para executar o workflow")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        query = st.text_area("Descrição do Caso Clínico",
                          "Descreva os sintomas, sinais vitais ou situação do paciente...",
                          height=100)

    with col2:
        if st.button("🚀 Executar Workflow", type="primary"):
            with st.spinner("Executando workflow automatizado..."):
                try:
                    summary = db.get_patient_summary(st.session_state.selected_patient)
                    if "error" in summary:
                        st.error(summary["error"])
                        return
                    vitals = summary["recent_vital_signs"]
                    patient_data = {
                        **summary["patient_info"],
                        "medical_history": summary["medical_history"],
                        "vital_signs": vitals[0] if vitals else {},
                        "completed_exams": [
                            exam["exam_type"]
                            for exam in db.get_exams(st.session_state.selected_patient)
                            if exam["result"] and exam["status"] in ("Normal", "Atenção", "Concluído")
                        ],
                    }
                    result = MedicalWorkflowGraph().execute_full_workflow(patient_data, query)
                    if not result["completed"]:
                        st.error("Workflow incompleto; requer revisão profissional.")
                        return

                    st.success("✅ Workflow executado com sucesso!")

                    # Exibir resultados
                    st.subheader("📊 Resultados do Workflow")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        priority = next(d["priority"] for d in result["decisions"]
                                        if d["step"] == "triage_classification")
                        st.metric("Prioridade", priority)
                    with col2:
                        st.metric("Alertas", len(result["alerts"]))
                    with col3:
                        st.metric("Recomendações", len(result["recommendations"]))

                    # Decisões tomadas
                    if result.get("decisions"):
                        st.subheader("📝 Decisões Tomadas")
                        for decision in result["decisions"]:
                            st.write(f"- {decision}")

                    # Alertas gerados
                    if result.get("alerts"):
                        st.warning("⚠️ Alertas Gerados:")
                        for alert in result["alerts"]:
                            st.warning(f"- {alert}")

                    # Recomendações
                    if result.get("recommendations"):
                        st.info("💡 Recomendações:")
                        for rec in result["recommendations"]:
                            st.info(f"- {rec}")

                    # Status de intervenção humana
                    if result.get("requires_human_intervention"):
                        st.error("⚠️ INTERVENÇÃO HUMANA NECESSÁRIA")
                    else:
                        st.success("✅ Sem intervenção humana necessária")

                except Exception as e:
                    st.error(f"❌ Erro ao executar workflow: {e}")

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

def render_home_page(db):
    st.title("🏥 Painel do Assistente Médico")
    st.markdown("Bem-vindo ao Sistema de Assistência Médica da Fase 3 do Tech Challenge FIAP")

    if st.session_state.selected_patient:
        st.success(f"✅ Paciente ativo: {st.session_state.selected_patient}")
        display_patient_info(db, st.session_state.selected_patient)

        st.markdown("---")
        st.info("💡 Use a navegação no topo para acessar Chat, Workflow ou Logs")
    else:
        st.warning("⚠️ Selecione um paciente no sidebar para começar")

def render_chat_page(db):
    st.title("💬 Chat com Assistente Médico")
    if st.session_state.selected_patient:
        display_patient_info(db, st.session_state.selected_patient)
        st.markdown("---")
        display_chat_interface(db)
    else:
        st.warning("⚠️ Selecione um paciente no sidebar para usar o chat")

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
    """Função principal da aplicação Streamlit"""

    # Navegação no topo da página (substitui o antigo st.radio no sidebar).
    # st.navigation(position="top") requer Streamlit >= 1.46 — confirme com
    # `pip show streamlit` antes de rodar.
    pages = [
        st.Page(lambda: render_home_page(db), url_path="home", title="Início", icon="🏠", default=True),
        st.Page(lambda: render_chat_page(db), url_path="chat", title="Chat", icon="💬"),
        st.Page(lambda: render_workflow_page(db), url_path="workflow", title="Workflow", icon="🔄"),
        st.Page(render_logs_page, title="Logs", icon="📋"),
        st.Page(lambda: render_full_data_page(db), url_path="data", title="Dados Completos", icon="📊"),
    ]
    pg = st.navigation(pages, position="top")

    # Sidebar (compartilhado por todas as páginas)
    with st.sidebar:
        st.title("🏥 Assistente Médico")
        st.markdown("---")

        # Inicialização do sistema
        if st.button("🔄 Inicializar Sistema", type="primary"):
            initialize_system()

        st.markdown("---")

        # Seleção de paciente
        st.subheader("👤 Selecionar Paciente")

        patients = get_patient_list(db)
        if patients:
            patient_options = {p['id']: f"{p['id']} - {p['name']}" for p in patients}
            selected = st.selectbox("Escolha um paciente:",
                                   options=list(patient_options.values()),
                                   format_func=lambda x: x)

            if selected:
                patient_id = [k for k, v in patient_options.items() if v == selected][0]
                st.session_state.selected_patient = patient_id
                st.success(f"✅ Paciente selecionado: {patient_id}")
        else:
            st.warning("Nenhum paciente encontrado no banco de dados")

        st.markdown("---")

        # Informações do sistema
        st.subheader("ℹ️ Informações do Sistema")
        st.info("""
        **Sistema:** Assistente Médico Fase 3 Tech Challenge
        **Chat:** Respostas simuladas
        **Workflow:** Regras sequenciais de demonstração
        **Revisão profissional:** Necessária; não registrada como aprovação
        """)

    pg.run()

def main():
    logging.basicConfig(level=logging.INFO)
    configure_page()
    initialize_session()
    db = PatientDatabase()
    try:
        render_app(db)
    finally:
        db.close()
