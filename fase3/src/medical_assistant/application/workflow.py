"""
Fluxos LangGraph para Coordenação de Decisões Médicas
Implementa workflows automatizados para triagem, sugestão de tratamentos e alertas
"""

import json
from copy import deepcopy
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
from enum import Enum
from langgraph.graph import END, START, StateGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """Estado do workflow médico"""
    patient_data: Dict[str, Any]
    query: str
    current_step: str
    collected_info: Dict[str, Any]
    decisions: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    requires_human_intervention: bool
    completed: bool
    error: str | None


class TriagePriority(Enum):
    """Prioridade de triagem"""
    LOW = "baixa"
    MEDIUM = "média"
    HIGH = "alta"
    EMERGENCY = "emergência"


class MedicalWorkflowGraph:
    """Grafo de workflow para decisões médicas automatizadas"""

    def __init__(self):
        self.graph = self._build_graph()
        self.state = self._initial_state()

    @staticmethod
    def _initial_state():
        return {
            "patient_data": {}, "query": "", "current_step": "initial",
            "collected_info": {}, "decisions": [], "alerts": [], "recommendations": [],
            "requires_human_intervention": False, "completed": False, "error": None,
        }

    def _build_graph(self):
        builder = StateGraph(WorkflowState)
        steps = ("initial_assessment", "triage_classification", "protocol_selection",
                 "exam_check", "treatment_suggestion", "completion")
        for name in steps:
            builder.add_node(name, self._node_for(name))
        builder.add_edge(START, steps[0])
        for current, following in zip(steps, steps[1:]):
            builder.add_conditional_edges(current, lambda state: "stop" if state.get("error") else "next",
                                          {"stop": END, "next": following})
        builder.add_edge(steps[-1], END)
        return builder.compile()

    @staticmethod
    def _node_for(name):
        def node(state):
            worker = object.__new__(MedicalWorkflowGraph)
            worker.state = deepcopy(state)
            try:
                return getattr(worker, "step_" + name)()
            except Exception as error:
                worker.state.update(error=str(error), completed=False, requires_human_intervention=True,
                                    current_step="failed")
                return worker.state
        return node

    def start_workflow(self, patient_data: Dict[str, Any], query: str) -> WorkflowState:
        """Inicia workflow com dados do paciente e query"""
        self.state = self._initial_state()
        self.state["patient_data"] = deepcopy(patient_data)
        self.state["query"] = query
        self.state["current_step"] = "initial"
        self.state["completed"] = False

        logger.info(f"Iniciando workflow para paciente: {patient_data.get('id', 'unknown')}")
        return self.state

    def step_initial_assessment(self) -> WorkflowState:
        """Passo 1: Avaliação inicial do paciente"""
        logger.info("Passo 1: Avaliação inicial")

        patient_data = self.state["patient_data"]

        # Avaliar sinais vitais básicos
        assessment = {
            "step": "initial_assessment",
            "timestamp": datetime.now().isoformat(),
            "vital_signs": self._evaluate_vital_signs(patient_data),
            "risk_factors": self._identify_risk_factors(patient_data)
        }

        self.state["collected_info"]["initial_assessment"] = assessment
        self.state["current_step"] = "triage_classification"

        return self.state

    def step_triage_classification(self) -> WorkflowState:
        """Passo 2: Classificação de triagem"""
        logger.info("Passo 2: Classificação de triagem")

        initial_assessment = self.state["collected_info"].get("initial_assessment", {})
        vital_signs = initial_assessment.get("vital_signs", {})

        # Classificar prioridade
        priority = self._classify_priority(vital_signs)

        triage_decision = {
            "step": "triage_classification",
            "timestamp": datetime.now().isoformat(),
            "priority": priority.value,
            "requires_immediate_attention": priority in [TriagePriority.HIGH, TriagePriority.EMERGENCY]
        }

        self.state["decisions"].append(triage_decision)

        # Gerar alertas se necessário
        if triage_decision["requires_immediate_attention"]:
            alert = {
                "type": "priority_alert",
                "priority": priority.value,
                "message": f"Paciente requer atenção {priority.value} imediata",
                "timestamp": datetime.now().isoformat()
            }
            self.state["alerts"].append(alert)
            self.state["requires_human_intervention"] = True

        self.state["current_step"] = "protocol_selection"
        return self.state

    def step_protocol_selection(self) -> WorkflowState:
        """Passo 3: Seleção de protocolo adequado"""
        logger.info("Passo 3: Seleção de protocolo")

        query = self.state["query"].lower()
        patient_data = self.state["patient_data"]

        # Selecionar protocolo baseado na query e dados do paciente
        protocol = self._select_protocol(query, patient_data)

        protocol_decision = {
            "step": "protocol_selection",
            "timestamp": datetime.now().isoformat(),
            "selected_protocol": protocol["name"],
            "protocol_id": protocol["id"],
            "rationale": protocol["rationale"]
        }

        self.state["decisions"].append(protocol_decision)
        self.state["collected_info"]["protocol"] = protocol

        self.state["current_step"] = "exam_check"
        return self.state

    def step_exam_check(self) -> WorkflowState:
        """Passo 4: Verificação de exames pendentes"""
        logger.info("Passo 4: Verificação de exames pendentes")

        patient_data = self.state["patient_data"]
        protocol = self.state["collected_info"].get("protocol", {})

        # Verificar exames necessários vs realizados
        required_exams = self._get_required_exams(protocol)
        completed_exams = patient_data.get("completed_exams", [])

        pending_exams = [exam for exam in required_exams if exam not in completed_exams]
        if 'exams' in patient_data:
            required_exams = [e['exam_type'] for e in patient_data['exams']]
            pending_exams = [e['exam_type'] for e in patient_data['exams'] if e.get('status') in ('Pendente','Vencido')]

        exam_check = {
            "step": "exam_check",
            "timestamp": datetime.now().isoformat(),
            "required_exams": required_exams,
            "completed_exams": completed_exams,
            "pending_exams": pending_exams,
            "all_exams_complete": len(pending_exams) == 0
        }

        self.state["collected_info"]["exam_check"] = exam_check

        # Gerar alertas para exames pendentes
        if pending_exams:
            alert = {
                "type": "pending_exams",
                "exams": pending_exams,
                "message": f"Exames pendentes: {', '.join(pending_exams)}",
                "timestamp": datetime.now().isoformat()
            }
            self.state["alerts"].append(alert)

        self.state["current_step"] = "treatment_suggestion"
        return self.state

    def step_treatment_suggestion(self) -> WorkflowState:
        """Passo 5: Sugestão de tratamento (com validação humana)"""
        logger.info("Passo 5: Sugestão de tratamento")

        patient_data = self.state["patient_data"]
        protocol = self.state["collected_info"].get("protocol", {})
        exam_check = self.state["collected_info"].get("exam_check", {})

        # Gerar sugestões de tratamento
        suggestions = self._generate_treatment_suggestions(patient_data, protocol, exam_check)

        treatment_decision = {
            "step": "treatment_suggestion",
            "timestamp": datetime.now().isoformat(),
            "suggestions": suggestions,
            "requires_human_validation": True,  # Sempre requer validação
            "validation_required": True
        }

        self.state["decisions"].append(treatment_decision)
        self.state["recommendations"] = suggestions
        self.state["requires_human_intervention"] = True

        self.state["current_step"] = "completion"
        return self.state

    def step_completion(self) -> WorkflowState:
        """Passo 6: Finalização do workflow"""
        logger.info("Passo 6: Finalização do workflow")

        completion_summary = {
            "step": "completion",
            "timestamp": datetime.now().isoformat(),
            "total_steps": len(self.state["decisions"]),
            "total_alerts": len(self.state["alerts"]),
            "total_recommendations": len(self.state["recommendations"]),
            "requires_human_intervention": self.state["requires_human_intervention"],
            "workflow_completed": True
        }

        self.state["decisions"].append(completion_summary)
        self.state["completed"] = True
        self.state["current_step"] = "completed"

        logger.info(f"Workflow concluído. Intervenção humana necessária: {self.state['requires_human_intervention']}")
        return self.state

    def _evaluate_vital_signs(self, patient_data: Dict) -> Dict[str, Any]:
        """Avalia sinais vitais do paciente"""
        vital_signs = patient_data.get("vital_signs", {})

        evaluation = {
            "blood_pressure": self._evaluate_blood_pressure(vital_signs.get("blood_pressure")),
            "heart_rate": self._evaluate_heart_rate(vital_signs.get("heart_rate")),
            "temperature": self._evaluate_temperature(vital_signs.get("temperature")),
            "oxygen_saturation": self._evaluate_oxygen_saturation(vital_signs.get("oxygen_saturation"))
        }

        return evaluation

    def _evaluate_blood_pressure(self, bp: str) -> Dict[str, str]:
        """Avalia pressão arterial"""
        if not bp:
            return {"status": "unknown", "interpretation": "Dados não disponíveis"}

        try:
            systolic, diastolic = map(int, bp.split("/"))

            if systolic >= 180 or diastolic >= 120:
                return {"status": "critical", "interpretation": "Crise hipertensiva"}
            elif systolic >= 140 or diastolic >= 90:
                return {"status": "high", "interpretation": "Hipertensão"}
            elif systolic >= 120 or diastolic >= 80:
                return {"status": "elevated", "interpretation": "Pré-hipertensão"}
            else:
                return {"status": "normal", "interpretation": "Normal"}
        except:
            return {"status": "error", "interpretation": "Formato inválido"}

    def _evaluate_heart_rate(self, hr: int) -> Dict[str, str]:
        """Avalia frequência cardíaca"""
        if not hr:
            return {"status": "unknown", "interpretation": "Dados não disponíveis"}

        if hr < 60:
            return {"status": "low", "interpretation": "Bradicardia"}
        elif hr > 100:
            return {"status": "high", "interpretation": "Taquicardia"}
        else:
            return {"status": "normal", "interpretation": "Normal"}

    def _evaluate_temperature(self, temp: float) -> Dict[str, str]:
        """Avalia temperatura"""
        if not temp:
            return {"status": "unknown", "interpretation": "Dados não disponíveis"}

        if temp >= 39:
            return {"status": "high", "interpretation": "Febre alta"}
        elif temp >= 38:
            return {"status": "elevated", "interpretation": "Febre"}
        elif temp < 36:
            return {"status": "low", "interpretation": "Hipotermia"}
        else:
            return {"status": "normal", "interpretation": "Normal"}

    def _evaluate_oxygen_saturation(self, sat: int) -> Dict[str, str]:
        """Avalia saturação de oxigênio"""
        if not sat:
            return {"status": "unknown", "interpretation": "Dados não disponíveis"}

        if sat < 90:
            return {"status": "critical", "interpretation": "Hipoxemia grave"}
        elif sat < 95:
            return {"status": "low", "interpretation": "Hipoxemia leve"}
        else:
            return {"status": "normal", "interpretation": "Normal"}

    def _identify_risk_factors(self, patient_data: Dict) -> List[str]:
        """Identifica fatores de risco"""
        risk_factors = []

        medical_history = patient_data.get("medical_history", [])
        if "hipertensão" in str(medical_history).lower():
            risk_factors.append("Hipertensão")
        if "diabetes" in str(medical_history).lower():
            risk_factors.append("Diabetes")
        if "cardíaco" in str(medical_history).lower():
            risk_factors.append("Histórico cardíaco")

        age = patient_data.get("age", 0)
        if age > 65:
            risk_factors.append("Idade avançada")

        return risk_factors

    def _classify_priority(self, vital_signs: Dict) -> TriagePriority:
        """Classifica prioridade de triagem"""
        critical_signs = [
            vital_signs.get("blood_pressure", {}).get("status") == "critical",
            vital_signs.get("heart_rate", {}).get("status") in ["critical", "high"],
            vital_signs.get("oxygen_saturation", {}).get("status") == "critical"
        ]

        if any(critical_signs):
            return TriagePriority.EMERGENCY

        high_priority = [
            vital_signs.get("blood_pressure", {}).get("status") == "high",
            vital_signs.get("temperature", {}).get("status") == "high"
        ]

        if any(high_priority):
            return TriagePriority.HIGH

        return TriagePriority.MEDIUM

    def _select_protocol(self, query: str, patient_data: Dict) -> Dict[str, str]:
        """Seleciona protocolo adequado"""
        query_lower = query.lower()
        if 'documentary_response' in patient_data:
            documents = [s for s in patient_data['documentary_response']['sources'] if s.get('source_type') == 'documento']
            if not documents:
                return {'id':'SEM_EVIDENCIA','name':'Sem orientação documental suficiente',
                        'rationale':'Não inferir conduta; conferir os documentos e o contexto com o responsável.'}
            source = documents[0]
            return {'id':source['id'],'name':source['title'],'rationale':source['quoted_text']}

        if "hipertensão" in query_lower or "pressão" in query_lower:
            return {
                "id": "PROTOC_001",
                "name": "Protocolo de Triagem de Hipertensão",
                "rationale": "Query relacionada a hipertensão/pressão arterial"
            }
        elif "diabetes" in query_lower or "glicemia" in query_lower:
            return {
                "id": "PROTOC_002",
                "name": "Protocolo de Manejo de Diabetes Tipo 2",
                "rationale": "Query relacionada a diabetes/glicemia"
            }
        elif "dor torácica" in query_lower or "peito" in query_lower:
            return {
                "id": "PROTOC_003",
                "name": "Protocolo de Dor Torácica",
                "rationale": "Query relacionada a dor torácica"
            }
        else:
            return {
                "id": "PROTOC_GENERAL",
                "name": "Protocolo Geral de Triagem",
                "rationale": "Protocolo genérico para casos não específicos"
            }

    def _get_required_exams(self, protocol: Dict) -> List[str]:
        """Retorna exames requeridos pelo protocolo"""
        protocol_id = protocol.get("id", "")

        exams_map = {
            "PROTOC_001": ["Pressão arterial", "Eletrocardiograma", "Exames de sangue"],
            "PROTOC_002": ["Glicemia de jejum", "Hemoglobina glicada", "Perfil lipídico"],
            "PROTOC_003": ["Eletrocardiograma", "Enzimas cardíacas", "Raio-X tórax"],
            "PROTOC_GENERAL": ["Exames básicos de triagem"]
        }

        return exams_map.get(protocol_id, ["Exames básicos"])

    def _generate_treatment_suggestions(self, patient_data: Dict, protocol: Dict, exam_check: Dict) -> List[Dict]:
        """Gera sugestões de tratamento"""
        suggestions = []
        if 'documentary_response' in patient_data:
            if exam_check.get('pending_exams'):
                suggestions.append({'type':'recorded_exam_review','priority':'revisão',
                    'description':'Revisar pendências administrativas registradas: '+', '.join(exam_check['pending_exams']),
                    'requires_validation':True,
                    'source_ids':[s['id'] for s in patient_data['record_sources'] if s['section_id']=='exams']})
            for source in patient_data['documentary_response']['sources']:
                suggestions.append({'type':'documentary_reference','priority':'revisão',
                    'description':source['quoted_text'],'requires_validation':True,'source_ids':[source['id']]})
            if not patient_data['documentary_response']['sources']:
                suggestions.append({'type':'missing_evidence','priority':'pendente',
                    'description':'Sem evidência documental para propor conduta. Completar dados/fontes e solicitar avaliação profissional.',
                    'requires_validation':True,'source_ids':[]})
            return suggestions

        # Sugestão baseada em exames pendentes
        if exam_check.get("pending_exams"):
            suggestions.append({
                "type": "exam_completion",
                "priority": "high",
                "description": f"Realizar exames pendentes: {', '.join(exam_check['pending_exams'])}",
                "requires_validation": True
            })

        # Sugestão baseada no protocolo
        protocol_id = protocol.get("id", "")
        if protocol_id == "PROTOC_001":
            suggestions.append({
                "type": "lifestyle_modification",
                "priority": "medium",
                "description": "Recomendar mudanças no estilo de vida: redução de sódio, exercícios físicos",
                "requires_validation": True
            })
        elif protocol_id == "PROTOC_002":
            suggestions.append({
                "type": "medication_review",
                "priority": "high",
                "description": "Revisar medicação atual para diabetes",
                "requires_validation": True
            })

        return suggestions

    def execute_full_workflow(self, patient_data: Dict, query: str) -> WorkflowState:
        """Executa workflow completo"""
        initial = self._initial_state()
        initial.update(patient_data=deepcopy(patient_data), query=query)
        result = self.graph.invoke(initial)
        self.state = deepcopy(result)
        return result


def main():
    """Função principal para testar o workflow"""

    # Criar grafo de workflow
    workflow = MedicalWorkflowGraph()

    # Dados de exemplo do paciente
    patient_data = {
        "id": "patient_001",
        "age": 45,
        "gender": "M",
        "vital_signs": {
            "blood_pressure": "145/95",
            "heart_rate": 88,
            "temperature": 36.5,
            "oxygen_saturation": 98
        },
        "medical_history": ["Hipertensão familiar"],
        "completed_exams": ["Pressão arterial"]
    }

    query = "Paciente com pressão alta, como proceder?"

    # Executar workflow completo
    result = workflow.execute_full_workflow(patient_data, query)

    # Exibir resultados
    print("\n" + "="*60)
    print("RESULTADO DO WORKFLOW MÉDICO")
    print("="*60)

    print(f"\nStatus: {'Concluído' if result['completed'] else 'Incompleto'}")
    print(f"Intervenção humana necessária: {result['requires_human_intervention']}")

    print(f"\nDecisões tomadas ({len(result['decisions'])}):")
    for decision in result['decisions']:
        print(f"- {decision['step']}: {decision.get('priority', decision.get('selected_protocol', 'N/A'))}")

    print(f"\nAlertas gerados ({len(result['alerts'])}):")
    for alert in result['alerts']:
        print(f"- [{alert['type']}]: {alert['message']}")

    print(f"\nRecomendações ({len(result['recommendations'])}):")
    for rec in result['recommendations']:
        print(f"- [{rec['type']}]: {rec['description']}")

    print(f"\nInformações coletadas:")
    for key, value in result['collected_info'].items():
        print(f"- {key}: {type(value).__name__}")


if __name__ == "__main__":
    main()
