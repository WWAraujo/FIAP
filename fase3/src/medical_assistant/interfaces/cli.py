"""
Interface Principal do Assistente Médico - Fase 3 Tech Challenge
Integra todos os componentes: RAG, LangChain, LangGraph, Base de Dados, Segurança
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
import sys

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from medical_assistant.application.assistant import MedicalAssistantLangChain
from medical_assistant.application.workflow import MedicalWorkflowGraph
from medical_assistant.clinical.patient_database import PatientDatabase, PatientContextBuilder
from medical_assistant.application.contracts import PatientContext

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MedicalAssistantInterface:
    """Interface principal do assistente médico"""

    def __init__(self):
        logger.info("Inicializando Assistente Médico - Fase 3 Tech Challenge")

        # Inicializar componentes
        self.langchain_assistant = MedicalAssistantLangChain()
        self.workflow_graph = MedicalWorkflowGraph()
        self.patient_db = PatientDatabase()
        self.context_builder = PatientContextBuilder(self.patient_db)

        logger.info("Todos os componentes inicializados com sucesso")

    def process_medical_query(self, query: str, patient_id: str = None) -> Dict[str, Any]:
        """Processa query médica completa com todos os componentes"""

        context = PatientContext.from_database(self.patient_db, patient_id) if patient_id else None
        response = self.langchain_assistant.process_query(query, context)
        return {
            "query": query,
            "patient_id": patient_id,
            "langchain_response": response,
            "components_used": ["FT-11", "RAG", "LangChain"] + (["patient_database"] if context else []),
            "summary": {"requires_validation": response["requires_validation"],
                        "security_blocked": response["security_blocked"]},
        }

    def interactive_mode(self):
        """Modo interativo para testes"""
        print("="*70)
        print("ASSISTENTE MÉDICO - FASE 3 TECH CHALLENGE")
        print("Integração: RAG + LangChain + LangGraph + Base de Dados + Segurança")
        print("="*70)

        while True:
            print("\n" + "-"*70)
            print("Opções:")
            print("1. Fazer pergunta médica")
            print("2. Consultar paciente")
            print("3. Executar workflow completo")
            print("4. Ver logs de auditoria")
            print("5. Sair")

            choice = input("\nEscolha uma opção (1-5): ").strip()

            if choice == "1":
                self._handle_medical_query()
            elif choice == "2":
                self._handle_patient_query()
            elif choice == "3":
                self._handle_workflow_execution()
            elif choice == "4":
                self._handle_audit_logs()
            elif choice == "5":
                print("Encerrando assistente médico...")
                break
            else:
                print("Opção inválida!")

    def _handle_medical_query(self):
        """Handle medical query option"""
        query = input("Digite sua pergunta médica: ").strip()
        if not query:
            print("Pergunta vazia!")
            return

        patient_id = input("ID do paciente (opcional, pressione Enter para pular): ").strip()
        patient_id = patient_id if patient_id else None

        try:
            result = self.process_medical_query(query, patient_id)
        except (ValueError, RuntimeError) as exc:
            print(f"Erro na consulta: {exc}")
            return

        print("\n" + "="*70)
        print("RESULTADO DA CONSULTA")
        print("="*70)

        print(f"\nQuery: {result['query']}")
        print(f"Componentes utilizados: {', '.join(result['components_used'])}")

        if "langchain_response" in result:
            lc_response = result["langchain_response"]
            print(f"\nResposta LangChain:")
            print(f"{'-'*70}")
            print(lc_response["response"])
            print(f"\nRequer validação: {lc_response['requires_validation']}")
            print(f"Fontes: {lc_response['sources']}")
            print(f"ID da query: {lc_response['query_id']}")

            if lc_response.get('security_blocked'):
                print("⚠️  RESPOSTA BLOQUEADA POR SEGURANÇA")

        if "workflow_result" in result:
            wf_result = result["workflow_result"]
            print(f"\nWorkflow executado:")
            print(f"Decisões: {len(wf_result['decisions'])}")
            print(f"Alertas: {len(wf_result['alerts'])}")
            print(f"Recomendações: {len(wf_result['recommendations'])}")

            if wf_result['alerts']:
                print("\nAlertas:")
                for alert in wf_result['alerts']:
                    print(f"- [{alert['type']}]: {alert['message']}")

    def _handle_patient_query(self):
        """Handle patient query option"""
        patient_id = input("ID do paciente: ").strip()
        if not patient_id:
            print("ID inválido!")
            return

        try:
            summary = self.patient_db.get_patient_summary(patient_id)

            if "error" in summary:
                print(f"Erro: {summary['error']}")
                return

            print("\n" + "="*70)
            print(f"PRONTUÁRIO DO PACIENTE: {summary['patient_info']['name']}")
            print("="*70)

            print(f"\nID: {summary['patient_info']['id']}")
            print(f"Idade: {summary['patient_info']['age']} anos")
            print(f"Gênero: {summary['patient_info']['gender']}")

            print("\nHistórico Médico:")
            for condition in summary["medical_history"]:
                print(f"- {condition['condition']}: {condition['status']}")

            print("\nSinais Vitais Recentes:")
            for vital in summary["recent_vital_signs"]:
                print(f"- {vital['recorded_date'] if 'recorded_date' in vital else vital['recorded_at']}: "
                      f"PA {vital['blood_pressure']}, FC {vital['heart_rate']}")

            print("\nMedicações Ativas:")
            for med in summary["current_medications"]:
                print(f"- {med['medication_name']} {med['dosage']} ({med['frequency']})")

        except Exception as e:
            print(f"Erro ao consultar paciente: {e}")

    def _handle_workflow_execution(self):
        """Handle workflow execution option"""
        patient_id = input("ID do paciente: ").strip()
        if not patient_id:
            print("ID inválido!")
            return

        query = input("Query/Descrição do caso: ").strip()
        if not query:
            print("Query vazia!")
            return

        try:
            patient_data = self.patient_db.get_patient_summary(patient_id)
            result = self.workflow_graph.execute_full_workflow(patient_data, query)

            print("\n" + "="*70)
            print("RESULTADO DO WORKFLOW")
            print("="*70)

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

        except Exception as e:
            print(f"Erro ao executar workflow: {e}")

    def _handle_audit_logs(self):
        """Handle audit logs option"""
        log_file = PROJECT_DIR / "logs" / "audit.log"

        if not log_file.exists():
            print("Nenhum log de auditoria encontrado.")
            return

        print("\n" + "="*70)
        print("LOGS DE AUDITORIA")
        print("="*70)

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Mostrar últimas 10 entradas
            recent_lines = lines[-10:] if len(lines) > 10 else lines

            for line in recent_lines:
                log_entry = json.loads(line)
                print(f"\n[{log_entry['timestamp']}] {log_entry['type'].upper()}")
                if log_entry['type'] == 'query':
                    print(f"Query: {log_entry['query']}")
                elif log_entry['type'] == 'response':
                    print(f"Query ID: {log_entry['query_id']}")
                    print(f"Fontes: {log_entry['sources']}")
                    print(f"Validação necessária: {log_entry['requires_validation']}")

        except Exception as e:
            print(f"Erro ao ler logs: {e}")


def main():
    """Função principal"""
    try:
        # Criar interface
        assistant = MedicalAssistantInterface()

        # Iniciar modo interativo
        assistant.interactive_mode()

    except KeyboardInterrupt:
        print("\n\nPrograma interrompido pelo usuário.")
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        print(f"Erro fatal: {e}")


if __name__ == "__main__":
    main()
