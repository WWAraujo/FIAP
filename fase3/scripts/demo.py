"""
Script de Demonstração Automatizada - Fase 3 Tech Challenge
Demonstra todas as funcionalidades do assistente médico sem interação manual
"""

import json
import logging
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from medical_assistant.application.assistant import MedicalAssistantLangChain
from medical_assistant.application.workflow import MedicalWorkflowGraph
from medical_assistant.clinical.patient_database import PatientDatabase, PatientContextBuilder
from medical_assistant.retrieval.ollama_adapter import OllamaMedicalAdapter
from medical_assistant.training.config import FineTuningConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_benchmark_results():
    """Demonstra resultados do benchmark"""
    print("\n" + "="*70)
    print("DEMONSTRAÇÃO 1: RESULTADOS DO BENCHMARK OLLAMA")
    print("="*70)

    print("\nModelo selecionado: gemma3:4b (vencedor do benchmark)")
    print("\nComparativo de modelos (benchmark 20260912T164628Z):")
    print("| Modelo      | Casos | Latência média | p95   | Tokens/s | Segurança |")
    print("|-------------|-------|----------------|-------|----------|-----------|")
    print("| gemma3:4b   | 21    | 3415 ms        | 3538 ms| 77.9     | 21/21     |")
    print("| mistral:7b  | 21    | 4647 ms        | 7086 ms| 58.4     | 21/21     |")
    print("| llama3.1:8b | 21    | 5616 ms        | 7970 ms| 55.1     | 21/21     |")
    print("| qwen3:4b    | 21    | 9254 ms        | 9039 ms| 78.5     | 21/21     |")
    print("| qwen:14b    | 21    | 6623 ms        | 9881 ms| 34.0     | 21/21     |")
    print("| qwen3:8b    | 21    | 9644 ms        | 12225 ms| 51.5     | 21/21     |")

    print("\nJustificativa da seleção:")
    print("- Melhor consistência (menor p95: 3538ms)")
    print("- 100% de segurança (21/21 checks)")
    print("- Respostas em português (diferente de outros modelos)")
    print("- Respostas diretas e concisas")
    print("- Melhor throughput entre os modelos rápidos")


def demo_dataset_preprocessing():
    """Demonstra dataset e preprocessing"""
    print("\n" + "="*70)
    print("DEMONSTRAÇÃO 2: DATASET E PREPROCESSING")
    print("="*70)

    # Carregar dataset anonimizado
    dataset_path = PROJECT_DIR / "data" / "curated" / "dataset_medico_anonimizado.json"
    if dataset_path.exists():
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)

            print(f"\nDataset carregado: {dataset_path}")
            print(f"Protocolos: {len(dataset.get('protocols', []))}")
            print(f"FAQs: {len(dataset.get('perguntas_frequentes', []))}")
            print(f"Procedimentos: {len(dataset.get('procedimentos_internos', []))}")

            protocols = dataset.get('protocols', [])
            if protocols:
                print("\nExemplo de protocolo:")
                print(f"Título: {protocols[0]['titulo']}")
                print(f"Conteúdo (primeiros 200 chars): {protocols[0]['conteudo'][:200]}...")

            print("\nDados de fine-tuning:")
            train_path = PROJECT_DIR / "data" / "splits" / "train_data.jsonl"
            test_path = PROJECT_DIR / "data" / "splits" / "test_data.jsonl"

            if train_path.exists():
                with open(train_path, 'r', encoding='utf-8') as f:
                    train_lines = f.readlines()
                print(f"Exemplos de treino: {len(train_lines)}")

            if test_path.exists():
                with open(test_path, 'r', encoding='utf-8') as f:
                    test_lines = f.readlines()
                print(f"Exemplos de teste: {len(test_lines)}")

        except Exception as e:
            print(f"Erro ao carregar dataset: {e}")
    else:
        print("Dataset não encontrado. Execute o preprocessing primeiro.")


def demo_langchain_pipeline():
    """Demonstra pipeline LangChain"""
    print("\n" + "="*70)
    print("DEMONSTRAÇÃO 3: PIPELINE LANGCHAIN")
    print("="*70)

    assistant = MedicalAssistantLangChain()

    test_queries = [
        "Quais são os sintomas de hipertensão?",
        "Como proceder na triagem de hipertensão?",
        "Preciso prescrever um medicamento para pressão alta"
    ]

    patient_context = {
        "idade": 45,
        "genero": "M",
        "queixa": "Dor de cabeça frequente",
        "historico": "Hipertensão familiar"
    }

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Teste {i}: {query} ---")
        result = assistant.process_query(query, patient_context)

        print(f"Resposta: {result['response'][:150]}...")
        print(f"Requer validação: {result['requires_validation']}")
        print(f"Fontes: {result['sources']}")
        print(f"ID da query: {result['query_id']}")

        if result.get('security_blocked'):
            print("⚠️  RESPOSTA BLOQUEADA POR SEGURANÇA")


def demo_workflow_graph():
    """Demonstra workflow LangGraph"""
    print("\n" + "="*70)
    print("DEMONSTRAÇÃO 4: WORKFLOW LANGGRAPH")
    print("="*70)

    workflow = MedicalWorkflowGraph()

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

    print(f"\nExecutando workflow para paciente: {patient_data['id']}")
    print(f"Query: {query}")

    result = workflow.execute_full_workflow(patient_data, query)

    print(f"\nStatus: {'Concluído' if result['completed'] else 'Incompleto'}")
    print(f"Intervenção humana necessária: {result['requires_human_intervention']}")

    print(f"\nDecisões tomadas ({len(result['decisions'])}):")
    for decision in result['decisions']:
        step = decision['step']
        priority = decision.get('priority', decision.get('selected_protocol', 'N/A'))
        print(f"- {step}: {priority}")

    print(f"\nAlertas gerados ({len(result['alerts'])}):")
    for alert in result['alerts']:
        print(f"- [{alert['type']}]: {alert['message']}")

    print(f"\nRecomendações ({len(result['recommendations'])}):")
    for rec in result['recommendations']:
        print(f"- [{rec['type']}]: {rec['description']}")


def demo_patient_database():
    """Demonstra base de dados de prontuários"""
    print("\n" + "="*70)
    print("DEMONSTRAÇÃO 5: BASE DE DADOS DE PRONTUÁRIOS")
    print("="*70)

    db = PatientDatabase()

    patient_id = "patient_001"

    print(f"\nConsultando paciente: {patient_id}")

    patient_info = db.get_patient_info(patient_id)
    if patient_info:
        print(f"Nome: {patient_info['name']}")
        print(f"Idade: {patient_info['age']} anos")
        print(f"Gênero: {patient_info['gender']}")

    print(f"\nHistórico médico:")
    medical_history = db.get_medical_history(patient_id)
    for condition in medical_history:
        print(f"- {condition['condition']}: {condition['status']}")

    print(f"\nSinais vitais recentes:")
    vital_signs = db.get_vital_signs(patient_id, limit=3)
    for vital in vital_signs:
        print(f"- {vital['recorded_at']}: PA {vital['blood_pressure']}, FC {vital['heart_rate']}")

    print(f"\nMedicações ativas:")
    medications = db.get_medications(patient_id, active_only=True)
    for med in medications:
        print(f"- {med['medication_name']} {med['dosage']} ({med['frequency']})")

    # Construir contexto para assistente
    context_builder = PatientContextBuilder(db)
    context = context_builder.build_context_for_assistant(patient_id)

    print(f"\nContexto formatado para assistente:")
    print(context[:500] + "...")


def demo_rag_system():
    """Demonstra sistema RAG"""
    print("\n" + "="*70)
    print("DEMONSTRAÇÃO 6: SISTEMA RAG (OLLAMA)")
    print("="*70)

    config = FineTuningConfig()
    adapter = OllamaMedicalAdapter(config)

    print(f"\nModelo base: {config.model_name}")
    print(f"Base de conhecimento carregada:")
    print(f"- Protocolos: {len(adapter.medical_knowledge_base['protocols'])}")
    print(f"- FAQs: {len(adapter.medical_knowledge_base['faqs'])}")
    print(f"- Procedimentos: {len(adapter.medical_knowledge_base['procedures'])}")

    print("\n⚠️  Nota: Ollama precisa estar rodando para respostas reais")
    print("Em modo de demonstração, o sistema está configurado mas não conectado")

    # Verificar configuração RAG
    rag_config_path = PROJECT_DIR / "models" / "fine_tuned" / "rag_system_config.json"
    if rag_config_path.exists():
        with open(rag_config_path, 'r', encoding='utf-8') as f:
            rag_config = json.load(f)
        print(f"\nConfiguração RAG carregada:")
        print(f"Modelo: {rag_config['model']}")
        print(f"Criação: {rag_config['creation_date']}")


def demo_audit_logs():
    """Demonstra logs de auditoria"""
    print("\n" + "="*70)
    print("DEMONSTRAÇÃO 7: LOGS DE AUDITORIA")
    print("="*70)

    log_file = PROJECT_DIR / "logs" / "audit.log"

    if log_file.exists():
        print(f"\nArquivo de log: {log_file}")

        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(f"Total de entradas: {len(lines)}")

        # Mostrar últimas 5 entradas
        recent_lines = lines[-5:] if len(lines) > 5 else lines

        print(f"\nÚltimas {len(recent_lines)} entradas:")
        for line in recent_lines:
            log_entry = json.loads(line)
            print(f"\n[{log_entry['timestamp']}] {log_entry['type'].upper()}")
            if log_entry['type'] == 'query':
                print(f"Query: {log_entry['query']}")
            elif log_entry['type'] == 'response':
                print(f"Query ID: {log_entry['query_id']}")
                print(f"Fontes: {log_entry['sources']}")
                print(f"Validação necessária: {log_entry['requires_validation']}")
    else:
        print("Nenhum log de auditoria encontrado ainda.")


def main():
    """Executa todas as demonstrações"""
    print("="*70)
    print("DEMONSTRAÇÃO COMPLETA - ASSISTENTE MÉDICO FASE 3")
    print("Tech Challenge FIAP - Pós-Tech em IA para Desenvolvedores")
    print("="*70)

    try:
        # Executar demonstrações
        demo_benchmark_results()
        demo_dataset_preprocessing()
        demo_langchain_pipeline()
        demo_workflow_graph()
        demo_patient_database()
        demo_rag_system()
        demo_audit_logs()

        print("\n" + "="*70)
        print("DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO")
        print("="*70)

        print("\nResumo dos componentes demonstrados:")
        print("✅ Seleção de modelo base (gemma3:4b - vencedor do benchmark)")
        print("✅ Dataset médico sintético e anonimizado")
        print("✅ Pipeline de preprocessing")
        print("✅ Sistema RAG (Ollama)")
        print("✅ Pipeline LangChain com segurança")
        print("✅ Workflow LangGraph automatizado")
        print("✅ Base de dados de prontuários")
        print("✅ Sistema de logging e auditoria")

        print("\nTodos os requisitos da Fase 3 foram implementados com sucesso!")

    except Exception as e:
        logger.error(f"Erro na demonstração: {e}")
        print(f"\nErro na demonstração: {e}")


if __name__ == "__main__":
    main()
