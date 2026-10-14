"""
Pipeline LangChain para Assistente Médico
Integra LLM customizada com base de conhecimento e sistemas de segurança
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from collections import deque

from medical_assistant.safety.audit import AuditLogger
from medical_assistant.safety.guardrails import SecuritySystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleMemory:
    """Simples sistema de memória para conversação"""

    def __init__(self, max_length: int = 10):
        self.conversation = deque(maxlen=max_length)

    def save_context(self, inputs: Dict, outputs: Dict):
        """Salva contexto da conversa"""
        self.conversation.append({
            "input": inputs.get("input", ""),
            "output": outputs.get("output", ""),
            "timestamp": datetime.now().isoformat()
        })

    def load_memory_variables(self, inputs: Dict = None) -> Dict:
        """Carrega variáveis de memória"""
        return {"chat_history": list(self.conversation)}

    def clear(self):
        """Limpa memória"""
        self.conversation.clear()


class MedicalAssistantLangChain:
    """Assistente médico usando LangChain"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.medical_knowledge = self._load_medical_knowledge()
        self.memory = SimpleMemory(max_length=10)
        self.security_system = SecuritySystem()
        self.audit_logger = AuditLogger()

    def _load_medical_knowledge(self) -> Dict[str, Any]:
        """Carrega base de conhecimento médico"""
        knowledge = {
            "protocols": [],
            "faqs": [],
            "procedures": []
        }

        base_dir = Path(__file__).resolve().parents[3]
        dataset_path = base_dir / "data/curated/dataset_medico_anonimizado.json"

        if dataset_path.exists():
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)

            knowledge["protocols"] = dataset.get("protocolos", [])
            knowledge["faqs"] = dataset.get("perguntas_frequentes", [])
            knowledge["procedures"] = dataset.get("procedimentos_internos", [])

            logger.info(f"Base de conhecimento carregada: {len(knowledge['protocols'])} protocolos, "
                       f"{len(knowledge['faqs'])} FAQs")

        return knowledge

    def _create_medical_prompt(self, user_query: str, patient_context: Dict = None) -> str:
        """Cria prompt especializado para contexto médico"""

        # Sistema base
        system_content = """Você é um assistente médico especializado do hospital.

SUA FUNÇÃO:
- Auxiliar na triagem e condutas clínicas baseado em protocolos internos
- Responder dúvidas sobre procedimentos e protocolos médicos
- Contextualizar informações com dados de pacientes quando fornecidos
- Sugerir abordagens baseadas em evidências, mas SEMPRE com validação humana

LIMITAÇÕES IMPORTANTES:
- NUNCA prescrever medicamentos diretamente sem validação médica
- NUNCA dar diagnóstico definitivo sem exames completos
- SEMPRE indicar quando validação humana é necessária
- SEMPRE citar fontes (protocolos, diretrizes) das informações
"""

        # Adicionar contexto do paciente se disponível
        if patient_context:
            system_content += f"""
DADOS DO PACIENTE:
- Idade: {patient_context.get('idade', 'N/A')}
- Gênero: {patient_context.get('genero', 'N/A')}
- Queixa principal: {patient_context.get('queixa', 'N/A')}
- Histórico: {patient_context.get('historico', 'N/A')}
"""

        # Adicionar conhecimento relevante
        relevant_knowledge = self._search_relevant_knowledge(user_query)
        if relevant_knowledge:
            system_content += "\nINFORMAÇÕES RELEVANTES DOS PROTOCOLOS:\n"
            for item in relevant_knowledge:
                system_content += f"[{item['type'].upper()}]: {item['content'][:500]}...\n"

        return system_content

    def _search_relevant_knowledge(self, query: str) -> List[Dict]:
        """Busca conhecimento relevante (RAG simples)"""
        relevant_items = []
        query_lower = query.lower()

        # Buscar em FAQs
        for faq in self.medical_knowledge["faqs"]:
            if any(keyword in faq["pergunta"].lower() for keyword in query_lower.split()):
                relevant_items.append({
                    "type": "faq",
                    "content": f"P: {faq['pergunta']}\nR: {faq['resposta']}"
                })

        # Buscar em protocolos
        for protocol in self.medical_knowledge["protocols"]:
            if any(keyword in protocol["titulo"].lower() or keyword in protocol["conteudo"].lower()
                   for keyword in query_lower.split()):
                relevant_items.append({
                    "type": "protocol",
                    "content": f"Protocolo: {protocol['titulo']}\n{protocol['conteudo'][:800]}"
                })

        return relevant_items[:3]

    def process_query(self, user_query: str, patient_context: Dict = None) -> Dict[str, Any]:
        """Processa query do usuário com pipeline LangChain completo"""

        # Registrar query para auditoria
        query_id = self.audit_logger.log_query(user_query, patient_context)

        # Verificar segurança
        security_check = self.security_system.check_query(user_query)
        if not security_check["allowed"]:
            logger.warning(f"Query bloqueada por segurança: {security_check['reason']}")
            return {
                "response": security_check["message"],
                "requires_validation": True,
                "sources": [],
                "security_blocked": True,
                "query_id": query_id
            }

        # Criar prompt
        system_prompt = self._create_medical_prompt(user_query, patient_context)

        # Simular resposta do LLM (substituir por chamada real ao modelo)
        llm_response = self._simulate_llm_response(system_prompt, user_query)

        # Verificar segurança da resposta
        response_check = self.security_system.check_response(llm_response)
        if not response_check["allowed"]:
            logger.warning(f"Resposta bloqueada por segurança: {response_check['reason']}")
            llm_response = response_check["message"]

        # Buscar fontes usadas
        sources = self._search_relevant_knowledge(user_query)
        source_types = [item["type"] for item in sources]

        # Determinar se requer validação humana
        requires_validation = self.security_system.requires_human_validation(user_query, llm_response)

        # Registrar resposta para auditoria
        self.audit_logger.log_response(query_id, llm_response, source_types, requires_validation)

        # Atualizar memória
        self.memory.save_context(
            {"input": user_query},
            {"output": llm_response}
        )

        return {
            "response": llm_response,
            "requires_validation": requires_validation,
            "sources": source_types,
            "security_blocked": False,
            "query_id": query_id,
            "confidence": 0.85  # Simulado
        }

    def _simulate_llm_response(self, system_prompt: str, user_query: str) -> str:
        """Simula resposta do LLM (substituir por chamada real)"""
        # Em produção, isso seria uma chamada ao modelo Ollama
        relevant_knowledge = self._search_relevant_knowledge(user_query)

        if relevant_knowledge:
            base_response = f"Com base nos protocolos do hospital, "
            if "hipertensão" in user_query.lower():
                base_response += "a hipertensão é frequentemente assintomática, por isso é chamada de 'assassina silenciosa'. O diagnóstico deve ser feito através da medição regular da pressão arterial. Valores acima de 140/90 mmHg indicam hipertensão e requerem acompanhamento médico."
            elif "diabetes" in user_query.lower():
                base_response += "o manejo da diabetes Tipo 2 envolve monitoramento da glicemia, mudanças no estilo de vida e, quando necessário, medicação como metformina. O controle da hemoglobina glicada (< 7%) é essencial."
            else:
                base_response += f"recomendo seguir os protocolos internos para {user_query}. Sempre valide com um médico antes de qualquer procedimento."
        else:
            base_response = "Baseado na sua pergunta, recomendo consultar os protocolos específicos do hospital. Para mais detalhes, preciso de informações adicionais sobre o contexto clínico."

        return base_response

    def get_conversation_history(self) -> List[Dict]:
        """Retorna histórico de conversação"""
        return self.memory.load_memory_variables({}).get("chat_history", [])


def main():
    """Função principal para testar o assistente médico"""

    # Criar assistente
    assistant = MedicalAssistantLangChain()

    # Testar queries
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

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        result = assistant.process_query(query, patient_context)

        print(f"Resposta: {result['response']}")
        print(f"Requer validação: {result['requires_validation']}")
        print(f"Fontes: {result['sources']}")
        print(f"ID da query: {result['query_id']}")

        if result.get('security_blocked'):
            print("⚠️  RESPOSTA BLOQUEADA POR SEGURANÇA")


if __name__ == "__main__":
    main()
