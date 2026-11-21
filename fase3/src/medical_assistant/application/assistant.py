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
from medical_assistant.application.contracts import AssistantRequest, AssistantResponse
from medical_assistant.retrieval.ft11_client import FT11Client, InferenceUnavailable
from medical_assistant.clinical.patient_queries import allergy_query, bmi_query
from medical_assistant.safety.reviews import ReviewStore

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
        self.backend = FT11Client()
        self.memory = SimpleMemory(max_length=10)
        self.security_system = SecuritySystem()
        self.audit_logger = AuditLogger()

    def _reviewable(self, result, request):
        for source in result.sources:
            if source.source_type == 'prontuario' and (request.patient_context is None or
                    source.document_id != 'prontuario:'+request.patient_context.patient_id):
                raise InferenceUnavailable('Fonte de prontuário pertence a outro paciente.')
        payload = {"kind":"chat", "response":result.response,
                   "sources":[s.model_dump() for s in result.sources], "model_usage":result.model_usage,
                   "query":request.query, "review_eligible":bool(result.sources) and not result.security_blocked}
        result.review_id = ReviewStore().create(payload,
            patient_id=request.patient_context.patient_id if request.patient_context else None,
            query_id=result.query_id)
        result.review_status = 'pending'
        return result

    def _load_medical_knowledge(self) -> Dict[str, Any]:
        """Carrega base de conhecimento médico"""
        knowledge = {
            "protocols": [],
            "faqs": [],
            "procedures": []
        }

        base_dir = Path(__file__).resolve().parents[3]
        dataset_path = base_dir / "llm/data/curated/dataset_medico_anonimizado.json"

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
        """Consulta FT-11/RAG com contexto separado e fontes identificáveis."""

        request = AssistantRequest(query=user_query, patient_context=patient_context)
        context = request.patient_context.model_dump() if request.patient_context else None

        # Registrar query para auditoria
        query_id = self.audit_logger.log_query(request.query, context)

        # Verificar segurança
        security_check = self.security_system.check_query(user_query)
        if not security_check["allowed"]:
            logger.warning(f"Query bloqueada por segurança: {security_check['reason']}")
            self.audit_logger.log_response(query_id, security_check["message"], [], True)
            return AssistantResponse(response=security_check["message"], query_id=query_id,
                                     security_blocked=True).model_dump()

        patient_result = allergy_query(request, query_id)
        if patient_result is None:
            patient_result = bmi_query(request, query_id)
        if patient_result is not None:
            patient_result = self._reviewable(patient_result, request)
            self.audit_logger.log_response(query_id, patient_result.response,
                [source.model_dump() for source in patient_result.sources], True,
                model_usage={"response_origin": "prontuario_sqlite", "model_id": None,
                             "review_id":patient_result.review_id, "review_status":'pending'})
            self.memory.save_context({"input": user_query}, {"output": patient_result.response})
            return patient_result.model_dump()

        try:
            result = self.backend.invoke(request)
        except InferenceUnavailable as exc:
            self.audit_logger.log_response(query_id, "", [], True, error=str(exc))
            raise

        answer = result.answer
        llm_response = answer.dados.get("trecho") or "\n".join(answer.pendencias)
        if not llm_response:
            llm_response = "Não encontrei evidência suficiente nos documentos disponíveis para responder."

        # Verificar segurança da resposta
        response_check = self.security_system.check_response(llm_response)
        if not response_check["allowed"]:
            logger.warning(f"Resposta bloqueada por segurança: {response_check['reason']}")
            llm_response = response_check["message"]

        sources = result.sources if response_check["allowed"] else []

        output = self._reviewable(AssistantResponse(response=llm_response, query_id=query_id, sources=sources,
                                 security_blocked=not response_check["allowed"], model_usage=result.model_usage,
                                 run_id=result.run_id, patient_context_used=result.patient_context_used), request)

        # Registrar resposta para auditoria
        self.audit_logger.log_response(query_id, llm_response,
                                       [source.model_dump() for source in sources], True,
                                       model_usage={**result.model_usage, "runtime_run_id": result.run_id,
                                                    "patient_context_used": result.patient_context_used,
                                                    "review_id":output.review_id, "review_status":'pending'})

        # Atualizar memória
        self.memory.save_context(
            {"input": user_query},
            {"output": llm_response}
        )

        return output.model_dump()

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
        "patient_id": "demo_001",
        "idade": 45,
        "genero": "M",
        "historico": [{"condition": "Hipertensão familiar"}],
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
