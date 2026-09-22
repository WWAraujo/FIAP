"""
Fine-tuning/Adaptação de Modelo Ollama para Assistente Médico
Como Ollama não suporta fine-tuning tradicional, implementamos adaptação via RAG e System Prompts
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import requests
from medical_assistant.training.config import FineTuningConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaMedicalAdapter:
    """Adaptador para criar assistente médico especializado usando Ollama"""

    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = config.model_name
        self.medical_knowledge_base = self._load_medical_knowledge()

    def _load_medical_knowledge(self) -> Dict[str, Any]:
        """Carrega base de conhecimento médico do dataset processado"""
        knowledge_base = {
            "protocols": [],
            "faqs": [],
            "procedures": []
        }

        # Carregar dataset anonimizado
        base_dir = Path(__file__).resolve().parents[3]
        dataset_path = base_dir / "data/curated/dataset_medico_anonimizado.json"

        if dataset_path.exists():
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)

            knowledge_base["protocols"] = dataset.get("protocolos", [])
            knowledge_base["faqs"] = dataset.get("perguntas_frequentes", [])
            knowledge_base["procedures"] = dataset.get("procedimentos_internos", [])

            logger.info(f"Base de conhecimento carregada: {len(knowledge_base['protocols'])} protocolos, "
                       f"{len(knowledge_base['faqs'])} FAQs, {len(knowledge_base['procedures'])} procedimentos")

        return knowledge_base

    def _create_system_prompt(self, context: str = "") -> str:
        """Cria system prompt especializado para assistente médico"""
        base_prompt = """Você é um assistente médico especializado do hospital, treinado para auxiliar médicos e equipe de saúde.

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

BASE DE CONHECIMENTO DISPONÍVEL:
"""

        # Adicionar resumo dos protocolos disponíveis
        if self.medical_knowledge_base["protocols"]:
            base_prompt += "\nProtocolos disponíveis:\n"
            for protocol in self.medical_knowledge_base["protocols"]:
                base_prompt += f"- {protocol['titulo']}\n"

        # Adicionar contexto específico se fornecido
        if context:
            base_prompt += f"\nCONTEXTO ATUAL:\n{context}\n"

        return base_prompt

    def _search_relevant_knowledge(self, query: str, top_k: int = 3) -> List[Dict]:
        """Busca conhecimento relevante baseado na query (RAG simples)"""
        relevant_items = []
        query_lower = query.lower()

        # Buscar em FAQs
        for faq in self.medical_knowledge_base["faqs"]:
            if any(keyword in faq["pergunta"].lower() for keyword in query_lower.split()):
                relevant_items.append({
                    "type": "faq",
                    "content": f"P: {faq['pergunta']}\nR: {faq['resposta']}"
                })

        # Buscar em protocolos
        for protocol in self.medical_knowledge_base["protocols"]:
            if any(keyword in protocol["titulo"].lower() or keyword in protocol["conteudo"].lower()
                   for keyword in query_lower.split()):
                relevant_items.append({
                    "type": "protocol",
                    "content": f"Protocolo: {protocol['titulo']}\n{protocol['conteudo']}"
                })

        # Buscar em procedimentos
        for proc in self.medical_knowledge_base["procedures"]:
            if any(keyword in proc["nome"].lower() for keyword in query_lower.split()):
                steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(proc["passos"])])
                relevant_items.append({
                    "type": "procedure",
                    "content": f"Procedimento: {proc['nome']}\n{steps_text}"
                })

        return relevant_items[:top_k]

    def generate_response(self, user_query: str, patient_context: Dict = None) -> Dict[str, Any]:
        """Gera resposta usando modelo Ollama com contexto médico"""

        # Buscar conhecimento relevante
        relevant_knowledge = self._search_relevant_knowledge(user_query)

        # Criar contexto do paciente se fornecido
        patient_context_str = ""
        if patient_context:
            patient_context_str = f"""
DADOS DO PACIENTE:
- Idade: {patient_context.get('idade', 'N/A')}
- Gênero: {patient_context.get('genero', 'N/A')}
- Queixa principal: {patient_context.get('queixa', 'N/A')}
- Histórico: {patient_context.get('historico', 'N/A')}
"""

        # Criar contexto de conhecimento
        knowledge_context = ""
        if relevant_knowledge:
            knowledge_context = "\nINFORMAÇÕES RELEVANTES DA BASE DE CONHECIMENTO:\n"
            for item in relevant_knowledge:
                knowledge_context += f"[{item['type'].upper()}]: {item['content']}\n\n"

        # Criar system prompt completo
        system_prompt = self._create_system_prompt(patient_context_str + knowledge_context)

        # Preparar payload para Ollama
        payload = {
            "model": self.model,
            "prompt": f"{system_prompt}\n\nPERGUNTA: {user_query}",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 1000
            }
        }

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=60)

            if response.status_code == 200:
                data = response.json()
                result = {
                    "response": data.get("response", ""),
                    "sources": [item["type"] for item in relevant_knowledge],
                    "model_used": self.model,
                    "context_used": bool(relevant_knowledge or patient_context)
                }

                logger.info(f"Resposta gerada com sucesso. Fontes: {result['sources']}")
                return result
            else:
                logger.error(f"Erro na API Ollama: {response.status_code} - {response.text}")
                return {
                    "response": "Erro ao gerar resposta. Tente novamente.",
                    "sources": [],
                    "model_used": self.model,
                    "context_used": False
                }

        except Exception as e:
            logger.error(f"Exceção ao chamar Ollama: {e}")
            return {
                "response": f"Erro de conexão com o modelo: {str(e)}",
                "sources": [],
                "model_used": self.model,
                "context_used": False
            }

    def create_rag_system(self) -> None:
        """Cria sistema RAG completo para o assistente médico"""
        logger.info("Criando sistema RAG para assistente médico...")

        # Salvar configuração do sistema
        system_config = {
            "model": self.model,
            "knowledge_base_size": {
                "protocols": len(self.medical_knowledge_base["protocols"]),
                "faqs": len(self.medical_knowledge_base["faqs"]),
                "procedures": len(self.medical_knowledge_base["procedures"])
            },
            "system_prompt": self._create_system_prompt(),
            "creation_date": str(Path.cwd())
        }

        base_dir = Path(__file__).resolve().parents[3]
        config_path = base_dir / self.config.output_dir / "rag_system_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(system_config, f, ensure_ascii=False, indent=2)

        logger.info(f"Sistema RAG configurado em {config_path}")


def main():
    """Função principal para criar o assistente médico adaptado"""

    # Configuração
    config = FineTuningConfig()

    # Criar adaptador
    adapter = OllamaMedicalAdapter(config)

    # Criar sistema RAG
    adapter.create_rag_system()

    # Testar o sistema
    logger.info("Testando sistema com perguntas de exemplo...")

    test_queries = [
        "Quais são os sintomas de hipertensão?",
        "Como devo proceder na triagem de hipertensão?",
        "Descreva o procedimento de coleta de sangue"
    ]

    for query in test_queries:
        logger.info(f"\nTestando: {query}")
        result = adapter.generate_response(query)
        print(f"Resposta: {result['response'][:200]}...")
        print(f"Fontes: {result['sources']}")

    logger.info("Sistema de assistente médico configurado com sucesso!")


if __name__ == "__main__":
    main()
