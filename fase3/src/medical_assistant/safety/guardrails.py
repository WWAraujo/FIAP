from typing import Dict, Any


class SecuritySystem:
    """Sistema de segurança e validação"""

    def __init__(self):
        self.forbidden_patterns = [
            "prescrever", "receitar", "diagnosticar definitivo",
            "procedimento invasivo sem supervisão"
        ]
        self.requires_validation_patterns = [
            "medicamento", "cirurgia", "diagnóstico", "tratamento"
        ]

    def check_query(self, query: str) -> Dict[str, Any]:
        """Verifica se a query é permitida"""
        query_lower = query.lower()

        for pattern in self.forbidden_patterns:
            if pattern in query_lower:
                return {
                    "allowed": False,
                    "reason": f"Contém termo proibido: {pattern}",
                    "message": f"Esta solicitação contém termos que requerem validação médica direta. Por favor, consulte um médico."
                }

        return {"allowed": True, "reason": "", "message": ""}

    def check_response(self, response: str) -> Dict[str, Any]:
        """Verifica se a resposta é segura"""
        response_lower = response.lower()

        for pattern in self.forbidden_patterns:
            if pattern in response_lower:
                return {
                    "allowed": False,
                    "reason": f"Resposta contém termo proibido: {pattern}",
                    "message": "Esta resposta foi modificada por segurança. Consulte um médico para orientação específica."
                }

        return {"allowed": True, "reason": "", "message": ""}

    def requires_human_validation(self, query: str, response: str) -> bool:
        """Determina se a resposta requer validação humana"""
        combined_text = (query + " " + response).lower()

        for pattern in self.requires_validation_patterns:
            if pattern in combined_text:
                return True

        return False
