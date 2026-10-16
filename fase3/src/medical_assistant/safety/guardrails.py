from typing import Dict, Any
import re
import unicodedata


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
        query_lower = ''.join(c for c in unicodedata.normalize('NFD', query.lower())
                              if unicodedata.category(c) != 'Mn')

        patterns = self.forbidden_patterns + [
            r'\bprescrev\w*', r'\breceit(?:e|ar|ando)\b', r'\bdiagnostiqu\w*',
            r'qual (?:e )?(?:o )?diagnostico', r'qual doenca (?:eu|ele|ela|esse paciente)',
            r'(?:qual|quanto|indique|recomende).{0,40}(?:dose|dosagem)',
            r'(?:devo|deve|posso).{0,30}(?:tomar|aumentar|reduzir)',
            r'ignore.{0,30}(?:instruc|regras)', r'(?:finja|declare).{0,40}aprovad',
        ]
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return {
                    "allowed": False,
                    "reason": f"Contém termo proibido: {pattern}",
                    "message": f"Esta solicitação contém termos que requerem validação médica direta. Por favor, consulte um médico."
                }

        return {"allowed": True, "reason": "", "message": ""}

    def check_response(self, response: str) -> Dict[str, Any]:
        """Verifica se a resposta é segura"""
        check = self.check_query(response)
        if not check['allowed']:
            return {
                    "allowed": False,
                    "reason": check['reason'],
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
