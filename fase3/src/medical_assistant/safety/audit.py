import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class AuditLogger:
    """Sistema de logging para auditoria"""

    def __init__(self):
        self.log_dir = Path(__file__).resolve().parents[3] / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.audit_file = self.log_dir / "audit.log"

    def log_query(self, query: str, patient_context: Dict = None) -> str:
        """Registra query para auditoria"""
        query_id = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query_id": query_id,
            "query": query,
            "patient_context": patient_context,
            "type": "query"
        }

        self._write_log(log_entry)
        return query_id

    def log_response(self, query_id: str, response: str, sources: List[str], requires_validation: bool,
                     *, model_usage: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        """Registra resposta para auditoria"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query_id": query_id,
            "response": response[:500],  # Limitar tamanho
            "sources": sources,
            "requires_validation": requires_validation,
            "type": "response"
        }

        if model_usage is not None:
            log_entry.update(model_usage)
        if error is not None:
            log_entry["error"] = error

        self._write_log(log_entry)

    def _write_log(self, log_entry: Dict):
        """Escreve entrada no log de auditoria"""
        with open(self.audit_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
