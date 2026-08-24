import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict

from .config import INSTANCE_ID, LOG_LEVEL


# ============================================================
# LOGGING ESTRUTURADO (JSON)
# ============================================================
# Logs em JSON facilitam a coleta/parsing por ferramentas de
# observabilidade (Loki, ELK, CloudWatch, etc.) quando a API roda
# em múltiplos containers e os logs precisam ser correlacionados.

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "instance_id": INSTANCE_ID,
        }

        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields:
            payload.update(extra_fields)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def configurar_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger("api_modelo")
    logger.setLevel(LOG_LEVEL)
    logger.handlers = [handler]
    logger.propagate = False

    return logger


logger = configurar_logging()


def log_evento(mensagem: str, nivel: str = "info", **campos: Any) -> None:
    log_fn = getattr(logger, nivel, logger.info)
    log_fn(mensagem, extra={"extra_fields": campos})