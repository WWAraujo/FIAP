import os

import requests

from medical_assistant.application.contracts import AssistantRequest, RuntimeResponse


class InferenceUnavailable(RuntimeError):
    pass


class FT11Client:
    def __init__(self):
        self.url = os.environ.get("FT11_RUNTIME_URL", "http://127.0.0.1:8001").rstrip("/")

    def invoke(self, request: AssistantRequest) -> RuntimeResponse:
        try:
            response = requests.post(f"{self.url}/query", json=request.model_dump(), timeout=(5, 600))
            response.raise_for_status()
            result = RuntimeResponse.model_validate(response.json())
            if result.error:
                raise InferenceUnavailable(str(result.error))
            return result
        except (requests.RequestException, ValueError) as exc:
            raise InferenceUnavailable(
                "FT-11/RAG indisponível. Confira o processo runtime_server.py e seus logs."
            ) from exc
