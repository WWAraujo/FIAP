import os
import socket

VERSAO_API = "2.0.0"

INSTANCE_ID = os.environ.get("HOSTNAME", socket.gethostname())

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

PASTA_ARQUIVO_ATUAL = os.path.dirname(os.path.abspath(__file__))
PASTA_PROJETO = os.path.dirname(os.path.dirname(PASTA_ARQUIVO_ATUAL))
PASTA_MODELO_API = os.path.join(PASTA_PROJETO, "modelo_api")

NOME_ARQUIVO_MODELO = os.environ.get(
    "NOME_ARQUIVO_MODELO",
    "modelo_genetico_vencedor.joblib"
)

CAMINHO_MODELO = os.path.join(PASTA_MODELO_API, NOME_ARQUIVO_MODELO)
CAMINHO_METADATA = os.path.join(PASTA_MODELO_API, "metadata_modelo_api.json")