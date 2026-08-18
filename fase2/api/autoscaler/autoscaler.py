"""
Autoscaler baseado em CPU para o serviço "api" rodando sob Docker Compose
(sem Swarm/Kubernetes).

Docker Compose puro não tem um controlador de autoscaling nativo (o que
existe é escala MANUAL via `--scale`). Este serviço supre essa lacuna:

1. A cada ciclo, lista os containers da API (label `hipertensao.role=api`)
   e lê o uso de CPU de cada um via Docker Engine API.
2. Se a média de CPU ultrapassar o limite superior, cria uma nova réplica
   (container adicional, mesma imagem, mesma rede, mesmo alias DNS "api")
   até o limite máximo configurado.
3. Se a média de CPU cair abaixo do limite inferior, remove a réplica
   extra mais recente até voltar ao mínimo configurado (as réplicas
   "baseline" definidas em deploy.replicas no docker-compose.yml nunca
   são removidas, pois não carregam o label hipertensao.managed_by=autoscaler).
4. Escreve o conjunto atual de réplicas em um arquivo de Service Discovery
   consumido pelo Prometheus (file_sd), então o monitoramento também
   acompanha a escala.
5. Expõe suas próprias métricas Prometheus em /metrics (porta 9200).

O alias de rede "api" é reaplicado a cada novo container, então o nginx
(que resolve "api" via DNS interno do Docker) já enxerga as novas réplicas
sem qualquer reconfiguração manual.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import docker
from docker.errors import APIError, NotFound
from prometheus_client import Gauge, start_http_server


# ============================================================
# CONFIGURAÇÃO
# ============================================================

TARGET_LABEL = "hipertensao.role=api"
MANAGED_LABEL_KEY = "hipertensao.managed_by"
MANAGED_LABEL_VALUE = "autoscaler"

API_IMAGE = os.environ.get("API_IMAGE", "hipertensao-api:latest")
NETWORK_NAME = os.environ.get("DOCKER_NETWORK_NAME", "hipertensao_net")
APP_PORT = int(os.environ.get("APP_PORT", "8000"))

MIN_REPLICAS = int(os.environ.get("MIN_REPLICAS", "2"))
MAX_REPLICAS = int(os.environ.get("MAX_REPLICAS", "6"))
CPU_SCALE_UP_THRESHOLD = float(os.environ.get("CPU_SCALE_UP_THRESHOLD", "70"))
CPU_SCALE_DOWN_THRESHOLD = float(os.environ.get("CPU_SCALE_DOWN_THRESHOLD", "25"))
POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", "15"))
COOLDOWN_SECONDS = int(os.environ.get("COOLDOWN_SECONDS", "60"))
METRICS_PORT = int(os.environ.get("METRICS_PORT", "9200"))
TARGETS_FILE = os.environ.get("TARGETS_FILE", "/targets/api-targets.json")

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


# ============================================================
# LOGGING JSON
# ============================================================

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields:
            payload.update(extra_fields)
        return json.dumps(payload, ensure_ascii=False)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger = logging.getLogger("autoscaler")
logger.setLevel(LOG_LEVEL)
logger.handlers = [handler]
logger.propagate = False


def log_evento(mensagem, nivel="info", **campos):
    getattr(logger, nivel, logger.info)(mensagem, extra={"extra_fields": campos})


# ============================================================
# MÉTRICAS PROMETHEUS
# ============================================================

REPLICAS_ATUAIS = Gauge(
    "autoscaler_current_replicas", "Número atual de réplicas da API"
)
REPLICAS_MIN = Gauge(
    "autoscaler_min_replicas", "Número mínimo de réplicas configurado"
)
REPLICAS_MAX = Gauge(
    "autoscaler_max_replicas", "Número máximo de réplicas configurado"
)
CPU_MEDIA = Gauge(
    "autoscaler_avg_cpu_percent", "Média de uso de CPU (%) entre as réplicas da API"
)
ULTIMA_ACAO_TIMESTAMP = Gauge(
    "autoscaler_last_scale_action_timestamp", "Timestamp Unix da última ação de escala"
)

REPLICAS_MIN.set(MIN_REPLICAS)
REPLICAS_MAX.set(MAX_REPLICAS)


# ============================================================
# CÁLCULO DE CPU (mesma fórmula usada por `docker stats`)
# ============================================================

def calcular_cpu_percent(stats: dict) -> float:
    try:
        cpu_stats = stats["cpu_stats"]
        precpu_stats = stats["precpu_stats"]

        cpu_delta = (
            cpu_stats["cpu_usage"]["total_usage"]
            - precpu_stats["cpu_usage"].get("total_usage", 0)
        )
        system_delta = (
            cpu_stats.get("system_cpu_usage", 0)
            - precpu_stats.get("system_cpu_usage", 0)
        )

        online_cpus = cpu_stats.get("online_cpus") or len(
            cpu_stats["cpu_usage"].get("percpu_usage") or [1]
        )

        if system_delta <= 0 or cpu_delta < 0:
            return 0.0

        return (cpu_delta / system_delta) * online_cpus * 100.0
    except (KeyError, ZeroDivisionError, TypeError):
        return 0.0


def listar_containers_api(client: docker.DockerClient):
    return client.containers.list(
        filters={"label": TARGET_LABEL, "status": "running"}
    )


def coletar_cpu_media(containers) -> float:
    valores = []

    for container in containers:
        try:
            stats = container.stats(stream=False)
            valores.append(calcular_cpu_percent(stats))
        except (APIError, NotFound) as erro:
            log_evento(
                "Falha ao coletar stats do container",
                nivel="warning",
                container=container.name,
                erro=str(erro)
            )

    if not valores:
        return 0.0

    return sum(valores) / len(valores)


# ============================================================
# SERVICE DISCOVERY PARA O PROMETHEUS
# ============================================================

def atualizar_arquivo_service_discovery(containers) -> None:
    targets = []

    for container in containers:
        redes = container.attrs.get("NetworkSettings", {}).get("Networks", {})
        rede = redes.get(NETWORK_NAME)

        if rede and rede.get("IPAddress"):
            targets.append(f"{rede['IPAddress']}:{APP_PORT}")

    conteudo = [{"targets": targets, "labels": {"job": "api"}}]

    os.makedirs(os.path.dirname(TARGETS_FILE), exist_ok=True)

    caminho_temporario = f"{TARGETS_FILE}.tmp"
    with open(caminho_temporario, "w", encoding="utf-8") as arquivo:
        json.dump(conteudo, arquivo)

    os.replace(caminho_temporario, TARGETS_FILE)


# ============================================================
# AÇÕES DE ESCALA
# ============================================================

def criar_replica(client: docker.DockerClient) -> None:
    sufixo = int(time.time() * 1000)
    nome = f"hipertensao-api-scaled-{sufixo}"

    container = client.containers.create(
        image=API_IMAGE,
        name=nome,
        detach=True,
        labels={
            "hipertensao.role": "api",
            MANAGED_LABEL_KEY: MANAGED_LABEL_VALUE,
        },
    )

    rede = client.networks.get(NETWORK_NAME)
    rede.connect(container, aliases=["api"])

    container.start()

    log_evento("Réplica criada pelo autoscaler", nome=nome)


def remover_replica_gerenciada(client: docker.DockerClient) -> bool:
    candidatos = client.containers.list(
        all=True,
        filters={
            "label": f"{MANAGED_LABEL_KEY}={MANAGED_LABEL_VALUE}",
        },
    )

    if not candidatos:
        return False

    candidatos.sort(key=lambda c: c.attrs["Created"], reverse=True)
    alvo = candidatos[0]

    try:
        alvo.stop(timeout=10)
        alvo.remove()
        log_evento("Réplica removida pelo autoscaler", nome=alvo.name)
        return True
    except (APIError, NotFound) as erro:
        log_evento(
            "Falha ao remover réplica",
            nivel="warning",
            nome=alvo.name,
            erro=str(erro)
        )
        return False


# ============================================================
# LOOP PRINCIPAL
# ============================================================

def main() -> None:
    client = docker.from_env()

    log_evento(
        "Autoscaler iniciado",
        min_replicas=MIN_REPLICAS,
        max_replicas=MAX_REPLICAS,
        limite_scale_up=CPU_SCALE_UP_THRESHOLD,
        limite_scale_down=CPU_SCALE_DOWN_THRESHOLD,
        intervalo_segundos=POLL_INTERVAL_SECONDS,
    )

    start_http_server(METRICS_PORT)

    ultima_acao = 0.0

    while True:
        try:
            containers = listar_containers_api(client)
            total_replicas = len(containers)

            atualizar_arquivo_service_discovery(containers)

            cpu_media = coletar_cpu_media(containers) if containers else 0.0

            REPLICAS_ATUAIS.set(total_replicas)
            CPU_MEDIA.set(cpu_media)

            agora = time.time()
            em_cooldown = (agora - ultima_acao) < COOLDOWN_SECONDS

            log_evento(
                "Ciclo de avaliação concluído",
                total_replicas=total_replicas,
                cpu_media_percent=round(cpu_media, 2),
                em_cooldown=em_cooldown,
            )

            if em_cooldown or total_replicas == 0:
                pass
            elif cpu_media > CPU_SCALE_UP_THRESHOLD and total_replicas < MAX_REPLICAS:
                criar_replica(client)
                ultima_acao = agora
                ULTIMA_ACAO_TIMESTAMP.set(agora)
            elif cpu_media < CPU_SCALE_DOWN_THRESHOLD and total_replicas > MIN_REPLICAS:
                if remover_replica_gerenciada(client):
                    ultima_acao = agora
                    ULTIMA_ACAO_TIMESTAMP.set(agora)

        except Exception as erro:
            log_evento(
                "Erro inesperado no ciclo do autoscaler",
                nivel="error",
                erro=str(erro)
            )

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
