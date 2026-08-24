from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS_TOTAL = Counter(
    "api_http_requests_total",
    "Total de requisições HTTP recebidas pela API",
    ["method", "path", "status_code"]
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "api_http_request_duration_seconds",
    "Duração das requisições HTTP em segundos",
    ["method", "path"]
)

PREDICOES_TOTAL = Counter(
    "api_predicoes_total",
    "Total de predições realizadas pelo modelo",
    ["classe_prevista"]
)

PREDICAO_PROBABILIDADE = Histogram(
    "api_predicao_probabilidade",
    "Distribuição das probabilidades de hipertensão previstas",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

PREDICOES_ERRO_TOTAL = Counter(
    "api_predicoes_erro_total",
    "Total de erros ao executar predição no modelo"
)

MODELO_INFO = Gauge(
    "api_modelo_info",
    "Informações estáticas do modelo carregado (valor sempre 1)",
    ["nome_modelo", "algoritmo", "versao_api", "instance_id"]
)