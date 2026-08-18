#!/usr/bin/env bash
#
# Gera carga concorrente contra POST /prever para forçar o autoscaler a
# escalar a API horizontalmente. Usa apenas curl (sem dependências extras),
# então roda em qualquer máquina com bash + curl.
#
# Uso:
#   ./scripts/teste_escalabilidade.sh [duracao_segundos] [concorrencia] [url_base]
#
# Exemplos:
#   ./scripts/teste_escalabilidade.sh                  # 90s, 40 requisições concorrentes, localhost:8000
#   ./scripts/teste_escalabilidade.sh 120 60            # 120s, 60 requisições concorrentes
#   ./scripts/teste_escalabilidade.sh 90 40 http://localhost:8000

set -euo pipefail

DURACAO_SEGUNDOS="${1:-90}"
CONCORRENCIA="${2:-40}"
API_URL="${3:-http://localhost:8000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD_FILE="$SCRIPT_DIR/../modelo_api/exemplo_entrada_api.json"

if [ ! -f "$PAYLOAD_FILE" ]; then
  echo "Payload de exemplo não encontrado em: $PAYLOAD_FILE" >&2
  exit 1
fi

echo "=================================================================="
echo "Teste de escalabilidade"
echo "=================================================================="
echo "Alvo:         $API_URL/prever"
echo "Duração:      ${DURACAO_SEGUNDOS}s"
echo "Concorrência: $CONCORRENCIA requisições simultâneas por lote"
echo ""
echo "Acompanhe em outros terminais enquanto o teste roda:"
echo "  docker compose logs -f autoscaler      # decisões de escala"
echo "  watch docker compose ps                # nº de containers 'api'"
echo "  http://localhost:3000                  # dashboard Grafana"
echo "=================================================================="
echo ""

if ! curl -sf -o /dev/null "$API_URL/health"; then
  echo "Não consegui acessar $API_URL/health. A stack está no ar? (docker compose up -d)" >&2
  exit 1
fi

inicio=$(date +%s)
fim=$((inicio + DURACAO_SEGUNDOS))
total_enviadas=0
total_ok=0
lote=0

while [ "$(date +%s)" -lt "$fim" ]; do
  lote=$((lote + 1))
  codigos_lote="$(mktemp)"

  for _ in $(seq 1 "$CONCORRENCIA"); do
    curl -s -o /dev/null -w "%{http_code}\n" -X POST "$API_URL/prever" \
      -H "Content-Type: application/json" \
      -d @"$PAYLOAD_FILE" >> "$codigos_lote" &
  done
  wait

  ok_lote=$(grep -c '^200$' "$codigos_lote" || true)
  total_ok=$((total_ok + ok_lote))
  total_enviadas=$((total_enviadas + CONCORRENCIA))
  rm -f "$codigos_lote"

  decorrido=$(( $(date +%s) - inicio ))
  echo "[${decorrido}s] lote $lote: $ok_lote/$CONCORRENCIA OK (total: $total_ok/$total_enviadas)"
done

echo ""
echo "=================================================================="
echo "Concluído em $(( $(date +%s) - inicio ))s — $total_ok/$total_enviadas requisições OK"
echo ""
echo "A carga parou agora. Se o autoscaler já criou réplicas extras, elas"
echo "devem ser removidas automaticamente após o cooldown (padrão: 60s)"
echo "quando a CPU média cair abaixo do limite de scale-down."
echo "=================================================================="
