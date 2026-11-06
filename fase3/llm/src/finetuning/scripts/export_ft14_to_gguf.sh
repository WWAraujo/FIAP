#!/usr/bin/env bash
set -euo pipefail

# Exporta o candidato FT-14 para GGUF Q4_K_M. Requer llama.cpp e espaço livre
# para os artefatos intermediários; não altera os pesos de treino nem o FT-11.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$ROOT/.tools/llama.cpp}"
QUANTIZE_BIN="${QUANTIZE_BIN:-/usr/local/lib/ollama/llama-quantize}"
RUN="$ROOT/artifacts/experiments/20261020T_ft14_medpt_full_restart"
ADAPTER="$RUN/training/20261020T102347Z_qwen3.5_9b_qlora_rocm/adapter"
BASE="$ROOT/models/base/Qwen3.5-9B"
MERGED="$ROOT/artifacts/exports/ft14_medpt_20261020_merged_hf"
BF16="$ROOT/artifacts/exports/FT14_MedPT_20261020-BF16.gguf"
GGUF="$ROOT/artifacts/shares/FT14_MedPT_20261020-Q4_K_M.gguf"

[[ -x "$PYTHON_BIN" ]] || { echo "Python LLM não encontrado: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" ]] || { echo "Conversor llama.cpp não encontrado: $LLAMA_CPP_DIR" >&2; exit 1; }
[[ -x "$QUANTIZE_BIN" ]] || { echo "Quantizador não encontrado: $QUANTIZE_BIN" >&2; exit 1; }
[[ ! -e "$MERGED" && ! -e "$BF16" && ! -e "$GGUF" ]] || { echo "Há artefato de exportação existente; use outro destino ou remova-o conscientemente." >&2; exit 1; }

mkdir -p "$ROOT/artifacts/exports" "$ROOT/artifacts/shares"
"$PYTHON_BIN" -B "$ROOT/src/finetuning/scripts/merge_ft14_adapter.py" \
  --base-dir "$BASE" --adapter-dir "$ADAPTER" --output-dir "$MERGED"
"$PYTHON_BIN" -B "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED" --outfile "$BF16" --outtype bf16
"$QUANTIZE_BIN" "$BF16" "$GGUF" Q4_K_M
sha256sum "$GGUF" > "$GGUF.sha256"
printf 'GGUF criado: %s\n' "$GGUF"
