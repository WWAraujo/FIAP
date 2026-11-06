#!/usr/bin/env bash
set -euo pipefail

# Cria um pacote portátil do adapter FT-14. Não inclui a base Qwen de ~19 GB.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN="$ROOT/artifacts/experiments/20261020T_ft14_medpt_full_restart"
TRAINING="$RUN/training/20261020T102347Z_qwen3.5_9b_qlora_rocm"
ADAPTER="$TRAINING/adapter"
PROMPT="$ROOT/data/medpt_ft14_full_20261019/system_prompt.txt"
OUT="$ROOT/artifacts/shares/FT14_MedPT_20261020.tar.zst"
STAGE="$(mktemp -d)"
PACKAGE="$STAGE/FT14_MedPT_20261020"

cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT

for file in "$ADAPTER/adapter_model.safetensors" "$ADAPTER/adapter_config.json" "$TRAINING/manifest.json" "$PROMPT" "$RUN/comparison.json"; do
  [[ -f "$file" ]] || { echo "Arquivo obrigatório ausente: $file" >&2; exit 1; }
done

mkdir -p "$PACKAGE/adapter" "$(dirname "$OUT")"
cp -a "$ADAPTER/." "$PACKAGE/adapter/"
cp "$TRAINING/manifest.json" "$PACKAGE/manifest.json"
cp "$TRAINING/environment.json" "$PACKAGE/environment.json"
cp "$RUN/comparison.json" "$PACKAGE/comparison.json"
cp "$RUN/status.json" "$PACKAGE/status.json"
cp "$PROMPT" "$PACKAGE/system_prompt.txt"

cat > "$PACKAGE/README.md" <<'EOF'
# FT-14 MedPT — pacote para execução local

Este pacote contém somente o adapter treinado em 20/10/2026. Ele precisa do repositório FIAP Fase 3, da base pública `Qwen/Qwen3.5-9B` (cerca de 19 GB) e de uma GPU AMD/ROCm compatível. O ambiente validado foi Linux, Python 3.14.4 e AMD RX 9060 XT de 16 GB.

1. Extraia este arquivo na raiz do clone do projeto:

   ```bash
   tar --use-compress-program=unzstd -xf FT14_MedPT_20261020.tar.zst -C /caminho/para/FIAP/fase3/llm/artifacts/shares
   ```

2. Confira os arquivos:

   ```bash
   cd /caminho/para/FIAP/fase3/llm/artifacts/shares/FT14_MedPT_20261020
   sha256sum -c SHA256SUMS
   ```

3. Instale o ambiente LLM e baixe a base conforme `fase3/llm/README.md`. Não é preciso treinar novamente.

4. Execute uma pergunta de teste a partir de `fase3/llm`:

   ```bash
   .venv/bin/python -B src/finetuning/scripts/run_qwen3_adapter_rocm.py \
     --adapter-dir artifacts/shares/FT14_MedPT_20261020/adapter \
     --model-dir models/base/Qwen3.5-9B \
     --system-prompt-file artifacts/shares/FT14_MedPT_20261020/system_prompt.txt \
     --prompt "Explique de forma geral o que é hipertensão, sem diagnosticar ou prescrever." \
     --max-new-tokens 128
   ```

O FT-14 é candidato experimental. Ele não substitui o FT-11 selecionado pela aplicação e não deve ser usado para cuidado clínico.
EOF

(cd "$PACKAGE" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
tar --use-compress-program='zstd -T0 -19' -cf "$OUT" -C "$STAGE" "FT14_MedPT_20261020"
sha256sum "$OUT" > "$OUT.sha256"
printf 'Pacote criado: %s\nChecksum: %s\n' "$OUT" "$OUT.sha256"
