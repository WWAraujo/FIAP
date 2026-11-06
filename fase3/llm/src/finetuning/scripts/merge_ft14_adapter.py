"""Mescla o adapter FT-14 em uma cópia local da base para exportação externa.

O resultado é um modelo Hugging Face completo, destinado exclusivamente à
conversão subsequente. Não altera a base, o adapter ou `best_model.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    base = args.base_dir.resolve()
    adapter = args.adapter_dir.resolve()
    output = args.output_dir.resolve()
    weights = adapter / "adapter_model.safetensors"
    if not (base / "config.json").is_file() or not weights.is_file():
        raise FileNotFoundError("Base ou adapter FT-14 incompleto.")
    if output.exists():
        raise FileExistsError(f"Destino já existe: {output}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Carregando a base em CPU para mesclar o adapter FT-14...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True, local_files_only=True)
    merged = PeftModel.from_pretrained(model, adapter, local_files_only=True).merge_and_unload(safe_merge=True)
    output.mkdir(parents=True)
    print("Salvando pesos mesclados para conversão GGUF...", flush=True)
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(output)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "temporary_hf_export_for_ft14_gguf_conversion",
        "base": str(base),
        "adapter": str(adapter),
        "adapter_weights_sha256": sha256(weights),
        "output_format": "safetensors_bfloat16",
        "merged": True,
    }
    (output / "ft14_merge_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
