"""Recarrega um adapter QLoRA salvo e gera uma resposta em GPU AMD/ROCm.

Este comando não treina nem altera o adapter. Ele é a validação de recarga em
processo novo exigida por FT-02 e imprime um JSON com o ``run_id`` do adapter.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "Você é um assistente médico acadêmico. Use apenas informações documentadas, "
    "não prescreva medicamentos, não faça diagnóstico definitivo e indique a necessidade "
    "de validação humana."
)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True, help="Diretório adapter/ gerado pelo treino.")
    parser.add_argument("--model-dir", type=Path, default=project_dir / "models" / "base" / "Qwen3-8B")
    parser.add_argument("--prompt", default="Quais são os sintomas de hipertensão?")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def require_rocm() -> tuple[Any, Any, int]:
    import torch

    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm com GPU disponível é obrigatório para recarregar este adapter.")
    index = torch.cuda.current_device()
    return torch, torch.device("cuda"), index


def load_run_id(adapter_dir: Path) -> str:
    config_path = adapter_dir / "adapter_config.json"
    manifest_path = adapter_dir.parent / "manifest.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"adapter_config.json não encontrado: {config_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json não encontrado: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"Manifesto sem run_id: {manifest_path}")
    return run_id


def main() -> None:
    args = parse_args()
    adapter_dir = args.adapter_dir.resolve()
    model_dir = args.model_dir.resolve()
    run_id = load_run_id(adapter_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Modelo-base não encontrado: {model_dir}")

    torch, device, device_index = require_rocm()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    started = time.perf_counter()
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quantization,
        dtype=compute_dtype,
        device_map={"": device_index},
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    inputs = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
    if not response:
        raise RuntimeError("O adapter recarregado retornou uma resposta vazia.")
    print(
        json.dumps(
            {
                "run_id": run_id,
                "adapter_dir": str(adapter_dir),
                "model_dir": str(model_dir),
                "prompt": args.prompt,
                "response": response,
                "latency_seconds": round(time.perf_counter() - started, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
