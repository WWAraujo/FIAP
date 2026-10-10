"""Compara Qwen3 base e adapter QLoRA no split de teste reservado."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

SYSTEM_PROMPT = (
    "Você é um assistente médico acadêmico. Use apenas informações documentadas, "
    "não prescreva medicamentos, não faça diagnóstico definitivo e indique a necessidade "
    "de validação humana."
)


def arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--base-only", action="store_true", help="Avalia somente o controle sem adapter.")
    parser.add_argument("--adapter-only", action="store_true", help="Avalia somente o adapter; o controle comparativo deve ser documentado separadamente.")
    parser.add_argument("--model-dir", type=Path, default=root / "models" / "base" / "Qwen3-8B")
    parser.add_argument("--test-file", type=Path, default=root / "data" / "splits" / "test.jsonl")
    parser.add_argument("--output-dir", type=Path, default=root / "artifacts" / "evaluations")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--system-prompt-file", type=Path, help="Prompt alternativo para uma ablação registrada.")
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Máximo de casos reservados a comparar; use um valor maior para ampliar a avaliação.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed da seleção estratificada de casos.")
    args = parser.parse_args()
    if args.base_only and args.adapter_only:
        parser.error("--base-only e --adapter-only são mutuamente exclusivos")
    return args


def select_records(records: list[dict], limit: int | None, seed: int) -> list[dict]:
    """Limita a avaliação sem eliminar por completo uma categoria do conjunto reservado."""
    if limit is None or limit >= len(records):
        return records
    if limit < 1:
        raise ValueError("--limit deve ser positivo")

    by_category: dict[str, list[dict]] = {}
    for record in records:
        by_category.setdefault(record["category"], []).append(record)
    if limit < len(by_category):
        raise ValueError(
            f"--limit ({limit}) é menor que as {len(by_category)} categorias do teste reservado."
        )

    selected: list[dict] = []
    selected_ids: set[str] = set()
    for category in sorted(by_category):
        record = random.Random(f"{seed}:{category}").choice(by_category[category])
        selected.append(record)
        selected_ids.add(record["source_id"])
    remaining = [record for record in records if record["source_id"] not in selected_ids]
    selected.extend(random.Random(f"{seed}:remaining").sample(remaining, limit - len(selected)))
    return sorted(selected, key=lambda record: record["source_id"])


def main() -> None:
    args = arguments()
    system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip() if args.system_prompt_file else SYSTEM_PROMPT
    if not system_prompt:
        raise ValueError("O prompt de sistema não pode estar vazio.")
    adapter_dir = args.adapter_dir.resolve() if args.adapter_dir else None
    model_dir, test_file = args.model_dir.resolve(), args.test_file.resolve()
    if not args.base_only and (adapter_dir is None or not (adapter_dir / "adapter_config.json").is_file()):
        raise FileNotFoundError(f"Adapter inválido: {adapter_dir}")
    if not model_dir.is_dir() or not test_file.is_file():
        raise FileNotFoundError("Modelo-base ou split de teste não encontrado.")
    manifest = json.loads((adapter_dir.parent / "manifest.json").read_text(encoding="utf-8")) if adapter_dir else {"run_id": None}
    records = [json.loads(line) for line in test_file.read_text(encoding="utf-8").splitlines() if line]
    total_test_count = len(records)
    records = select_records(records, args.limit, args.seed)
    if not records:
        raise ValueError("O split de teste está vazio.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_eval03"
    output = args.output_dir.resolve() / run_id
    output.mkdir(parents=True, exist_ok=False)
    request = {"arguments": vars(args), "input_file": str(test_file), "input_sha256": hashlib.sha256(test_file.read_bytes()).hexdigest(), "selected_ids": [row["source_id"] for row in records], "system_prompt": system_prompt}
    (output / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Artefatos: {output}", flush=True)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm com GPU disponível é obrigatório.")
    index, device = torch.cuda.current_device(), torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    base = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=quantization, dtype=dtype, device_map={"": index})
    base.eval()

    def evaluate(model, label: str) -> list[dict]:
        rows = []
        for case_index, record in enumerate(records, 1):
            prompt = record["instruction"] + ("\n\n" + record["input"] if record.get("input") else "")
            inputs = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                add_generation_prompt=True, enable_thinking=False, tokenize=True, return_tensors="pt", return_dict=True,
            ).to(device)
            started = time.perf_counter()
            with torch.inference_mode():
                sequences = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            response = tokenizer.decode(sequences[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            rows.append({"model": label, "source_id": record["source_id"], "reference": record["output"], "response": response, "latency_seconds": round(time.perf_counter() - started, 3), "nonempty": bool(response), "source_available": False})
            generated_tokens = sequences.shape[-1] - inputs["input_ids"].shape[-1]
            rows[-1].update({"generated_tokens": generated_tokens, "hit_token_limit": generated_tokens >= args.max_new_tokens})
            with (output / f"{label}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
            print(f"[{label} {case_index}/{len(records)}] {record['source_id']} | {generated_tokens} tokens | {rows[-1]['latency_seconds']} s", flush=True)
        return rows

    model_rows = {} if args.adapter_only else {"base": evaluate(base, "base")}
    if not args.base_only:
        adapter = PeftModel.from_pretrained(base, adapter_dir)
        adapter.eval()
        model_rows["adapter"] = evaluate(adapter, "adapter")
    summary = {"run_id": run_id, "adapter_run_id": manifest["run_id"], "test_count": len(records), "test_file_count": total_test_count, "selection": {"strategy": "amostra estratificada por categoria", "seed": args.seed, "limit": args.limit}, "max_new_tokens": args.max_new_tokens, "models": {name: {"nonempty_rate": sum(row["nonempty"] for row in rows) / len(rows), "mean_latency_seconds": sum(row["latency_seconds"] for row in rows) / len(rows), "source_available": False} for name, rows in model_rows.items()}, "limitations": ["Respostas e referências exigem revisão humana.", "Este ensaio não usa RAG; source_available é false."]}
    summary.update({"input_file": str(test_file), "input_sha256": hashlib.sha256(test_file.read_bytes()).hexdigest(), "generation": {"do_sample": False, "enable_thinking": False}})
    summary["system_prompt"] = {"text": system_prompt, "sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(), "file": str(args.system_prompt_file.resolve()) if args.system_prompt_file else None}
    for name, rows in model_rows.items():
        summary["models"][name]["token_limit_count"] = sum(row["hit_token_limit"] for row in rows)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(output / "summary.json"), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
