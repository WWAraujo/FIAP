"""Avalia um modelo Ollama na mesma seleção estratificada do controle ROCm."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from evaluate_qwen3_base_vs_adapter_rocm import select_records


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-prompt-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, default=root / "data/splits/validation.jsonl")
    parser.add_argument("--output-dir", type=Path, default=root / "artifacts/evaluations")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-ctx", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("Prompt vazio.")
    records = [json.loads(line) for line in args.test_file.read_text(encoding="utf-8").splitlines() if line]
    selected = select_records(records, args.limit, args.seed)
    if not selected:
        raise ValueError("Split vazio.")

    def api(endpoint: str, payload: dict | None = None) -> dict:
        url = args.host.rstrip("/") + "/api/" + endpoint
        response = requests.get(url, timeout=args.timeout) if payload is None else requests.post(url, json=payload, timeout=args.timeout)
        response.raise_for_status()
        return response.json()

    tags = api("tags")
    tag = next((m for m in tags["models"] if m["name"] == args.model), None)
    if tag is None:
        raise ValueError(f"Modelo não instalado: {args.model}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_ollama_validation"
    output = args.output_dir.resolve() / run_id
    output.mkdir(parents=True, exist_ok=False)
    options = {"temperature": 0, "seed": args.seed, "num_predict": args.max_new_tokens, "num_ctx": args.num_ctx}
    manifest = {"run_id": run_id, "model": tag, "model_info": api("show", {"model": args.model}), "ollama_version": api("version"), "input_file": str(args.test_file.resolve()), "input_sha256": hashlib.sha256(args.test_file.read_bytes()).hexdigest(), "system_prompt": {"text": prompt, "sha256": hashlib.sha256(prompt.encode()).hexdigest()}, "options": options, "think": False, "selection": {"seed": args.seed, "limit": args.limit, "ids": [r["source_id"] for r in selected]}, "limitations": ["Sem RAG ou validação clínica.", "GGUF/Ollama e NF4/Transformers são sistemas diferentes."]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Artefatos: {output}", flush=True)
    warmup = api("chat", {"model": args.model, "messages": [{"role": "user", "content": "Responda somente OK."}], "stream": False, "think": False, "keep_alive": "30m", "options": {**options, "num_predict": 8}})
    (output / "warmup.json").write_text(json.dumps(warmup, ensure_ascii=False, indent=2) + "\n")
    (output / "loaded_models_after_warmup.json").write_text(json.dumps(api("ps"), ensure_ascii=False, indent=2) + "\n")
    rows = []
    with (output / "responses.jsonl").open("w", encoding="utf-8") as stream:
        for index, record in enumerate(selected, 1):
            content = record["instruction"] + ("\n\n" + record["input"] if record.get("input") else "")
            started = time.perf_counter()
            row = {"model": args.model, "source_id": record["source_id"], "reference": record["output"], "source_available": False}
            try:
                body = api("chat", {"model": args.model, "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": content}], "stream": False, "think": False, "keep_alive": "30m", "options": options})
                row.update({"response": body["message"].get("content", "").strip(), "generated_tokens": body.get("eval_count", 0), "done_reason": body.get("done_reason"), "server_metrics": {k: body.get(k) for k in ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_duration")}, "unexpected_thinking": bool(body["message"].get("thinking")), "error": None})
            except (requests.RequestException, ValueError, KeyError) as error:
                row.update({"response": "", "generated_tokens": 0, "error": str(error)})
            row.update({"latency_seconds": round(time.perf_counter() - started, 3), "nonempty": bool(row["response"]), "hit_token_limit": row["generated_tokens"] >= args.max_new_tokens})
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{index}/{len(selected)}] {row['source_id']} | {row['generated_tokens']} tokens | {row['latency_seconds']} s | error={row['error']}", flush=True)
    summary = {"run_id": run_id, "count": len(rows), "errors": sum(r["error"] is not None for r in rows), "nonempty_count": sum(r["nonempty"] for r in rows), "token_limit_count": sum(r["hit_token_limit"] for r in rows), "mean_tokens": sum(r["generated_tokens"] for r in rows) / len(rows), "mean_latency_seconds": sum(r["latency_seconds"] for r in rows) / len(rows), "source_available": False, "warmup_excluded": True}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
