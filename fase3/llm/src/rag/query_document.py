"""Consulta local com a cadeia documental corrigida, sem ler splits de avaliação."""

import argparse
from datetime import datetime, timezone
import json
import importlib.metadata
from pathlib import Path

from document_chain_v3 import DocumentChainV3, FT11BackendV3, INDEX, TOTAL_TOKENS, OUTPUT_TOKENS
from ft11_backend import ROOT, read, sha


def check_readiness():
    receipt = read(INDEX / "readiness.json")
    decision_path = ROOT / receipt["test_decision_path"]
    if sha(decision_path) != receipt["test_decision_sha256"] or read(decision_path).get("rag_preparation_accepted") is not True:
        raise ValueError("Preparação documental ainda não aprovada ou decisão alterada.")
    for name, expected in receipt["runtime_script_hashes"].items():
        if sha(Path(__file__).parent / name) != expected:
            raise ValueError(f"Módulo diverge da versão avaliada: {name}")
    if sha(INDEX / "frozen_config.json") != receipt["index_config_sha256"]:
        raise ValueError("Configuração do índice alterada.")
    for name, version in receipt["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Ambiente diverge da versão avaliada: {name}")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("Pergunta vazia.")
    receipt = check_readiness()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_rag_query_v3")
    output = ROOT / "artifacts/experiments" / run_id
    output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    save("request.json", {"question": args.question, "chain_version": "v3", "max_total_tokens": TOTAL_TOKENS,
                          "max_new_tokens": OUTPUT_TOKENS, "evaluation_splits_used": False,
                          "chain_sha256": sha(Path(__file__).parent / "document_chain_v3.py"),
                          "retrieval_config_sha256": sha(INDEX / "frozen_config.json"),
                          "readiness_sha256": sha(INDEX / "readiness.json")})
    print(f"Artefatos: {output}", flush=True)
    try:
        backend = FT11BackendV3()
        for key, expected in receipt["model_identity"].items():
            if backend.identity[key] != expected:
                raise ValueError(f"Modelo diverge da versão avaliada: {key}")
        save("identity.json", backend.identity)
        result = DocumentChainV3(backend).invoke(args.question)
        save("result.json", result)
        save("resources.json", {"gpu": backend.torch.cuda.get_device_name(), "load_seconds": backend.load_seconds,
                                "peak_allocated_vram_bytes": backend.torch.cuda.max_memory_allocated()})
        print(json.dumps({"answer": result["answer"], "sources": result["sources"],
                          "error": result.get("error")}, ensure_ascii=False, indent=2))
        if result.get("error"):
            raise SystemExit(1)
    except Exception as error:
        save("error.json", {"type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
