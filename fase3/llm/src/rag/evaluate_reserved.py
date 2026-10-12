"""RAG-08: teste reservado da cadeia congelada após validação aprovada."""

import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time

from document_chain_v2 import FT11BackendV2, INSTRUCTION, TOTAL_TOKENS, OUTPUT_TOKENS
from evaluate_validation import CONDITIONS, CORPUS, automatic_observations
from evaluate_validation_v2 import EvaluationChainV2
from ft11_backend import ROOT, read, sha

VALIDATION_RUN = ROOT / "artifacts/experiments/20261011T235027290179Z_rag07_validation"
REQUIRED_CRITERIA = ("ft11_rag_more_adequate_than_without", "no_increase_raw_unsupported_document_quotes",
                     "no_increase_visible_semantically_inadequate_answers", "all_special_cases_passed",
                     "retrieval_and_citation_validity")


def approved_config(directory):
    """Verificar o gate sem abrir nenhum arquivo de perguntas de teste."""
    decision = read(directory / "decision.json")
    if (decision.get("validation_complete") is not True or decision.get("advance_to_reserved_test") is not True or
            decision.get("partial_probe") is not False or decision.get("records_reviewed") != 160 or
            any(decision.get("criteria", {}).get(key) is not True for key in REQUIRED_CRITERIA)):
        raise ValueError("Validação completa aprovada é necessária antes de abrir o teste.")
    for filename, key in (("documentary_review.json", "review_sha256"), ("reviewed_summary.json", "reviewed_summary_sha256"),
                          ("chain_config.json", "chain_config_sha256"), ("protocol.json", "protocol_sha256")):
        if sha(directory / filename) != decision.get(key):
            raise ValueError(f"Artefato de validação divergente: {filename}")
    review = read(directory / "documentary_review.json")
    if Counter(c["condition"] for c in review["cases"]) != {condition: 40 for condition in CONDITIONS}:
        raise ValueError("Revisão de validação incompleta.")
    frozen = read(directory / "chain_config.json")
    if frozen["instruction"] != INSTRUCTION or frozen["total_token_budget"] != TOTAL_TOKENS or frozen["max_new_tokens"] != OUTPUT_TOKENS:
        raise ValueError("Instrução/orçamento difere da versão aprovada.")
    for name, expected in frozen["script_hashes"].items():
        if sha(Path(__file__).parent / name) != expected:
            raise ValueError(f"Módulo congelado alterado: {name}")
    for name, expected in frozen["packages"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Pacote congelado alterado: {name}")
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-run", type=Path, default=VALIDATION_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    frozen = approved_config(args.validation_run)
    validation_protocol = read(args.validation_run / "protocol.json")
    specification_path = CORPUS / "evaluation/evaluation_spec.json"
    if sha(specification_path) != validation_protocol["evaluation_spec_sha256"]:
        raise ValueError("Rubrica/especificação difere da validação aprovada.")
    specification = read(specification_path)
    if sha(CORPUS / "manifest.json") != specification["manifest_sha256"]:
        raise ValueError("Corpus alterado após a validação.")
    test_path = CORPUS / "evaluation/test.jsonl"
    if sha(test_path) != specification["splits_sha256"]["test"]:
        raise ValueError("Teste difere do split congelado.")
    cases = [json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines()]
    if (len(cases) != 40 or any(c["split"] != "test" for c in cases) or
            Counter(c["category"] for c in cases) != specification["splits"]["test"]["categories"]):
        raise ValueError("Split/categorias do teste divergentes.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_rag08_test")
    output = args.output or ROOT / "artifacts/experiments" / run_id
    output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    save("chain_config.json", frozen)
    save("protocol.json", {"run_id": run_id, "split": "test", "conditions": CONDITIONS, "case_count": 40,
         "case_ids": [c["id"] for c in cases], "partial_probe": False,
         "approved_validation_run": str(args.validation_run.resolve()),
         "validation_decision_sha256": sha(args.validation_run / "decision.json"),
         "chain_config_sha256": sha(output / "chain_config.json"), "test_sha256": sha(test_path),
         "evaluation_spec_sha256": sha(specification_path), "max_total_tokens": TOTAL_TOKENS,
         "max_new_tokens": OUTPUT_TOKENS, "do_sample": False, "test_evaluated": True, "retuning_allowed": False,
         "grading": specification["response_rubric"], "criteria": specification["comparison_acceptance"],
         "fixture_policy": validation_protocol["fixture_policy"], "exposure_policy": validation_protocol["exposure_policy"],
         "model_policy": validation_protocol["model_policy"], "script_sha256": sha(Path(__file__))})
    print(f"Artefatos: {output}", flush=True)
    backend = FT11BackendV2()
    previous_identity = read(args.validation_run / "identity.json")
    for key in ("model_id", "model_revision", "adapter_weights_sha256", "adapter_config_sha256", "manifest_sha256",
                "system_prompt_sha256", "base_files_sha256"):
        if backend.identity[key] != previous_identity[key]:
            raise ValueError(f"Modelo/pesos/template diferem da validação: {key}")
    save("identity.json", backend.identity)
    save("model_environment.json", {"gpu": backend.torch.cuda.get_device_name(), "load_seconds": backend.load_seconds,
                                    "base_files_sha256": backend.identity["base_files_sha256"], "packages": frozen["packages"]})
    totals = []
    for condition in CONDITIONS:
        with_rag = condition.endswith("with_rag") and not condition.endswith("without_rag")
        adapted = condition.startswith("ft11")
        chain = EvaluationChainV2(backend, cases, with_rag)
        (output / condition).mkdir()
        rows = []
        with nullcontext() if adapted else backend.model.disable_adapter():
            status = backend.model.get_model_status()
            if status.enabled is not adapted:
                raise RuntimeError("Estado do adapter diverge do controle.")
            save(f"{condition}/model_identity.json", {"model_id": backend.identity["model_id"],
                 "model_revision": backend.identity["model_revision"], "adapter_enabled": status.enabled,
                 "adapter_weights_sha256_effective": backend.identity["adapter_weights_sha256"] if adapted else None,
                 "loaded_adapter_weights_sha256": backend.identity["adapter_weights_sha256"]})
            for i, case in enumerate(cases, 1):
                backend.torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                try:
                    result = chain.invoke(case["question"])
                except Exception as error:
                    result = chain.failure({"question": case["question"], "provided": [], "omitted": [], "retrieved": [], "sources": []},
                                           "execution_error", f"{type(error).__name__}: {error}")
                observations = automatic_observations(case, result)
                row = {"condition": condition, "id": case["id"], "category": case["category"], "result": result,
                       "automatic_observations": observations, "total_seconds": time.perf_counter() - start,
                       "peak_allocated_vram_bytes": backend.torch.cuda.max_memory_allocated(),
                       "peak_reserved_vram_bytes": backend.torch.cuda.max_memory_reserved()}
                save(f"{condition}/{case['id']}.json", row)
                rows.append(row)
                print(f"{condition} {i}/40 {case['id']}: {observations['error_code'] or result.get('outcome')}", flush=True)
        totals.append({"condition": condition, "count": len(rows),
                       "generations": sum(r["automatic_observations"]["generation_performed"] for r in rows),
                       "resolved_sources": sum(r["automatic_observations"]["source_resolved"] for r in rows),
                       "errors": dict(Counter(r["automatic_observations"]["error_code"] for r in rows)),
                       "generation_seconds": sum(r["result"].get("generation", {}).get("latency_seconds", 0) for r in rows)})
        save(f"{condition}/automatic_summary.json", totals[-1])
    save("automatic_summary.json", {"conditions": totals, "semantic_review_pending": True, "test_evaluated": True})
    print("160 registros de teste preservados; revisão documental pendente.", flush=True)


if __name__ == "__main__":
    main()
