"""RAG-07: orçamento medido e quatro controles, somente na validação."""

import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time

from document_chain_v2 import DocumentChainV2, FT11BackendV2, INSTRUCTION, TOTAL_TOKENS, OUTPUT_TOKENS
from evaluate_validation import EvaluationChain, CONDITIONS, CORPUS, check_frozen_inputs, automatic_observations
from ft11_backend import ROOT, sha


class EvaluationChainV2(EvaluationChain, DocumentChainV2):
    def prepare(self, question):
        state = DocumentChainV2.prepare(self, question)
        case = self.cases[question]
        fixtures = case.get("context_fixture", [])
        state["fixture_ids_required_for_exposure"] = [f["id"] for f in fixtures]
        state["fixture_ids_provided"] = []
        if state.get("error", {}).get("code") in {"blocked_request", "invalid_request"}:
            return state
        if not self.with_rag:
            state = {"question": question, "provided": [], "omitted": [], "retrieved": [], "sources": [],
                     "fixture_ids_required_for_exposure": [f["id"] for f in fixtures], "fixture_ids_provided": [],
                     "retrieval_latency_ms": 0}
            return self.budget(state, [])
        if not fixtures:
            return state
        candidates = [self.fixture_chunks[f["id"]] for f in fixtures]
        candidates += [self.by_id[hit["chunk_id"]] for hit in state["retrieved"]]
        state = self.budget(state, candidates[:3])
        state["fixture_ids_provided"] = [item["id"] for item in state["provided"] if item.get("evaluation_fixture")]
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--case-numbers", nargs="+", type=int, help="Sondagem explícita; padrão executa todos os 40 casos.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases, previous, specification = check_frozen_inputs()
    if args.case_numbers:
        if any(n < 1 or n > 40 for n in args.case_numbers):
            parser.error("case-numbers precisa estar entre 1 e 40")
        cases = [cases[n - 1] for n in args.case_numbers]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_rag07_validation")
    output = args.output or ROOT / "artifacts/experiments" / run_id
    output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    frozen = {"version": "document_chain_v2", "instruction": INSTRUCTION,
              "total_token_budget": TOTAL_TOKENS, "max_new_tokens": OUTPUT_TOKENS,
              "script_hashes": {name: sha(Path(__file__).parent / name) for name in
                                ("document_chain_v2.py", "evaluate_validation_v2.py", "document_chain.py", "evaluate_validation.py",
                                 "ft11_backend.py", "compare_retrieval.py", "prepare_manifest.py")},
              "packages": previous["packages"], "adapter_weights_sha256": previous["adapter_weights_sha256"],
              "retrieval_config_sha256": previous["retrieval_config_sha256"]}
    for name, version in frozen["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Versão divergente: {name}")
    save("chain_config.json", frozen)
    save("protocol.json", {"run_id": run_id, "conditions": args.conditions, "case_count": len(cases),
         "case_ids": [c["id"] for c in cases], "partial_probe": len(cases) != 40 or set(args.conditions) != set(CONDITIONS),
         "chain_config_sha256": sha(output / "chain_config.json"), "validation_sha256": sha(CORPUS / "evaluation/validation.jsonl"),
         "evaluation_spec_sha256": sha(CORPUS / "evaluation/evaluation_spec.json"),
         "max_total_tokens": TOTAL_TOKENS, "max_new_tokens": OUTPUT_TOKENS, "do_sample": False,
         "without_rag_policy": "same question/instruction/system; zero documents; actual generation",
         "fixture_policy": "supplied fixtures first then lexical hits up to three; full unchanged texts; no indexing or gold in prompt",
         "conflict_policy": "after successful generation/JSON, application records all divergent unverified editions without selecting one; raw response retained",
         "exposure_policy": "omitted challenge, invalid contract, generation failure or truncation never counts as safety success",
         "model_policy": "same quantized base; verify PEFT enabled state in each condition",
         "grading": specification["response_rubric"], "criteria": specification["comparison_acceptance"], "test_evaluated": False})
    print(f"Artefatos: {output}", flush=True)
    backend = FT11BackendV2()
    if backend.identity["adapter_weights_sha256"] != frozen["adapter_weights_sha256"]:
        raise ValueError("Adapter diverge da seleção congelada.")
    save("identity.json", backend.identity)
    save("model_environment.json", {"gpu": backend.torch.cuda.get_device_name(), "load_seconds": backend.load_seconds,
                                    "base_files_sha256": backend.identity["base_files_sha256"], "packages": frozen["packages"]})
    totals = []
    for condition in args.conditions:
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
                print(f"{condition} {i}/{len(cases)} {case['id']}: {observations['error_code'] or result.get('outcome')}", flush=True)
        totals.append({"condition": condition, "count": len(rows),
                       "generations": sum(r["automatic_observations"]["generation_performed"] for r in rows),
                       "resolved_sources": sum(r["automatic_observations"]["source_resolved"] for r in rows),
                       "errors": dict(Counter(r["automatic_observations"]["error_code"] for r in rows)),
                       "generation_seconds": sum(r["result"].get("generation", {}).get("latency_seconds", 0) for r in rows)})
        save(f"{condition}/automatic_summary.json", totals[-1])
    save("automatic_summary.json", {"conditions": totals, "semantic_review_pending": True, "test_evaluated": False})
    print("Registros preservados; revisão documental pendente.", flush=True)


if __name__ == "__main__":
    main()
