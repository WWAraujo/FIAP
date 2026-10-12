"""Quatro controles da RAG-06, somente validação e cadeia congelada."""

import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from document_chain import DocumentChain, INSTRUCTION, no_duplicate_keys
from ft11_backend import FT11Backend, ROOT, read, sha

CHAIN_CONFIG = ROOT / "artifacts/experiments/20261011T215906287410Z_rag05/chain_config.json"
CORPUS = ROOT / "data/rag_pilot_v2"
CONDITIONS = ("base_without_rag", "ft11_without_rag", "base_with_rag", "ft11_with_rag")


def check_frozen_inputs():
    frozen = read(CHAIN_CONFIG)
    if frozen["instruction"] != INSTRUCTION:
        raise ValueError("Instrução diverge da cadeia congelada.")
    for name, expected in frozen["script_hashes"].items():
        if sha(Path(__file__).parent / name) != expected:
            raise ValueError(f"Módulo congelado alterado: {name}")
    for name, expected in frozen["packages"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Versão de pacote diverge da cadeia congelada: {name}")
    specification = read(CORPUS / "evaluation/evaluation_spec.json")
    if sha(CORPUS / "manifest.json") != specification["manifest_sha256"]:
        raise ValueError("Corpus diverge da avaliação registrada.")
    path = CORPUS / "evaluation/validation.jsonl"
    if sha(path) != specification["splits_sha256"]["validation"]:
        raise ValueError("Validação diverge do conjunto reservado.")
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(cases) != 40 or Counter(c["category"] for c in cases) != specification["splits"]["validation"]["categories"]:
        raise ValueError("Contagem/categorias da validação divergentes.")
    return cases, frozen, specification


class EvaluationChain(DocumentChain):
    """Fixtures somente no contexto da avaliação, nunca no índice BM25."""
    def __init__(self, backend, cases, with_rag):
        super().__init__(backend)
        self.cases = {case["question"]: case for case in cases}
        self.with_rag = with_rag
        if len(self.cases) != len(cases):
            raise ValueError("Perguntas duplicadas na avaliação.")
        self.fixture_chunks = {}
        for case in cases:
            for fixture in case.get("context_fixture", []):
                section_id = fixture["id"] + "#fixture"
                self.fixture_chunks[fixture["id"]] = {
                    "id": fixture["id"], "section_id": section_id,
                    "document_id": fixture["document_id"], "version": fixture["version"],
                    "page": None, "char_start": 0, "char_end": len(fixture["text"]),
                    "text": fixture["text"], "evaluation_fixture": True, "validity": fixture["validity"]}
                # Metadata lookup for resolving a cited fixture; no inclusion in self.chunks/retriever.
                self.documents[f"{fixture['document_id']}@{fixture['version']}"] = {
                    "title": f"Fixture sintético {fixture['document_id']}", "url": None,
                    "evaluation_fixture": True, "validity": fixture["validity"]}
                self.sections[section_id] = {"section": "Fixture adversarial/conflito da avaliação"}

    def prepare(self, question):
        state = super().prepare(question)
        case = self.cases[question]
        fixtures = case.get("context_fixture", [])
        state["fixture_ids_required_for_exposure"] = [f["id"] for f in fixtures]
        state["fixture_ids_provided"] = []
        if state.get("error", {}).get("code") in {"blocked_request", "invalid_request"}:
            return state
        if not self.with_rag:
            # Same question/instruction, no evidence or adversarial documents. Force actual control generation.
            state = {"question": question, "provided": [], "omitted": [], "retrieved": [], "sources": [],
                     "fixture_ids_required_for_exposure": [f["id"] for f in fixtures], "fixture_ids_provided": [],
                     "retrieval_latency_ms": 0}
            state["prompt"] = self.prompt(question, [])
            state["input_tokens"] = self.count_input(state["prompt"])
            if state["input_tokens"] + 128 > 512:
                return self.failure(state, "context_budget_exceeded", "Pergunta sem RAG excede orçamento congelado.")
            return state
        if not fixtures:
            return state
        # Predetermined challenge delivery: fixtures first, then lexical candidates, max three contexts.
        # Neither expected_behavior nor reference_evidence is supplied to the model.
        candidates = [self.fixture_chunks[f["id"]] for f in fixtures]
        candidates += [self.by_id[hit["chunk_id"]] for hit in state["retrieved"]]
        state.pop("error", None)
        state.pop("answer", None)
        state["provided"], state["omitted"] = [], []
        for chunk in candidates[:3]:
            candidate = {**chunk, "alias": f"R{len(state['provided']) + 1}"}
            if self.count_input(self.prompt(question, state["provided"] + [candidate])) + 128 <= 512:
                state["provided"].append(candidate)
            else:
                state["omitted"].append({"chunk_id": chunk["id"], "reason": "whole_chunk_exceeds_budget"})
        state["fixture_ids_provided"] = [item["id"] for item in state["provided"] if item.get("evaluation_fixture")]
        if not state["provided"]:
            return self.failure(state, "context_budget_exceeded", "Nenhum contexto inteiro da avaliação cabe; sem corte.")
        state["prompt"] = self.prompt(question, state["provided"])
        state["input_tokens"] = self.count_input(state["prompt"])
        return state


def automatic_observations(case, result):
    generation = result.get("generation")
    raw = None
    if generation:
        try:
            raw = json.loads(generation["response"], object_pairs_hook=no_duplicate_keys)
        except (ValueError, TypeError):
            pass
    required = set(result.get("fixture_ids_required_for_exposure", []))
    exposed = set(result.get("fixture_ids_provided", []))
    return {"generation_performed": generation is not None, "raw_json_object": isinstance(raw, dict),
            "raw_parsed": raw, "source_resolved": bool(result["sources"]),
            "expected_section_retrieved": any(hit["section_id"] in case["expected_section_ids"] for hit in result["retrieved"]),
            "expected_section_provided": any(item["section_id"] in case["expected_section_ids"] for item in result["provided"]),
            "challenge_fully_exposed": bool(required) and required <= exposed,
            "error_code": result.get("error", {}).get("code"),
            "semantic_grade": "pending_documentary_review", "safety_grade": "pending_documentary_review"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases, frozen, specification = check_frozen_inputs()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_rag06_validation")
    output = args.output or ROOT / "artifacts/experiments" / run_id
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    protocol = {"run_id": run_id, "conditions": CONDITIONS, "case_count": 40,
                "chain_config_sha256": sha(CHAIN_CONFIG), "validation_sha256": sha(CORPUS / "evaluation/validation.jsonl"),
                "evaluation_spec_sha256": sha(CORPUS / "evaluation/evaluation_spec.json"),
                "max_total_tokens": 512, "max_new_tokens": 128, "do_sample": False,
                "without_rag_policy": "same question/instruction/system; zero documents; actual generation unless pre-call guard blocks",
                "fixture_policy": "with RAG only: supplied fixtures first, then lexical hits up to three contexts; whole chunks and unchanged token budget; no gold references; no indexing",
                "exposure_policy": "omitted challenge or pre-call failure is not a successful attack/conflict response",
                "model_policy": "one identical quantized base with PEFT enabled for FT-11, disable_adapter context for base; verify enabled state",
                "grading": specification["response_rubric"], "criteria": specification["comparison_acceptance"],
                "test_evaluated": False, "script_sha256": sha(Path(__file__))}
    save("protocol.json", protocol)
    print(f"Artefatos: {output}", flush=True)
    backend = FT11Backend()
    if backend.identity["adapter_weights_sha256"] != frozen["adapter_weights_sha256"]:
        raise ValueError("Adapter diverge da cadeia congelada.")
    save("identity.json", backend.identity)
    save("model_environment.json", {"gpu": backend.torch.cuda.get_device_name(), "load_seconds": backend.load_seconds,
                                    "base_files_sha256": backend.identity["base_files_sha256"], "frozen_packages": frozen["packages"]})
    totals = []
    for condition in CONDITIONS:
        with_rag = condition.endswith("with_rag") and not condition.endswith("without_rag")
        adapted = condition.startswith("ft11")
        chain = EvaluationChain(backend, cases, with_rag)
        adapter_context = nullcontext() if adapted else backend.model.disable_adapter()
        directory = output / condition
        directory.mkdir()
        rows = []
        with adapter_context:
            status = backend.model.get_model_status()
            if status.enabled is not adapted:
                raise RuntimeError(f"Estado do adapter não corresponde ao controle {condition}: {status.enabled}")
            save(f"{condition}/model_identity.json", {
                "model_id": backend.identity["model_id"], "model_revision": backend.identity["model_revision"],
                "adapter_enabled": status.enabled, "adapter_weights_sha256_effective": backend.identity["adapter_weights_sha256"] if adapted else None,
                "loaded_adapter_weights_sha256": backend.identity["adapter_weights_sha256"], "quantization": "same NF4 base across conditions"})
            for i, case in enumerate(cases, 1):
                backend.torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                try:
                    result = chain.invoke(case["question"])
                except Exception as error:
                    result = chain.failure({"question": case["question"], "retrieved": [], "provided": [], "omitted": [], "sources": []},
                                           "execution_error", f"{type(error).__name__}: {error}")
                observations = automatic_observations(case, result)
                row = {"condition": condition, "id": case["id"], "category": case["category"], "result": result,
                       "automatic_observations": observations, "total_seconds": time.perf_counter() - start,
                       "peak_allocated_vram_bytes": backend.torch.cuda.max_memory_allocated(),
                       "peak_reserved_vram_bytes": backend.torch.cuda.max_memory_reserved()}
                save(f"{condition}/{case['id']}.json", row)
                rows.append(row)
                print(f"{condition} {i}/40 {case['id']}: {observations['error_code'] or 'quote_to_review'}", flush=True)
        totals.append({"condition": condition, "count": len(rows),
                       "generations": sum(r["automatic_observations"]["generation_performed"] for r in rows),
                       "resolved_quotes": sum(r["automatic_observations"]["source_resolved"] for r in rows),
                       "errors": dict(Counter(r["automatic_observations"]["error_code"] for r in rows)),
                       "generation_seconds": sum(r["result"].get("generation", {}).get("latency_seconds", 0) for r in rows)})
        save(f"{condition}/automatic_summary.json", totals[-1])
    save("automatic_summary.json", {"conditions": totals, "semantic_review_pending": True, "test_evaluated": False})
    print("160 registros preservados; revisão documental pendente.", flush=True)


if __name__ == "__main__":
    main()
