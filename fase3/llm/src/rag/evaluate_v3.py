"""Quatro controles v3, configuração congelada e teste liberado somente após validação."""
import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time

from document_chain_v3 import DocumentChainV3, FT11BackendV3, INDEX, INSTRUCTION, VERIFY_INSTRUCTION, OUTPUT_TOKENS, TOTAL_TOKENS
from evaluate_validation import CONDITIONS, CORPUS, check_frozen_inputs, automatic_observations
from ft11_backend import ROOT, read, sha

RESERVED = ROOT / "data/rag_reserved_v5"


class EvaluationChainV3(DocumentChainV3):
    def __init__(self, backend, cases, with_rag):
        super().__init__(backend)
        self.cases = {c["question"]: c for c in cases}
        self.with_rag = with_rag
        self.fixture_chunks = {}
        for case in cases:
            for f in case.get("context_fixture", []):
                sid = f["id"] + "#fixture"
                self.fixture_chunks[f["id"]] = {"id": f["id"], "section_id": sid, "document_id": f["document_id"],
                    "version": f["version"], "page": None, "char_start": 0, "char_end": len(f["text"]),
                    "text": f["text"], "evaluation_fixture": True, "validity": f["validity"]}
                self.documents[f"{f['document_id']}@{f['version']}"] = {
                    "title": f.get("title", f["document_id"]), "url": None, "validity": f["validity"]}
                self.sections[sid] = {"section": "Documento fornecido na avaliação"}

    def prepare(self, question):
        state = super().prepare(question)
        fixtures = self.cases[question].get("context_fixture", [])
        state.update(fixture_ids_required_for_exposure=[f["id"] for f in fixtures], fixture_ids_provided=[])
        if state.get("error", {}).get("code") in {"blocked_request", "invalid_request"}:
            return state
        if not self.with_rag:
            state.update(provided=[], omitted=[], retrieved=[], sources=[], retrieval_latency_ms=0)
            return self.budget(state, [])
        if fixtures:
            candidates = [self.fixture_chunks[f["id"]] for f in fixtures]
            candidates += [self.by_id[h["chunk_id"]] for h in state["retrieved"]]
            state = self.budget(state, candidates[:3])
            state["fixture_ids_provided"] = [c["id"] for c in state["provided"] if c.get("evaluation_fixture")]
        return state


def freeze():
    _, previous, _ = check_frozen_inputs()
    return {"version": "document_chain_v3", "instruction": INSTRUCTION, "verification_instruction": VERIFY_INSTRUCTION, "total_tokens": TOTAL_TOKENS,
            "support_verifier": "same quantized base with adapter disabled, shared across conditions; selector uses condition's adapter state",
            "decoding": "greedy JSON choices from provided paragraph IDs or absence; support boolean; explicit chat EOS",
            "output_tokens": OUTPUT_TOKENS, "index_config_sha256": sha(INDEX / "frozen_config.json"),
            "reserved_protocol_sha256": sha(RESERVED / "protocol.json"),
            "packages": previous["packages"], "adapter_weights_sha256": previous["adapter_weights_sha256"],
            "script_hashes": {name: sha(Path(__file__).parent / name) for name in
                ("document_chain_v3.py", "document_chain_v2.py", "document_chain.py", "evaluate_v3.py",
                 "evaluate_validation.py", "ft11_backend.py", "compare_retrieval.py", "prepare_manifest.py", "review_v3.py", "prepare_reserved_v5.py")}}


def approved_validation(directory, frozen):
    decision = read(directory / "decision.json")
    protocol = read(directory / "protocol.json")
    summary = read(directory / "reviewed_summary.json")
    reviewed = read(directory / "documentary_review.json")
    if (decision.get("advance_to_reserved_test") is not True or decision.get("documentary_ai_signoff") is not True or
            not decision.get("checks") or not all(decision["checks"].values()) or
            protocol["split"] != "validation" or protocol["partial_probe"] or protocol["case_count"] != 40 or
            set(protocol["conditions"]) != set(CONDITIONS) or read(directory / "chain_config.json") != frozen or
            protocol["chain_config_sha256"] != sha(directory / "chain_config.json") or
            decision["review_sha256"] != sha(directory / "documentary_review.json") or
            decision["summary_sha256"] != sha(directory / "reviewed_summary.json") or
            len(reviewed["records"]) != 160 or len(summary["conditions"]) != 4 or
            any(s["count"] != 40 for s in summary["conditions"])):
        raise ValueError("Validação incompleta/não aprovada ou configuração/revisão divergente; teste não lido.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--approved-validation", type=Path)
    p.add_argument("--case-numbers", nargs="+", type=int, help="Sondagem na validação; nunca no teste.")
    args = p.parse_args()
    if args.split == "test" and args.case_numbers:
        p.error("Não são permitidas sondagens no teste reservado.")
    frozen = freeze()
    if args.split == "test":
        if not args.approved_validation:
            p.error("Teste exige --approved-validation.")
        approved = args.approved_validation
        approved_validation(approved, frozen)
        if (RESERVED / "consumed.json").exists():
            raise ValueError("Teste já consumido; não repetir como avaliação independente.")
        path = RESERVED / "test.jsonl"
        if sha(path) != read(RESERVED / "protocol.json")["test_sha256"]:
            raise ValueError("Teste reservado alterado.")
        cases = [json.loads(line) for line in path.read_text().splitlines()]
    else:
        cases, _, _ = check_frozen_inputs()
        path = CORPUS / "evaluation/validation.jsonl"
        if args.case_numbers:
            if any(n < 1 or n > 40 for n in args.case_numbers):
                p.error("case-numbers deve estar entre 1 e 40")
            cases = [cases[n - 1] for n in args.case_numbers]
    run_id = datetime.now(timezone.utc).strftime(f"%Y%m%dT%H%M%S%fZ_rag09_{args.split}")
    output = ROOT / "artifacts/experiments" / run_id
    output.mkdir(exist_ok=False)
    def save(name, value):
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    save("chain_config.json", frozen)
    (output / "source").mkdir()
    for name in frozen["script_hashes"]:
        (output / "source" / name).write_bytes((Path(__file__).parent / name).read_bytes())
    if args.split == "test":
        with (RESERVED / "consumed.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"run_id": run_id, "validation": str(approved), "configuration_sha256": sha(output / "chain_config.json")}) + "\n")
    save("protocol.json", {"conditions": args.conditions, "split": args.split, "case_count": len(cases),
         "cases_sha256": sha(path), "chain_config_sha256": sha(output / "chain_config.json"),
         "partial_probe": set(args.conditions) != set(CONDITIONS) or len(cases) != 40,
         "case_ids": [c["id"] for c in cases], "do_sample": False,
         "gold_in_prompt": False, "test_consumed": args.split == "test",
         "exposure": "full fixtures, no indexing, no truncation; errors/omission not safety approval",
         "review": "documentary AI review, not clinical validation"})
    print(f"Artefatos: {output}", flush=True)
    backend = FT11BackendV3()
    if backend.identity["adapter_weights_sha256"] != frozen["adapter_weights_sha256"]:
        raise ValueError("Adapter divergente.")
    for name, version in frozen["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Pacote divergente: {name}")
    save("identity.json", backend.identity)
    if args.split == "test":
        previous_identity = read(approved / "identity.json")
        for key in ("model_id", "model_revision", "base_files_sha256", "adapter_weights_sha256", "system_prompt"):
            if backend.identity[key] != previous_identity[key]:
                raise ValueError(f"Identidade do modelo diverge da validação: {key}")
    for condition in args.conditions:
        adapted = condition.startswith("ft11")
        chain = EvaluationChainV3(backend, cases, condition.endswith("with_rag") and not condition.endswith("without_rag"))
        (output / condition).mkdir()
        with nullcontext() if adapted else backend.model.disable_adapter():
            if backend.model.get_model_status().enabled is not adapted:
                raise RuntimeError("Estado do adapter divergente.")
            save(f"{condition}/model_identity.json", {"adapter_enabled": adapted,
                 "identity_stage": "selector", "support_verifier_adapter_enabled": False,
                 "effective_adapter_sha256": frozen["adapter_weights_sha256"] if adapted else None})
            for n, case in enumerate(cases, 1):
                backend.torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                try:
                    result = chain.invoke(case["question"])
                except Exception as error:
                    result = chain.failure({"question": case["question"], "provided": [], "omitted": [], "retrieved": [], "sources": []},
                                           "execution_error", f"{type(error).__name__}: {error}")
                obs = automatic_observations(case, result)
                save(f"{condition}/{case['id']}.json", {"condition": condition, "id": case["id"], "category": case["category"],
                     "result": result, "automatic_observations": obs, "total_seconds": time.perf_counter() - start,
                     "peak_allocated_vram_bytes": backend.torch.cuda.max_memory_allocated()})
                print(f"{condition} {n}/{len(cases)} {case['id']}: {obs['error_code'] or result.get('outcome')}", flush=True)
    save("automatic_summary.json", {"semantic_review_pending": True, "test_consumed": args.split == "test"})


if __name__ == "__main__":
    main()
