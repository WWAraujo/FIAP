"""Triagem técnica do treino; não aprova conteúdo clínico nem lê o teste."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def groups(rows: list[dict], field: str) -> list[list[str]]:
    grouped = defaultdict(list)
    for row in rows:
        if field == "pair":
            key = (normalize(row["instruction"] + "\n" + row.get("input", "")), normalize(row["output"]))
        else:
            key = normalize(row.get(field, ""))
        grouped[key].append(row["source_id"])
    return [sorted(ids) for ids in grouped.values() if len(ids) > 1]


def technical_flags(row: dict) -> tuple[list[str], dict]:
    prompt = row["instruction"] + ("\n\n" + row["input"] if row.get("input") else "")
    answer = row["output"].strip()
    words = normalize(answer).split()
    ngrams = Counter(tuple(words[i:i + 6]) for i in range(max(0, len(words) - 5)))
    peak = max(ngrams.values(), default=0)
    repeated_fraction = sum(count - 1 for count in ngrams.values()) / max(1, len(words) - 5)
    sentences = [normalize(s) for s in re.split(r"[.!?]+(?:\s|$)", answer) if normalize(s)]
    repeated_sentences = max(Counter(sentences).values(), default=0)
    flags = []
    if len(normalize(prompt).split()) < 5:
        flags.append("short_prompt")
    if len(words) < 12:
        flags.append("short_answer")
    if len(words) > 180:
        flags.append("long_answer")
    if not re.search(r"[.!?][\s\"'”’)]*$", answer):
        flags.append("missing_terminal_punctuation")
    if re.search(r"(?:\.\.\.|…|[:,;\-])\s*[\"'”’)]*$", answer):
        flags.append("possible_unfinished_answer")
    if peak >= 3 or repeated_fraction >= 0.25 or repeated_sentences >= 2:
        flags.append("repetitive_answer")
    if not row.get("origin") or not row.get("source_id"):
        flags.append("missing_provenance")
    if row.get("origin") == "medpt" and not all(row.get(k) for k in ("source_url", "license", "question_type")):
        flags.append("missing_external_metadata")
    if re.search(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}|\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", prompt + "\n" + answer):
        flags.append("possible_pii")
    if re.search(r"doctoralia|agend\w*\s+consulta|reserv\w*\s+(?:uma\s+)?consulta|te\s+convidamos|r\$\s*\d", answer, re.I):
        flags.append("commercial_call_to_action")
    if re.search(r"(?:^|[.!?\n])\s*(?:[ÁáAa]t\.|[Aa]tenciosamente[:,])\s*[A-ZÀ-Ý][\wÀ-ÿ]+(?:\s+[A-ZÀ-Ý][\wÀ-ÿ]+){1,3}", answer):
        flags.append("possible_named_signature")
    return flags, {
        "prompt_chars": len(prompt), "answer_chars": len(answer),
        "prompt_words": len(normalize(prompt).split()), "answer_words": len(words),
        "max_sixgram_occurrences": peak, "repeated_sixgram_fraction": repeated_fraction,
        "max_sentence_occurrences": repeated_sentences,
    }


def distribution(values: list[int]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {}
    return {"min": ordered[0], "median": ordered[len(ordered) // 2],
            "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
            "max": ordered[-1], "mean": sum(ordered) / len(ordered)}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=root / "data/splits/train.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--medpt-limit", type=int, default=50)
    args = parser.parse_args()
    expected_train = (root / "data/splits/train.jsonl").resolve()
    if args.train_file.resolve() != expected_train:
        parser.error("Esta auditoria aceita exclusivamente o treino DAT-06 existente.")
    if args.medpt_limit < 1:
        parser.error("--medpt-limit deve ser positivo")
    from transformers import AutoTokenizer
    import runpy
    training = runpy.run_path(str(root / "src/finetuning/scripts/train_qwen3_8b_rocm.py"))
    model_dir = root / "models/base/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    rows = [json.loads(line) for line in args.train_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({row["source_id"] for row in rows}) != len(rows):
        raise ValueError("source_id duplicado: requer decisão explícita antes da triagem")
    duplicates = {field: groups(rows, field) for field in ("instruction", "output", "pair")}
    redundant_answers = {sid for group in duplicates["output"] for sid in group[1:]}
    redundant_pairs = {sid for group in duplicates["pair"] for sid in group[1:]}
    audits, eligible = [], []
    for row in sorted(rows, key=lambda r: r["source_id"]):
        flags, metrics = technical_flags(row)
        prompt = row["instruction"] + ("\n\n" + row["input"] if row.get("input") else "")
        messages = [{"role": "system", "content": training["SYSTEM_PROMPT"]}, {"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": row["output"]}], tokenize=False, enable_thinking=False, add_generation_prompt=False)
        metrics.update({"prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
                        "answer_tokens": len(tokenizer.encode(row["output"], add_special_tokens=False)),
                        "chat_tokens": len(tokenizer.encode(rendered, add_special_tokens=False))})
        if metrics["chat_tokens"] > 384:
            flags.append("chat_exceeds_384")
        if row["source_id"] in redundant_answers:
            flags.append("duplicate_answer")
        if row["source_id"] in redundant_pairs:
            flags.append("duplicate_pair")
        audits.append({"source_id": row["source_id"], "origin": row.get("origin"), "category": row.get("category"),
                       "answer_sha256": hashlib.sha256(row["output"].encode()).hexdigest(),
                       "flags": flags, "metrics": metrics, "technical_eligible": not flags,
                       "clinical_review": "pending"})
        if not flags:
            eligible.append(row)
    # Limite por categoria, round-robin e seed fixa; nenhuma métrica de validação seleciona exemplos.
    by_category = defaultdict(list)
    for row in eligible:
        if row.get("origin") == "medpt":
            by_category[row["category"]].append(row)
    for category, items in by_category.items():
        random.Random(f"{args.seed}:{category}").shuffle(items)
    selected_medpt = []
    while any(by_category.values()) and len(selected_medpt) < args.medpt_limit:
        for category in sorted(by_category):
            if by_category[category] and len(selected_medpt) < args.medpt_limit:
                selected_medpt.append(by_category[category].pop())
    candidate = sorted([row for row in eligible if row.get("origin") == "synthetic"] + selected_medpt, key=lambda r: r["source_id"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / "audit.jsonl", audits)
    write_jsonl(args.output_dir / "candidate_train.jsonl", candidate)
    write_jsonl(args.output_dir / "review.jsonl", [{"source_id": r["source_id"], "decision": "pending", "reviewer": None,
                 "evidence": [], "reason": None, "answer_sha256": hashlib.sha256(r["output"].encode()).hexdigest()} for r in candidate])
    (args.output_dir / "duplicates.json").write_text(json.dumps(duplicates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    provenance = root / "data/external/medpt/manifest.json"
    source_manifest = json.loads(provenance.read_text(encoding="utf-8"))
    summary = {
        "run_id": args.output_dir.name, "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "environment": {"executable": sys.executable, "python": sys.version,
            "transformers": importlib.metadata.version("transformers"), "tokenizers": importlib.metadata.version("tokenizers")},
        "train_file": str(args.train_file), "train_sha256": sha256(args.train_file),
        "script_sha256": sha256(Path(__file__)), "training_script_sha256": sha256(root / "src/finetuning/scripts/train_qwen3_8b_rocm.py"),
        "source_manifest_sha256": sha256(provenance), "source_manifest": source_manifest,
        "source_card_sha256": sha256(root / "data/external/medpt" / f"MedPT-{source_manifest['revision']}" / "README.md"),
        "tokenizer_sha256": {p.name: sha256(p) for p in sorted(model_dir.glob("*")) if p.is_file() and ("token" in p.name or "template" in p.name)},
        "seed": args.seed, "medpt_limit": args.medpt_limit, "input_count": len(rows),
        "origin_counts": dict(Counter(r.get("origin") for r in rows)),
        "flag_counts": dict(Counter(f for a in audits for f in a["flags"])),
        "flag_counts_by_origin": {origin: dict(Counter(f for a in audits if a["origin"] == origin for f in a["flags"]))
            for origin in sorted({a["origin"] for a in audits})},
        "duplicate_group_counts": {field: len(g) for field, g in duplicates.items()},
        "technical_eligible_count": len(eligible), "candidate_count": len(candidate),
        "candidate_origin_counts": dict(Counter(r.get("origin") for r in candidate)),
        "candidate_category_counts": dict(Counter(r.get("category") for r in candidate)),
        "lengths_by_origin": {origin: {key: distribution([a["metrics"][key] for a in audits if a["origin"] == origin])
            for key in ("prompt_chars", "answer_chars", "prompt_words", "answer_words", "prompt_tokens", "answer_tokens", "chat_tokens")}
            for origin in sorted({a["origin"] for a in audits})},
        "rules": {"min_prompt_words": 5, "min_answer_words": 12, "max_answer_words": 180,
            "max_chat_tokens": 384, "max_sixgram_occurrences_exclusive": 3,
            "max_repeated_sixgram_fraction_exclusive": 0.25, "max_sentence_occurrences_exclusive": 2,
            "duplicate_policy": "keep lexicographically first source_id for each normalized answer/pair",
            "terminal_punctuation": ".!? followed only by whitespace or closing quotes/parentheses",
            "commercial_pattern": "doctoralia|agend\\w*\\s+consulta|reserv\\w*\\s+(?:uma\\s+)?consulta|te\\s+convidamos|r\\$\\s*\\d (case insensitive)",
            "signature_pattern": "salutation At./Át./Atenciosamente: followed by at least two capitalized name components",
            "selection": "all technically eligible synthetic + category round-robin capped MedPT"},
        "clinical_approval": False, "training_authorized_by_audit": False,
        "limitations": ["Punctuation is a heuristic, not proof of semantic completeness.",
            "Automatic filters do not validate clinical facts, safety, anonymization or licensing.",
            "MedPT question types are model-generated labels, not clinical curation.",
            "CC-BY-4.0 is inherited from ingestion metadata; the local source card does not establish that license.",
            "Synthetic templates are correlated; deduplication reduces but does not remove that correlation.",
            "Candidate requires independent semantic review with identifiable evidence before training."],
        "reserved_test_accessed": False, "validation_accessed": False,
        "artifact_sha256": {p.name: sha256(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("run_id", "input_count", "flag_counts", "technical_eligible_count", "candidate_count", "candidate_origin_counts", "clinical_approval")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
