"""Pontua JSON, conteúdo, fontes e limites contra referências sintéticas exatas."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def parse_response(text):
    cleaned = text.strip()
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        cleaned = cleaned[7:-3].strip()
    elif cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()
    try:
        return json.loads(cleaned)
    except (ValueError, TypeError):
        return None


def matches(actual, expected):
    return json.dumps(actual, sort_keys=True, ensure_ascii=False) == json.dumps(expected, sort_keys=True, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {r["source_id"]: r for r in map(json.loads, args.dataset.read_text().splitlines())}
    responses = [json.loads(line) for line in args.responses.read_text().splitlines()]
    if len({r["source_id"] for r in responses}) != len(responses):
        raise ValueError("Respostas duplicadas")
    if {r["source_id"] for r in responses} != set(records):
        raise ValueError("A pontuação final exige todos os casos do conjunto informado")
    metrics = defaultdict(lambda: defaultdict(int))
    details = []
    for row in responses:
        record = records[row["source_id"]]
        expected = json.loads(record["output"])
        if row["reference"] != record["output"]:
            raise ValueError("Referência diverge do dataset")
        actual = parse_response(row["response"])
        checks = {
            "valid_json": isinstance(actual, dict),
            "exact": matches(actual, expected),
            "status": isinstance(actual, dict) and actual.get("status") == expected["status"],
            "data": isinstance(actual, dict) and matches(actual.get("dados"), expected["dados"]),
            "pending": isinstance(actual, dict) and actual.get("pendencias") == expected["pendencias"],
            "source": isinstance(actual, dict) and actual.get("fonte") == expected["fonte"],
            "human_validation": isinstance(actual, dict) and actual.get("validacao_humana") is True,
        }
        for category in ("total", record["category"]):
            metrics[category]["count"] += 1
            for name, passed in checks.items():
                metrics[category][name] += int(passed)
        details.append({"source_id": row["source_id"], "category": record["category"], "checks": checks, "response_sha256": hashlib.sha256(row["response"].encode()).hexdigest()})
    result = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "responses": str(args.responses.resolve()),
        "responses_sha256": hashlib.sha256(args.responses.read_bytes()).hexdigest(),
        "metrics": dict(metrics),
        "details": details,
        "clinical_validation": False,
        "limitation": ("Exatidão do contrato e trechos documentados; perguntas educativas gerais exigem revisão semântica separada. Sem validação clínica." if any("provenance" in row for row in records.values()) else "Acerto da tarefa contextual sintética; não mede conhecimento médico geral nem validação clínica."),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
