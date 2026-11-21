"""Consolida revisão documental já concluída, sem inferir notas pelo texto do modelo."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "mapping", "grades", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    records = {r["source_id"]: r for r in map(json.loads, args.dataset.read_text().splitlines()) if r["category"].startswith("educativo_")}
    mapping = json.loads(args.mapping.read_text())
    grades = [json.loads(line) for line in args.grades.read_text().splitlines()]
    if len({r["id"] for r in grades}) != len(grades) or {r["id"] for r in grades} != set(records) or set(mapping) != set(records):
        raise ValueError("Revisão deve conter exatamente todos os casos educativos, sem duplicados")
    metrics = defaultdict(Counter)
    categories = defaultdict(lambda: defaultdict(Counter))
    details = []
    for row in grades:
        if set(row["grades"]) != set(mapping[row["id"]]):
            raise ValueError("Rótulos da revisão não correspondem aos modelos ocultados")
        for label, verdict in row["grades"].items():
            if verdict not in ("adequado", "parcial", "inadequado"):
                raise ValueError("Nota desconhecida")
            system = mapping[row["id"]][label]
            metrics[system][verdict] += 1
            categories[system][records[row["id"]]["category"]][verdict] += 1
            details.append({"source_id": row["id"], "system": system, "grade": verdict, "note": row["notes"][label], "evidence": row.get("evidence", [])})
    result = {"unblinded_at": datetime.now(timezone.utc).isoformat(), "count_per_system": len(records), "metrics": metrics, "categories": categories, "details": details, "reviewer": "Revisão documental por IA; não é validação clínica", "clinical_validation": False}
    result["inputs"] = {name: {"path": str(getattr(args, name).resolve()), "sha256": hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()} for name in ("dataset", "mapping", "grades")}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
