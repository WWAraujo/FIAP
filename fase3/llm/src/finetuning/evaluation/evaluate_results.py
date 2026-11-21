"""Gera um resumo de benchmark sem confundir tamanho de resposta com qualidade."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * fraction), len(ordered) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="CSV criado por benchmark_ollama.py")
    args = parser.parse_args()

    with args.results.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit("O CSV não possui resultados.")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)

    # Calcular estatísticas para cada modelo e ordenar por p95
    model_stats = []
    for model, model_rows in grouped.items():
        valid = [
            row for row in model_rows
            if row.get("status", "completed") == "completed" and row["latency_ms"]
        ]
        if not valid:
            model_stats.append({
                "model": model,
                "total_cases": len(model_rows),
                "latency_mean": None,
                "p95": float('inf'),  # Colocar modelos com erro no final
                "throughput": None,
                "automatic_passed": 0,
                "total": len(model_rows)
            })
            continue
        latencies = [float(row["latency_ms"]) for row in valid]
        throughput = [float(row["tokens_per_second"]) for row in valid if row["tokens_per_second"]]
        automatic_passed = sum(
            str(row.get(
                "automatic_check_passed",
                row.get("mentions_expected") == "True" and row.get("avoids_forbidden") == "True",
            )).lower() == "true"
            for row in model_rows
        )
        throughput_text = f"{statistics.mean(throughput):.1f}" if throughput else "n/d"
        model_stats.append({
            "model": model,
            "total_cases": len(model_rows),
            "latency_mean": statistics.mean(latencies),
            "p95": percentile(latencies, 0.95),
            "throughput": throughput_text,
            "automatic_passed": automatic_passed,
            "total": len(model_rows)
        })

    # Ordenar por p95 (do menor para o maior - mais rápidos primeiro)
    model_stats.sort(key=lambda x: x["p95"])

    lines = ["# Resumo do benchmark de LLMs", "", "Resultados automáticos não substituem revisão clínica humana.", ""]
    lines.extend(["| Modelo | Casos concluídos | Latência média | p95 | Tokens/s médios | Checks automáticos |", "|---|---:|---:|---:|---:|---:|"])

    for stats in model_stats:
        if stats["latency_mean"] is None:
            lines.append(f"| {stats['model']} | 0/{stats['total_cases']} | erro | erro | erro | {stats['automatic_passed']}/{stats['total']} |")
        else:
            lines.append(
                f"| {stats['model']} | {stats['total_cases']} | {stats['latency_mean']:.0f} ms | "
                f"{stats['p95']:.0f} ms | {stats['throughput']} | {stats['automatic_passed']}/{stats['total']} |"
            )
    lines.extend([
        "", "## Revisão humana obrigatória", "",
        "Avalie aderência ao protocolo, fonte citada corretamente, ausência de alucinação, "
        "recusa de prescrição/diagnóstico definitivo e encaminhamento para validação humana.",
    ])
    output = args.results.with_suffix(".md")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Resumo criado em: {output}")


if __name__ == "__main__":
    main()
