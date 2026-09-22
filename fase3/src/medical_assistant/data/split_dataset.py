"""Cria splits determinísticos e sem vazamento para o dataset sintético."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medical_assistant.data.preprocessing import MedicalDataPreprocessor


SPLIT_NAMES = ("train", "validation", "test")


def split_examples(examples: list[dict[str, str]], seed: int = 42) -> dict[str, list[dict[str, str]]]:
    """Divide por categoria e `source_id`, reservando teste de todos os tipos."""
    by_category: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for example in examples:
        source_group = example.get("source_group", example["source_id"])
        example["source_group"] = source_group
        by_category[example["category"]][source_group].append(example)

    splits: dict[str, list[dict[str, str]]] = {name: [] for name in SPLIT_NAMES}
    for category in sorted(by_category):
        category_groups = [
            by_category[category][group_id] for group_id in sorted(by_category[category])
        ]
        random.Random(f"{seed}:{category}").shuffle(category_groups)

        # Todo tipo aparece no teste. Categorias maiores usam 80/10/10;
        # categorias pequenas preservam ao menos um exemplo de avaliação.
        test_count = max(1, round(len(category_groups) * 0.1))
        validation_count = max(1, round(len(category_groups) * 0.1))
        if len(category_groups) - test_count - validation_count < 1:
            validation_count = 0
        splits["test"].extend(
            example for group in category_groups[:test_count] for example in group
        )
        splits["validation"].extend(
            example
            for group in category_groups[test_count : test_count + validation_count]
            for example in group
        )
        splits["train"].extend(
            example for group in category_groups[test_count + validation_count :] for example in group
        )

    _validate_splits(splits)
    return splits


def _validate_splits(splits: dict[str, list[dict[str, str]]]) -> None:
    """Falha cedo se um exemplo vazar entre os splits ou faltar no teste."""
    source_ids_by_split = {
        name: {example["source_id"] for example in examples}
        for name, examples in splits.items()
    }
    source_groups_by_split = {
        name: {example.get("source_group", example["source_id"]) for example in examples}
        for name, examples in splits.items()
    }
    for left_index, left_name in enumerate(SPLIT_NAMES):
        for right_name in SPLIT_NAMES[left_index + 1 :]:
            overlap = source_ids_by_split[left_name] & source_ids_by_split[right_name]
            if overlap:
                raise ValueError(f"Vazamento entre {left_name} e {right_name}: {sorted(overlap)}")
            group_overlap = source_groups_by_split[left_name] & source_groups_by_split[right_name]
            if group_overlap:
                raise ValueError(
                    f"Vazamento de grupo entre {left_name} e {right_name}: {sorted(group_overlap)}"
                )

    all_categories = {
        example["category"]
        for examples in splits.values()
        for example in examples
    }
    test_categories = {example["category"] for example in splits["test"]}
    missing_categories = all_categories - test_categories
    if missing_categories:
        raise ValueError(f"Teste sem categorias: {sorted(missing_categories)}")


def build_manifest(
    splits: dict[str, list[dict[str, str]]],
    source_metadata: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Registra receita, distribuição e limitações do split."""
    categories = sorted(
        {example["category"] for examples in splits.values() for example in examples}
    )
    distribution = {
        category: {
            split_name: sum(
                example["category"] == category for example in splits[split_name]
            )
            for split_name in SPLIT_NAMES
        }
        for category in categories
    }
    sparse_categories = [
        category
        for category, counts in distribution.items()
        if any(counts[split_name] == 0 for split_name in SPLIT_NAMES)
    ]
    record_counts = source_metadata.get("record_counts") or source_metadata.get(
        "synthetic", {}
    ).get("record_counts", {})
    source_count = sum(record_counts.values())

    def distribution_by(field: str, default: str) -> dict[str, dict[str, int]]:
        values = sorted(
            {
                example.get(field, default)
                for examples in splits.values()
                for example in examples
            }
        )
        return {
            value: {
                split_name: sum(
                    example.get(field, default) == value for example in splits[split_name]
                )
                for split_name in SPLIT_NAMES
            }
            for value in values
        }

    return {
        "schema_version": "1.0",
        "seed": seed,
        "strategy": "categoria + source_group, sem duplicação entre splits",
        "source_dataset": source_metadata,
        "split_counts": {name: len(examples) for name, examples in splits.items()},
        "category_distribution": distribution,
        "origin_distribution": distribution_by("origin", "external"),
        "language_distribution": distribution_by("language", "unknown"),
        "reserved_test_source_ids": sorted(
            example["source_id"] for example in splits["test"]
        ),
        "limitations": (
            f"Dataset sintético de {source_count} exemplos. Categorias com menos de três exemplos "
            "não aparecem em todos os splits; o teste, porém, contém todos os tipos."
            if sparse_categories
            else "Nenhuma limitação de cobertura por categoria detectada."
        ),
        "sparse_categories": sparse_categories,
    }


def save_jsonl(examples: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for example in examples:
            output_file.write(json.dumps(example, ensure_ascii=False) + "\n")


def main() -> None:
    project_dir = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Cria splits determinísticos para treino, validação e teste.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_dir / "data/curated/dataset_medico_anonimizado.json",
    )
    parser.add_argument("--output-dir", type=Path, default=project_dir / "data/splits")
    parser.add_argument(
        "--external-input",
        type=Path,
        help="JSONL externo já curado, como o sample.jsonl gerado por ingest_medquad.py.",
    )
    parser.add_argument(
        "--external-manifest",
        type=Path,
        help="Manifesto de proveniência do JSONL externo.",
    )
    parser.add_argument(
        "--external-limit",
        type=int,
        help="Quantidade máxima de exemplos externos a incluir, escolhidos deterministicamente.",
    )
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as input_file:
        curated_dataset = json.load(input_file)
    examples = MedicalDataPreprocessor().format_for_fine_tuning(curated_dataset)
    external_metadata: dict[str, Any] = {}
    if args.external_input:
        with args.external_input.open(encoding="utf-8") as external_file:
            external_examples = [json.loads(line) for line in external_file if line.strip()]
        if args.external_limit is not None:
            if args.external_limit < 1 or args.external_limit > len(external_examples):
                raise ValueError("--external-limit deve estar entre 1 e o total de exemplos externos")
            external_examples = random.Random(f"{args.seed}:external-limit").sample(
                sorted(external_examples, key=lambda example: example["source_id"]),
                args.external_limit,
            )
        examples.extend(external_examples)
        if args.external_manifest:
            with args.external_manifest.open(encoding="utf-8") as manifest_file:
                external_metadata = json.load(manifest_file)
        external_metadata["selected_for_corpus"] = len(external_examples)
    splits = split_examples(examples, seed=args.seed)

    for split_name, split_examples_data in splits.items():
        save_jsonl(split_examples_data, args.output_dir / f"{split_name}.jsonl")
    manifest = build_manifest(
        splits,
        {
            "synthetic": curated_dataset.get("dataset_metadata", {}),
            "external": external_metadata,
        },
        seed=args.seed,
    )
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)

    print("Splits criados:", manifest["split_counts"])


if __name__ == "__main__":
    main()
