"""Baixa e cura uma amostra reproduzível do MedPT em português brasileiro."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


DATASET_ID = "AKCIT/MedPT"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
LICENSE = "CC-BY-4.0"
DEFAULT_SAMPLE_SIZE = 5_000
EXCLUDED_QUESTION_TYPES = {"diagnosis", "diagnostico", "treatment", "tratamento"}


def normalize(value: str) -> str:
    """Normaliza texto para comparações estáveis de categoria e duplicidade."""
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", normalized.casefold()).split())


def is_allowed_question_type(question_type: str) -> bool:
    """Exclui perguntas rotuladas como diagnóstico ou tratamento direto."""
    return normalize(question_type) not in EXCLUDED_QUESTION_TYPES


def record_from_row(row: dict[str, Any]) -> dict[str, str] | None:
    """Converte uma linha MedPT em exemplo de SFT, preservando sua proveniência."""
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    question_type = str(row.get("question_type", "")).strip()
    if not question or not answer or not is_allowed_question_type(question_type):
        return None

    record_id = str(row.get("id", "")).strip()
    if not record_id:
        return None
    question_type_slug = normalize(question_type).replace(" ", "_") or "other"
    return {
        "source_id": f"MEDPT:{record_id}",
        "category": f"external_medpt_{question_type_slug}",
        "instruction": question,
        "input": "",
        "output": answer,
        "source": "MedPT",
        "source_url": DATASET_URL,
        "license": LICENSE,
        "language": "pt-BR",
        "origin": "medpt",
        "condition": str(row.get("condition", "")).strip(),
        "medical_specialty": str(row.get("medical_specialty", "")).strip(),
        "question_type": question_type,
    }


def extract_records(rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """Remove registros inelegíveis e perguntas duplicadas da fonte externa."""
    records: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    for row in rows:
        record = record_from_row(row)
        if record is None:
            continue
        normalized_question = normalize(record["instruction"])
        if normalized_question in seen_questions:
            continue
        seen_questions.add(normalized_question)
        records.append(record)
    return sorted(records, key=lambda item: item["source_id"])


def select_sample(records: list[dict[str, str]], sample_size: int, seed: int) -> list[dict[str, str]]:
    """Seleciona uma amostra determinística sem alterar a ordem de saída."""
    if sample_size > len(records):
        raise ValueError(
            f"Amostra solicitada ({sample_size}) maior que fonte elegível ({len(records)})"
        )
    return sorted(random.Random(seed).sample(records, sample_size), key=lambda item: item["source_id"])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_jsonl(records: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def download_rows(output_dir: Path, revision: str | None) -> tuple[list[dict[str, Any]], str, dict[str, str]]:
    """Baixa um snapshot fixado e lê os Parquet locais, sem depender do HEAD posterior."""
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    resolved_revision = revision or api.dataset_info(DATASET_ID).sha
    snapshot_dir = output_dir / f"MedPT-{resolved_revision}"
    if not snapshot_dir.exists():
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            revision=resolved_revision,
            local_dir=str(snapshot_dir),
            allow_patterns=["README.md", "data/**"],
        )
    parquet_files = sorted(snapshot_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("Snapshot MedPT sem arquivos Parquet.")
    dataset = load_dataset(
        "parquet",
        data_files=[str(path) for path in parquet_files],
        split="train",
        cache_dir=str(output_dir / ".datasets_cache"),
    )
    return list(dataset), resolved_revision, {
        path.relative_to(snapshot_dir).as_posix(): file_sha256(path) for path in parquet_files
    }


def main() -> None:
    project_dir = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", help="SHA do dataset; sem valor, resolve o HEAD uma única vez.")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "data" / "external" / "medpt")
    args = parser.parse_args()

    if args.sample_size < 1:
        raise ValueError("--sample-size deve ser positivo")
    rows, revision, file_hashes = download_rows(args.output_dir, args.revision)
    eligible_records = extract_records(rows)
    sample = select_sample(eligible_records, args.sample_size, args.seed)
    sample_path = args.output_dir / "sample.jsonl"
    save_jsonl(sample, sample_path)
    question_types = sorted({record["question_type"] for record in eligible_records})
    manifest = {
        "source": "MedPT",
        "dataset_id": DATASET_ID,
        "dataset_url": DATASET_URL,
        "revision": revision,
        "license": LICENSE,
        "language": "pt-BR",
        "seed": args.seed,
        "source_pair_count": len(rows),
        "eligible_pair_count": len(eligible_records),
        "excluded_question_types": sorted(EXCLUDED_QUESTION_TYPES),
        "eligible_question_types": question_types,
        "sample_count": len(sample),
        "sample_sha256": file_sha256(sample_path),
        "parquet_sha256": file_hashes,
        "sample_path": "sample.jsonl",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"sample": str(sample_path), **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
