"""Baixa uma amostra reproduzível do MedQuAD oficial para uso acadêmico."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import requests


REPOSITORY_URL = "https://github.com/abachaa/MedQuAD"
API_COMMIT_URL = "https://api.github.com/repos/abachaa/MedQuAD/commits/master"
LICENSE = "CC-BY-4.0"
DEFAULT_SAMPLE_SIZE = 1_000


def resolve_revision(revision: str | None, timeout: int) -> str:
    """Usa uma revisão explícita ou consulta o commit atual da fonte oficial."""
    if revision:
        return revision
    response = requests.get(API_COMMIT_URL, timeout=timeout)
    response.raise_for_status()
    return response.json()["sha"]


def download_archive(revision: str, destination: Path, timeout: int) -> str:
    """Baixa o ZIP da revisão fixada e retorna seu SHA-256."""
    archive_url = f"https://github.com/abachaa/MedQuAD/archive/{revision}.zip"
    response = requests.get(archive_url, stream=True, timeout=timeout)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with destination.open("wb") as archive_file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                archive_file.write(chunk)
                digest.update(chunk)
    return digest.hexdigest()


def extract_archive(archive_path: Path, destination: Path) -> Path:
    """Extrai ZIP confiável sem permitir caminhos fora do diretório de destino."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination_root):
                raise ValueError(f"Caminho inválido no ZIP: {member.filename}")
        archive.extractall(destination)

    roots = [path for path in destination.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise ValueError("Estrutura inesperada no arquivo MedQuAD")
    return roots[0]


def _child_text(node: ElementTree.Element, name: str) -> str:
    for child in node:
        tag = child.tag.rsplit("}", 1)[-1].lower()
        if tag == name:
            return " ".join(part.strip() for part in child.itertext() if part.strip())
    return ""


def extract_records(dataset_root: Path) -> list[dict[str, str]]:
    """Extrai pares de pergunta/resposta dos XMLs que ainda contêm resposta."""
    records: list[dict[str, str]] = []
    for xml_path in sorted(dataset_root.rglob("*.xml")):
        root = ElementTree.parse(xml_path).getroot()
        for node in root.iter():
            question = _child_text(node, "question")
            answer = _child_text(node, "answer")
            if not question or not answer:
                continue
            relative_path = xml_path.relative_to(dataset_root).as_posix()
            records.append(
                {
                    "source_id": f"MEDQUAD:{relative_path}:{len(records)}",
                    "category": "external_medquad_qa",
                    "instruction": question,
                    "input": "",
                    "output": answer,
                    "source": "MedQuAD",
                    "source_url": REPOSITORY_URL,
                    "license": LICENSE,
                    "language": "en",
                }
            )
    return records


def select_sample(records: list[dict[str, str]], sample_size: int, seed: int) -> list[dict[str, str]]:
    """Remove perguntas duplicadas e seleciona uma amostra estável por seed."""
    unique_records: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    for record in sorted(records, key=lambda item: item["source_id"]):
        normalized_question = " ".join(record["instruction"].casefold().split())
        if normalized_question not in seen_questions:
            seen_questions.add(normalized_question)
            unique_records.append(record)
    if sample_size > len(unique_records):
        raise ValueError(f"Amostra solicitada ({sample_size}) maior que fonte única ({len(unique_records)})")
    selected = random.Random(seed).sample(unique_records, sample_size)
    return sorted(selected, key=lambda item: item["source_id"])


def save_jsonl(records: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    project_dir = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Baixa e amostra o MedQuAD oficial.")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", help="Commit Git SHA; sem valor, consulta o HEAD atual.")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=project_dir / "data/external/medquad")
    args = parser.parse_args()

    revision = resolve_revision(args.revision, args.timeout)
    archive_path = args.output_dir / f"MedQuAD-{revision}.zip"
    extracted_root = args.output_dir / f"MedQuAD-{revision}"
    archive_hash = download_archive(revision, archive_path, args.timeout)
    dataset_root = extract_archive(archive_path, extracted_root)
    records = extract_records(dataset_root)
    sample = select_sample(records, args.sample_size, args.seed)
    sample_path = args.output_dir / "sample.jsonl"
    save_jsonl(sample, sample_path)

    manifest = {
        "source": "MedQuAD",
        "repository_url": REPOSITORY_URL,
        "revision": revision,
        "archive_sha256": archive_hash,
        "license": LICENSE,
        "language": "en",
        "seed": args.seed,
        "source_pair_count": len(records),
        "sample_count": len(sample),
        "sample_path": "sample.jsonl",
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
    print(f"MedQuAD importado: {len(sample)} exemplos, revisão {revision}")


if __name__ == "__main__":
    main()
