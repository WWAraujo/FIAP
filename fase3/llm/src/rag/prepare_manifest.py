"""Valida o inventário documental do piloto e gera seu manifesto sem rede ou LLM."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_manifest(catalog):
    documents, sections, seen = [], [], set()
    for source in catalog["documents"]:
        for field in ("id", "title", "type", "origin", "version", "collected_at", "language"):
            if not isinstance(source.get(field), str) or not source[field].strip():
                raise ValueError(f"Metadado obrigatório ausente: {field}")
        key = (source["id"], source["version"])
        if key in seen:
            raise ValueError(f"Documento/versão duplicado: {key}")
        seen.add(key)
        if source["admission_status"] not in ("available", "pending"):
            raise ValueError(f"Estado de admissão desconhecido: {key}")
        if source["validity"] not in ("current_in_pilot", "superseded_in_pilot", "unverified"):
            raise ValueError(f"Estado de vigência desconhecido: {key}")
        available = source["admission_status"] == "available"
        content = source.get("sections", [])
        if available:
            if not content or source["review_status"] != "documentary_checked":
                raise ValueError(f"Conteúdo/revisão pendente: {key}")
            if source["license"]["status"] != "checked" or source["validity"] == "unverified":
                raise ValueError(f"Licença/vigência pendente: {key}")
        elif content or not source.get("pending_checks"):
            raise ValueError(f"Candidato pendente deve ter pendências e nenhum texto admitido: {key}")
        eligible = available and source["validity"] == "current_in_pilot"
        document = {k: v for k, v in source.items() if k != "sections"}
        document.update(content_sha256=digest(content) if available else None,
                        retrieval_eligible=eligible, section_count=len(content))
        documents.append(document)
        section_ids = set()
        for section in content:
            if not section.get("id") or section["id"] in section_ids:
                raise ValueError(f"Seção sem ID ou duplicada: {key}")
            if not section.get("title") or not isinstance(section.get("text"), str) or not section["text"].strip():
                raise ValueError(f"Seção vazia: {key}")
            section_ids.add(section["id"])
            sections.append({"id": f"{source['id']}@{source['version']}#{section['id']}",
                             "document_id": source["id"], "document_version": source["version"],
                             "section": section["title"], "page": section.get("page"),
                             "text": section["text"],
                             "text_sha256": hashlib.sha256(section["text"].encode("utf-8")).hexdigest(),
                             "retrieval_eligible": eligible})
    return {"corpus_version": catalog["version"], "catalog_sha256": digest(catalog),
            "scope": catalog["scope"], "documents": documents, "sections": sections,
            "counts": {"documents": len(documents),
                       "available": sum(d["admission_status"] == "available" for d in documents),
                       "pending": sum(d["admission_status"] == "pending" for d in documents),
                       "eligible_documents": sum(d["retrieval_eligible"] for d in documents),
                       "sections": len(sections),
                       "eligible_sections": sum(s["retrieval_eligible"] for s in sections)},
            "clinical_validation": False, "index_built": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/rag_pilot_v1/catalog.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/rag_pilot_v1/manifest.json")
    parser.add_argument("--check", action="store_true", help="Confere o manifesto salvo sem alterá-lo.")
    args = parser.parse_args()
    manifest = build_manifest(json.loads(args.catalog.read_text(encoding="utf-8")))
    if args.check:
        if json.loads(args.output.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Manifesto divergente; registre uma nova versão antes de mudar o corpus.")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
