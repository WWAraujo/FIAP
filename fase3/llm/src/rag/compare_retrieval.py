"""Compara BM25 e E5/CPU somente na validação, sem geração de respostas."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
import unicodedata

from prepare_manifest import build_manifest

ROOT = Path(__file__).resolve().parents[2]
MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
STOPWORDS = set("a o as os de da do das dos e em no na nos nas um uma uns umas para por com que qual quais como ao aos se sua seu suas seus esta este esse essa foi ser deve pode segundo diante conforme do da pelo pela".split())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def peak_rss_mib():
    # Linux getrusage can retain the launcher's pre-exec peak; VmHWM belongs to this process image.
    status = Path("/proc/self/status")
    if status.exists():
        match = re.search(r"^VmHWM:\s+(\d+) kB", status.read_text(), re.MULTILINE)
        if match:
            return int(match[1]) / 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, obj):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def tokens(text):
    text = "".join(c for c in unicodedata.normalize("NFD", text.lower()) if not unicodedata.combining(c))
    return [t for t in re.findall(r"\w+", text) if t not in STOPWORDS]


def make_chunks(manifest, tokenizer, size=192, overlap=32):
    if not 0 <= overlap < size:
        raise ValueError("Sobreposição precisa ser menor que o trecho.")
    titles = {f"{d['id']}@{d['version']}": d["title"] for d in manifest["documents"]}
    chunks = []
    for section in manifest["sections"]:
        if not section["retrieval_eligible"]:
            continue
        text = section["text"]
        offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        for n, start_token in enumerate(range(0, len(offsets), size - overlap)):
            end_token = min(start_token + size, len(offsets))
            start = 0 if start_token == 0 else offsets[start_token][0]
            end = len(text) if end_token == len(offsets) else offsets[end_token - 1][1]
            # Expand to word boundaries: never split a number/unit or negation in half.
            while start > 0 and not text[start - 1].isspace():
                start -= 1
            while end < len(text) and not text[end].isspace():
                end += 1
            chunk = {"id": f"{section['id']}~{n:03d}", "section_id": section["id"],
                     "document_id": section["document_id"], "version": section["document_version"],
                     "page": section["page"], "char_start": start, "char_end": end,
                     "text": text[start:end], "text_sha256": hashlib.sha256(text[start:end].encode()).hexdigest()}
            chunk["search_text"] = titles[f"{chunk['document_id']}@{chunk['version']}"] + "\n" + section["section"] + "\n" + chunk["text"]
            if len(tokenizer("passage: " + chunk["search_text"])["input_ids"]) > 512:
                raise ValueError(f"Trecho excede limite E5: {chunk['id']}")
            chunks.append(chunk)
            if end_token == len(offsets):
                break
    return chunks


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.counts = [Counter(tokens(c["search_text"])) for c in chunks]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = statistics.mean(self.lengths)
        self.k1, self.b = k1, b
        df = Counter(t for c in self.counts for t in c)
        self.idf = {t: math.log(1 + (len(chunks) - count + 0.5) / (count + 0.5)) for t, count in df.items()}

    def score(self, question):
        terms = set(tokens(question))
        return [sum(self.idf.get(t, 0) * (counts[t] * (self.k1 + 1)) /
                    (counts[t] + self.k1 * (1 - self.b + self.b * length / self.average))
                    for t in terms if counts[t]) for counts, length in zip(self.counts, self.lengths)]


def ranking(chunks, scores, k=3):
    # Lexical zero is no match; semantic cosine can still retrieve irrelevant passages.
    order = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), chunks[i]["id"]))
    return [{"chunk_id": chunks[i]["id"], "section_id": chunks[i]["section_id"],
             "score": float(scores[i])} for i in order if float(scores[i]) > 0][:k]


def encode(texts, tokenizer, model, torch):
    vectors = []
    for start in range(0, len(texts), 8):
        batch = texts[start:start + 8]
        if any(len(tokenizer(text)["input_ids"]) > 512 for text in batch):
            raise ValueError("Entrada E5 excede 512 tokens; truncamento proibido.")
        inputs = tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            vectors.append(torch.nn.functional.normalize(pooled, p=2, dim=1))
    return torch.cat(vectors)


def worker(args):
    chunks = read(args.output / "index.json")["chunks"]
    questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines()]
    start = time.perf_counter()
    if args.worker == "lexical":
        retriever = BM25(chunks)
        score = retriever.score
    else:
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(4)
        torch.manual_seed(0)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        model = AutoModel.from_pretrained(args.model_path, local_files_only=True).to("cpu").eval()
        passages = encode(["passage: " + c["search_text"] for c in chunks], tokenizer, model, torch)
        np.save(args.output / "embeddings.npy", passages.numpy())
        def score(question):
            query = encode(["query: " + question], tokenizer, model, torch)
            return (query @ passages.T)[0].tolist()
    build_seconds = time.perf_counter() - start
    # One warm-up query; measured queries include encoding plus scoring/ranking.
    score("Como registrar uma pendência documental?")
    results, durations = [], []
    for question in questions:
        start = time.perf_counter()
        retrieved = ranking(chunks, score(question["question"]))
        duration = (time.perf_counter() - start) * 1000
        durations.append(duration)
        expected = set(question["expected_section_ids"])
        ranks = [i + 1 for i, item in enumerate(retrieved) if item["section_id"] in expected]
        results.append({"id": question["id"], "category": question["category"], "question": question["question"],
                        "retrieved": retrieved, "latency_ms": duration,
                        "section_hit_at_3": bool(ranks) if question["category"] == "answerable" else None,
                        "reciprocal_rank_at_3": 1 / ranks[0] if ranks else 0})
    answerable = [r for r in results if r["category"] == "answerable"]
    stats = {"engine": args.worker, "answerable_hits_at_3": sum(r["section_hit_at_3"] for r in answerable),
             "answerable_count": len(answerable), "mrr_at_3": statistics.mean(r["reciprocal_rank_at_3"] for r in answerable),
             "median_latency_ms": statistics.median(durations), "p95_latency_ms": sorted(durations)[math.ceil(len(durations) * .95) - 1],
             "build_seconds": build_seconds, "peak_process_rss_mib": peak_rss_mib(),
             "errors": 0, "truncations": 0, "results": results}
    write(args.output / f"{args.worker}.json", stats)
    print(json.dumps({k: v for k, v in stats.items() if k != "results"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/rag_pilot_v2")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=["lexical", "semantic"], help=argparse.SUPPRESS)
    parser.add_argument("--questions", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/embeddings/multilingual-e5-small" )
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    manifest_path = args.corpus / "manifest.json"
    manifest = read(manifest_path)
    if build_manifest(read(args.corpus / "catalog.json")) != manifest:
        raise ValueError("Catálogo/manifesto divergentes.")
    specification = read(args.corpus / "evaluation/evaluation_spec.json")
    questions = args.corpus / "evaluation/validation.jsonl"
    if sha(manifest_path) != specification["manifest_sha256"] or sha(questions) != specification["splits_sha256"]["validation"]:
        raise ValueError("Corpus/validação divergem da versão congelada.")
    # Deliberately never open test.jsonl. Gold references are used only by scoring.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_rag03")
    output = args.output or ROOT / "artifacts/experiments" / run_id
    output.mkdir(parents=True, exist_ok=False)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    print(f"run_id={run_id}; baixando/conferindo E5 {REVISION}", flush=True)
    snapshot_download(MODEL, revision=REVISION, local_dir=args.model_path,
                      allow_patterns=["config.json", "model.safetensors", "tokenizer*", "special_tokens_map.json", "sentencepiece.bpe.model", "README.md"])
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    chunks = make_chunks(manifest, tokenizer)
    index = {"corpus_version": manifest["corpus_version"], "manifest_sha256": sha(manifest_path), "chunks": chunks}
    write(output / "index.json", index)
    config = {"corpus_version": manifest["corpus_version"], "manifest_sha256": sha(manifest_path),
              "index_sha256": sha(output / "index.json"), "validation_sha256": sha(questions),
              "chunk_tokens": 192, "overlap_tokens": 32, "top_k": 3,
              "search_input": "título + seção + texto original; sem QAs/gabaritos/fixtures",
              "lexical": {"algorithm": "BM25", "k1": 1.5, "b": .75, "normalization": "minúsculas, sem acentos, palavras; stopwords PT fixas"},
              "semantic": {"model": MODEL, "revision": REVISION, "license": "MIT", "device": "cpu", "threads": 4,
                           "pooling": "attention-masked mean; L2", "query_prefix": "query: ", "passage_prefix": "passage: ",
                           "max_tokens": 512, "truncation": False, "dtype": "float32", "batch_size": 8},
              "selection_rule": "maior hit@3 respondível; depois MRR@3; empate favorece lexical",
              "abstention_threshold": None, "test_evaluated": False, "generation_performed": False}
    write(output / "comparison_config.json", config)
    for engine in ("lexical", "semantic"):
        subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker", engine,
                        "--output", str(output), "--questions", str(questions), "--model-path", str(args.model_path)], check=True)
    scores = [read(output / f"{engine}.json") for engine in ("lexical", "semantic")]
    best = max(scores, key=lambda s: (s["answerable_hits_at_3"], s["mrr_at_3"], s["engine"] == "lexical"))
    config["selected_engine"] = best["engine"]
    config["comparison_config_sha256"] = sha(output / "comparison_config.json")
    write(output / "selected_config.json", config)
    import importlib.metadata
    environment = {"run_id": run_id, "python": sys.version, "platform": platform.platform(),
                   "cpu": platform.processor(), "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "huggingface_hub")},
                   "script_sha256": sha(Path(__file__)), "model_files": {p.name: sha(p) for p in args.model_path.iterdir() if p.is_file()},
                   "index_chunks": len(chunks), "embeddings_sha256": sha(output / "embeddings.npy"),
                   "memory_scope": "Linux VmHWM de workers separados; inclui imports/modelo, exclui pico pré-exec do launcher; não é memória incremental", "gpu_used": False}
    write(output / "environment.json", environment)
    summary = {"run_id": run_id, "selected_engine": best["engine"],
               "minimum_required_hits": 22, "selected_meets_section_hit_target": best["answerable_hits_at_3"] >= 22,
               "engines": [{k: v for k, v in s.items() if k != "results"} for s in scores],
               "limitations": ["Hit de seção não prova suporte do trecho nem adequação da resposta; revisar trechos.",
                               "Ausência, conflito e ataques não avaliam comportamento da LLM nesta tarefa.",
                               "Sem limiar de abstenção; similaridade não comprova evidência.",
                               "Corpus pequeno documental, sem anexos clínicos; teste reservado não executado."]}
    write(output / "summary.json", summary)
    print(f"Resultado: {output}; selecionado: {best['engine']}", flush=True)


if __name__ == "__main__":
    main()
