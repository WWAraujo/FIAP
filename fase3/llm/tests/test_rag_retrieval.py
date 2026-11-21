"""Verifica filtro de fontes, cortes rastreáveis e ranking sem rede/modelo."""

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/rag"))
spec = importlib.util.spec_from_file_location("rag_retrieval", ROOT / "src/rag/compare_retrieval.py")
retrieval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrieval)


class WordTokenizer:
    def __call__(self, text, **kwargs):
        import re
        offsets = [m.span() for m in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


def test_index_filters_versions_and_preserves_text_offsets():
    text = "Exame não concluído: 4,7 mmol/L. Copiar a unidade sem conversão."
    section = dict(id="D@2#s", document_id="D", document_version="2", section="Unidades",
                   page=1, text=text, retrieval_eligible=True)
    manifest = {"documents": [dict(id="D", version="2", title="Documento")],
                "sections": [section, dict(section, id="D@1#s", retrieval_eligible=False)]}
    chunks = retrieval.make_chunks(manifest, WordTokenizer(), size=4, overlap=1)
    assert len(chunks) > 1
    coverage = set()
    for chunk in chunks:
        assert chunk["section_id"] == "D@2#s"
        assert chunk["text"] == text[chunk["char_start"]:chunk["char_end"]]
        coverage.update(range(chunk["char_start"], chunk["char_end"]))
    assert all(i in coverage for i, c in enumerate(text) if not c.isspace())
    assert any("4,7 mmol/L." in c["text"] for c in chunks)


def test_bm25_handles_accents_negations_and_unmatched_queries():
    chunks = [dict(id="a", section_id="A", search_text="Registro de pendências de exame não concluído"),
              dict(id="b", section_id="B", search_text="Controle de estoque e formulários")]
    scores = retrieval.BM25(chunks).score("pendencias exame")
    assert retrieval.ranking(chunks, scores)[0]["section_id"] == "A"
    assert "nao" in retrieval.tokens("não informado")
    assert retrieval.ranking(chunks, retrieval.BM25(chunks).score("astronomia")) == []


def test_top_three_has_deterministic_ties_and_no_extra_gold_input():
    chunks = [dict(id=str(i), section_id=str(i), search_text="texto") for i in range(5)]
    ranked = retrieval.ranking(chunks, [1] * 5)
    assert [r["chunk_id"] for r in ranked] == ["0", "1", "2"]
    assert retrieval.ranking(chunks, [0] * 5) == []
