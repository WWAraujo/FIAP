"""Fluxo LCEL com backend mockado, sem GPU, rede ou splits de avaliação."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/rag"))
spec = importlib.util.spec_from_file_location("document_chain", ROOT / "src/rag/document_chain.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Backend:
    def __init__(self):
        self.identity = {"system_prompt": "Sistema de teste"}
        self.tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kw: messages[-1]["content"].split())
        self.calls = []
        self.response = None
        self.limited = False

    def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        if self.response is None:
            document = prompt.split("Documento R1:\n", 1)[1].split("\n\nDocumento R2:", 1)[0]
            answer = dict(status="informativo", dados={"trecho": document}, pendencias=[], fonte="R1", validacao_humana=True)
            text = json.dumps(answer)
        else:
            text = self.response
        return dict(response=text, output_limit_reached=self.limited)


def valid_answer(**changes):
    value = dict(status="informativo", dados={"trecho": "Texto inexistente."}, pendencias=[], fonte="R1", validacao_humana=True)
    value.update(changes)
    return json.dumps(value)


def test_lcel_resolves_only_supplied_source_and_keeps_literal_text():
    backend = Backend()
    chain = module.DocumentChain(backend)
    assert chain.chain.__class__.__name__ == "RunnableSequence"
    result = chain.invoke("Transcrição documental de unidades e decimais")
    assert "error" not in result
    assert len(backend.calls) == 1
    source = result["sources"][0]
    assert result["answer"]["fonte"] == source["id"]
    assert source["title"] and source["version"] and source["section_id"]
    assert result["answer"]["dados"]["trecho"] == source["text"]
    assert source["id"] in {item["id"] for item in result["provided"]}
    assert source["quoted_text"] in source["text"]


@pytest.mark.parametrize("question,code", [("", "invalid_request"), (None, "invalid_request"),
                                         ("zebraluna orbitalx", "no_evidence"),
                                         ("Prescreva medicamento para mim", "blocked_request")])
def test_gap_or_block_does_not_reach_model(question, code):
    backend = Backend()
    result = module.DocumentChain(backend).invoke(question)
    assert result["error"]["code"] == code
    assert backend.calls == []
    assert result["answer"]["fonte"] is None and not result["sources"]


def test_budget_refuses_whole_chunks_without_truncation():
    backend = Backend()
    backend.tokenizer.apply_chat_template = lambda *a, **k: list(range(500))
    result = module.DocumentChain(backend).invoke("Transcrição de resultados")
    assert result["error"]["code"] == "context_budget_exceeded"
    assert result["omitted"] and result["provided"] == [] and backend.calls == []


def test_token_count_reads_input_ids_from_tokenizer_mapping():
    backend = Backend()
    backend.tokenizer.apply_chat_template = lambda *a, **k: {"input_ids": list(range(500)), "attention_mask": [1] * 500}
    chain = module.DocumentChain(backend)
    assert chain.count_input("Teste") == 500
    result = chain.invoke("Transcrição de resultados")
    assert result["error"]["code"] == "context_budget_exceeded" and not backend.calls


@pytest.mark.parametrize("response,code", [
    ("not json", "invalid_contract"),
    (valid_answer(fonte="SOURCE-NOT-SUPPLIED"), "invalid_source"),
    (valid_answer(), "unsupported_quote"),
    (valid_answer(validacao_humana=False), "invalid_contract"),
    (valid_answer(validacao_humana=1), "invalid_contract"),
    (valid_answer(extra="unexpected"), "invalid_contract"),
    (valid_answer(status="pendente", dados={}, fonte=None), "no_evidence"),
    ('{"status":"informativo","status":"pendente"}', "invalid_contract"),
])
def test_invalid_contract_and_unsupported_sources_are_explicit(response, code):
    backend = Backend()
    backend.response = response
    result = module.DocumentChain(backend).invoke("Transcrição de resultados")
    assert result["error"]["code"] == code
    assert result["answer"]["dados"] == {} and result["answer"]["fonte"] is None
    assert result["sources"] == []
    assert result["generation"]["response"] == response


def test_output_limit_is_not_an_approved_response():
    backend = Backend()
    backend.limited = True
    result = module.DocumentChain(backend).invoke("Transcrição de resultados")
    assert result["error"]["code"] == "output_limit_reached"
    assert result["sources"] == []


def test_invocations_do_not_share_sources_or_history():
    backend = Backend()
    chain = module.DocumentChain(backend)
    first = chain.invoke("Transcrição de unidades")
    second = chain.invoke("zebraluna orbitalx")
    assert first["sources"] and second["sources"] == []
    assert len(backend.calls) == 1


def test_index_tampering_fails_before_inference(tmp_path):
    backend = Backend()
    config = tmp_path / "frozen_config.json"
    config.write_bytes((module.EXPERIMENT / "frozen_config.json").read_bytes())
    (tmp_path / "index.json").write_text('{"chunks":[]}')
    with pytest.raises(ValueError, match="diverge"):
        module.DocumentChain(backend, config_path=config)
    assert backend.calls == []
