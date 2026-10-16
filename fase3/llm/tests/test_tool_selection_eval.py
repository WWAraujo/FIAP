import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/rag"))
from evaluate_tool_selection import cases, parse_response, score


@pytest.mark.parametrize("mode,response", [
    ("native", '<tool_call>\n<function=calcular_imc>\n</function>\n</tool_call>'),
    ("json", '{"calls":[{"name":"calcular_imc","arguments":{}}]}'),
])
def test_native_and_json_selection_are_scored_without_execution(mode, response):
    calls = parse_response(response, mode)
    assert score(cases()[0], calls)
    assert not score(next(c for c in cases() if c["id"] == "CTL06"), calls)


@pytest.mark.parametrize("response", [
    '{"calls":[{"name":"executar_sql","arguments":{}}]}',
    '{"calls":[{"name":"calcular_imc","arguments":{"patient_id":"other"}}]}',
    '{"calls":[],"calls":[{"name":"calcular_imc","arguments":{}}]}',
    '{"calls":[{"name":"consultar_prontuario","arguments":{"campos":["senha"]}}]}',
])
def test_unsupported_tool_argument_and_duplicate_keys_are_invalid(response):
    with pytest.raises(ValueError):
        parse_response(response, "json")


def test_prose_or_trained_answer_is_not_a_tool_call():
    for response in ['Vou calcular o IMC.', '{"status":"informativo","dados":{"imc":28.2}}']:
        with pytest.raises(ValueError):
            parse_response(response, "native")
    assert parse_response("SEM_FERRAMENTA", "native") == []


def test_multiple_terminal_tokens_do_not_invalidate_native_or_json_calls():
    ending = '<|im_end|>\n<|endoftext|>'
    assert parse_response('<tool_call>\n<function=calcular_imc>\n</function>\n</tool_call>' + ending,
                          'native')[0]['name'] == 'calcular_imc'
    assert parse_response('{"calls":[]}' + ending, 'json') == []


def test_patient_fields_must_match_and_controls_are_frozen():
    rows = cases()
    assert len(rows) == 24 and sum(r["critical"] for r in rows) == 6
    case = next(c for c in rows if c["id"] == "SQL01")
    assert not score(case, [{"name":"consultar_prontuario", "arguments":{"campos":["medicamentos"]}}])
