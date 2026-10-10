import importlib.util
from pathlib import Path


path = Path(__file__).resolve().parents[1] / "src/finetuning/data/audit_training_corpus.py"
spec = importlib.util.spec_from_file_location("audit_training_corpus", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def row(answer, source_id="A"):
    return {"source_id": source_id, "origin": "synthetic", "instruction": "Como registrar a avaliação humana no fluxo?", "input": "", "output": answer}


def test_repetition_and_unfinished_answer_are_flagged():
    flags, _ = audit.technical_flags(row("Registre a avaliação humana no fluxo. " * 4 + "Depois disso:"))
    assert {"repetitive_answer", "missing_terminal_punctuation", "possible_unfinished_answer"} <= set(flags)


def test_nonrepetitive_complete_answer_and_closing_quote():
    flags, _ = audit.technical_flags(row('Registre a decisão com identificação do responsável, data e justificativa antes de prosseguir com o fluxo institucional."'))
    assert flags == []


def test_duplicates_normalize_spacing_case_and_punctuation():
    rows = [row("Registre a decisão.", "A"), row("REGISTRE  a decisão!", "B")]
    assert audit.groups(rows, "output") == [["A", "B"]]
    assert audit.groups(rows, "pair") == [["A", "B"]]


def test_contact_information_is_flagged():
    flags, _ = audit.technical_flags(row("Para registrar a decisão no fluxo institucional envie os dados para contato@example.com."))
    assert "possible_pii" in flags


def test_commercial_boilerplate_is_flagged_without_blocking_professional_referral():
    flags, _ = audit.technical_flags(row("Te convidamos para uma consulta e você pode reservar uma consulta pelo site Doctoralia."))
    assert "commercial_call_to_action" in flags
    flags, _ = audit.technical_flags(row("Encaminhe a decisão para avaliação pelo profissional responsável e registre os dados disponíveis no fluxo institucional."))
    assert "commercial_call_to_action" not in flags


def test_named_signature_is_flagged():
    flags, _ = audit.technical_flags(row("Registre a decisão com identificação do responsável, data e justificativa antes de prosseguir. Át.Exemplo Sintético!"))
    assert "possible_named_signature" in flags
