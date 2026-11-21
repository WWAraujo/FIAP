"""Testes do preparo de exemplos supervisionados para QLoRA."""

import importlib.util
from pathlib import Path


def _training_script():
    path = Path(__file__).resolve().parents[1] / "src/finetuning/scripts" / "train_qwen3_8b_rocm.py"
    spec = importlib.util.spec_from_file_location("train_qwen3_8b_rocm", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_long_answer_keeps_prompt_and_only_supervises_answer():
    script = _training_script()

    input_ids, labels = script.truncate_supervised_sequence(
        list(range(200)), list(range(200, 700)), max_length=384
    )

    assert input_ids[:192] == list(range(8, 200))
    assert input_ids[192:] == list(range(200, 392))
    assert labels[:192] == [-100] * 192
    assert labels[192:] == list(range(200, 392))


def test_short_answer_retains_the_entire_prompt_when_it_fits():
    script = _training_script()

    input_ids, labels = script.truncate_supervised_sequence([1, 2, 3], [4, 5], max_length=8)

    assert input_ids == [1, 2, 3, 4, 5]
    assert labels == [-100, -100, -100, 4, 5]
