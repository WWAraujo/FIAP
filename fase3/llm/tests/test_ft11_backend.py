"""Falhas de seleção e orçamento devem ocorrer antes da geração, sem GPU/rede."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ft11_backend", ROOT / "src/rag/ft11_backend.py")
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


def selection(tmp_path):
    base = tmp_path / "base"
    adapter = tmp_path / "run/adapter"
    base.mkdir()
    adapter.mkdir(parents=True)
    (base / "part.safetensors").write_bytes(b"synthetic")
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "part.safetensors"}}))
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"synthetic adapter")
    prompt = tmp_path / "system.txt"
    prompt.write_text("Sistema documental")
    manifest = {"run_id": "FT11-MOCK", "model": {"id": "test/base", "revision": "revision-fixed"},
                "environment": {"system_prompt": {"sha256": hashlib.sha256(prompt.read_bytes()).hexdigest()}}}
    (adapter.parent / "manifest.json").write_text(json.dumps(manifest))
    (adapter / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "base_model_name_or_path": str(base)}))
    path = tmp_path / "best.json"
    path.write_text(json.dumps({"eligible": True, "adapter": str(adapter), "base": str(base),
                              "adapter_weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                              "model_id": "test/base", "model_revision": "revision-fixed", "system_prompt": str(prompt)}))
    return path, adapter, base


def test_selection_matches_identity_without_loading_model(tmp_path):
    path, adapter, _ = selection(tmp_path)
    identity = backend.resolve_selection(path)
    assert identity["run_id"] == "FT11-MOCK"
    assert identity["adapter"] == str(adapter)
    assert identity["model_revision"] == "revision-fixed"


@pytest.mark.parametrize("failure", ["missing_adapter", "corrupt_adapter", "missing_base", "wrong_revision"])
def test_invalid_weights_or_revision_fail_without_fallback(tmp_path, failure):
    path, adapter, base = selection(tmp_path)
    if failure == "missing_adapter":
        (adapter / "adapter_model.safetensors").unlink()
    elif failure == "corrupt_adapter":
        (adapter / "adapter_model.safetensors").write_bytes(b"corrupt")
    elif failure == "missing_base":
        (base / "part.safetensors").unlink()
    else:
        value = json.loads(path.read_text())
        value["model_revision"] = "other"
        path.write_text(json.dumps(value))
    with pytest.raises((FileNotFoundError, ValueError)):
        backend.resolve_selection(path)


def test_excessive_budget_is_rejected_before_model_call():
    instance = backend.FT11Backend.__new__(backend.FT11Backend)
    instance.identity = {"system_prompt": "Sistema"}
    class InputIds:
        shape = (1, 500)
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": InputIds()}
    instance.tokenizer = Tokenizer()
    with pytest.raises(ValueError, match="sem truncamento"):
        instance.generate("Conteúdo longo", 128)


def test_shared_loader_keeps_cli_attention_default_and_backend_eager(tmp_path, monkeypatch):
    path, adapter, base = selection(tmp_path)
    loader = sys.modules[backend.load_adapter.__module__]
    fake_torch = SimpleNamespace(bfloat16="bf16", float16="fp16",
                                 cuda=SimpleNamespace(is_bf16_supported=lambda: True))
    monkeypatch.setattr(loader, "require_rocm", lambda: (fake_torch, "cuda", 0))
    calls = []
    class BaseModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(kwargs)
            return "base-model"
    token = SimpleNamespace(pad_token=None, eos_token="eos")
    transformers = SimpleNamespace(AutoModelForCausalLM=BaseModel,
                                    AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: token),
                                    BitsAndBytesConfig=lambda **kwargs: kwargs)
    model = SimpleNamespace(eval=lambda: None)
    peft = SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=lambda *a, **k: model))
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)
    assert loader.load_adapter(base, adapter)[0] is model
    assert calls[-1]["attn_implementation"] == "eager"
    loader.load_adapter(base, adapter, attention_implementation=None)
    assert calls[-1]["attn_implementation"] is None
