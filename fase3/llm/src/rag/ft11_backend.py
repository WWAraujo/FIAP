"""Backend local do adapter selecionado, sem API, recuperação ou avaliação de splits."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from finetuning.scripts.run_qwen3_adapter_rocm import load_adapter


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_selection(selection_path):
    selected = read(selection_path)
    if selected.get("eligible") is not True:
        raise ValueError("Seleção não elegível; nenhum modelo alternativo será carregado.")
    adapter, base = Path(selected["adapter"]).resolve(), Path(selected["base"]).resolve()
    weights = adapter / "adapter_model.safetensors"
    if sha(weights) != selected["adapter_weights_sha256"]:
        raise ValueError("Hash do adapter diverge da seleção FT-11.")
    manifest_path = adapter.parent / "manifest.json"
    manifest = read(manifest_path)
    if not manifest.get("run_id") or manifest["model"]["id"] != selected["model_id"] or manifest["model"]["revision"] != selected["model_revision"]:
        raise ValueError("Identidade/revisão do manifesto diverge da seleção.")
    config = read(adapter / "adapter_config.json")
    if Path(config["base_model_name_or_path"]).resolve() != base or config["peft_type"] != "LORA":
        raise ValueError("Configuração do adapter incompatível com a base selecionada.")
    index = read(base / "model.safetensors.index.json")
    shards = sorted(set(index["weight_map"].values()))
    if not shards or any(Path(name).name != name for name in shards):
        raise ValueError("Índice de pesos da base inválido.")
    for name in shards:
        if not (base / name).is_file():
            raise FileNotFoundError(f"Shard da base indisponível: {base / name}")
    prompt_path = Path(selected["system_prompt"])
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    if not prompt or prompt_hash != manifest["environment"]["system_prompt"]["sha256"]:
        raise ValueError("Prompt de sistema diverge do usado em FT-11.")
    return {"run_id": manifest["run_id"], "model_id": selected["model_id"],
            "model_revision": selected["model_revision"], "base": str(base), "adapter": str(adapter),
            "selection_sha256": sha(selection_path), "adapter_weights_sha256": sha(weights),
            "adapter_config_sha256": sha(adapter / "adapter_config.json"), "manifest_sha256": sha(manifest_path),
            "system_prompt_sha256": prompt_hash, "system_prompt": prompt, "base_shards": shards}


class FT11Backend:
    def __init__(self, selection_path=ROOT / "artifacts/best_model.json"):
        self.identity = resolve_selection(selection_path)
        base = Path(self.identity["base"])
        print("Conferindo hashes dos pesos-base locais...", flush=True)
        self.identity["base_files_sha256"] = {name: sha(base / name) for name in self.identity["base_shards"] +
                                             ["model.safetensors.index.json", "config.json", "tokenizer.json", "tokenizer_config.json"]}
        self.identity["revision_verification"] = "Revisão declarada na seleção/manifesto; hashes locais registrados, sem comparação remota dos shards."
        os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        started = time.perf_counter()
        self.model, self.tokenizer, self.torch, self.device = load_adapter(base, Path(self.identity["adapter"]))
        if self.model.active_adapters != ["default"]:
            raise RuntimeError("Adapter esperado não está ativo.")
        self.load_seconds = time.perf_counter() - started

    def generate(self, prompt, max_new_tokens=128):
        if not isinstance(prompt, str) or not prompt.strip() or not 1 <= max_new_tokens <= 192:
            raise ValueError("Prompt vazio ou orçamento de saída inválido.")
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.identity["system_prompt"]}, {"role": "user", "content": prompt}],
            add_generation_prompt=True, enable_thinking=False, tokenize=True,
            return_tensors="pt", return_dict=True)
        count = inputs["input_ids"].shape[-1]
        if count + max_new_tokens > 512:
            raise ValueError(f"Orçamento total excede 512 tokens: entrada={count}, reserva={max_new_tokens}; sem truncamento.")
        inputs = inputs.to(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                         pad_token_id=self.tokenizer.pad_token_id)
        self.torch.cuda.synchronize()
        generated = output[0][count:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not text:
            raise RuntimeError("FT-11 retornou uma resposta vazia.")
        return {"response": text, "input_tokens": count, "output_tokens": len(generated),
                "latency_seconds": time.perf_counter() - started,
                "output_limit_reached": len(generated) >= max_new_tokens,
                "identity": {k: self.identity[k] for k in ("run_id", "model_id", "model_revision", "adapter_weights_sha256")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT / "artifacts/best_model.json")
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_rag04")
    output = ROOT / "artifacts/experiments" / run_id
    output.mkdir(parents=True, exist_ok=False)
    print(f"Artefatos: {output}", flush=True)
    def save(name, value):
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    try:
        backend = FT11Backend(args.selection)
        save("identity.json", backend.identity)
        torch = backend.torch
        probes = [
            {"id": "SMOKE-FT11-01", "prompt": "Copie codigo e unidade do contexto: codigo DEMO-001; unidade mmol/L; fonte SMOKE-FT11-01.", "expected_data": {"codigo": "DEMO-001", "unidade": "mmol/L"}},
            {"id": "SMOKE-FT11-02", "prompt": "Copie etiqueta do contexto: etiqueta TESTE-LOCAL-02; fonte SMOKE-FT11-02.", "expected_data": {"etiqueta": "TESTE-LOCAL-02"}},
        ]
        save("request.json", {"cases": probes, "max_new_tokens": 128, "total_token_budget": 512, "do_sample": False, "rag_evaluation_used": False})
        results = []
        for probe in probes:
            response = backend.generate(probe["prompt"])
            try:
                parsed = json.loads(response["response"])
                passed = (set(parsed) == {"status", "dados", "pendencias", "fonte", "validacao_humana"} and
                          parsed["dados"] == probe["expected_data"] and parsed["fonte"] == probe["id"] and
                          parsed["validacao_humana"] is True and not response["output_limit_reached"])
            except (ValueError, TypeError):
                passed = False
            results.append({"id": probe["id"], **response, "documentary_probe_passed": passed})
        save("results.json", results)
        save("summary.json", {"run_id": run_id, "backend_loaded": True, "adapter_active": backend.model.active_adapters,
                             "load_seconds": backend.load_seconds, "gpu": torch.cuda.get_device_name(),
                             "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
                             "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(),
                             "passed": sum(r["documentary_probe_passed"] for r in results), "count": len(results),
                             "clinical_validation": False, "python": sys.version, "script_sha256": sha(Path(__file__))})
        if not all(r["documentary_probe_passed"] for r in results):
            raise RuntimeError("Carga real concluída, mas algum probe documental falhou; confira results.json.")
        print("FT-11 carregado e 2/2 probes documentais aprovados.", flush=True)
    except Exception as error:
        save("error.json", {"type": type(error).__name__, "message": str(error), "fallback_used": False})
        raise


if __name__ == "__main__":
    main()
