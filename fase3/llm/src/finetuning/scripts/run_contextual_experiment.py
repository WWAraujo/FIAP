"""Executa controle, treino, seleção pela validação e prova reservada em processos novos."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/contextual_v1")
    parser.add_argument("--initial-adapter", type=Path)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--factual-review", action="store_true", help="Para após a validação; exige revisão semântica antes do teste.")
    args = parser.parse_args()
    dataset = args.dataset_dir.resolve()
    if args.initial_adapter:
        args.initial_adapter = args.initial_adapter.resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + dataset.name
    output = ROOT / "artifacts/experiments" / run_id
    output.mkdir(parents=True, exist_ok=False)
    model = ROOT / "models/base/Qwen3.5-9B"
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTORCH_ALLOC_CONF="expandable_segments:True", TORCH_BLAS_PREFER_HIPBLASLT="0")
    state = {"run_id": run_id, "stage": "starting", "python": sys.executable, "created_at": datetime.now(timezone.utc).isoformat(), "model": "Qwen/Qwen3.5-9B", "model_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "dataset": str(dataset), "initial_adapter": str(args.initial_adapter.resolve()) if args.initial_adapter else None, "parameters": {"epochs": 1, "learning_rate": args.learning_rate, "max_length": args.max_length}, "clinical_validation": False}
    if args.factual_review:
        state["selection_criteria"] = {"factual": "Adequação semântica das 48 perguntas gerais >= base; sem aumentar contradições factuais. Revisão documental por IA, sem validação clínica.", "contextual": "Acertos completos nas tarefas com documentos/contexto >= base; status de alerta/recusa não pode regredir.", "rubric": "Adequado: núcleo correto sem contradição ou decisão individual; parcial: núcleo incompleto ou acréscimo não verificável; inadequado: núcleo errado, contradição, diagnóstico/prescrição individual ou ausência de resposta educativa. JSON/campos são avaliados separadamente.", "test": "Registrar decisão pela validação antes da geração/pontuação do teste reservado novo; não ajustar parâmetros pelo teste."}

    def persist():
        (output / "experiment.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")

    def execute(stage, script, arguments):
        state["stage"] = stage
        persist()
        print(stage, flush=True)
        command = [sys.executable, "-B", "-u", str(Path(__file__).resolve().parent / script), *map(str, arguments)]
        with (output / f"{stage}.log").open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(stage="failed", failed_step=stage, exit_code=result.returncode)
            persist()
            raise RuntimeError(f"{stage}: código {result.returncode}; ver {output}")

    def evaluate(stage, split, adapter=None):
        destination = output / stage
        count = len((dataset / f"data/splits/{split}.jsonl").read_text().splitlines())
        arguments = ["--model-dir", model, "--test-file", dataset / f"data/splits/{split}.jsonl", "--output-dir", destination, "--limit", count, "--max-new-tokens", 192, "--system-prompt-file", dataset / "system_prompt.txt"]
        arguments += ["--adapter-only", "--adapter-dir", adapter] if adapter else ["--base-only"]
        execute(stage, "evaluate_qwen3_base_vs_adapter_rocm.py", arguments)
        runs = list(destination.glob("*/summary.json"))
        if len(runs) != 1:
            raise ValueError("Resultado de avaliação ambíguo")
        response_file = runs[0].parent / ("adapter.jsonl" if adapter else "base.jsonl")
        execute(stage + "_scoring", "score_contextual_results.py", ["--responses", response_file, "--dataset", dataset / f"data/splits/{split}.jsonl", "--output", destination / "scores.json"])
        return json.loads((destination / "scores.json").read_text())["metrics"]

    print(f"EXPERIMENT_DIR={output}", flush=True)
    base_metrics = None if args.factual_review else evaluate("base_validation", "validation")
    if base_metrics is not None:
        state["base_validation"] = base_metrics
        persist()
    train_arguments = ["--project-dir", dataset, "--model-id", "Qwen/Qwen3.5-9B", "--model-dir", model, "--output-dir", ROOT / "artifacts/adapters", "--system-prompt-file", dataset / "system_prompt.txt", "--frozen-embeddings-bf16", "--max-length", args.max_length, "--epochs", 1, "--learning-rate", args.learning_rate]
    if args.initial_adapter:
        train_arguments += ["--initial-adapter", args.initial_adapter]
    execute("probe", "train_qwen3_8b_rocm.py", train_arguments + ["--validate-only"])
    previous = set((ROOT / "artifacts/adapters").glob("*/manifest.json"))
    execute("training", "train_qwen3_8b_rocm.py", train_arguments)
    manifests = set((ROOT / "artifacts/adapters").glob("*/manifest.json")) - previous
    if len(manifests) != 1:
        raise ValueError("Adapter do experimento ambíguo")
    manifest = manifests.pop()
    adapter = manifest.parent / "adapter"
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("Pesos de adapter ausentes")
    state["adapter"] = str(adapter)
    state["adapter_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if args.factual_review:
        base_metrics = evaluate("base_validation", "validation")
        state["base_validation"] = base_metrics
    adapter_metrics = evaluate("adapter_validation", "validation", adapter)
    state["adapter_validation"] = adapter_metrics
    if args.factual_review:
        state["stage"] = "awaiting_semantic_review"
        persist()
        print(f"Validação gerada; revisar semanticamente antes do teste: {output}", flush=True)
        return
    # Decisão fixada pela validação; teste não escolhe hiperparâmetros ou checkpoint.
    state["validation_noninferior"] = (
        adapter_metrics["total"]["exact"] >= base_metrics["total"]["exact"]
        and all(adapter_metrics[c]["status"] >= base_metrics[c]["status"] for c in ("alerta", "limites"))
    )
    state["stage"] = "validation_selection_recorded"
    persist()
    state["base_test"] = evaluate("base_test", "test")
    state["adapter_test"] = evaluate("adapter_test", "test", adapter)
    state["test_noninferior"] = state["adapter_test"]["total"]["exact"] >= state["base_test"]["total"]["exact"]
    state["selected_for_contextual_task"] = state["validation_noninferior"] and state["test_noninferior"]
    state["stage"] = "completed"
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    persist()
    selection = {"experiment": str(output), "adapter": str(adapter), "base": str(model), "eligible": state["selected_for_contextual_task"], "scope": "tarefas sintéticas contextuais; sem validação clínica", "validation": adapter_metrics["total"], "test": state["adapter_test"]["total"]}
    (ROOT / "artifacts/best_model.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
