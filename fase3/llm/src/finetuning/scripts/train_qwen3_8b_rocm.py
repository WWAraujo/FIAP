"""Treina Qwen3-8B com QLoRA localmente em GPU AMD/ROCm.

Execute primeiro com ``--validate-only``. O script requer uma build ROCm do
PyTorch: ``torch.version.hip`` e ``torch.cuda.is_available()`` devem existir.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "9c925d64d72725edaf899c6cb9c377fd0709d9c5"
MODEL_REVISIONS = {
    MODEL_ID: MODEL_REVISION,
    "Qwen/Qwen3-14B": "40c069824f4251a91eefaf281ebe4c544efd3e18",
    "unsloth/Qwen3-14B-bnb-4bit": "b455922269e4678c84f854fe38a6993d73614a1e",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
}
MEDPT_SAMPLE_SIZE = 5_000
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
SYSTEM_PROMPT = (
    "Você é um assistente médico acadêmico. Use apenas informações documentadas, "
    "não prescreva medicamentos, não faça diagnóstico definitivo e indique a necessidade "
    "de validação humana."
)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=project_dir)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--model-id", choices=MODEL_REVISIONS, default=MODEL_ID)
    parser.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--initial-adapter", type=Path, help="Continua pesos LoRA da mesma base, conservando a proveniência.")
    parser.add_argument("--frozen-embeddings-bf16", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--optimizer",
        default="adamw_torch",
        choices=("adamw_torch", "paged_adamw_8bit"),
        help="adamw_torch evita o otimizador paginado experimental do bitsandbytes em ROCm.",
    )
    parser.add_argument(
        "--attention-implementation",
        default="eager",
        choices=("eager", "sdpa"),
        help="eager evita os kernels de atenção experimentais da GPU AMD durante o treino.",
    )
    parser.add_argument(
        "--blas-backend",
        default="hipblas",
        choices=("hipblas", "default"),
        help="hipblas evita hipBLASLt, que é selecionado por padrão no gfx1200.",
    )
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Limita passos do treino para teste; -1 executa a época completa.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ativo por padrão para caber na VRAM; usa a variante não reentrante.",
    )
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_rocm(blas_backend: str) -> tuple[Any, Any, Any]:
    if blas_backend == "hipblas":
        # Deve ser definido antes de importar torch para que a seleção inicial
        # não escolha hipBLASLt no gfx1200.
        os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch não está instalado neste ambiente Python.") from error

    if torch.version.hip is None:
        raise RuntimeError(
            "Este PyTorch não usa ROCm (torch.version.hip é None). Instale a build ROCm "
            "compatível com ROCm 10.0; builds +cpu e +cu* não acessam uma Radeon."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch ROCm não encontrou a GPU. Verifique rocminfo, /dev/kfd, os grupos "
            "render/video e a instalação oficial do ROCm antes de treinar."
        )

    try:
        torch.backends.cuda.preferred_blas_library(blas_backend)
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError(f"Não foi possível selecionar o backend BLAS {blas_backend}.") from error

    device_index = torch.cuda.current_device()
    return torch, torch.device("cuda"), device_index


def rebuild_data(project_dir: Path, seed: int) -> dict[str, Any]:
    """Recria curadoria e splits com a amostra local e fixa do MedPT."""
    sys.path.insert(0, str(project_dir / "src"))
    from finetuning.data.preprocessing import MedicalDataPreprocessor
    from finetuning.data.split_dataset import (
        build_manifest,
        save_jsonl as save_split_jsonl,
        split_examples,
    )

    curated_path = project_dir / "data" / "curated" / "dataset_medico_anonimizado.json"
    splits_dir = project_dir / "data" / "splits"
    medpt_dir = project_dir / "data" / "external" / "medpt"
    raw_path = project_dir / "data" / "raw" / "dataset_medico.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"Fonte sintética ausente: {raw_path}")

    raw_dataset = json.loads(raw_path.read_text(encoding="utf-8"))
    preprocessor = MedicalDataPreprocessor()
    curated_dataset = preprocessor.curate_dataset(raw_dataset)
    curated_path.parent.mkdir(parents=True, exist_ok=True)
    curated_path.write_text(
        json.dumps(curated_dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    sample_path = medpt_dir / "sample.jsonl"
    external_manifest_path = medpt_dir / "manifest.json"
    if not sample_path.exists() or not external_manifest_path.exists():
        raise FileNotFoundError(
            "Amostra MedPT ausente. Execute primeiro "
            "src/finetuning/data/ingest_medpt.py --sample-size 5000 --seed 42."
        )
    medpt_sample = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines() if line]
    external_manifest = json.loads(external_manifest_path.read_text(encoding="utf-8"))
    if len(medpt_sample) != MEDPT_SAMPLE_SIZE:
        raise ValueError(f"Amostra MedPT deve conter {MEDPT_SAMPLE_SIZE} pares; encontrou {len(medpt_sample)}.")
    synthetic_examples = preprocessor.format_for_fine_tuning(curated_dataset)
    splits = split_examples(synthetic_examples + medpt_sample, seed=seed)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, examples in splits.items():
        save_split_jsonl(examples, splits_dir / f"{name}.jsonl")

    manifest = build_manifest(
        splits,
        {
            "synthetic": curated_dataset["dataset_metadata"],
            "external": {**external_manifest, "selected_for_corpus": len(medpt_sample)},
        },
        seed=seed,
    )
    (splits_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_records(splits_dir: Path) -> dict[str, list[dict[str, str]]]:
    records: dict[str, list[dict[str, str]]] = {}
    for name in ("train", "validation", "test"):
        path = splits_dir / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Split ausente: {path}")
        records[name] = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        source_overlap = {row["source_id"] for row in records[left]} & {
            row["source_id"] for row in records[right]
        }
        group_overlap = {row["source_group"] for row in records[left]} & {
            row["source_group"] for row in records[right]
        }
        if source_overlap or group_overlap:
            raise ValueError(f"Vazamento entre {left} e {right}")
    return records


def resolve_model(model_dir: Path, model_id: str = MODEL_ID) -> str:
    if model_dir.exists():
        return str(model_dir)
    from huggingface_hub import snapshot_download

    revision = MODEL_REVISIONS[model_id]
    print(f"Baixando {model_id} na revisão {revision}...")
    return snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=str(model_dir),
    )


def make_tokenizer_and_datasets(model_path: str, records: dict[str, list[dict[str, str]]], max_length: int, system_prompt: str = SYSTEM_PROMPT):
    from datasets import Dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(record: dict[str, str]) -> dict[str, list[int]]:
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": record["instruction"] + ("\n\n" + record["input"] if record["input"] else ""),
            },
        ]
        # O tokenizador BPE pode fundir o final do prompt com o primeiro token
        # da resposta. Por isso, localizamos a resposta pelos offsets do texto
        # renderizado, e não comparando listas de tokens separadas.
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_text = tokenizer.apply_chat_template(
            prompt_messages + [{"role": "assistant", "content": record["output"]}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        if not full_text.startswith(prompt_text):
            raise ValueError(f"Template de chat desalinhado: {record['source_id']}")

        encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        full_ids = encoded["input_ids"]
        answer_start = len(prompt_text)
        answer_start_index = next(
            (
                index
                for index, (start, end) in enumerate(encoded["offset_mapping"])
                if start >= answer_start and end > start
            ),
            None,
        )
        if answer_start_index is None:
            raise ValueError(f"Resposta vazia após tokenização: {record['source_id']}")
        prompt_ids = full_ids[:answer_start_index]
        answer_ids = full_ids[answer_start_index:]

        input_ids, labels = truncate_supervised_sequence(prompt_ids, answer_ids, max_length)
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

    datasets = {}
    for name, rows in records.items():
        dataset = Dataset.from_list(rows)
        datasets[name] = dataset.map(tokenize, remove_columns=dataset.column_names)
    return tokenizer, datasets


def truncate_supervised_sequence(
    prompt_ids: list[int], answer_ids: list[int], max_length: int
) -> tuple[list[int], list[int]]:
    """Mantém contexto e rótulos da resposta dentro do comprimento fixo.

    Respostas longas não podem ser treinadas isoladamente, pois isso elimina a
    relação entre a pergunta e a conclusão a ser aprendida. Para preservar os
    dois lados do exemplo, reserva-se metade da sequência ao prompt quando
    ambos excedem o limite; a resposta continua sendo o único trecho rotulado.
    """
    if max_length < 2:
        raise ValueError("max_length deve permitir ao menos um token de prompt e um de resposta.")
    if not prompt_ids or not answer_ids:
        raise ValueError("Exemplo supervisionado exige prompt e resposta não vazios.")

    prompt_budget = min(len(prompt_ids), max_length // 2)
    answer_budget = max_length - prompt_budget
    if len(answer_ids) < answer_budget:
        prompt_budget = min(len(prompt_ids), max_length - len(answer_ids))
        answer_budget = max_length - prompt_budget

    retained_prompt = prompt_ids[-prompt_budget:]
    retained_answer = answer_ids[:answer_budget]
    input_ids = retained_prompt + retained_answer
    labels = [-100] * len(retained_prompt) + retained_answer
    return input_ids, labels


def main() -> None:
    args = parse_args()
    system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip() if args.system_prompt_file else SYSTEM_PROMPT
    if not system_prompt:
        raise ValueError("O prompt de sistema não pode estar vazio.")
    project_dir = args.project_dir.resolve()
    model_name = args.model_id.split("/")[-1]
    model_dir = (args.model_dir or project_dir / "models" / "base" / model_name).resolve()
    output_root = (args.output_dir or project_dir / "artifacts" / "adapters").resolve()
    splits_dir = project_dir / "data" / "splits"

    torch, device, device_index = require_rocm(args.blas_backend)
    if args.rebuild_data or not all((splits_dir / f"{name}.jsonl").exists() for name in ("train", "validation", "test")):
        manifest = rebuild_data(project_dir, args.seed)
        print("Dados recriados:", manifest["split_counts"])
    records = load_records(splits_dir)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + model_name.lower().replace("-", "_") + "_qlora_rocm"
    run_dir = output_root / run_id
    adapter_dir = run_dir / "adapter"
    run_dir.mkdir(parents=True, exist_ok=False)

    gpu = torch.cuda.get_device_properties(device_index)
    free_before, total_vram = torch.cuda.mem_get_info(device_index)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    environment = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "local_rocm",
        "allocator_config": os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF")),
        "model": {"id": args.model_id, "revision": MODEL_REVISIONS[args.model_id], "license": "Apache-2.0"},
        "system_prompt": {"text": system_prompt, "sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(), "file": str(args.system_prompt_file.resolve()) if args.system_prompt_file else None},
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
        "gpu": {"name": gpu.name, "vram_bytes": total_vram, "free_before_bytes": free_before},
        "torch": {
            "version": torch.__version__,
            "hip_version": torch.version.hip,
            "blas_backend": str(torch.backends.cuda.preferred_blas_library()),
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("transformers", "peft", "accelerate", "bitsandbytes", "datasets", "huggingface_hub")
        },
    }

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    probe = torch.randn((256, 256), device=device, dtype=torch.float32, requires_grad=True)
    probe_loss = (probe @ probe.T).square().mean()
    probe_loss.backward()
    torch.cuda.synchronize(device)
    if not torch.isfinite(probe_loss) or not torch.isfinite(probe.grad).all():
        raise RuntimeError("Forward/backward mínimo falhou na GPU ROCm.")
    environment["accelerator_probe"] = {
        "loss": probe_loss.item(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    del probe, probe_loss
    torch.cuda.empty_cache()
    environment["stage"] = "accelerator_validated"
    (run_dir / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    model_path = resolve_model(model_dir, args.model_id)
    tokenizer, datasets = make_tokenizer_and_datasets(model_path, records, args.max_length, system_prompt)

    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments

    def data_collator(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        def padded(field: str, padding_value: int) -> torch.Tensor:
            rows = [
                torch.tensor(item[field], dtype=torch.long)
                for item in features
            ]
            return torch.stack(
                [
                    torch.nn.functional.pad(
                        row,
                        (0, args.max_length - row.shape[-1]),
                        value=padding_value,
                    )
                    for row in rows
                ]
            )

        return {
            "input_ids": padded("input_ids", tokenizer.pad_token_id),
            "attention_mask": padded("attention_mask", 0),
            "labels": padded("labels", -100),
        }

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    torch.cuda.reset_peak_memory_stats(device)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization,
        dtype=compute_dtype,
        device_map={"": device_index},
        attn_implementation=args.attention_implementation,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if args.gradient_checkpointing else None
        ),
    )
    environment["frozen_embeddings_bf16"] = args.frozen_embeddings_bf16
    environment["probe_autocast"] = args.frozen_embeddings_bf16
    if args.frozen_embeddings_bf16:
        if compute_dtype != torch.bfloat16:
            raise RuntimeError("Embeddings BF16 exigem suporte BF16 da GPU.")
        precision_changes = []
        for name, module in (("input_embeddings", model.get_input_embeddings()), ("output_embeddings", model.get_output_embeddings())):
            if module is None or any(parameter.requires_grad for parameter in module.parameters()):
                raise RuntimeError("A redução de precisão exige embeddings e head congelados.")
            before_bytes = sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())
            before_dtypes = sorted({str(parameter.dtype) for parameter in module.parameters()})
            module.to(dtype=torch.bfloat16)
            precision_changes.append({"module": name, "before_dtypes": before_dtypes, "after_dtype": "torch.bfloat16", "before_bytes": before_bytes, "after_bytes": sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())})
        environment["frozen_embedding_precision_changes"] = precision_changes
    parent_adapter = None
    if args.initial_adapter:
        parent_path = args.initial_adapter.resolve()
        parent_manifest_path = parent_path.parent / "manifest.json"
        parent_manifest = json.loads(parent_manifest_path.read_text())
        if parent_manifest["model"]["id"] != args.model_id or parent_manifest["model"]["revision"] != MODEL_REVISIONS[args.model_id]:
            raise ValueError("Adapter inicial pertence a outra base/revisão")
        parent_adapter = {"path": str(parent_path), "manifest_sha256": sha256(parent_manifest_path), "weights_sha256": sha256(parent_path / "adapter_model.safetensors")}
        model = PeftModel.from_pretrained(model, str(parent_path), is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=LORA_TARGET_MODULES + (["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"] if args.model_id == "Qwen/Qwen3.5-9B" else []),
            ),
        )

    torch.cuda.empty_cache()
    environment["parameter_devices"] = dict(Counter(str(parameter.device) for parameter in model.parameters()))
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("O probe exige todos os parâmetros na GPU, sem offload.")
    environment["stage"] = "qlora_loaded"
    (run_dir / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    batch = {key: value.to(device) for key, value in data_collator([datasets["train"][0]]).items()}
    model.train()
    with torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=args.frozen_embeddings_bf16):
        qlora_loss = model(**batch).loss
    qlora_loss.backward()
    torch.cuda.synchronize(device)
    if not torch.isfinite(qlora_loss):
        raise RuntimeError("Loss não finita no forward/backward QLoRA.")
    environment["qlora_forward_backward"] = {
        "loss": qlora_loss.item(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "max_length": args.max_length,
            "lora": {"r": 8, "alpha": 32, "dropout": 0.1},
    }
    if args.validate_only:
        # O forward/backward não aciona o otimizador usado no treino. Este passo
        # curto detecta falhas HIP desse caminho antes da execução longa.
        optimizer_probe = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
        )
        optimizer_probe.step()
        optimizer_probe.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        environment["qlora_forward_backward"]["optimizer_probe"] = "adamw_torch"
        environment["qlora_forward_backward"]["peak_with_optimizer_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        environment["qlora_forward_backward"]["peak_with_optimizer_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    environment["stage"] = "qlora_probe_completed"
    (run_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    model.zero_grad(set_to_none=True)
    del batch, qlora_loss
    torch.cuda.empty_cache()
    print(f"QLoRA validada; pico de VRAM: {environment['qlora_forward_backward']['peak_allocated_bytes'] / 2**30:.2f} GiB")

    if args.validate_only:
        print(f"Validação concluída. Evidência: {run_dir / 'environment.json'}")
        return

    steps_per_epoch = math.ceil(len(datasets["train"]) / (args.batch_size * args.gradient_accumulation))
    warmup_steps = max(1, math.ceil(steps_per_epoch * args.epochs * 0.05))
    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if args.gradient_checkpointing else None
        ),
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        optim=args.optimizer,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
    )
    torch.cuda.reset_peak_memory_stats(device)
    train_metrics = trainer.train().metrics
    validation_metrics = trainer.evaluate()
    environment["training_and_validation_memory"] = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    adapter_dir.mkdir(parents=True, exist_ok=False)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    training_manifest = {
        "run_id": run_id,
        "initial_adapter": parent_adapter,
        "model": {"id": args.model_id, "revision": MODEL_REVISIONS[args.model_id], "license": "Apache-2.0"},
        "dataset": {
            "seed": args.seed,
            "splits": {
                name: {
                    "count": len(rows),
                    "sha256": sha256(splits_dir / f"{name}.jsonl"),
                    "origins": dict(Counter(row.get("origin", "external") for row in rows)),
                }
                for name, rows in records.items()
            },
        },
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "optimizer": args.optimizer,
            "attention_implementation": args.attention_implementation,
            "blas_backend": args.blas_backend,
            "max_length": args.max_length,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "gradient_checkpointing": args.gradient_checkpointing,
            "compute_dtype": str(compute_dtype),
            "frozen_embeddings_bf16": args.frozen_embeddings_bf16,
        },
        "metrics": {"train": train_metrics, "validation": validation_metrics},
        "environment": environment,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(training_manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization,
        dtype=compute_dtype,
        device_map={"": device_index},
        attn_implementation=args.attention_implementation,
    )
    from peft import PeftModel

    reloaded_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    reloaded_model.eval()
    example = records["validation"][0]
    inputs = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": example["instruction"] + ("\n\n" + example["input"] if example.get("input") else "")}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)
    with torch.inference_mode():
        output = reloaded_model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    if not response.strip():
        raise RuntimeError("O adapter recarregado retornou uma resposta vazia.")
    print(f"Treino concluído. Adapter: {adapter_dir}")
    print(f"Recarga validada com o exemplo: {example['source_id']}")


if __name__ == "__main__":
    main()
