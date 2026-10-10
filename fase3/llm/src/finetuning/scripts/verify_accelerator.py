"""Registra forward/backward na GPU para ORG-04; não valida capacidade do LLM."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from verify_environment import command_output, installed_versions


def probe_accelerator() -> dict:
    report = {"passed": False}
    try:
        import torch

        report.update(
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            hip_version=torch.version.hip,
            cuda_available=torch.cuda.is_available(),
        )
        if not report["cuda_available"]:
            report["error"] = "GPU indisponível no PyTorch; FT-01 permanece bloqueada."
            return report

        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        report.update(
            device=device, gpu=properties.name, vram_bytes=total,
            free_before_bytes=free, dtype="float32", shape=[256, 256], seed=42,
        )
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats(device)
        inputs = torch.randn(256, 256, device=device, requires_grad=True)
        loss = (inputs @ inputs.T).square().mean()
        loss.backward()
        torch.cuda.synchronize(device)
        report.update(
            loss=loss.item() if torch.isfinite(loss).item() else None,
            forward_finite=bool(torch.isfinite(loss).item()),
            backward_finite=bool(torch.isfinite(inputs.grad).all().item()),
            gradient_nonzero=bool(inputs.grad.abs().sum().item() > 0),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
        report["passed"] = all(report[key] for key in (
            "forward_finite", "backward_finite", "gradient_nonzero",
        ))
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("local", "colab"), default="local")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[3] / "artifacts/environment")
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    report = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
        "packages": installed_versions(),
        "nvidia_driver": command_output(["nvidia-smi"]),
        "amd_driver": command_output(["rocminfo"]),
        "accelerator": probe_accelerator(),
        "org04_complete": False,
        "remaining_validation": [
            "Medir memória de LoRA com o modelo candidato; o tensor mínimo não comprova capacidade.",
        ],
    }
    if args.provider == "colab":
        report["remaining_validation"].append(
            "Salvar e recarregar checkpoint no Drive em uma nova sessão."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"org04_{run_id}.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Evidência escrita em: {output}")
    if not report["accelerator"]["passed"]:
        raise SystemExit("Validação do acelerador falhou; consulte a evidência.")
    print("Forward/backward passou. ORG-04 ainda requer as validações listadas no relatório.")


if __name__ == "__main__":
    main()
