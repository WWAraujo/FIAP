"""Verifica se o host local está pronto para treinar Qwen3-8B com ROCm.

Não baixa modelos nem cria checkpoints. O relatório aponta a configuração que
precisa ser corrigida antes de executar ``train_qwen3_8b_rocm.py``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def command_output(command: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def device_report(path: Path) -> dict[str, object]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "read_write_access": os.access(path, os.R_OK | os.W_OK) if exists else False,
        "mode": oct(path.stat().st_mode & 0o777) if exists else None,
        "group_id": path.stat().st_gid if exists else None,
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def torch_report() -> dict[str, object]:
    try:
        import torch
    except ImportError as error:
        return {"installed": False, "error": str(error)}

    report: dict[str, object] = {
        "installed": True,
        "version": torch.__version__,
        "hip_version": torch.version.hip,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        report["gpu"] = {
            "index": device,
            "name": properties.name,
            "vram_bytes": total,
            "free_bytes": free,
        }
    return report


def python_development_report() -> dict[str, object]:
    include_dir = Path(sysconfig.get_paths()["include"])
    header = include_dir / "Python.h"
    return {
        "include_dir": str(include_dir),
        "python_h": str(header),
        "python_h_exists": header.is_file(),
        "gcc_path": shutil.which("gcc"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "artifacts" / "environment",
    )
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    torch_info = torch_report()
    python_development = python_development_report()
    devices = {
        "kfd": device_report(Path("/dev/kfd")),
        "renderD128": device_report(Path("/dev/dri/renderD128")),
    }
    rocminfo = command_output(["rocminfo"])
    torch_ready = (
        torch_info.get("installed")
        and torch_info.get("hip_version") is not None
        and torch_info.get("cuda_available") is True
    )
    device_ready = all(device["exists"] and device["read_write_access"] for device in devices.values())
    rocminfo_ready = rocminfo.get("returncode") == 0 and "gfx1200" in str(rocminfo.get("stdout", ""))
    python_development_ready = python_development["python_h_exists"] and python_development["gcc_path"] is not None
    report = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
        "user": {"uid": os.getuid(), "groups": os.getgroups()},
        "devices": devices,
        "tools": {
            "rocminfo": rocminfo,
            "rocminfo_path": shutil.which("rocminfo"),
            "rocm_smi_path": shutil.which("rocm-smi"),
            "amd_smi_path": shutil.which("amd-smi"),
        },
        "packages": {name: package_version(name) for name in ("torch", "transformers", "peft", "bitsandbytes")},
        "python_development": python_development,
        "torch": torch_info,
        "checks": {
            "device_access": device_ready,
            "rocminfo_detects_gfx1200": rocminfo_ready,
            "pytorch_rocm_gpu": torch_ready,
            "python_development_headers": python_development_ready,
        },
        "ready_for_training": device_ready and torch_ready and python_development_ready,
        "next_steps": [],
    }
    if not device_ready:
        report["next_steps"].append(
            "Garanta acesso de leitura e escrita a /dev/kfd e /dev/dri/renderD128; normalmente o usuário precisa dos grupos render e video e uma nova sessão."
        )
    if rocminfo.get("error"):
        report["next_steps"].append(
            "Instale as ferramentas ROCm 10.0 oficiais para disponibilizar rocminfo; isso melhora o diagnóstico, mas não substitui a validação do PyTorch ROCm."
        )
    elif not rocminfo_ready:
        report["next_steps"].append(
            "Confirme que rocminfo mostra gfx1200 e que as bibliotecas ROCm pertencem à instalação 10.0."
        )
    if not torch_ready:
        report["next_steps"].append(
            "Use uma venv separada com PyTorch ROCm: torch.version.hip deve ter uma versão e torch.cuda.is_available() deve ser True."
        )
    if not python_development["python_h_exists"]:
        report["next_steps"].append(
            "Instale os cabeçalhos para a versão da venv: sudo apt install python3.14-dev."
        )
    if python_development["gcc_path"] is None:
        report["next_steps"].append("Instale um compilador C: sudo apt install build-essential.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"rocm_preflight_{run_id}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Relatório salvo em: {output}")
    for name, passed in report["checks"].items():
        print(f"{'OK' if passed else 'FALHOU'}: {name}")
    if report["ready_for_training"]:
        print("Ambiente apto para validar QLoRA com train_qwen3_8b_rocm.py --validate-only")
        return
    print("Ambiente ainda não está apto para treino; veja next_steps no relatório.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
