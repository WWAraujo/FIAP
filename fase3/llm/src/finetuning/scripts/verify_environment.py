"""Coleta a evidência de ambiente exigida pela tarefa ORG-03."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PACKAGES = (
    "accelerate", "bitsandbytes", "datasets", "langchain", "langchain-community",
    "langchain-ollama", "langgraph", "peft", "torch", "transformers",
)


def command_output(command: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def installed_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def torch_environment() -> dict[str, object]:
    try:
        import torch
    except ImportError as error:
        return {"available": False, "error": str(error)}

    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "hip_version": torch.version.hip,
        "device_count": torch.cuda.device_count(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    display_devices = command_output(["pnputil", "/enum-devices", "/connected", "/class", "Display"])
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
        "packages": installed_versions(),
        "torch": torch_environment(),
        "ollama": command_output(["ollama", "--version"]),
        "display_devices": display_devices,
        "pip_check": command_output([sys.executable, "-m", "pip", "check"]),
        "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze", "--all"]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Evidência escrita em: {args.output}")
    if report["pip_check"].get("returncode"):
        raise SystemExit("pip check falhou; consulte a evidência gerada.")


if __name__ == "__main__":
    main()
