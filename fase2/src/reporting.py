from __future__ import annotations

# ============================================================
# IMPORTAÇÕES
# ============================================================

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .genetic.optimizer import OptimizationResult


# ============================================================
# EXPORTAÇÃO DE ARQUIVOS JSON
# ============================================================

def salvar_json(caminho: Path, dados: Any) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as arquivo:
        json.dump(dados, arquivo, ensure_ascii=False, indent=2, default=str)


# ============================================================
# RELATÓRIO DE CADA EXPERIMENTO GENÉTICO
# ============================================================

def salvar_resultado_experimento(
    resultado: OptimizationResult,
    pasta: str | Path,
) -> None:
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    salvar_json(pasta / "configuracao.json", asdict(resultado.config))
    salvar_json(
        pasta / "melhor_individuo.json",
        {**resultado.best_chromosome, **resultado.best_metrics.to_dict()},
    )
    pd.DataFrame(resultado.history).to_csv(pasta / "historico_geracoes.csv", index=False)
    pd.DataFrame(resultado.evaluations).to_csv(pasta / "individuos_avaliados.csv", index=False)

    # Gera o gráfico de convergência usado no relatório técnico da Fase 2.
    historico = pd.DataFrame(resultado.history)
    plt.figure(figsize=(9, 5))
    plt.plot(historico["geracao"], historico["melhor_fitness"], marker="o", label="Melhor")
    plt.plot(historico["geracao"], historico["fitness_medio"], marker=".", label="Media")
    plt.xlabel("Geracao")
    plt.ylabel("Fitness")
    plt.title(f"Convergencia - {resultado.config.nome}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pasta / "convergencia_fitness.png", dpi=180)
    plt.close()
