from __future__ import annotations

# ============================================================
# IMPORTAÇÕES
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techchallenge_fase2.genetic import (  # noqa: E402
    GeneticConfig,
    GeneticOptimizer,
    MedicalFitnessEvaluator,
    SearchSpace,
)
from techchallenge_fase2.genetic.chromosome import Gene  # noqa: E402


# ============================================================
# TESTE RÁPIDO COM UMA BASE SINTÉTICA
# ============================================================

def main() -> None:
    # Cria uma base binária pequena para validar o fluxo sem usar o Vigitel.
    X_array, y_array = make_classification(
        n_samples=500,
        n_features=8,
        n_informative=5,
        weights=[0.7, 0.3],
        random_state=42,
    )
    X = pd.DataFrame(X_array, columns=[f"x{i}" for i in range(X_array.shape[1])])
    y = pd.Series(y_array)

    # Reduz o espaço de busca para o teste terminar em poucos segundos.
    espaco = SearchSpace(
        genes=(
            Gene("n_estimators", (15, 25)),
            Gene("max_depth", (4, 6, None)),
            Gene("min_samples_split", (2, 6)),
            Gene("min_samples_leaf", (1, 3)),
            Gene("max_features", ("sqrt",)),
            Gene("criterion", ("gini",)),
            Gene("class_weight", ("balanced",)),
            Gene("threshold", (0.4, 0.5, 0.6)),
        )
    )
    config = GeneticConfig(
        nome="smoke",
        population_size=6,
        generations=3,
        mutation_rate=0.2,
        crossover_rate=0.8,
        elite_size=1,
        tournament_size=2,
        patience=3,
        random_state=42,
    )
    resultado = GeneticOptimizer(
        config,
        MedicalFitnessEvaluator(X, y, folds=2, random_state=42),
        espaco,
    ).executar()

    # O teste falha imediatamente se o melhor cromossomo for inválido.
    assert espaco.validar(resultado.best_chromosome)
    print("Teste rápido concluído.")
    print(resultado.best_chromosome)
    print(resultado.best_metrics.to_dict())


if __name__ == "__main__":
    main()
