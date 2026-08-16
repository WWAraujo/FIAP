from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techchallenge_fase2.data import separar_treino_teste  # noqa: E402
from techchallenge_fase2.experiment import executar_experimentos  # noqa: E402


class IntegrationTests(unittest.TestCase):
    def test_tiny_experiment_generates_final_artifacts(self) -> None:
        X_array, y_array = make_classification(
            n_samples=180,
            n_features=6,
            n_informative=4,
            weights=[0.7, 0.3],
            random_state=42,
        )
        X = pd.DataFrame(X_array, columns=[f"x{i}" for i in range(6)])
        y = pd.Series(y_array)
        X_train, X_test, y_train, y_test = separar_treino_teste(X, y)

        with tempfile.TemporaryDirectory() as temp:
            pasta = Path(temp)
            config = pasta / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "nome": "integration",
                        "population_size": 2,
                        "generations": 1,
                        "mutation_rate": 0.1,
                        "crossover_rate": 0.8,
                        "elite_size": 1,
                        "tournament_size": 2,
                        "patience": 1,
                        "random_state": 42,
                    }
                ),
                encoding="utf-8",
            )
            baseline = {
                "n_estimators": 20,
                "max_depth": 5,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "criterion": "gini",
                "class_weight": "balanced",
                "threshold": 0.5,
            }
            resumo = executar_experimentos(
                [config],
                X_train,
                y_train,
                X_test,
                y_test,
                pasta / "output",
                baseline_chromosome=baseline,
                sample_size=100,
                folds=2,
            )
            self.assertEqual(resumo["experimento_vencedor"], "integration")
            self.assertTrue((pasta / "output" / "resumo_final.json").exists())
            self.assertTrue((pasta / "output" / "modelo_genetico_vencedor.joblib").exists())
            self.assertTrue((pasta / "output" / "comparacao_baseline_genetico.csv").exists())


if __name__ == "__main__":
    unittest.main()

