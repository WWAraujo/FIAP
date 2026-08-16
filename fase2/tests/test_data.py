from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techchallenge_fase2.data import amostra_estratificada, separar_treino_teste  # noqa: E402


class DataTests(unittest.TestCase):
    def test_train_and_test_indices_are_disjoint(self) -> None:
        X = pd.DataFrame({"x": range(100)})
        y = pd.Series([0] * 70 + [1] * 30)
        X_train, X_test, y_train, y_test = separar_treino_teste(X, y)
        self.assertTrue(set(X_train.index).isdisjoint(X_test.index))
        self.assertEqual(len(X_train) + len(X_test), 100)
        self.assertAlmostEqual(y_train.mean(), y_test.mean(), places=2)

    def test_stratified_sample_has_requested_size(self) -> None:
        X = pd.DataFrame({"x": range(100)})
        y = pd.Series([0] * 70 + [1] * 30)
        X_sample, y_sample = amostra_estratificada(X, y, 40)
        self.assertEqual(len(X_sample), 40)
        self.assertAlmostEqual(y_sample.mean(), y.mean(), places=2)


if __name__ == "__main__":
    unittest.main()

