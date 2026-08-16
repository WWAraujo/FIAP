from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from techchallenge_fase2.genetic import (  # noqa: E402
    FitnessResult,
    GeneticConfig,
    GeneticOptimizer,
    SearchSpace,
)
from techchallenge_fase2.genetic.chromosome import Gene  # noqa: E402


class GeneticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.space = SearchSpace(
            genes=(
                Gene("n_estimators", (10, 20, 30)),
                Gene("max_depth", (3, 5)),
                Gene("threshold", (0.4, 0.5, 0.6)),
            )
        )

    def test_create_crossover_and_mutation_are_valid(self) -> None:
        rng = random.Random(42)
        pai_a = self.space.criar(rng)
        pai_b = self.space.criar(rng)
        filho_a, filho_b = self.space.crossover(pai_a, pai_b, rng)
        self.assertTrue(self.space.validar(filho_a))
        self.assertTrue(self.space.validar(filho_b))
        self.assertTrue(self.space.validar(self.space.mutar(filho_a, 1.0, rng)))

    def test_optimizer_keeps_valid_best_individual(self) -> None:
        config = GeneticConfig(
            nome="unit",
            population_size=8,
            generations=5,
            mutation_rate=0.3,
            crossover_rate=0.8,
            elite_size=2,
            tournament_size=3,
            patience=5,
            random_state=7,
        )

        def evaluator(cromossomo):
            score = cromossomo["n_estimators"] / 100 + cromossomo["threshold"] / 10
            return FitnessResult(score, score, score, score, score, score)

        resultado = GeneticOptimizer(config, evaluator, self.space).executar()
        self.assertTrue(self.space.validar(resultado.best_chromosome))
        self.assertGreater(len(resultado.history), 0)
        melhores = [linha["melhor_fitness"] for linha in resultado.history]
        self.assertGreaterEqual(max(melhores), melhores[0])

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GeneticConfig(
                nome="invalid",
                population_size=1,
                generations=2,
                mutation_rate=0.1,
                crossover_rate=0.8,
                elite_size=1,
                tournament_size=2,
                patience=2,
                random_state=1,
            ).validar()


if __name__ == "__main__":
    unittest.main()

