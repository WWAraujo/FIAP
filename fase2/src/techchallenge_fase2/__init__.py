"""Tech Challenge Fase 2: otimizacao genetica de modelos medicos."""

from .genetic.optimizer import (
    GeneticConfig,
    GeneticOptimizer,
    OptimizationResult,
)
from .genetic.fitness import (
    MedicalFitnessEvaluator,
    FitnessResult,
    avaliar_no_teste,
)

__version__ = "0.1.0"

__all__ = [
    "GeneticConfig",
    "GeneticOptimizer",
    "OptimizationResult",
    "MedicalFitnessEvaluator",
    "FitnessResult",
    "avaliar_no_teste",
]

