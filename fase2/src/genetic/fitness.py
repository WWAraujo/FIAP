from __future__ import annotations

# ============================================================
# IMPORTAÇÕES
# ============================================================

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ..modeling import construir_pipeline


# ============================================================
# MÉTRICAS PRODUZIDAS PELA FUNÇÃO FITNESS
# ============================================================

@dataclass(frozen=True)
class FitnessResult:
    fitness: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


# ============================================================
# FUNÇÃO FITNESS PARA O CONTEXTO MÉDICO
# ============================================================

class MedicalFitnessEvaluator:
    """Avalia indivíduos apenas com predições out-of-fold do conjunto de treino."""

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        folds: int = 3,
        random_state: int = 42,
        cv_jobs: int = 1,
        precision_minima: float = 0.50,
    ) -> None:
        self.X = X
        self.y = y
        self.folds = folds
        self.random_state = random_state
        self.cv_jobs = cv_jobs
        self.precision_minima = precision_minima

    def __call__(self, cromossomo: dict[str, Any]) -> FitnessResult:
        # O pipeline completo é recriado em cada fold. Isso impede que imputação,
        # escala ou One-Hot Encoding aprendam informações da validação.
        pipeline = construir_pipeline(self.X, cromossomo, self.random_state)
        cv = StratifiedKFold(
            n_splits=self.folds,
            shuffle=True,
            random_state=self.random_state,
        )
        probabilidades = cross_val_predict(
            pipeline,
            self.X,
            self.y,
            cv=cv,
            method="predict_proba",
            n_jobs=self.cv_jobs,
        )[:, 1]
        threshold = float(cromossomo["threshold"])
        predicoes = (probabilidades >= threshold).astype(int)

        # Calcula as métricas exigidas pelo enunciado da Fase 2.
        accuracy = accuracy_score(self.y, predicoes)
        precision = precision_score(self.y, predicoes, zero_division=0)
        recall = recall_score(self.y, predicoes, zero_division=0)
        f1 = f1_score(self.y, predicoes, zero_division=0)
        roc_auc = roc_auc_score(self.y, probabilidades)

        # Recall recebe maior peso por se tratar de uma ferramenta de triagem.
        fitness = 0.45 * recall + 0.30 * f1 + 0.25 * roc_auc

        # Evita soluções que aumentem o recall classificando quase todos os
        # pacientes como positivos e gerando excesso de falsos alarmes.
        if precision < self.precision_minima:
            fitness -= 2.0 * (self.precision_minima - precision)

        return FitnessResult(
            fitness=float(fitness),
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            roc_auc=float(roc_auc),
        )


# ============================================================
# AVALIAÇÃO ÚNICA NO CONJUNTO DE TESTE ISOLADO
# ============================================================

def avaliar_no_teste(
    pipeline: Any,
    X_teste: pd.DataFrame,
    y_teste: pd.Series,
    threshold: float,
) -> FitnessResult:
    probabilidades = pipeline.predict_proba(X_teste)[:, 1]
    predicoes = (probabilidades >= threshold).astype(int)
    precision = precision_score(y_teste, predicoes, zero_division=0)
    recall = recall_score(y_teste, predicoes, zero_division=0)
    f1 = f1_score(y_teste, predicoes, zero_division=0)
    roc_auc = roc_auc_score(y_teste, probabilidades)
    fitness = 0.45 * recall + 0.30 * f1 + 0.25 * roc_auc
    if precision < 0.50:
        fitness -= 2.0 * (0.50 - precision)
    return FitnessResult(
        fitness=float(fitness),
        accuracy=float(accuracy_score(y_teste, predicoes)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=float(roc_auc),
    )
