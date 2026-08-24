from __future__ import annotations

# ============================================================
# IMPORTAÇÕES
# ============================================================

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..modeling import construir_modelo, construir_preprocessador


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
    """Avalia indivíduos apenas com predições out-of-fold do conjunto de treino.

    O pré-processador (imputação/escala/one-hot) é ajustado uma única vez
    por fold no __init__, não a cada cromossomo avaliado. Como só os
    hiperparâmetros da RandomForest mudam entre indivíduos, isso elimina
    o custo repetido de recriar o ColumnTransformer centenas de vezes
    durante a busca genética.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        folds: int = 3,
        random_state: int = 42,
        n_jobs_modelo: int = 1,
        precision_minima: float = 0.50,
    ) -> None:
        self.X = X
        self.y = y
        self.folds = folds
        self.random_state = random_state
        self.n_jobs_modelo = n_jobs_modelo
        self.precision_minima = precision_minima

        # --------------------------------------------------------
        # PRÉ-PROCESSAMENTO CALCULADO UMA ÚNICA VEZ POR FOLD
        # --------------------------------------------------------
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        self._folds: list[tuple[np.ndarray, np.ndarray, Any, Any]] = []
        for treino_idx, val_idx in cv.split(X, y):
            prep = construir_preprocessador(X)
            X_treino_prep = prep.fit_transform(X.iloc[treino_idx])
            X_val_prep = prep.transform(X.iloc[val_idx])
            self._folds.append((treino_idx, val_idx, X_treino_prep, X_val_prep))

    def __call__(self, cromossomo: dict[str, Any]) -> FitnessResult:
        probabilidades = np.empty(len(self.y), dtype=float)

        # Cada fold reaproveita o pré-processamento já ajustado; só a
        # RandomForest é recriada e treinada para este cromossomo.
        for treino_idx, val_idx, X_treino_prep, X_val_prep in self._folds:
            modelo = construir_modelo(
                cromossomo,
                random_state=self.random_state,
                n_jobs=self.n_jobs_modelo,
            )
            modelo.fit(X_treino_prep, self.y.iloc[treino_idx])
            probabilidades[val_idx] = modelo.predict_proba(X_val_prep)[:, 1]

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