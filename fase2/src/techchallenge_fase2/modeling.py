from __future__ import annotations

# ============================================================
# IMPORTAÇÕES
# ============================================================

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PIPELINE DE PRÉ-PROCESSAMENTO
# ============================================================

def construir_preprocessador(X: pd.DataFrame) -> ColumnTransformer:
    """Replica o tratamento numérico e categórico usado na Fase 1."""
    numericas = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categoricas = [coluna for coluna in X.columns if coluna not in numericas]

    transformadores = []
    if numericas:
        transformadores.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numericas,
            )
        )
    if categoricas:
        transformadores.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categoricas,
            )
        )

    if not transformadores:
        raise ValueError("Nenhuma variavel de entrada foi encontrada.")
    return ColumnTransformer(transformers=transformadores)


# ============================================================
# CONVERSÃO DO CROMOSSOMO EM HIPERPARÂMETROS
# ============================================================

def parametros_modelo(cromossomo: dict[str, Any]) -> dict[str, Any]:
    # O threshold pertence à regra de decisão e não ao construtor da Random Forest.
    return {
        chave: valor
        for chave, valor in cromossomo.items()
        if chave != "threshold"
    }


# ============================================================
# CONSTRUÇÃO DO MODELO (SEM PRÉ-PROCESSADOR)
# ============================================================
# Separado do pipeline completo porque, no GA, o pré-processador é sempre
# o mesmo — só a RandomForest muda entre cromossomos. Recriar o
# ColumnTransformer/OneHotEncoder a cada avaliação era custo repetido
# desnecessário; ver fitness.py, que agora faz fit_transform uma única vez.

def construir_modelo(
    cromossomo: dict[str, Any],
    random_state: int = 42,
    n_jobs: int = 1,
) -> RandomForestClassifier:
    parametros = parametros_modelo(cromossomo)
    return RandomForestClassifier(
        **parametros,
        random_state=random_state,
        n_jobs=n_jobs,
    )


# ============================================================
# CONSTRUÇÃO DO PIPELINE COMPLETO
# ============================================================
# Mantido para uso fora do GA (treino final, baseline, avaliação em teste
# isolado), onde não há reaproveitamento de pré-processamento entre
# múltiplos cromossomos e o pipeline único é mais simples de persistir
# com joblib.

def construir_pipeline(
    X: pd.DataFrame,
    cromossomo: dict[str, Any],
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    modelo = construir_modelo(cromossomo, random_state=random_state, n_jobs=n_jobs)
    return Pipeline(
        [
            ("prep", construir_preprocessador(X)),
            ("modelo", modelo),
        ]
    )


# ============================================================
# BASELINE OFICIAL PUBLICADO NA FASE 1
# ============================================================

def cromossomo_baseline(metadata: dict[str, Any]) -> dict[str, Any]:
    parametros = dict(metadata.get("hiperparametros", {}))
    parametros.pop("random_state", None)
    parametros.setdefault("min_samples_split", 2)
    parametros.setdefault("max_features", "sqrt")
    parametros.setdefault("criterion", "gini")
    parametros["threshold"] = float(metadata.get("threshold_classificacao", 0.5))
    return parametros