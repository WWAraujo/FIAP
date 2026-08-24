from __future__ import annotations

# ============================================================
# IMPORTAÇÕES
# ============================================================

import json
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# FUNÇÕES DE NORMALIZAÇÃO DOS DADOS DO VIGITEL
# ============================================================

def normalizar_texto(valor: Any) -> Any:
    if pd.isna(valor):
        return valor
    texto = unicodedata.normalize("NFKD", str(valor).strip().lower())
    return "".join(caractere for caractere in texto if not unicodedata.combining(caractere))


def converter_sim_nao(valor: Any) -> Any:
    if pd.isna(valor):
        return valor
    mapa = {
        "sim": 1,
        "s": 1,
        "yes": 1,
        "y": 1,
        "nao": 0,
        "n": 0,
        "no": 0,
    }
    return mapa.get(normalizar_texto(valor), valor)


# ============================================================
# CARREGAMENTO DOS METADADOS DA FASE 1
# ============================================================

def carregar_metadata_fase1(caminho: str | Path) -> dict[str, Any]:
    with Path(caminho).open(encoding="utf-8") as arquivo:
        return json.load(arquivo)


# ============================================================
# CARREGAMENTO DA BASE VIGITEL
# ============================================================

def carregar_vigitel(
    caminho: str | Path,
    variaveis: list[str],
    target: str = "hart",
) -> tuple[pd.DataFrame, pd.Series]:
    """Carrega somente o target e as 20 colunas do modelo final da Fase 1."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {caminho}")

    colunas = list(dict.fromkeys([*variaveis, target]))
    df = pd.read_csv(
        caminho,
        encoding="latin1",
        sep=",",
        usecols=lambda coluna: coluna in colunas,
        on_bad_lines="skip",
        low_memory=False,
    )

    ausentes = sorted(set(colunas) - set(df.columns))
    if ausentes:
        raise ValueError(f"Colunas obrigatorias ausentes no dataset: {ausentes}")

    colunas_texto = df.select_dtypes(include=["object", "string"]).columns
    if len(colunas_texto):
        df[colunas_texto] = df[colunas_texto].apply(
            lambda coluna: coluna.map(converter_sim_nao)
        )

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target]).copy()
    df[target] = df[target].astype(int)

    classes = set(df[target].unique())
    if not classes.issubset({0, 1}) or len(classes) != 2:
        raise ValueError(f"O target deve ser binario (0/1); classes encontradas: {classes}")

    return df[variaveis].copy(), df[target].copy()


# ============================================================
# SEPARAÇÃO ENTRE TREINO E TESTE
# ============================================================

def separar_treino_teste(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# ============================================================
# AMOSTRAGEM ESTRATIFICADA PARA A BUSCA GENÉTICA
# ============================================================

def amostra_estratificada(
    X: pd.DataFrame,
    y: pd.Series,
    tamanho: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    if tamanho <= 0 or tamanho >= len(X):
        return X.copy(), y.copy()

    # A proporção entre as classes é preservada para que a amostra represente
    # corretamente o problema original de triagem de hipertensão.
    X_amostra, _, y_amostra, _ = train_test_split(
        X,
        y,
        train_size=tamanho,
        random_state=random_state,
        stratify=y,
    )
    return X_amostra, y_amostra
