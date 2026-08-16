import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# FUNÇÕES DE LOG
# ============================================================

def log_etapa(mensagem):
    agora = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{agora}] {mensagem}", flush=True)


def log_info(mensagem):
    agora = datetime.now().strftime("%H:%M:%S")
    print(f"[{agora}] {mensagem}", flush=True)


def imprimir_linha():
    print("=" * 80, flush=True)


# ============================================================
# FUNÇÃO PARA RECUPERAR VARIÁVEL ORIGINAL
# ============================================================

def obter_variavel_original(nome_feature, num_cols, cat_cols):
    nome = str(nome_feature)

    if nome.startswith("num__"):
        return nome.replace("num__", "", 1)

    if nome.startswith("cat__"):
        nome_sem_prefixo = nome.replace("cat__", "", 1)

        cat_cols_ordenadas = sorted(
            list(cat_cols),
            key=len,
            reverse=True
        )

        for col in cat_cols_ordenadas:
            if nome_sem_prefixo == col or nome_sem_prefixo.startswith(col + "_"):
                return col

        return nome_sem_prefixo

    return nome


# ============================================================
# INÍCIO
# ============================================================

inicio_execucao = datetime.now()

log_etapa("Iniciando treinamento do modelo final da API com Random Forest...")


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_PROJETO = os.path.dirname(os.path.abspath(__file__))

ARQUIVO_BASE_TRATADA = os.path.join(
    PASTA_PROJETO,
    "base_tratada_sem_vazamento.csv"
)

TARGET = "hart"

RANDOM_STATE = 42

# Hiperparâmetros alinhados com o Jupyter
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 12
RF_MIN_SAMPLES_LEAF = 5

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

PASTA_RESULTADOS = os.path.join(
    PASTA_PROJETO,
    "resultados_modelo_api_random_forest",
    RUN_ID
)

PASTA_API = Path(PASTA_RESULTADOS) / "modelo_api"

os.makedirs(PASTA_RESULTADOS, exist_ok=True)
PASTA_API.mkdir(parents=True, exist_ok=True)

CAMINHO_MODELO_API = PASTA_API / "modelo_hipertensao_api.joblib"
CAMINHO_METADATA_API = PASTA_API / "metadata_modelo_api.json"
CAMINHO_EXEMPLO_API = PASTA_API / "exemplo_entrada_api.json"
CAMINHO_BASE_TESTE_API = PASTA_API / "base_teste_modelo_api.csv"
CAMINHO_FEATURE_IMPORTANCE = Path(PASTA_RESULTADOS) / "feature_importance_random_forest_api.csv"
CAMINHO_TOP20 = Path(PASTA_RESULTADOS) / "top20_variaveis_api.txt"
CAMINHO_METRICAS = Path(PASTA_RESULTADOS) / "metricas_modelo_api.csv"

log_info(f"Pasta do projeto: {PASTA_PROJETO}")
log_info(f"Base tratada: {ARQUIVO_BASE_TRATADA}")
log_info(f"Pasta de resultados: {PASTA_RESULTADOS}")
log_info(f"Pasta modelo API: {PASTA_API}")


# ============================================================
# VALIDAÇÃO DA BASE
# ============================================================

log_etapa("Validando existência da base tratada...")

if not os.path.exists(ARQUIVO_BASE_TRATADA):
    raise FileNotFoundError(
        f"Base tratada não encontrada: {ARQUIVO_BASE_TRATADA}\n"
        "Execute primeiro o programa completo que gera a base_tratada_sem_vazamento.csv."
    )

log_info("Base tratada encontrada.")


# ============================================================
# CARREGAMENTO DA BASE TRATADA
# ============================================================

log_etapa("Carregando base_tratada_sem_vazamento.csv...")

df = pd.read_csv(
    ARQUIVO_BASE_TRATADA,
    encoding="utf-8-sig"
)

log_info(f"Linhas carregadas: {df.shape[0]}")
log_info(f"Colunas carregadas: {df.shape[1]}")

if TARGET not in df.columns:
    raise ValueError(f"A coluna target '{TARGET}' não foi encontrada na base tratada.")

df = df.dropna(subset=[TARGET])

log_info("Distribuição do target:")
print(df[TARGET].value_counts(), flush=True)

log_info("Distribuição percentual do target:")
print(df[TARGET].value_counts(normalize=True) * 100, flush=True)


# ============================================================
# SEPARAÇÃO X / Y
# ============================================================

log_etapa("Separando features e target...")

y = df[TARGET]
X = df.drop(columns=[TARGET])

log_info(f"Total de registros utilizados: {X.shape[0]}")
log_info(f"Total de variáveis disponíveis: {X.shape[1]}")


# ============================================================
# SEPARAÇÃO TREINO / TESTE
# ============================================================

log_etapa("Separando dados de treino e teste...")

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

log_info(f"X_treino: {X_treino.shape}")
log_info(f"X_teste: {X_teste.shape}")


# ============================================================
# IDENTIFICAR VARIÁVEIS NUMÉRICAS E CATEGÓRICAS
# ============================================================

log_etapa("Identificando variáveis numéricas e categóricas...")

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns

log_info(f"Variáveis numéricas: {len(num_cols)}")
log_info(f"Variáveis categóricas: {len(cat_cols)}")


# ============================================================
# PRÉ-PROCESSAMENTO COMPLETO
# ============================================================

log_etapa("Montando pré-processador para o modelo completo...")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            num_cols
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            cat_cols
        )
    ]
)


# ============================================================
# RANDOM FOREST COMPLETO PARA CALCULAR FEATURE IMPORTANCE
# ============================================================

log_etapa("Treinando Random Forest completo para calcular importância das variáveis...")

modelo_rf_completo = RandomForestClassifier(
    n_estimators=RF_N_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    min_samples_leaf=RF_MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE,
    n_jobs=1,
    class_weight="balanced"
)

pipeline_rf_completo = Pipeline([
    ("prep", preprocessor),
    ("modelo", modelo_rf_completo)
])

pipeline_rf_completo.fit(X_treino, y_treino)

log_info("Random Forest completo treinado com sucesso.")


# ============================================================
# AVALIAÇÃO DO RANDOM FOREST COMPLETO
# ============================================================

log_etapa("Avaliando Random Forest completo no conjunto de teste...")

pred_completo = pipeline_rf_completo.predict(X_teste)
proba_completo = pipeline_rf_completo.predict_proba(X_teste)[:, 1]

acc_completo = accuracy_score(y_teste, pred_completo)
precision_completo = precision_score(y_teste, pred_completo, zero_division=0)
recall_completo = recall_score(y_teste, pred_completo, zero_division=0)
f1_completo = f1_score(y_teste, pred_completo, zero_division=0)
auc_completo = roc_auc_score(y_teste, proba_completo)

log_info("Métricas do Random Forest completo:")
print(f"Accuracy:  {acc_completo:.4f}", flush=True)
print(f"Precision: {precision_completo:.4f}", flush=True)
print(f"Recall:    {recall_completo:.4f}", flush=True)
print(f"F1-score:  {f1_completo:.4f}", flush=True)
print(f"AUC-ROC:   {auc_completo:.4f}", flush=True)

print("\nClassification Report - Random Forest completo:", flush=True)
print(classification_report(y_teste, pred_completo, zero_division=0), flush=True)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

log_etapa("Calculando feature importance do Random Forest completo...")

feature_names = pipeline_rf_completo.named_steps["prep"].get_feature_names_out()
importances = pipeline_rf_completo.named_steps["modelo"].feature_importances_

importance_df = pd.DataFrame({
    "variavel_transformada": feature_names,
    "importancia": importances
})

importance_df["variavel_original"] = importance_df["variavel_transformada"].apply(
    lambda nome: obter_variavel_original(nome, num_cols, cat_cols)
)

importance_df = importance_df.sort_values(
    by="importancia",
    ascending=False
)

importance_df.to_csv(
    CAMINHO_FEATURE_IMPORTANCE,
    index=False,
    encoding="utf-8-sig"
)

log_info(f"Feature importance exportada em: {CAMINHO_FEATURE_IMPORTANCE}")


# ============================================================
# TOP 20 VARIÁVEIS ORIGINAIS
# ============================================================

log_etapa("Selecionando as 20 variáveis originais mais importantes...")

top20_variaveis_api = (
    importance_df["variavel_original"]
    .drop_duplicates()
    .head(20)
    .tolist()
)

print("\n", flush=True)
imprimir_linha()
print("Top 20 variáveis selecionadas para o modelo da API", flush=True)
imprimir_linha()

for indice, variavel in enumerate(top20_variaveis_api, start=1):
    print(f"{indice}. {variavel}", flush=True)

with open(CAMINHO_TOP20, "w", encoding="utf-8") as arquivo:
    for variavel in top20_variaveis_api:
        arquivo.write(f"{variavel}\n")

log_info(f"Lista Top 20 exportada em: {CAMINHO_TOP20}")


# ============================================================
# CRIAR BASE COM AS 20 VARIÁVEIS
# ============================================================

log_etapa("Criando base reduzida com as 20 variáveis da API...")

X_api = X[top20_variaveis_api].copy()

X_api_treino, X_api_teste, y_api_treino, y_api_teste = train_test_split(
    X_api,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

log_info(f"X_api_treino: {X_api_treino.shape}")
log_info(f"X_api_teste: {X_api_teste.shape}")


# ============================================================
# IDENTIFICAR TIPOS DAS 20 VARIÁVEIS
# ============================================================

log_etapa("Identificando tipos das 20 variáveis da API...")

num_cols_api = X_api.select_dtypes(include=["int64", "float64"]).columns
cat_cols_api = X_api.select_dtypes(include=["object", "category", "string"]).columns

log_info("Variáveis numéricas da API:")
print(list(num_cols_api), flush=True)

log_info("Variáveis categóricas da API:")
print(list(cat_cols_api), flush=True)


# ============================================================
# PRÉ-PROCESSAMENTO DA API
# ============================================================

log_etapa("Montando pré-processador exclusivo do modelo da API...")

preprocessor_api = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            num_cols_api
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            cat_cols_api
        )
    ]
)


# ============================================================
# MODELO FINAL DA API - RANDOM FOREST
# ============================================================

log_etapa("Configurando Random Forest final da API com hiperparâmetros alinhados ao Jupyter...")

modelo_api = RandomForestClassifier(
    n_estimators=RF_N_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    min_samples_leaf=RF_MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE,
    n_jobs=1,
    class_weight="balanced"
)

pipeline_api = Pipeline([
    ("prep", preprocessor_api),
    ("modelo", modelo_api)
])


# ============================================================
# TREINAMENTO DO MODELO DA API
# ============================================================

log_etapa("Treinando Random Forest final da API com as 20 variáveis...")

pipeline_api.fit(X_api_treino, y_api_treino)

log_info("Modelo final da API treinado com sucesso.")


# ============================================================
# AVALIAÇÃO DO MODELO DA API
# ============================================================

log_etapa("Avaliando modelo final da API no conjunto de teste...")

pred_api = pipeline_api.predict(X_api_teste)
proba_api = pipeline_api.predict_proba(X_api_teste)[:, 1]

acc_api = accuracy_score(y_api_teste, pred_api)
precision_api = precision_score(y_api_teste, pred_api, zero_division=0)
recall_api = recall_score(y_api_teste, pred_api, zero_division=0)
f1_api = f1_score(y_api_teste, pred_api, zero_division=0)
auc_api = roc_auc_score(y_api_teste, proba_api)

print("\n", flush=True)
imprimir_linha()
print("Resultado do modelo final da API - Random Forest com 20 variáveis", flush=True)
imprimir_linha()

print(f"Accuracy:  {acc_api:.4f}", flush=True)
print(f"Precision: {precision_api:.4f}", flush=True)
print(f"Recall:    {recall_api:.4f}", flush=True)
print(f"F1-score:  {f1_api:.4f}", flush=True)
print(f"AUC-ROC:   {auc_api:.4f}", flush=True)

print("\nClassification Report - Modelo API:", flush=True)
print(classification_report(y_api_teste, pred_api, zero_division=0), flush=True)


# ============================================================
# SALVAR MÉTRICAS
# ============================================================

log_etapa("Salvando métricas dos modelos...")

metricas_df = pd.DataFrame([
    {
        "modelo": "Random_Forest_Completo",
        "quantidade_variaveis": X.shape[1],
        "accuracy": acc_completo,
        "precision": precision_completo,
        "recall": recall_completo,
        "f1_score": f1_completo,
        "roc_auc": auc_completo
    },
    {
        "modelo": "Random_Forest_API_Top20",
        "quantidade_variaveis": X_api.shape[1],
        "accuracy": acc_api,
        "precision": precision_api,
        "recall": recall_api,
        "f1_score": f1_api,
        "roc_auc": auc_api
    }
])

metricas_df.to_csv(
    CAMINHO_METRICAS,
    index=False,
    encoding="utf-8-sig"
)

log_info(f"Métricas exportadas em: {CAMINHO_METRICAS}")


# ============================================================
# MATRIZ DE CONFUSÃO DO MODELO DA API
# ============================================================

log_etapa("Gerando matriz de confusão do modelo final da API...")

ConfusionMatrixDisplay.from_predictions(
    y_api_teste,
    pred_api,
    display_labels=["Sem Hipertensão", "Com Hipertensão"],
    values_format="d",
    cmap="Blues"
)

plt.title("Matriz de Confusão - Random Forest API Top 20")
plt.tight_layout()

CAMINHO_MATRIZ_API = Path(PASTA_RESULTADOS) / "matriz_confusao_modelo_api_random_forest.png"

plt.savefig(CAMINHO_MATRIZ_API, dpi=300)
plt.close()

log_info(f"Matriz de confusão salva em: {CAMINHO_MATRIZ_API}")


# ============================================================
# GRÁFICO TOP 20 FEATURE IMPORTANCE
# ============================================================

log_etapa("Gerando gráfico Top 20 Feature Importance...")

top20_plot = (
    importance_df
    .drop_duplicates(subset=["variavel_original"])
    .head(20)
    .copy()
)

plt.figure(figsize=(10, 8))

plt.barh(
    top20_plot["variavel_original"],
    top20_plot["importancia"]
)

plt.gca().invert_yaxis()
plt.title("Top 20 Variáveis Mais Importantes - Random Forest")
plt.xlabel("Importância")
plt.tight_layout()

CAMINHO_GRAFICO_TOP20 = Path(PASTA_RESULTADOS) / "top20_feature_importance_random_forest.png"

plt.savefig(CAMINHO_GRAFICO_TOP20, dpi=300)
plt.close()

log_info(f"Gráfico Top 20 salvo em: {CAMINHO_GRAFICO_TOP20}")


# ============================================================
# EXPORTAR MODELO FINAL DA API
# ============================================================

log_etapa("Exportando modelo final da API em joblib...")

joblib.dump(pipeline_api, CAMINHO_MODELO_API)

log_info("Modelo da API exportado em:")
print(CAMINHO_MODELO_API, flush=True)


# ============================================================
# EXPORTAR METADATA DA API
# ============================================================

log_etapa("Exportando metadata do modelo da API...")

metadata_api = {
    "nome_modelo": "modelo_hipertensao_api",
    "algoritmo": "RandomForestClassifier",
    "descricao": "Modelo Random Forest treinado com as 20 variáveis mais importantes, alinhado ao resultado do Jupyter.",
    "target": TARGET,
    "descricao_target": {
        "0": "Sem hipertensão",
        "1": "Com hipertensão"
    },
    "threshold_classificacao": 0.5,
    "hiperparametros": {
        "n_estimators": RF_N_ESTIMATORS,
        "max_depth": RF_MAX_DEPTH,
        "min_samples_leaf": RF_MIN_SAMPLES_LEAF,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE
    },
    "variaveis_entrada": top20_variaveis_api,
    "variaveis_numericas": list(num_cols_api),
    "variaveis_categoricas": list(cat_cols_api),
    "metricas_modelo_completo": {
        "quantidade_variaveis": int(X.shape[1]),
        "accuracy": round(acc_completo, 4),
        "precision": round(precision_completo, 4),
        "recall": round(recall_completo, 4),
        "f1_score": round(f1_completo, 4),
        "roc_auc": round(auc_completo, 4)
    },
    "metricas_modelo_api_top20": {
        "quantidade_variaveis": int(X_api.shape[1]),
        "accuracy": round(acc_api, 4),
        "precision": round(precision_api, 4),
        "recall": round(recall_api, 4),
        "f1_score": round(f1_api, 4),
        "roc_auc": round(auc_api, 4)
    }
}

with open(CAMINHO_METADATA_API, "w", encoding="utf-8") as arquivo:
    json.dump(
        metadata_api,
        arquivo,
        ensure_ascii=False,
        indent=4,
        allow_nan=False
    )

log_info("Metadata da API exportado em:")
print(CAMINHO_METADATA_API, flush=True)


# ============================================================
# EXPORTAR EXEMPLO DE ENTRADA
# ============================================================

log_etapa("Exportando exemplo de entrada da API...")

exemplo_entrada = (
    X_api_teste
    .head(1)
    .replace({np.nan: None})
    .to_dict(orient="records")[0]
)

with open(CAMINHO_EXEMPLO_API, "w", encoding="utf-8") as arquivo:
    json.dump(
        exemplo_entrada,
        arquivo,
        ensure_ascii=False,
        indent=4,
        allow_nan=False
    )

log_info("Exemplo de entrada exportado em:")
print(CAMINHO_EXEMPLO_API, flush=True)


# ============================================================
# EXPORTAR BASE DE TESTE DA API
# ============================================================

log_etapa("Exportando base de teste reduzida da API...")

base_teste_api = X_api_teste.copy()
base_teste_api[TARGET] = y_api_teste

base_teste_api.to_csv(
    CAMINHO_BASE_TESTE_API,
    index=False,
    encoding="utf-8-sig"
)

log_info("Base de teste da API exportada em:")
print(CAMINHO_BASE_TESTE_API, flush=True)


# ============================================================
# RESUMO FINAL
# ============================================================

fim_execucao = datetime.now()
tempo_total = fim_execucao - inicio_execucao

print("\n", flush=True)
imprimir_linha()
print("Resumo final da execução", flush=True)
imprimir_linha()

print(f"Modelo final da API: Random Forest", flush=True)
print(f"Hiperparâmetros: n_estimators={RF_N_ESTIMATORS}, max_depth={RF_MAX_DEPTH}, min_samples_leaf={RF_MIN_SAMPLES_LEAF}", flush=True)
print(f"Total de variáveis disponíveis na base: {X.shape[1]}", flush=True)
print(f"Total de variáveis usadas na API: {X_api.shape[1]}", flush=True)
print(f"AUC-ROC modelo completo: {auc_completo:.4f}", flush=True)
print(f"AUC-ROC modelo API Top 20: {auc_api:.4f}", flush=True)
print(f"F1-score modelo API Top 20: {f1_api:.4f}", flush=True)
print(f"Pasta de resultados: {PASTA_RESULTADOS}", flush=True)
print(f"Modelo exportado: {CAMINHO_MODELO_API}", flush=True)
print(f"Início: {inicio_execucao.strftime('%d/%m/%Y %H:%M:%S')}", flush=True)
print(f"Fim: {fim_execucao.strftime('%d/%m/%Y %H:%M:%S')}", flush=True)
print(f"Tempo total: {tempo_total}", flush=True)

print("\nPrograma finalizado com sucesso.", flush=True)