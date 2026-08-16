from __future__ import annotations

# ============================================================
# IMPORTAÇÕES
# ============================================================

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .data import amostra_estratificada
from .genetic import GeneticConfig, GeneticOptimizer, MedicalFitnessEvaluator
from .genetic.fitness import FitnessResult, avaliar_no_teste
from .logging_utils import imprimir_linha, log_etapa, log_info
from .modeling import construir_pipeline
from .reporting import salvar_json, salvar_resultado_experimento


# ============================================================
# CARREGAMENTO DAS CONFIGURAÇÕES DOS EXPERIMENTOS
# ============================================================

def carregar_config(caminho: str | Path) -> GeneticConfig:
    with Path(caminho).open(encoding="utf-8") as arquivo:
        return GeneticConfig.from_dict(json.load(arquivo))


def executar_experimentos(
    configs: list[str | Path],
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    X_teste: pd.DataFrame,
    y_teste: pd.Series,
    pasta_resultados: str | Path,
    baseline_chromosome: dict[str, Any] | None = None,
    sample_size: int = 120_000,
    folds: int = 3,
    cv_jobs: int = 1,
) -> dict[str, Any]:
    # --------------------------------------------------------
    # PREPARAÇÃO DA PASTA E DA AMOSTRA DE BUSCA
    # --------------------------------------------------------
    pasta_resultados = Path(pasta_resultados)
    pasta_resultados.mkdir(parents=True, exist_ok=True)
    X_busca, y_busca = amostra_estratificada(X_treino, y_treino, sample_size, 42)
    log_info(f"Amostra estratificada usada na busca: {len(X_busca)} registros")

    # --------------------------------------------------------
    # EXECUÇÃO DAS TRÊS CONFIGURAÇÕES DO ALGORITMO GENÉTICO
    # --------------------------------------------------------
    vencedores = []
    for caminho_config in configs:
        config = carregar_config(caminho_config)
        imprimir_linha()
        log_etapa(f"Iniciando o experimento genético: {config.nome}")
        log_info(f"População: {config.population_size}")
        log_info(f"Gerações máximas: {config.generations}")
        log_info(f"Taxa de mutação: {config.mutation_rate:.0%}")
        log_info(f"Taxa de crossover: {config.crossover_rate:.0%}")

        evaluator = MedicalFitnessEvaluator(
            X_busca,
            y_busca,
            folds=folds,
            random_state=config.random_state,
            cv_jobs=cv_jobs,
        )
        resultado = GeneticOptimizer(config, evaluator).executar()
        salvar_resultado_experimento(resultado, pasta_resultados / config.nome)
        vencedores.append((resultado.best_chromosome, resultado.best_metrics, config.nome))
        log_info(f"Melhor fitness do experimento: {resultado.best_metrics.fitness:.4f}")
        log_info(f"Melhor cromossomo: {resultado.best_chromosome}")

    # --------------------------------------------------------
    # REVALIDAÇÃO DOS VENCEDORES NO TREINO COMPLETO
    # --------------------------------------------------------
    # Os três vencedores da busca amostral são reavaliados no treino completo.
    # O conjunto de teste continua completamente fora da decisão.
    log_etapa("Reavaliando os vencedores no conjunto de treino completo...")
    vencedores_revalidados = []
    for cromossomo, metricas_amostra, nome in vencedores:
        log_info(f"Reavaliando o vencedor de {nome}...")
        reavaliador = MedicalFitnessEvaluator(
            X_treino,
            y_treino,
            folds=folds,
            random_state=42,
            cv_jobs=cv_jobs,
        )
        metricas_treino_completo = reavaliador(cromossomo)
        vencedores_revalidados.append(
            {
                "experimento": nome,
                **cromossomo,
                **{f"amostra_{k}": v for k, v in metricas_amostra.to_dict().items()},
                **{f"treino_completo_{k}": v for k, v in metricas_treino_completo.to_dict().items()},
            }
        )

    pd.DataFrame(vencedores_revalidados).to_csv(
        pasta_resultados / "comparacao_vencedores_validacao.csv",
        index=False,
    )
    melhor_revalidado = max(
        vencedores_revalidados,
        key=lambda item: item["treino_completo_fitness"],
    )
    nome_experimento = melhor_revalidado["experimento"]
    vencedor, _, _ = next(item for item in vencedores if item[2] == nome_experimento)
    metricas_validacao = FitnessResult(
        **{
            chave: melhor_revalidado[f"treino_completo_{chave}"]
            for chave in ("fitness", "accuracy", "precision", "recall", "f1", "roc_auc")
        }
    )

    # --------------------------------------------------------
    # TREINAMENTO E AVALIAÇÃO FINAL DO MODELO GENÉTICO
    # --------------------------------------------------------
    log_etapa(f"Treinando o vencedor do {nome_experimento} no treino completo...")
    pipeline = construir_pipeline(X_treino, vencedor, random_state=42)
    pipeline.fit(X_treino, y_treino)

    log_etapa("Avaliando o modelo genético no teste isolado...")
    metricas_teste = avaliar_no_teste(
        pipeline,
        X_teste,
        y_teste,
        threshold=float(vencedor["threshold"]),
    )
    joblib.dump(pipeline, pasta_resultados / "modelo_genetico_vencedor.joblib")
    log_info(f"AUC-ROC do modelo genético no teste: {metricas_teste.roc_auc:.4f}")
    log_info(f"Recall do modelo genético no teste: {metricas_teste.recall:.4f}")

    # --------------------------------------------------------
    # COMPARAÇÃO COM O BASELINE OFICIAL DA FASE 1
    # --------------------------------------------------------
    metricas_baseline = None
    if baseline_chromosome is not None:
        log_etapa("Treinando e avaliando o baseline oficial da Fase 1...")
        pipeline_baseline = construir_pipeline(X_treino, baseline_chromosome, random_state=42)
        pipeline_baseline.fit(X_treino, y_treino)
        metricas_baseline = avaliar_no_teste(
            pipeline_baseline,
            X_teste,
            y_teste,
            threshold=float(baseline_chromosome["threshold"]),
        )
        pd.DataFrame(
            [
                {"modelo": "baseline_fase1", **metricas_baseline.to_dict()},
                {"modelo": "genetico_fase2", **metricas_teste.to_dict()},
            ]
        ).to_csv(pasta_resultados / "comparacao_baseline_genetico.csv", index=False)

    # --------------------------------------------------------
    # GERAÇÃO DO RESUMO FINAL
    # --------------------------------------------------------
    resumo = {
        "experimento_vencedor": nome_experimento,
        "melhor_cromossomo": vencedor,
        "metricas_validacao": metricas_validacao.to_dict(),
        "metricas_teste": metricas_teste.to_dict(),
        "metricas_baseline_teste": metricas_baseline.to_dict() if metricas_baseline else None,
        "amostra_busca": len(X_busca),
        "treino_total": len(X_treino),
        "teste_total": len(X_teste),
    }
    salvar_json(pasta_resultados / "resumo_final.json", resumo)
    return resumo
