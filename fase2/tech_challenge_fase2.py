"""
Tech Challenge - Fase 2

Programa principal para otimização genética do modelo Random Forest de
triagem de hipertensão desenvolvido na Fase 1.

Fluxo executado:
1. carrega os metadados e as 20 variáveis do modelo da Fase 1;
2. carrega e separa a base Vigitel em treino e teste;
3. executa três configurações do algoritmo genético;
4. revalida os vencedores usando somente o conjunto de treino;
5. avalia o melhor modelo no teste isolado;
6. compara o resultado com o baseline da Fase 1 e salva os artefatos.
"""

from __future__ import annotations

# ============================================================
# IMPORTAÇÕES
# ============================================================

import argparse
import sys
from datetime import datetime
from pathlib import Path


# ============================================================
# CONFIGURAÇÃO DOS CAMINHOS DO PROJETO
# ============================================================

PASTA_PROJETO = Path(__file__).resolve().parent
PASTA_CODIGO_FONTE = PASTA_PROJETO / "src"

# Permite executar este arquivo diretamente, sem instalar o pacote.
sys.path.insert(0, str(PASTA_CODIGO_FONTE))

from techchallenge_fase2.data import (  # noqa: E402
    carregar_metadata_fase1,
    carregar_vigitel,
    separar_treino_teste,
)
from techchallenge_fase2.experiment import executar_experimentos  # noqa: E402
from techchallenge_fase2.logging_utils import imprimir_linha, log_etapa, log_info  # noqa: E402
from techchallenge_fase2.modeling import cromossomo_baseline  # noqa: E402


# ============================================================
# CAMINHOS PADRÃO
# ============================================================

ARQUIVO_BASE_PADRAO = PASTA_PROJETO / "vigitel.csv"
PASTA_FASE1 = PASTA_PROJETO / "techchallenge_fase1"
ARQUIVO_METADATA_FASE1_PADRAO = PASTA_FASE1 / "modelo_api" / "metadata_modelo_api.json"
PASTA_CONFIGURACOES = PASTA_PROJETO / "configs"
PASTA_RESULTADOS = PASTA_PROJETO / "resultados"


# ============================================================
# ARGUMENTOS DE EXECUÇÃO
# ============================================================

def ler_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa os três experimentos de otimização genética da Fase 2."
    )
    parser.add_argument(
        "--data",
        default=str(ARQUIVO_BASE_PADRAO),
        help="Caminho do vigitel.csv. Padrão: arquivo na raiz da Fase 2.",
    )
    parser.add_argument(
        "--metadata-fase1",
        default=str(ARQUIVO_METADATA_FASE1_PADRAO),
        help="Caminho do metadata_modelo_api.json produzido na Fase 1.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=120_000,
        help="Quantidade máxima de registros de treino usada na busca genética.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Quantidade de folds da validação cruzada.",
    )
    parser.add_argument(
        "--cv-jobs",
        type=int,
        default=1,
        help="Quantidade de processos paralelos na validação cruzada.",
    )
    parser.add_argument(
        "--output",
        default=str(PASTA_RESULTADOS),
        help="Pasta onde os resultados serão gravados.",
    )
    return parser.parse_args()


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:
    inicio = datetime.now()
    argumentos = ler_argumentos()

    imprimir_linha()
    log_etapa("INICIANDO O TECH CHALLENGE DA FASE 2")
    imprimir_linha()

    # --------------------------------------------------------
    # 1) CARREGAMENTO DOS METADADOS DA FASE 1
    # --------------------------------------------------------
    log_etapa("Carregando os metadados do modelo de referência da Fase 1...")
    metadata = carregar_metadata_fase1(argumentos.metadata_fase1)
    variaveis_entrada = metadata["variaveis_entrada"]
    target = metadata.get("target", "hart")
    log_info(f"Modelo de referência: {metadata.get('algoritmo', 'RandomForestClassifier')}")
    log_info(f"Target: {target}")
    log_info(f"Quantidade de variáveis de entrada: {len(variaveis_entrada)}")

    # --------------------------------------------------------
    # 2) CARREGAMENTO E SEPARAÇÃO DA BASE
    # --------------------------------------------------------
    log_etapa("Carregando a base Vigitel...")
    X, y = carregar_vigitel(argumentos.data, variaveis_entrada, target)
    log_info(f"Registros válidos: {len(X)}")
    log_info(f"Distribuição do target: {y.value_counts().to_dict()}")

    log_etapa("Separando os conjuntos de treino e teste...")
    X_treino, X_teste, y_treino, y_teste = separar_treino_teste(X, y)
    log_info(f"Treino: {X_treino.shape}")
    log_info(f"Teste isolado: {X_teste.shape}")

    # --------------------------------------------------------
    # 3) CONFIGURAÇÃO DOS EXPERIMENTOS GENÉTICOS
    # --------------------------------------------------------
    configuracoes = sorted(PASTA_CONFIGURACOES.glob("experimento_*.json"))
    if len(configuracoes) != 3:
        raise ValueError(
            f"Eram esperadas 3 configurações em {PASTA_CONFIGURACOES}; "
            f"foram encontradas {len(configuracoes)}."
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    pasta_execucao = Path(argumentos.output) / run_id
    log_info(f"Execução: {run_id}")
    log_info(f"Pasta de resultados: {pasta_execucao}")

    # --------------------------------------------------------
    # 4) OTIMIZAÇÃO, REVALIDAÇÃO E TESTE FINAL
    # --------------------------------------------------------
    resumo = executar_experimentos(
        configuracoes,
        X_treino,
        y_treino,
        X_teste,
        y_teste,
        pasta_execucao,
        baseline_chromosome=cromossomo_baseline(metadata),
        sample_size=argumentos.sample_size,
        folds=argumentos.folds,
        cv_jobs=argumentos.cv_jobs,
    )

    # --------------------------------------------------------
    # 5) RESUMO FINAL
    # --------------------------------------------------------
    duracao = datetime.now() - inicio
    imprimir_linha()
    log_etapa("OTIMIZAÇÃO GENÉTICA CONCLUÍDA COM SUCESSO")
    imprimir_linha()
    print(f"Experimento vencedor: {resumo['experimento_vencedor']}")
    print(f"Melhor cromossomo: {resumo['melhor_cromossomo']}")
    print(f"Métricas no teste: {resumo['metricas_teste']}")
    print(f"Métricas do baseline: {resumo['metricas_baseline_teste']}")
    print(f"Tempo total: {duracao}")
    print(f"Resultados salvos em: {pasta_execucao}")


if __name__ == "__main__":
    main()

