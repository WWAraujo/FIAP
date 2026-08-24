from __future__ import annotations

# ============================================================
# IMPORTAÇÕES
# ============================================================

import random
import statistics
from dataclasses import dataclass
from typing import Any, Callable

from joblib import Parallel, delayed

from .chromosome import SearchSpace
from .fitness import FitnessResult
from ..logging_utils import log_info


# ============================================================
# CONFIGURAÇÃO DO ALGORITMO GENÉTICO
# ============================================================

@dataclass(frozen=True)
class GeneticConfig:
    nome: str
    population_size: int
    generations: int
    mutation_rate: float
    crossover_rate: float
    elite_size: int
    tournament_size: int
    patience: int
    random_state: int

    @classmethod
    def from_dict(cls, dados: dict[str, Any]) -> "GeneticConfig":
        config = cls(**dados)
        config.validar()
        return config

    def validar(self) -> None:
        if self.population_size < 2:
            raise ValueError("A população deve possuir ao menos 2 indivíduos.")
        if not 0 <= self.mutation_rate <= 1 or not 0 <= self.crossover_rate <= 1:
            raise ValueError("Taxas de mutação e crossover devem estar entre 0 e 1.")
        if not 1 <= self.elite_size < self.population_size:
            raise ValueError("Elitismo deve estar entre 1 e population_size - 1.")
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError("Tamanho do torneio inválido.")


# ============================================================
# RESULTADO COMPLETO DA OTIMIZAÇÃO
# ============================================================

@dataclass
class OptimizationResult:
    config: GeneticConfig
    best_chromosome: dict[str, Any]
    best_metrics: FitnessResult
    history: list[dict[str, Any]]
    evaluations: list[dict[str, Any]]


class GeneticOptimizer:
    """Executa seleção, crossover, mutação, elitismo e parada antecipada."""

    def __init__(
        self,
        config: GeneticConfig,
        evaluator: Callable[[dict[str, Any]], FitnessResult],
        search_space: SearchSpace | None = None,
        n_jobs_populacao: int = -1,
    ) -> None:
        self.config = config
        self.evaluator = evaluator
        self.search_space = search_space or SearchSpace()
        self.rng = random.Random(config.random_state)
        self.cache: dict[tuple[Any, ...], FitnessResult] = {}
        self.evaluations: list[dict[str, Any]] = []
        self.n_jobs_populacao = n_jobs_populacao

    def _avaliar_populacao(
        self,
        populacao: list[dict[str, Any]],
        geracao: int,
    ) -> list[FitnessResult]:
        # Só os cromossomos ainda não vistos entram na avaliação paralela;
        # o cache evita retrabalho entre gerações e dentro da mesma geração
        # (comum após crossover, quando filhos repetem combinações dos pais).
        pendentes: list[dict[str, Any]] = []
        chaves_pendentes: list[tuple[Any, ...]] = []
        for individuo in populacao:
            chave = self.search_space.chave(individuo)
            if chave not in self.cache:
                pendentes.append(individuo)
                chaves_pendentes.append(chave)

        if pendentes:
            resultados_novos = Parallel(n_jobs=self.n_jobs_populacao)(
                delayed(self.evaluator)(individuo) for individuo in pendentes
            )
            for individuo, chave, resultado in zip(pendentes, chaves_pendentes, resultados_novos):
                self.cache[chave] = resultado
                self.evaluations.append(
                    {"geracao_primeira_avaliacao": geracao, **individuo, **resultado.to_dict()}
                )

        return [self.cache[self.search_space.chave(individuo)] for individuo in populacao]

    def _torneio(
        self,
        populacao: list[dict[str, Any]],
        resultados: list[FitnessResult],
    ) -> dict[str, Any]:
        # Seleciona aleatoriamente alguns indivíduos e devolve o mais apto.
        indices = self.rng.sample(range(len(populacao)), self.config.tournament_size)
        vencedor = max(indices, key=lambda indice: resultados[indice].fitness)
        return dict(populacao[vencedor])

    def executar(self) -> OptimizationResult:
        # --------------------------------------------------------
        # 1) CRIAÇÃO DA POPULAÇÃO INICIAL
        # --------------------------------------------------------
        populacao = [
            self.search_space.criar(self.rng)
            for _ in range(self.config.population_size)
        ]
        historico: list[dict[str, Any]] = []
        melhor_global: tuple[dict[str, Any], FitnessResult] | None = None
        sem_melhora = 0

        for geracao in range(self.config.generations):
            # ----------------------------------------------------
            # 2) AVALIAÇÃO (PARALELA) E RANKING DA GERAÇÃO ATUAL
            # ----------------------------------------------------
            resultados = self._avaliar_populacao(populacao, geracao)
            ranking = sorted(
                range(len(populacao)),
                key=lambda indice: resultados[indice].fitness,
                reverse=True,
            )
            melhor_indice = ranking[0]
            melhor_geracao = resultados[melhor_indice]
            fitness_populacao = [resultado.fitness for resultado in resultados]
            fitness_medio = statistics.fmean(fitness_populacao)

            historico.append(
                {
                    "geracao": geracao,
                    "melhor_fitness": melhor_geracao.fitness,
                    "fitness_medio": fitness_medio,
                    "fitness_desvio": statistics.pstdev(fitness_populacao),
                    "avaliacoes_unicas_acumuladas": len(self.cache),
                    **{f"melhor_{k}": v for k, v in populacao[melhor_indice].items()},
                    **{f"melhor_{k}": v for k, v in melhor_geracao.to_dict().items() if k != "fitness"},
                }
            )
            log_info(
                f"[{self.config.nome}] Geração {geracao + 1}/{self.config.generations} | "
                f"melhor fitness: {melhor_geracao.fitness:.4f} | "
                f"fitness médio: {fitness_medio:.4f} | "
                f"avaliações únicas: {len(self.cache)}"
            )

            # ----------------------------------------------------
            # 3) ATUALIZAÇÃO DO MELHOR INDIVÍDUO GLOBAL
            # ----------------------------------------------------
            if melhor_global is None or melhor_geracao.fitness > melhor_global[1].fitness:
                melhor_global = (dict(populacao[melhor_indice]), melhor_geracao)
                sem_melhora = 0
            else:
                sem_melhora += 1

            if sem_melhora >= self.config.patience:
                log_info(
                    f"[{self.config.nome}] Parada antecipada: "
                    f"{self.config.patience} gerações sem melhora."
                )
                break

            # ----------------------------------------------------
            # 4) ELITISMO, SELEÇÃO, CROSSOVER E MUTAÇÃO
            # ----------------------------------------------------
            nova_populacao = [dict(populacao[i]) for i in ranking[: self.config.elite_size]]
            while len(nova_populacao) < self.config.population_size:
                pai_a = self._torneio(populacao, resultados)
                pai_b = self._torneio(populacao, resultados)
                if self.rng.random() < self.config.crossover_rate:
                    filho_a, filho_b = self.search_space.crossover(pai_a, pai_b, self.rng)
                else:
                    filho_a, filho_b = dict(pai_a), dict(pai_b)
                filho_a = self.search_space.mutar(filho_a, self.config.mutation_rate, self.rng)
                filho_b = self.search_space.mutar(filho_b, self.config.mutation_rate, self.rng)
                nova_populacao.append(filho_a)
                if len(nova_populacao) < self.config.population_size:
                    nova_populacao.append(filho_b)
            populacao = nova_populacao

        if melhor_global is None:
            raise RuntimeError("O algoritmo genético não avaliou nenhum indivíduo.")

        # --------------------------------------------------------
        # 5) RESULTADO FINAL DO EXPERIMENTO
        # --------------------------------------------------------
        return OptimizationResult(
            config=self.config,
            best_chromosome=melhor_global[0],
            best_metrics=melhor_global[1],
            history=historico,
            evaluations=self.evaluations,
        )