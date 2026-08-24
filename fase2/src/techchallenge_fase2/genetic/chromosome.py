from __future__ import annotations

# ============================================================
# IMPORTAÇÕES
# ============================================================

import random
from dataclasses import dataclass, field
from typing import Any


# ============================================================
# REPRESENTAÇÃO DE UM GENE
# ============================================================

@dataclass(frozen=True)
class Gene:
    nome: str
    valores: tuple[Any, ...]

    def sortear(self, rng: random.Random) -> Any:
        return rng.choice(self.valores)


# ============================================================
# ESPAÇO DE BUSCA DOS HIPERPARÂMETROS
# ============================================================

@dataclass
class SearchSpace:
    # Cada indivíduo possui uma opção válida para cada gene abaixo.
    genes: tuple[Gene, ...] = field(default_factory=lambda: (
        Gene("n_estimators", (100, 200, 300, 400, 500, 650, 800)),
        Gene("max_depth", (6, 8, 10, 12, 16, 20, 24, 30, None)),
        Gene("min_samples_split", (2, 4, 6, 8, 12, 16, 24, 30)),
        Gene("min_samples_leaf", (1, 2, 4, 5, 8, 10, 15, 20)),
        Gene("max_features", ("sqrt", "log2", 0.3, 0.5)),
        Gene("criterion", ("gini", "entropy", "log_loss")),
        Gene("class_weight", ("balanced", "balanced_subsample")),
        Gene("threshold", (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)),
    ))

    @property
    def nomes(self) -> tuple[str, ...]:
        return tuple(gene.nome for gene in self.genes)

    def criar(self, rng: random.Random) -> dict[str, Any]:
        return {gene.nome: gene.sortear(rng) for gene in self.genes}

    def validar(self, cromossomo: dict[str, Any]) -> bool:
        if set(cromossomo) != set(self.nomes):
            return False
        return all(cromossomo[gene.nome] in gene.valores for gene in self.genes)

    def chave(self, cromossomo: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(cromossomo[nome] for nome in self.nomes)

    def crossover(
        self,
        pai_a: dict[str, Any],
        pai_b: dict[str, Any],
        rng: random.Random,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Crossover uniforme: cada gene tem 50% de chance de vir de cada pai.
        filho_a: dict[str, Any] = {}
        filho_b: dict[str, Any] = {}
        for nome in self.nomes:
            if rng.random() < 0.5:
                filho_a[nome], filho_b[nome] = pai_a[nome], pai_b[nome]
            else:
                filho_a[nome], filho_b[nome] = pai_b[nome], pai_a[nome]
        return filho_a, filho_b

    def mutar(
        self,
        cromossomo: dict[str, Any],
        taxa: float,
        rng: random.Random,
    ) -> dict[str, Any]:
        # Cada gene sofre mutação de forma independente conforme a taxa definida.
        mutante = dict(cromossomo)
        for gene in self.genes:
            if rng.random() < taxa:
                alternativas = tuple(
                    valor for valor in gene.valores if valor != mutante[gene.nome]
                )
                if alternativas:
                    mutante[gene.nome] = rng.choice(alternativas)
        return mutante
