# Fase 2 — Otimização Genética de Hiperparâmetros

Otimiza os hiperparâmetros do modelo RandomForest de triagem de
hipertensão (treinado na Fase 1) usando um algoritmo genético
customizado, comparando o resultado com o baseline original.

Para o deploy do modelo já otimizado como API (com LLM, autoscaling e
monitoramento), veja [api/README.md](./api/README.md).

## Estrutura

```
fase2/
├── tech_challenge_fase2.py       Script principal
├── configs/
│   ├── experimento_a.json        3 configurações do GA testadas
│   ├── experimento_b.json
│   └── experimento_c.json
├── scripts/
│   ├── smoke_test.py             Teste rápido com dados sintéticos
│   └── run_experiments.py        Alias de compatibilidade (chama o script principal)
├── tests/
│   ├── test_data.py              Split treino/teste e amostragem estratificada
│   ├── test_genetic.py           Crossover, mutação e o otimizador em si
│   └── test_integration.py       Fluxo completo ponta a ponta (dados sintéticos)
├── src/techchallenge_fase2/      Código-fonte do pacote
│   ├── data.py
│   ├── experiment.py
│   ├── modeling.py
│   ├── reporting.py
│   ├── logging_utils.py
│   └── genetic/
│       ├── chromosome.py
│       ├── fitness.py
│       └── optimizer.py
├── resultados/                   Outputs de cada execução (gerado em runtime)
└── api/                          Deploy do modelo otimizado (ver api/README.md)
```

## Pré-requisitos

- Python 3.10+
- Base `vigitel.csv` em `shared/data/vigitel.csv` (na raiz do repositório,
  fora de `fase2/`)
- Docker instalado para rodar localmente

## Instalação

```bash
cd fase2
pip install -r requirements.txt
```

## Como rodar o treino do modelo

```bash
python tech_challenge_fase2.py
```



Parâmetros opcionais:

| Argumento | Padrão | Descrição |
|---|---|---|
| `--data` | `../shared/data/vigitel.csv` | Caminho da base Vigitel |
| `--metadata-fase1` | `api/modelo_api/metadata_modelo_api.json` | Metadata do baseline da Fase 1 |
| `--sample-size` | `120000` | Registros usados na busca genética (amostra estratificada) |
| `--folds` | `3` | Folds da validação cruzada |
| `--output` | `resultados/` | Pasta onde os resultados são salvos |

O script carrega os dados, roda os três experimentos configurados em
`configs/`, revalida os vencedores no treino completo, treina o modelo
final, avalia no teste isolado e compara com o baseline da Fase 1.
Resultados de cada execução ficam em `resultados/<timestamp>/`.

## Os três experimentos

`configs/experimento_{a,b,c}.json` variam o tamanho da população e as
taxas de mutação/crossover, do mais conservador ao mais exploratório:

| | `experimento_a` | `experimento_b` | `experimento_c` |
|---|---|---|---|
| População | 12 | 20 | 30 |
| Gerações | 8 | 12 | 15 |
| Taxa de mutação | 0.10 | 0.20 | 0.30 |
| Taxa de crossover | 0.80 | 0.85 | 0.90 |
| Elitismo | 2 | 2 | 3 |
| Torneio | 3 | 3 | 4 |
| Paciência (early stop) | 5 | 6 | 7 |

## Testes

```bash
python -m pytest tests/
```

- **`test_data.py`** — garante que treino/teste não têm índices em comum
  e que a amostragem estratificada preserva a proporção de classes.
- **`test_genetic.py`** — valida que crossover e mutação sempre produzem
  cromossomos válidos, e que o otimizador nunca piora o melhor fitness
  ao longo das gerações.
- **`test_integration.py`** — roda um experimento genético mínimo
  ponta a ponta (dados sintéticos, população/gerações reduzidas) e
  confirma que os artefatos finais (`resumo_final.json`, modelo
  `.joblib`, comparação com baseline) são gerados corretamente.

### Smoke test (sem precisar da base Vigitel)

```bash
python scripts/smoke_test.py
```

Roda uma otimização genética pequena e rápida sobre dados sintéticos
(`sklearn.datasets.make_classification`), útil para validar que o
pipeline do GA está íntegro sem precisar baixar/ter a base Vigitel.

### `scripts/run_experiments.py`

Alias de compatibilidade com versões anteriores do projeto — apenas
encaminha para `tech_challenge_fase2.py` na raiz. Prefira rodar o
script principal diretamente.