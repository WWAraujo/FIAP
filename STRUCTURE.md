# Estrutura do Projeto FIAP Tech Challenge

## Visão Geral

O repositório foi reorganizado para separar os projetos de **Fase 1** e **Fase 2**, enquanto mantém recursos compartilhados centralizados.

```
FIAP/
├── fase1/                          # Projeto Fase 1: API de Triagem
│   ├── src/techchallenge_fase1/   # Código-fonte da API
│   ├── modelo_api/                # Artefatos do modelo treinado
│   ├── Dockerfile                 # Docker configuration
│   ├── requirements.txt           # Dependências específicas
│   └── README.md                  # Instruções de Fase 1
│
├── fase2/                          # Projeto Fase 2: Otimização Genética
│   ├── src/techchallenge_fase2/   # Código-fonte da otimização
│   ├── tests/                     # Testes unitários
│   ├── scripts/                   # Scripts executáveis
│   │   ├── run_experiments.py    # Encaminhador para programa principal
│   │   └── smoke_test.py         # Testes de fumaça
│   ├── tech_challenge_fase2.py   # Programa principal
│   ├── pyproject.toml            # Configuração setuptools
│   ├── requirements.txt          # Dependências
│   └── README.md                 # Instruções de Fase 2
│
├── shared/                        # Recursos compartilhados
│   ├── configs/                  # Configurações de experimentos
│   │   ├── experimento_a.json
│   │   ├── experimento_b.json
│   │   └── experimento_c.json
│   ├── scripts/                  # Scripts utilitários compartilhados
│   └── data/                     # Datasets compartilhados
│
├── resultados/                    # Outputs dos experimentos
│   └── [timestamp]/              # Resultados por data/hora
│
└── pyproject.toml                # Configuração raiz (desenvolvimento)
```

## Fase 1: API de Triagem de Hipertensão

Localização: `./fase1/`

**Objetivo:** Servir um modelo Random Forest pré-treinado via API FastAPI com interface HTML.

**Stack:**
- FastAPI + Uvicorn
- scikit-learn (modelo)
- pandas + joblib

**Como usar:**
```bash
cd fase1
pip install -r requirements.txt
python src/techchallenge_fase1/api_modelo.py
```

**Arquivos principais:**
- `src/techchallenge_fase1/api_modelo.py` - Aplicação principal
- `modelo_api/modelo_hipertensao_api.joblib` - Modelo treinado
- `formulario.html` - Interface web

## Fase 2: Otimização Genética

Localização: `./fase2/`

**Objetivo:** Otimizar o modelo da Fase 1 usando algoritmo genético para encontrar o melhor subconjunto de variáveis.

**Stack:**
- Algoritmo genético customizado
- scikit-learn (para comparações)
- pandas + numpy (processamento)
- matplotlib (visualizações)

**Como usar:**
```bash
cd fase2
pip install -r requirements.txt
python tech_challenge_fase2.py
```

**Fluxo de execução:**
1. Carrega metadados e variáveis do modelo da Fase 1
2. Carrega dataset Vigitel e separa em treino/teste
3. Executa 3 configurações do algoritmo genético
4. Revalida vencedores no conjunto de treino
5. Avalia melhor modelo no teste isolado
6. Compara com baseline e salva artefatos

**Testes:**
```bash
python -m pytest tests/
python scripts/smoke_test.py  # Teste de fumaça rápido
```

## Recursos Compartilhados

Localização: `./shared/`

### Configurações (`shared/configs/`)
Arquivo YAML/JSON com parâmetros para experimentos:
- `experimento_a.json` - Config 1 do algoritmo genético
- `experimento_b.json` - Config 2
- `experimento_c.json` - Config 3

### Resultados (`./resultados/`)
Saídas dos experimentos organizadas por timestamp:
```
resultados/
└── 20260810_104008/
    ├── experimento_a/configuracao.json
    ├── experimento_b/...
    └── experimento_c/...
```

## Dependências

### Fase 1 (`fase1/requirements.txt`)
```
fastapi==0.111.0
uvicorn==0.30.1
joblib
pandas
scikit-learn==1.5.0
```

### Fase 2 (`fase2/requirements.txt`)
```
joblib>=1.4,<2
matplotlib>=3.8,<4
numpy>=1.26,<3
pandas>=2.2,<3
scikit-learn>=1.5,<2
```

## Desenvolvimento

Para trabalhar no projeto como um todo:

```bash
# Instalar ambas as fases
cd fase1 && pip install -r requirements.txt
cd ../fase2 && pip install -r requirements.txt

# Ou usar pyproject.toml em cada uma
cd fase1 && pip install -e .
cd fase2 && pip install -e .
```

## Fluxo de Dados

```
fase1/modelo_api/
    ├── dicionario-vigitel-2006-2024.xlsx
    ├── metadata_modelo_api.json        ─────┐
    ├── modelo_api_random_forest.py           │
    └── modelo_hipertensao_api.joblib        │
                                              │ Fase 2 lê metadados e carrega modelo
                                              │
shared/configs/                              │
    ├── experimento_a.json ────────────────┐ │
    ├── experimento_b.json                 │ │
    └── experimento_c.json  ──────────────┼─┴─> fase2/tech_challenge_fase2.py
                                           │         │
shared/data/                               │         │
    └── vigitel-2024.csv ───────────────────────> Processamento e otimização
                                                      │
                                                      v
                                                resultados/[timestamp]/
```

## Próximos Passos

- [ ] Adicionar testes para Fase 1 em `fase1/tests/`
- [ ] Documentação de API (FastAPI docs, Swagger)
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Containerização completa (docker-compose)
- [ ] Notebook de análise em `shared/notebooks/`
