# FIAP Tech Challenge

Projeto multi-fase para triagem de hipertensão a partir dos dados do VIGITEL
(Vigilância de Fatores de Risco e Proteção para Doenças Crônicas por
Inquérito Telefônico), evoluindo de um modelo de machine learning básico
(Fase 1) para um pipeline otimizado por algoritmo genético e com
interpretação das predições via LLM (Fase 2).

## Estrutura do Projeto

```
├── fase1/        Projeto de API FastAPI (Fase 1)
├── fase2/        Projeto de otimização genética (Fase 2)
├── shared/       Configurações e dados compartilhados
├── resultados/   Outputs dos experimentos
└── STRUCTURE.md  Documentação detalhada da estrutura
```

## Quick Start

Cada fase pode ser executada localmente (com pip) ou via Docker. Em ambos
os casos, a API fica disponível em **http://localhost:8000**.

### Fase 1: API

**Localmente:**
```bash
cd fase1
pip install -r requirements.txt
python src/techchallenge_fase1/api_modelo.py
```

**Via Docker:**
```bash
cd fase1
docker build -t hipertensao-api-fase1 .
docker run -d -p 8000:8000 hipertensao-api-fase1
```
Acesse http://localhost:8000

### Fase 2: Otimização + API com LLM

**Rodar a otimização genética (gera o modelo):**
```bash
cd fase2
pip install -r requirements.txt
python tech_challenge_fase2.py
```

**Subir a API do modelo otimizado (Docker, com autoscaling e monitoramento):**
```bash
cd fase2/api
docker compose up -d --build
```
Acesse http://localhost:8000 — sobe também nginx (load balancer),
autoscaler (baseado em CPU), Prometheus, cAdvisor e Grafana
(http://localhost:3000). Veja [fase2/api/README.md](fase2/api/README.md)
para o passo a passo completo, incluindo configuração da interpretação
via LLM (Google Gemini) e teste de escalabilidade automática.

## Fases do Projeto

| Fase | Localização | Objetivo | Stack |
|------|-------------|----------|-------|
| **1** | `./fase1/` | Treinar e servir via API um modelo RandomForest de triagem de hipertensão, a partir de 20 variáveis selecionadas da base VIGITEL | FastAPI, scikit-learn |
| **2** | `./fase2/` | Otimizar os hiperparâmetros do modelo da Fase 1 com algoritmo genético e servir a API com escalabilidade automática (Docker + nginx + autoscaler), monitoramento (Prometheus/Grafana) e interpretação das predições em linguagem natural via LLM (Google Gemini) | Algoritmo genético customizado, scikit-learn, FastAPI, Docker Compose |

## Documentação

- **[STRUCTURE.md](STRUCTURE.md)** - Estrutura completa, fluxos de dados e desenvolvimento
- **[fase1/README.md](fase1/README.md)** - Instruções específicas da API
- **[fase2/README.md](fase2/README.md)** - Instruções de otimização e testes

## Desenvolvimento

### Instalar dependências
```bash
# Fase 1
cd fase1 && pip install -r requirements.txt

# Fase 2
cd fase2 && pip install -r requirements.txt
```

### Executar testes (Fase 2)
```bash
cd fase2
python -m pytest tests/
python scripts/smoke_test.py
```

## Arquivos Compartilhados

- **Configurações:** `shared/configs/` (experimento_a.json, b, c)
- **Dados:** `shared/data/`
- **Scripts:** `shared/scripts/`

## Resultados

Os resultados dos experimentos são salvos em `resultados/[timestamp]/` com outputs de cada experimento.

---

Para mais detalhes sobre a estrutura, veja [STRUCTURE.md](STRUCTURE.md).