# AGENTS.md - Guia para Desenvolvimento com Devin

## Visão Geral do Projeto

Projeto FIAP Tech Challenge para triagem de hipertensão com machine learning e otimização genética. Organizado em fases independentes com recursos compartilhados.

**Contexto:** Este é um projeto da Pós-Tech em IA para Desenvolvedores da FIAP (Faculdade de Informática e Administração Paulista).

## Estrutura de Diretórios

```
FIAP/
├── fase1/                    # API de Triagem (baseline)
│   ├── src/techchallenge_fase1/
│   │   ├── api_modelo.py     # API FastAPI principal
│   │   └── formulario.html   # Interface web
│   ├── modelo_api/           # Modelo treinado e metadados
│   ├── data/                 # Dados de treinamento (vigitel.csv, dicionário, variáveis)
│   ├── requirements.txt
│   └── Dockerfile
│
├── fase2/                    # Otimização Genética + API escalável
│   ├── tech_challenge_fase2.py  # Programa principal
│   ├── src/
│   │   ├── genetic/          # Algoritmo genético (chromosome, fitness, optimizer)
│   │   ├── data.py           # Processamento de dados
│   │   ├── experiment.py     # Configuração de experimentos
│   │   ├── modeling.py       # Modelagem ML
│   │   └── reporting.py      # Geração de relatórios
│   ├── configs/              # Configurações de experimentos (a, b, c)
│   ├── data/                 # Dados de treinamento da fase2
│   ├── tests/                # Testes unitários
│   ├── scripts/              # Scripts executáveis
│   ├── api/                  # API v2.0.0 escalável
│   │   ├── src/app/api_modelo.py
│   │   ├── docker-compose.yml
│   │   ├── nginx/            # Load balancer
│   │   ├── autoscaler/       # Autoscaler CPU-based
│   │   └── monitoring/       # Prometheus + Grafana
│   └── requirements.txt
│
├── resultados/               # Outputs dos experimentos
└── fase3/                    # (vazio - fase futura)
```

## Como Executar

### Fase 1: API Baseline
```bash
cd fase1
pip install -r requirements.txt
python src/techchallenge_fase1/api_modelo.py
# Acesse http://localhost:8000
```

### Fase 2: Otimização Genética
```bash
cd fase2
pip install -r requirements.txt
python tech_challenge_fase2.py
```

### Fase 2: Testes
```bash
cd fase2
python -m pytest tests/
python scripts/smoke_test.py
```

### Fase 2: API Escalável (Docker Compose)
```bash
cd fase2/api
docker compose up -d --build
# API: http://localhost:8000
# Grafana: http://localhost:3000
```

## Dependências Principais

### Fase 1
- fastapi==0.111.0
- uvicorn==0.30.1
- scikit-learn==1.5.0
- pandas
- joblib

### Fase 2
- scikit-learn>=1.5,<2
- pandas>=2.2,<3
- numpy>=1.26,<3
- matplotlib>=3.8,<4
- joblib>=1.4,<2

### Fase 2 API
- fastapi==0.111.0
- uvicorn==0.30.1
- scikit-learn==1.5.0
- prometheus-client==0.20.0

## Fluxo de Dados

1. Fase 1: Dados de treinamento em `fase1/data/` (vigitel.csv, dicionário, variáveis)
2. Fase 1: Modelo Random Forest treinado salvo em `fase1/modelo_api/modelo_hipertensao_api.joblib`
3. Fase 2: Lê metadados do modelo da Fase 1 + configurações de `fase2/configs/`
4. Fase 2: Executa 3 configurações do algoritmo genético
5. Fase 2: Gera modelo_genetico_vencedor.joblib
6. Fase 2 API: Copia modelo vencedor para deploy escalável

## Comandos Úteis

### Ver estrutura do projeto
```powershell
Get-ChildItem -Recurse -Depth 2
```

### Rodar testes específicos
```bash
cd fase2
python -m pytest tests/test_genetic.py
python -m pytest tests/test_data.py
python -m pytest tests/test_integration.py
```

### Ver resultados dos experimentos
```powershell
Get-ChildItem fase2/resultados
```

## Padrões e Convenções

- Python como linguagem principal
- FastAPI para APIs
- scikit-learn para ML
- Estrutura modular com separação de responsabilidades
- Configurações externalizadas em JSON
- Testes unitários em Fase 2
- Docker para deploy da API escalável
- Logging estruturado e monitoramento (Prometheus/Grafana)

## Arquivos Importantes

- `fase1/modelo_api/metadata_modelo_api.json` - Metadados do modelo baseline
- `fase2/api/modelo_api/metadata_modelo_api.json` - Metadados do modelo otimizado
- `fase2/configs/experimento_*.json` - Configurações do algoritmo genético
- `fase2/api/ARCHITECTURE.md` - Arquitetura da API escalável
- `STRUCTURE.md` - Documentação detalhada da estrutura

## Ambientes Virtuais

- `.venv/` já existe na raiz do projeto
- Cada fase tem seu próprio requirements.txt
- Use `pip install -r requirements.txt` em cada fase conforme necessário

## Observações

- Fase 1 é o baseline histórico e permanece inalterada
- Fase 2 é independente e gera modelo otimizado
- Fase 2 API é um deploy escalável do modelo vencedor
- Cada fase tem seus próprios dados e configurações
- fase3/ existe mas está vazio (planejamento futuro)
