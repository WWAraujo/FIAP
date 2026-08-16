# FIAP Tech Challenge

Projeto multi-fase para triagem de hipertensão com machine learning e otimização genética.

## Estrutura do Projeto

```
├── fase1/        Projeto de API FastAPI (Fase 1)
├── fase2/        Projeto de otimização genética (Fase 2)
├── shared/       Configurações e dados compartilhados
├── resultados/   Outputs dos experimentos
└── STRUCTURE.md  Documentação detalhada da estrutura
```

## Quick Start

### Fase 1: API
```bash
cd fase1
pip install -r requirements.txt
python src/techchallenge_fase1/api_modelo.py
# Acesse http://localhost:8000
```

### Fase 2: Otimização
```bash
cd fase2
pip install -r requirements.txt
python tech_challenge_fase2.py
```

## Fases do Projeto

| Fase | Localização | Objetivo | Stack |
|------|-------------|----------|-------|
| **1** | `./fase1/` | API REST para triagem de hipertensão | FastAPI, scikit-learn |
| **2** | `./fase2/` | Otimizar modelo com algoritmo genético | Genético customizado, scikit-learn |

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
