# Modelo API - Artefatos

Este diretório contém os artefatos do modelo Random Forest treinado na Fase 1.

## Arquivos Necessários

### ⚠️ Arquivos Grandes (não versionados no git)

Os seguintes arquivos precisam ser baixados ou gerados localmente:

1. **`modelo_hipertensao_api.joblib`** (~100+ MB)
   - Modelo Random Forest treinado
   - Necessário para rodar a API

2. **`dicionario-vigitel-2006-2024.xlsx`** (~50+ MB)
   - Dicionário de variáveis do Vigitel
   - Usado como referência

### ✅ Arquivos Versionados no Git

- `metadata_modelo_api.json` - Metadados do modelo
- `exemplo_entrada_api.json` - Exemplo de entrada para testes
- `base_teste_modelo_api.csv` - Base de testes

## Como Obter os Arquivos Grandes

### Opção 1: Download do Google Drive/OneDrive
Solicitar ao grupo responsável os links para download:

```bash
# Fazer download dos arquivos
# modelo_hipertensao_api.joblib -> fase1/modelo_api/
# dicionario-vigitel-2006-2024.xlsx -> fase1/modelo_api/
```

### Opção 2: Retraining
Se o notebook estiver disponível, reexecutar:

```bash
cd fase1
jupyter notebook src/techchallenge_fase1/tech_challenge.ipynb
```

Os modelos treinados serão salvos em `modelo_api/`.

## Estrutura de Arquivos

```
modelo_api/
├── README.md (este arquivo)
├── metadata_modelo_api.json ✅
├── exemplo_entrada_api.json ✅
├── base_teste_modelo_api.csv ✅
├── modelo_api_random_forest.py
├── modelo_hipertensao_api.joblib ⚠️ (não versionado)
└── dicionario-vigitel-2006-2024.xlsx ⚠️ (não versionado)
```

## Para Rodar a API

Certifique-se de que `modelo_hipertensao_api.joblib` está presente:

```bash
cd fase1
pip install -r requirements.txt
python src/techchallenge_fase1/api_modelo.py
```

A API ficará disponível em `http://localhost:8000`.
