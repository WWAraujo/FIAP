# Modelo API - Artefatos Treinados

Este diretório contém os artefatos do modelo Random Forest treinado na Fase 1 do projeto FIAP Tech Challenge.

## Arquivos

### modelo_hipertensao_api.joblib
- **Descrição:** Modelo Random Forest treinado com 20 variáveis mais importantes
- **Algoritmo:** RandomForestClassifier
- **Tamanho:** ~57 MB
- **Variáveis:** 20 features selecionadas (diab, iddpapa, imc, excpeso, etc.)
- **Métricas:**
  - Accuracy: 0.6955
  - Precision: 0.5021
  - Recall: 0.6979
  - F1-Score: 0.584
  - ROC-AUC: 0.7672

### metadata_modelo_api.json
- **Descrição:** Metadados completos do modelo
- **Conteúdo:**
  - Hiperparâmetros do modelo
  - Lista de variáveis de entrada
  - Métricas de performance
  - Threshold de classificação (0.5)
  - Descrição do target

### exemplo_entrada_api.json
- **Descrição:** Exemplo de entrada para testes da API
- **Uso:** Referência para formatar requisições à API

### modelo_api_random_forest.py
- **Descrição:** Script de treinamento do modelo
- **Uso:** Referência para como o modelo foi treinado

## Como Usar

### Via API FastAPI
```bash
cd fase1
pip install -r requirements.txt
python src/techchallenge_fase1/api_modelo.py
```

A API ficará disponível em `http://localhost:8000`

### Via Python (direto)
```python
import joblib
import pandas as pd

# Carregar modelo
modelo = joblib.load('fase1/modelo_api/modelo_hipertensao_api.joblib')

# Carregar metadados
import json
with open('fase1/modelo_api/metadata_modelo_api.json', 'r') as f:
    metadata = json.load(f)

# Fazer predição
# ... seu código de predição
```

## Retreinamento

Para treinar novamente o modelo, use o notebook:
```bash
cd fase1/src/techchallenge_fase1
jupyter notebook tech_challenge.ipynb
```

Siga as instruções no notebook para executar os blocos de treinamento.

## Notas

- Este modelo é o baseline da Fase 1
- Para otimização genética, consulte a Fase 2
- O modelo usa 20 variáveis selecionadas por feature importance
- Threshold de classificação: 0.5
- Tratamento de data leakage aplicado durante treinamento
