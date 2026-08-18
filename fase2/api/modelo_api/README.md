# Modelo API - Artefatos (Fase 2)

Artefatos do modelo servido pela API de deploy da Fase 2: RandomForest com
hiperparâmetros otimizados por algoritmo genético (`experimento_a`), usando
as mesmas 20 variáveis de entrada do modelo original da Fase 1.

## Arquivos (todos versionados no git)

- **`modelo_genetico_vencedor.joblib`** (~70 MB) — modelo em produção,
  pronto para uso. Veio no `git clone`, nenhum passo manual necessário.
- `metadata_modelo_api.json` — hiperparâmetros, lista de variáveis de
  entrada, métricas (validação e teste) e o registro da poda pós-treino
  (chave `poda_pos_treino`)
- `exemplo_entrada_api.json` — exemplo de payload para `POST /prever`

## Poda do modelo

O vencedor original do algoritmo genético (`experimento_a`, ver
[resultados](../../../resultados)) tem 400 árvores e ~550 MB —
grande demais para o limite de 100 MB do GitHub, e lento para carregar
sob CPU limitada em produção (~10s por réplica, e pior ainda sob
compressão, que testamos e descartamos — decompressão é CPU-bound e
piora justamente sob o `cpus` limitado do `docker-compose.yml`).

Em vez de comprimir ou retreinar, o modelo foi **podado pós-treino**:
mantidas as primeiras 50 das 400 árvores já ajustadas. Isso é válido
porque as árvores de um Random Forest são estatisticamente
intercambiáveis — cada uma foi treinada em uma amostra bootstrap
independente, então um subconjunto delas já é representativo do
ensemble completo.

Resultado, medido no mesmo conjunto de teste (`base_teste_modelo_api.csv`
da Fase 1, 166.644 linhas) usado para as métricas oficiais:

| | 400 árvores (original) | 50 árvores (produção) |
|---|---|---|
| Tamanho | 549 MB | 70 MB |
| F1-score | 0.5836 | 0.5837 |
| ROC-AUC | 0.7655 | 0.7651 |
| Carregamento (cpus=1.0) | 9.83s | 2.72s |

Degradação de métrica desprezível (diferença de milésimos, dentro do
ruído estatístico), arquivo cabe folgado no limite do GitHub, e carrega
mais rápido em produção — inclusive mais rápido que o original mesmo
sem limite de CPU. Detalhes completos em `poda_pos_treino` dentro de
[metadata_modelo_api.json](./metadata_modelo_api.json).

> Se precisar reverter para o modelo completo de 400 árvores ou
> retreinar do zero, rode `fase2/tech_challenge_fase2.py` (requer a base
> Vigitel) e substitua este arquivo antes de rebuildar a imagem da API.

## Para rodar a API

Veja [../README.md](../README.md) — recomendado subir via `docker compose`
a partir de `fase2/api/`.
