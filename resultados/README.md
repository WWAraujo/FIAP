# Resultados dos Experimentos

Este diretório contém os outputs dos experimentos executados, especialmente da Fase 2 (Otimização Genética).

## Estrutura

```
resultados/
├── README.md (este arquivo)
├── .gitkeep (para manter o diretório versionado)
└── [timestamp]/
    ├── comparacao_baseline_genetico.csv
    ├── comparacao_vencedores_validacao.csv
    ├── modelo_genetico_vencedor.joblib ⚠️ (arquivo grande, não versionado)
    ├── resumo_final.json
    ├── experimento_a/
    │   ├── configuracao.json
    │   ├── convergencia_fitness.png
    │   ├── historico_geracoes.csv
    │   ├── individuos_avaliados.csv
    │   └── melhor_individuo.json
    ├── experimento_b/
    │   └── ...
    └── experimento_c/
        └── ...
```

## Arquivos Gerados

### Arquivos de Configuração
- `configuracao.json` - Parâmetros usados no experimento

### Arquivos de Resultados
- `melhor_individuo.json` - Cromossomo do melhor indivíduo encontrado
- `resumo_final.json` - Resumo agregado de todos os experimentos
- `modelo_genetico_vencedor.joblib` - ⚠️ Modelo treinado (não versionado)

### Históricas e Dados
- `historico_geracoes.csv` - Evolução por geração
- `individuos_avaliados.csv` - Todos os indivíduos avaliados
- `convergencia_fitness.png` - Gráfico de convergência

### Comparativas
- `comparacao_baseline_genetico.csv` - Comparação Fase 1 vs Fase 2
- `comparacao_vencedores_validacao.csv` - Validação dos vencedores

## Nota sobre Arquivos Grandes

O arquivo `modelo_genetico_vencedor.joblib` não é versionado no git por ser muito grande (>100MB).

Se precisar recuperar:
1. Reexecute a Fase 2
2. Ou faça download manualmente se estiver em um repositório remoto com Git LFS

## Como Gerar Novos Resultados

```bash
cd fase2
python tech_challenge_fase2.py

# Os resultados serão salvos em:
# resultados/[ano][mês][dia]_[hora][minuto][segundo]/
```

## Limpeza

Para remover resultados antigos (mantendo a estrutura):

```bash
# Remover apenas um diretório específico
rm -rf resultados/20260810_104008/

# Limpar todos os resultados (mas manter .gitkeep)
find resultados -mindepth 1 -not -name '.gitkeep' -delete
```

O arquivo `.gitkeep` garante que o diretório seja mantido versionado mesmo vazio.
