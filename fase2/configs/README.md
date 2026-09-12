# Configurações de Experimentos - Fase 2

Este diretório contém as configurações para os experimentos de otimização genética da Fase 2.

## Arquivos

### experimento_a.json
- **Objetivo:** Segurança Clínica (Baseline)
- **Foco:** Alto Recall para capturar máximo de doentes
- **Threshold:** 0.5
- **Class Weight:** balanced

### experimento_b.json
- **Objetivo:** Eficiência Médica (Alta Precisão)
- **Foco:** Eliminar alarmes falsos
- **Threshold:** 0.6
- **Class Weight:** null

### experimento_c.json
- **Objetivo:** Validação Científica (Impacto Puro)
- **Foco:** Isolar ganho das novas variáveis
- **Threshold:** 0.5
- **Class Weight:** balanced

## Uso

As configurações são carregadas automaticamente pelo programa principal:

```bash
cd fase2
python tech_challenge_fase2.py
```

O programa busca automaticamente os arquivos JSON neste diretório.

## Estrutura do JSON

Cada configuração contém:
- `nome`: Identificador do experimento
- `descricao`: Descrição do objetivo
- `algoritmo_genetico`: Parâmetros do algoritmo genético
  - `populacao_tamanho`: Tamanho da população
  - `geracoes`: Número de gerações
  - `taxa_crossover`: Taxa de crossover
  - `taxa_mutacao`: Taxa de mutação
  - `elitismo`: Número de indivíduos elitistas
- `modelo`: Parâmetros do modelo
  - `class_weight`: Peso das classes
  - `threshold`: Threshold de classificação
