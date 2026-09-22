# Treinamento

Este pacote concentrará o fine-tuning real com PEFT/LoRA. Hoje ele contém somente `config.py`; não existe executor de treino, checkpoint ou adapter treinado.

Entradas planejadas: modelo-base identificável e dados em `data/splits/`.

Saídas planejadas: `artifacts/adapters/<run_id>/` com pesos do adapter, configuração, métricas e manifesto. A implementação é a tarefa **FT-01** do [plano](../../../docs/PLANO_EXECUCAO.md).
