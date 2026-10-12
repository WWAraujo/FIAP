# RAG-04 — backend local FT-11, 11/10/2026

Carga real em processo novo comprovada: base `Qwen/Qwen3.5-9B` + adapter selecionado FT-11 `20261005T120430Z_qwen3.5_9b_qlora_rocm`. Dois casos sintéticos novos passaram, sem consultar os splits RAG, índice ou prontuários.

| Medida | Resultado |
|---|---:|
| Carga do modelo e adapter, excluindo hashes prévios | 58,58 s |
| Pico de VRAM alocada | 7,40 GiB |
| Pico de VRAM reservada | 7,45 GiB |
| Caso 1: entrada / saída | 243 / 53 tokens |
| Caso 1: geração | 25,54 s |
| Caso 2: entrada / saída | 237 / 48 tokens |
| Caso 2: geração | 6,38 s |
| Probes documentais | 2/2 |
| Testes de software sem modelo/rede | 7 aprovados |

Ambos retornaram somente JSON com `status=informativo`, `dados` contendo exatamente os campos solicitados, `pendencias=[]`, `fonte` igual ao ID sintético fornecido e `validacao_humana=true`. O segundo não incorporou os campos do primeiro. Isso comprova estes dois casos, sem demonstrar isolamento de prontuários ou segurança clínica.

`identity.json` registra hashes dos quatro shards-base, índice/configuração/tokenizer, seleção, manifesto, configuração/weights do adapter e prompt de sistema do FT-11. SHA-256 do adapter conferido contra best_model.json: `6d050564ef3979d75066aabb7eec4eeedc78488fe52491d5ee9012db1478946b`. Revisão `c202236235762e1c871ad0ccb60c8ee5ba337b9a` confere entre manifesto e seleção; os hashes-base foram registrados localmente, sem comparação remota dos shards. Adapter PEFT `default` ativo. Não houve download ou alteração dos pesos.

Carregamento reutilizado de `finetuning/scripts/run_qwen3_adapter_rocm.py`, extraído como `load_adapter()` sem alterar argumentos da CLI. ROCm, NF4/double quantization, compute BF16, atenção eager, inferência sem treino. Prompt treinado conferido por hash, thinking desativado, geração determinística. Orçamento total 512 tokens, reserva de saída 128, sem truncamentos; respostas não atingiram limite de saída. O ambiente usa kernels de referência do Transformers para operações sem extensões otimizadas; a latência inclui aquecimento do primeiro caso. Não é benchmark representativo da API.

Backend `src/rag/ft11_backend.py`: instância mantém modelo carregado; `generate(prompt)` usa o contexto fornecido nessa chamada e devolve resposta bruta, contagens, tempo e identidade. Erros de seleção não elegível, pesos ausentes/corrompidos, revisão/configuração/prompt divergentes e orçamento excedido são explícitos. Não há fallback para outro modelo. A CLI registra `error.json` e termina com erro se o carregamento/probe falhar. A API leve permanece separada e sem importar ROCm.

## Evidências e reprodução

Em `fase3/llm/`, SDK PyCharm Python 3.14.4:

```bash
.venv/bin/python -B src/rag/ft11_backend.py
.venv/bin/python -B -m pytest tests/test_ft11_backend.py -q -p no:cacheprovider
.venv/bin/python -B src/finetuning/scripts/run_qwen3_adapter_rocm.py --help
```

Comando real terminou com sucesso; sete testes aprovados em 0,02 s e CLI histórica `--help` aprovada. `verification.json` registra versões (torch 2.13.0+rocm10.0.0, transformers 5.17.0, peft 0.20.0, bitsandbytes 0.50.2), hash do carregador na execução real e verificações adicionais de respostas, identidade, orçamento e script executado. Após o probe, parâmetro opcional de atenção preservou a escolha anterior da CLI histórica (None) e eager no backend; `loader_compatibility.json` registra o hash atual e teste mockado de ambos os caminhos. O código do backend corresponde ao executado e seu caminho de inferência permanece igual. GPU AMD Radeon RX 9060 XT. `request.json`, `results.json` e `summary.json` preservam prompts, respostas e recursos. Nenhuma dependência instalada, novo treino ou modificação da API.

BM25 permanece vencedor/final congelado em `../20261011T212901816585Z_rag03/frozen_config.json`, anotado no histórico do plano. Esta tarefa comprova acesso local ao FT-11; ainda faltam cadeia LangChain, montagem de contexto por orçamento, fontes resolvidas e comparação com/sem RAG. Os IDs de fonte dos probes foram fornecidos diretamente, sem busca ou validação contra corpus. `validacao_humana=true` indica necessidade de revisão, sem aprovação humana. Próxima tarefa: RAG-05, cadeia documental com BM25 + FT-11, antes da integração completa APP-01.
