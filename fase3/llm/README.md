# LLM — fine-tuning e RAG

O modelo selecionado é **FT-11**, com base `Qwen/Qwen3.5-9B` na revisão `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Caminhos, hash dos pesos e decisão estão em [artifacts/best_model.json](artifacts/best_model.json).

```text
llm/
├── src/
│   ├── finetuning/          # preparação, treino e avaliação
│   │   ├── data/
│   │   ├── training/
│   │   ├── evaluation/
│   │   └── scripts/
│   └── rag/                 # reservado; implementação ainda pendente
├── data/                    # corpus atual e entradas de avaliações históricas
├── models/base/Qwen3.5-9B/  # base necessária para carregar os adapters
├── artifacts/               # pesos finais, manifestos e resultados comparativos
└── tests/
```

## Arquivos preservados

- FT-11: `artifacts/adapters/20261005T120430Z_qwen3.5_9b_qlora_rocm/` e corpus `data/factual_v2/` (1.408/160/160 exemplos de treino/validação/teste).
- FT-10: `artifacts/adapters/20261005T101013Z_qwen3.5_9b_qlora_rocm/` e corpus `data/contextual_v1/` (960/120/120). Seus pesos são a origem do treino FT-11, necessários para reproduzi-lo.
- Respostas, métricas, revisões e manifestos dos modelos inferiores em `artifacts/evaluations/`, `artifacts/experiments/` e `artifacts/adapters/`; entradas reservadas, prompts e proveniência histórica também permanecem.
- `data/raw/` e `data/curated/`: fonte sintética e dados consumidos pelo protótipo da API. Ambiente `.venv` e dependências existentes mantidos.

A limpeza de 10/10/2026 enviou à lixeira os pesos rejeitados FT-12/FT-13, checkpoints dispensáveis, conjuntos antigos de treino e caches: **212,4 MiB retirados da árvore ativa**. Lista exata em [artifacts/cleanup_20261010.json](artifacts/cleanup_20261010.json). Os conjuntos históricos sem `train.jsonl` são evidências de avaliação, não bases completas para repetir os treinos rejeitados. Os comandos registrados em experimentos anteriores conservam os caminhos da época.

## Preparar um clone para treinamento

O caminho validado é Linux, Python 3.14.4 e GPU AMD Radeon RX 9060 XT de 16 GiB com ROCm. `requirements.txt` fixa o ambiente ROCm/gfx1200; estes comandos não configuram CUDA ou treinamento em CPU. Instale o driver/ROCm compatível no host antes do pré-check. Reserve pelo menos 30 GB livres para a base (cerca de 19 GB), adapters, checkpoints e temporários, além do ambiente Python.

Partindo da raiz do repositório:

```bash
cd fase3/llm
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B src/finetuning/scripts/check_rocm_before_training.py
.venv/bin/python -B -m pytest tests -q -p no:cacheprovider
```

No PyCharm, selecione `.venv/bin/python` como SDK deste módulo e consulte o ambiente antes dos comandos. A API permanece em ambiente separado.

Os corpora `data/contextual_v1/` e `data/factual_v2/` são pequenos e estão no Git, incluindo treino, validação, teste, desafios, prompts e manifestos. Não precisam ser baixados nem recriados. Os manifestos registram contagens, seeds, hashes e fontes. `contextual_v1` é sintético; `factual_v2` mistura exemplos sintéticos com trechos públicos traduzidos e curados, sem prontuários reais.

A parte externa de `factual_v2` deriva de [MedQuAD](https://github.com/abachaa/MedQuAD), por Asma Ben Abacha e Dina Demner-Fushman, revisão `577bd37b96c02d1833b2c9eed2de9f96964e96cb`, sob [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). A tradução usa [recogna-nlp/drbode_dataset](https://huggingface.co/datasets/recogna-nlp/drbode_dataset), revisão `5ce2ec87299adef39602d664209f79ed5914881a`, cuja licença declarada é MIT. Os trechos foram selecionados, traduzidos e modificados; URLs, hashes, transformações e limites estão em `data/factual_v2/data/splits/manifest.json` e na proveniência de cada exemplo. Preserve essa atribuição ao redistribuir.

## Baixar os arquivos grandes

A base pública é [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), licença Apache-2.0. Baixe a revisão utilizada no treino com a [CLI oficial Hugging Face](https://huggingface.co/docs/huggingface_hub/guides/cli), instalada pelas dependências:

```bash
.venv/bin/hf download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --local-dir models/base/Qwen3.5-9B
```

O treinador também baixa essa revisão quando o diretório do modelo não existe. Um diretório parcialmente baixado não é validado automaticamente; conclua o comando acima antes do treino. O download completo traz pesos, configuração, tokenizer e licença; mantenha esses arquivos juntos.

Os adapters FT-10/FT-11 produzidos neste projeto **não têm URL pública de download registrada**. Seus pesos ficam apenas localmente. Para obtê-los em outro computador, execute os dois treinos abaixo ou copie os diretórios completos de uma execução existente, com manifestos e configurações. Baixar a base Qwen não fornece os adapters do projeto. `best_model.json` e os registros históricos identificam a execução original e contêm caminhos do computador onde ela ocorreu.

## Treinar FT-10 e depois FT-11

Os comandos abaixo iniciam novos treinamentos com IDs próprios e não substituem os pesos históricos. Antes de cada treino, execute o respectivo comando acrescentando `--validate-only` para conferir forward/backward e memória. O treinador completo salva os pesos e faz recarga técnica ao terminar; validação técnica não equivale à aprovação das respostas.

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_BLAS_PREFER_HIPBLASLT=0

# FT-10: uma época a partir da base, 960 exemplos.
.venv/bin/python -B src/finetuning/scripts/train_qwen3_8b_rocm.py \
  --project-dir data/contextual_v1 \
  --model-id Qwen/Qwen3.5-9B --model-dir models/base/Qwen3.5-9B \
  --output-dir artifacts/adapters \
  --system-prompt-file data/contextual_v1/system_prompt.txt \
  --frozen-embeddings-bf16 --max-length 384 \
  --epochs 1 --learning-rate 2e-5 --seed 42
```

Ao concluir, o treinador imprime `Treino concluído. Adapter: <caminho>`. Use esse caminho, com o novo ID, no comando seguinte; não presuma que os IDs históricos existirão em um clone.

```bash
# Substitua pelo caminho impresso ao concluir o seu FT-10.
FT10_ADAPTER="artifacts/adapters/<run-id-do-seu-ft10>/adapter"

# FT-11: uma época de continuação, 1.408 exemplos.
.venv/bin/python -B src/finetuning/scripts/train_qwen3_8b_rocm.py \
  --project-dir data/factual_v2 \
  --model-id Qwen/Qwen3.5-9B --model-dir models/base/Qwen3.5-9B \
  --output-dir artifacts/adapters --initial-adapter "$FT10_ADAPTER" \
  --system-prompt-file data/factual_v2/system_prompt.txt \
  --frozen-embeddings-bf16 --max-length 512 \
  --epochs 1 --learning-rate 1e-5 --seed 42
```

Os demais parâmetros fixados no treinador são QLoRA NF4/BF16, LoRA r=8/alpha=32/dropout=0,1, batch 1, acumulação 8, AdamW e checkpointing. Treino usa `train.jsonl`; validação de perda usa `validation.jsonl`. O teste é carregado para conferir separação entre conjuntos, mas não entra no otimizador. Reproduzir dados e parâmetros não garante pesos ou métricas idênticos em outro ambiente.

Para avaliar as respostas de sua nova execução, substitua o caminho abaixo pelo FT-11 gerado:

```bash
FT11_ADAPTER="artifacts/adapters/<run-id-do-seu-ft11>/adapter"
.venv/bin/python -B src/finetuning/scripts/evaluate_qwen3_base_vs_adapter_rocm.py \
  --adapter-dir "$FT11_ADAPTER" --model-dir models/base/Qwen3.5-9B \
  --test-file data/factual_v2/data/splits/validation.jsonl \
  --system-prompt-file data/factual_v2/system_prompt.txt \
  --output-dir artifacts/evaluations --limit 99999 --max-new-tokens 192
```

Revise a validação e registre a seleção antes de avaliar o teste. `score_contextual_results.py` e `score_semantic_review.py` separam contrato/conteúdo contextual de adequação factual. Os conjuntos históricos já foram observados e não constituem uma nova prova independente de generalização.

No Git ficam código, dependências, corpora atuais, manifestos/configurações FT-10/FT-11 e evidências finais JSON/JSONL/Markdown de `experiments/`. Pesos, ambientes, caches, logs operacionais, scripts históricos de execução, `data_audits/` e `evaluations/` ficam locais e ignorados.

## Resultado e próximo passo

FT-11 teve 100/112 acertos documentais/contextuais no teste contra 29/112 da base; nas 48 perguntas factuais, 22 adequadas e 11 inadequadas contra 20 e 21. Na prova adicional, houve regressão de 6 para 5 adequadas em 12 perguntas factuais: generalização parcial, sem validação clínica. A [revisão FT-11](artifacts/experiments/20261005T120254Z_factual_v2/review.md) contém a decisão e as limitações; FT-12 e FT-13 não substituíram o modelo selecionado.

O RAG permanece planejado em [PLANO_RAG.md](../docs/PLANO_RAG.md). `src/rag/` ainda não contém recuperação ou integração; o runtime da API continua separado em `../src/medical_assistant/retrieval/`.
