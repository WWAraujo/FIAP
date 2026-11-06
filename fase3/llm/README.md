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
│   └── rag/                 # corpus, BM25, cadeia documental e avaliações
├── data/                    # corpus atual e entradas de avaliações históricas
├── models/base/Qwen3.5-9B/  # base necessária para carregar os adapters
├── artifacts/               # pesos finais, manifestos e resumos históricos
└── tests/
```

## Integração atual (16/10/2026)

O runtime `src/rag/runtime_server.py` atende a aplicação com a cadeia documental v3 e a nova cadeia factual `patient_facts_chain.py`, que usa registros atuais de medicamentos/exames/consultas/idade/sinais com fontes SQLite. O SDK API executa LangGraph e revisão persistente; o SDK LLM continua com as versões congeladas pelo recibo. Não instalar LangGraph ou atualizar LangChain nesse SDK para satisfazer a UI: os ambientes são separados. [Fluxo completo e validação](../docs/RELATORIO_TECNICO.md).

## Arquivos preservados

Avaliação de ferramentas do FT-11 em 15/10/2026: seleção nativa correta nas 18 perguntas operacionais, com três falhas no marcador de abstenção; **21/24 contratos/seleções corretos**. Estratégia JSON: **20/24 corretos**, incluindo falhas sintáticas e omissão em pedido misto. Nenhuma condição atingiu os critérios prévios para integração automática. Os relatórios brutos por execução foram removidos da cópia local durante a limpeza de `artifacts/experiments`; esta avaliação não executou ferramentas nem alterou o chat.

Para uma nova execução, no SDK LLM e com GPU livre, partindo da raiz: `fase3/llm/.venv/bin/python -B fase3/llm/src/rag/evaluate_tool_selection.py`. Cada execução cria artefatos novos; os casos já observados não constituem holdout independente. A correção de parser da execução original foi validada por nove testes em `llm/tests/test_tool_selection_eval.py`.

- FT-11: `artifacts/adapters/20261005T120430Z_qwen3.5_9b_qlora_rocm/` e corpus `data/factual_v2/` (1.408/160/160 exemplos de treino/validação/teste).
- FT-10: corpus histórico `data/contextual_v1/` (960/120/120), usado como origem do treino FT-11. Seus pesos foram removidos localmente nesta limpeza; o FT-11 selecionado continua disponível, mas a continuação original a partir do FT-10 não pode ser reproduzida com os artefatos locais restantes.
- O resumo dos treinamentos e resultados preservados está em [artifacts/training_history.md](artifacts/training_history.md). As pastas antigas de `artifacts/experiments/` e os adapters diferentes de FT-11 foram removidos localmente a pedido do usuário; logs detalhados dessas execuções não estão mais disponíveis nesta cópia.
- Avaliações e auditorias históricas foram condensadas em [artifacts/evaluation_audit_summary.md](artifacts/evaluation_audit_summary.md); os resultados brutos foram removidos. `artifacts/evaluations/` fica vazio como destino padrão das próximas avaliações, e `artifacts/data_audits/` foi removido.
- `data/raw/` e `data/curated/`: fonte sintética e dados consumidos pelo protótipo da API. Ambiente `.venv` e dependências existentes mantidos.

A limpeza de 10/10/2026 enviou à lixeira os pesos rejeitados FT-12/FT-13, checkpoints dispensáveis, conjuntos antigos de treino e caches: **212,4 MiB retirados da árvore ativa**. Lista exata em [artifacts/cleanup_20261010.json](artifacts/cleanup_20261010.json). Os conjuntos históricos sem `train.jsonl` são evidências de avaliação, não bases completas para repetir os treinos rejeitados. A tabela de treinamento registra parâmetros e métricas consolidadas disponíveis antes da limpeza dos relatórios brutos.

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

O Git contém código de execução/treino, dependências e corpora FT-10/FT-11 permitidos; os resumos históricos locais ficam fora do Git junto com pesos, ambientes, caches, logs operacionais, scripts históricos e novas saídas em `evaluations/`. Execuções novas não entram automaticamente; a seleção está no `.gitignore` e em `artifacts/versioning_policy.json`. A pasta `evaluations/` é mantida vazia como destino padrão do script; `data_audits/` foi removida após o resumo. A [organização dos artefatos](artifacts/README.md) identifica os resultados finais e o que permanece local.

## Resultado e próximo passo

FT-11 teve 100/112 acertos documentais/contextuais no teste contra 29/112 da base; nas 48 perguntas factuais, 22 adequadas e 11 inadequadas contra 20 e 21. Na prova adicional, houve regressão de 6 para 5 adequadas em 12 perguntas factuais: generalização parcial, sem validação clínica. A [tabela de histórico](artifacts/training_history.md) resume a decisão e as limitações; FT-12 e FT-13 não substituíram o modelo selecionado.

RAG-01/RAG-02 concluíram inventário e corpus `data/rag_pilot_v2/`, com 40 perguntas de validação/40 de teste e três portarias oficiais admitidas somente na primeira página dos PDFs. RAG-03 comparou BM25/E5-CPU na validação: suporte efetivo top-3 24/24 e 23/24; BM25 congelado para este piloto, 14 testes aprovados. Anexos clínicos fora do corpus; PDFs/extrações locais com hashes e download documentado. Comandos no [README RAG](src/rag/README.md); sequência no [PLANO_RAG.md](../docs/PLANO_RAG.md). `src/rag/` contém corpus, recuperação, backend FT-11 e cadeia documental local; o runtime da API continua separado em `../src/medical_assistant/retrieval/`.

RAG-04 comprovou o backend FT-11 e RAG-05 a cadeia LCEL local. Após a falha da RAG-06, RAG-07 preservou a v1 e comparou `src/rag/document_chain_v2.py`: nos 160 registros de validação, base/FT-11 sem RAG 0/24 adequadas, base+RAG 20/24 e uma parcial, FT-11+RAG 21/24. Zero sem RAG é limitação do modo extrativo com fonte fornecida. FT-11+RAG acertou 8/8 ausências, 4/4 conflitos e 4/4 ataques; evidência entregue 24/24, nenhuma omissão/truncamento, pico VRAM 7,71 GiB. Três abstenções indevidas no FT-11 e falhas especiais no base foram registradas. Os relatórios brutos foram removidos localmente na limpeza de `artifacts/experiments`.

No histórico RAG-08, uma pergunta de diabetes citou dislipidemia e três ataques geraram abstenção indevida, apesar de regras disponíveis. O relatório bruto foi removido localmente na limpeza; o resumo permanece aqui.

**RAG-09 concluída em 12/10/2026:** cadeia v3, BM25 sobre 15 seções integrais, seleção FT-11 e verificador comum na mesma base quantizada sem adapter. Escopo de fontes, JSON restrito, resolução literal/offsets e cobertura de todo texto útil, incluindo última linha. Validação final: ambos com RAG 22/24 completas, duas parciais e todos os 16 casos especiais corretos. No teste, FT-11+RAG teve 19/24 completas, duas parciais e três abstenções indevidas; base+RAG teve 18/24 completas, três parciais e as mesmas abstenções. Ambos acertaram 8/8 ausências, 4/4 conflitos, 4/4 instruções hostis e entregaram 24/24 spans. 42 testes de software foram aprovados. Os relatórios brutos RAG-09 foram removidos localmente; `data/rag_index_v3_final/readiness.json` mantém o vínculo de integridade dos módulos, índice, pacotes e pesos. [Comando de consulta e limites](src/rag/README.md).

Preparação documental acadêmica aceita; o verificador ainda falha em algumas formulações. Holdouts são paráfrases/cobertura de regressão do mesmo corpus/conceitos/famílias, sem independência temática ou clínica. Falhas e métricas consolidadas permanecem descritas neste README; arquivos brutos de `experiments/` foram removidos localmente. Sem treino novo ou mudança de pesos/seleção. APP-00 concluiu a política de seleção na API Ollama, com 21 testes mockados; [comandos e limites](../src/medical_assistant/application/README.md). APP-01, APP-02 e SAFE-01 estão implementadas no fluxo atual, com 82 testes API/UI e 124 testes LLM aprovados, mais seis verificações de integração real; `validacao_humana: true` indica necessidade de revisão, não aprovação efetiva.

## FT-14 — continuação com MedPT

O snapshot `AKCIT/MedPT` está fixado na revisão `24aeac298da0f1b46e6e7360eeeb0d36e711779d`. O corpus completo preparado contém 41.828 pares MedPT após filtros: 34.449 treino, 3.295 validação e 4.084 teste, mais 384 exemplos sintéticos de replay no treino. Os 384.095 pares originais não são todos usados; há exclusões registradas no manifesto do corpus. Não há revisão clínica nem garantia de qualidade factual.

A execução `artifacts/experiments/20261020T_ft14_medpt_full_restart/` concluiu com código zero em 21/10/2026. O adapter candidato está em `training/20261020T102347Z_qwen3.5_9b_qlora_rocm/adapter`; perdeu uma correspondência exata para o FT-11 na comparação técnica de 46 casos e aguarda revisão semântica. Não promove o candidato automaticamente. A exportação local `artifacts/shares/FT14_MedPT_20261020-Q4_K_M.gguf` mescla o candidato e o quantiza para distribuição multiplataforma; foi carregada e gerou `OK` pelo llama.cpp b12138 em CPU/Linux. A quantização não substitui a avaliação semântica nem a seleção FT-11. O histórico consolidado dos treinamentos está em [artifacts/training_history.md](artifacts/training_history.md). Use o [guia de compartilhamento](../docs/COMPARTILHAR_FT14.md).
