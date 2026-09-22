# Scorecard de candidatos para fine-tuning — EVAL-02

Atualizado em 20/09/2026. Este documento seleciona um **candidato para validação de treinamento**, não um modelo clinicamente aprovado. A aprovação clínica exige os dados curados, o modelo adaptado e a avaliação controlada de EVAL-03.

## Evidência de inferência

Fonte local: [benchmark `20260920T190753Z`](../artifacts/benchmarks/benchmark_20260920T190753Z.md), com 7 casos e 3 repetições por modelo. Os checks são automáticos e lexicais; não medem qualidade clínica nem segurança completa.

| Candidato Ollama | Modelo-base treinável candidato | Checks automáticos | Latência média | Situação |
|---|---|---:|---:|---|
| `gemma3:4b` | `google/gemma-3-4b-it` | 14/21 | 3,6 s | Alternativa leve para inferência; não é o candidato prioritário de LoRA. |
| `qwen2.5:7b-instruct` | `Qwen/Qwen2.5-7B-Instruct` | 9/21 | 5,7 s | Elegível tecnicamente, mas pior resultado automático nesta execução. |
| `qwen3:8b` | `Qwen/Qwen3-8B` | 14/21 | 10,2 s | Candidato prioritário para validação acelerada. |
| `qwen3.8:27b` | Não identificado com precisão | 18/21 | 66,1 s | Excluído: alias não mapeia uma revisão verificável e o custo local é desproporcional. |

O digest Ollama identifica somente o GGUF quantizado usado na inferência. Ele não substitui o commit do repositório Safetensors que será registrado em FT-01.

## Critérios de seleção

| Critério | Gemma 3 4B IT | Qwen 2.5 7B Instruct | Qwen3 8B | Resultado |
|---|---|---|---|---|
| Pesos para treino | Safetensors, acesso condicionado a aceite | Safetensors | Safetensors | Todos possíveis; Qwen evita acesso condicionado. |
| Licença | Termos Gemma e obrigações de redistribuição | Apache-2.0 | Apache-2.0 | Qwen é mais simples para uma entrega acadêmica reproduzível. |
| Template/tokenizer | Disponível, modelo multimodal | Modelo de instrução para `Transformers` | Modelo de instrução para `Transformers` | Todos exigem aplicar o template oficial; não reutilizar prompt Ollama como formato de treino. |
| LoRA/PEFT | Compatível segundo documentação oficial | A validar no ambiente acelerado | A validar no ambiente acelerado | A compatibilidade real depende do forward/backward de ORG-04. |
| Inferência local observada | Mais rápida; 14/21 checks | 9/21 checks | 14/21 checks | Não decide qualidade clínica. |
| Memória e tempo | Menor opção | Intermediário | Maior que 7B, porém ainda viável em hipótese para QLoRA | Medir; o Windows atual é CPU-only. |

## Fontes verificáveis

- [Qwen/Qwen3-8B — model card e licença Apache-2.0](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen/Qwen2.5-7B-Instruct — model card e licença Apache-2.0](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [google/gemma-3-4b-it — model card, pesos condicionados a aceite](https://huggingface.co/google/gemma-3-4b-it)
- [Termos de uso do Gemma](https://ai.google.dev/gemma/terms)
- [Fine-tuning oficial de Gemma, incluindo Transformers e PEFT](https://ai.google.dev/gemma/docs/tune)

As páginas externas devem ser revisadas novamente no momento do download, registrando URL, commit e licença efetivamente aceitos no manifesto de FT-01.

## Decisão registrada

**DEC-05 — Priorizar `Qwen/Qwen3-8B` como candidato para a validação de LoRA/QLoRA em ambiente CUDA.**

Motivos: licença Apache-2.0, pesos e template oficiais disponíveis, resultado automático empatado com Gemma e superior ao Qwen 2.5 neste conjunto, e tamanho muito mais razoável que o alias local de 27B. Isto **não** é uma aprovação clínica nem uma garantia de que caberá na GPU.

O candidato será aceito para FT-01 somente se ORG-04 comprovar CUDA, forward/backward, consumo de memória e persistência do checkpoint. Caso 8B exceda a VRAM disponível, a alternativa é `Qwen/Qwen2.5-7B-Instruct`; Gemma 3 4B só será retomado se a restrição de VRAM for determinante e os termos forem aceitos/documentados.

## Lacunas deliberadamente não escondidas

- Não houve revisão humana das respostas; logo, 14/21 ou 18/21 checks não são métricas clínicas.
- Os dados ainda não passaram por DAT-01 a DAT-03; não há conjunto reservado apto para escolher por qualidade.
- A configuração atual de treinamento aponta para `gemma3:4b`, um alias Ollama GGUF. Ela não deve ser usada por FT-01; o executor deverá receber o repositório Safetensors e uma revisão fixada.
- Não existe acelerador utilizável no Windows atual: `torch 2.14.0+cpu`, sem CUDA/HIP, conforme `artifacts/environment/org03_20260920.json`.
