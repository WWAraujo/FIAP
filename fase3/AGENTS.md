# Guia de desenvolvimento — Fase 3

## Fonte de verdade

Leia nesta ordem antes de alterar a fase:

1. [PROPOSTA.md](PROPOSTA.md), especialmente as cinco primeiras páginas;
2. [README.md](README.md), para o estado real e a estrutura;
3. [docs/PLANO_EXECUCAO.md](docs/PLANO_EXECUCAO.md), para selecionar uma única tarefa;
4. o README do pacote afetado.

O status da fase é **protótipo parcial**. Não afirmar que RAG equivale a fine-tuning, que LangChain/LangGraph estão integrados apenas por dependência/nome de arquivo, ou que segurança foi comprovada sem teste do caminho real.

## Modo de trabalho

- Execute uma tarefa do plano por vez; marque `EM ANDAMENTO` antes de começar.
- Não implemente uma tarefa dependente por antecipação.
- Preserve mudanças locais existentes e não faça refatorações fora da tarefa.
- Ao concluir, atualize a tarefa com data, comandos, ambiente, resultados e caminhos dos artefatos. Se falhar, registre a falha e mantenha o status adequado.
- Resultados de benchmark, adapters e logs são evidências. Nunca os substitua sem `run_id`, metadados e justificativa.

## Responsabilidades dos pacotes

| Local | Responsabilidade | Não deve fazer |
|---|---|---|
| `src/medical_assistant/evaluation/` | Inferência comparativa, respostas, métricas e suporte à decisão | Declarar treinabilidade só por latência |
| `src/medical_assistant/data/` | Dados sintéticos, PII, curadoria e splits | Misturar teste no treino |
| `src/medical_assistant/training/` | LoRA/PEFT, checkpoints, recarga e manifestos | Chamar RAG de checkpoint treinado |
| `src/medical_assistant/retrieval/` | Recuperação e rastreio das fontes | Ser o mecanismo de treinamento |
| `src/medical_assistant/clinical/` e `application/` | Domínio clínico e orquestração | Duplicar treino ou avaliação |
| `tests/` | Testes unitários e integrados, sem rede por padrão | Apenas imprimir demonstrações |

## Regras para dados e modelos

- Use somente dados sintéticos ou anonimizados dentro do repositório.
- Não inclua logs com PII, banco operacional, tokens, modelos-base ou pesos grandes no Git.
- Para cada dataset usado em treino, registrar origem, versão, contagem, split e seed.
- Para cada adapter, registrar modelo-base, revisão, licença, tokenizer/template, configuração, dataset, seed, hardware e métricas.

## Critérios de qualidade

### Benchmark

- Latência e tokens/s medem inferência, não adequação clínica nem treinabilidade.
- Preservar respostas brutas e parâmetros do experimento.
- Separar check automático, revisão humana e decisão final.
- Resposta vazia, erro ou falta de fontes não pode contar como aprovação de segurança.

### Fine-tuning

- Um treinamento válido gera adapter/pesos, configuração, métricas e teste de recarga em processo novo.
- Comparar o adapter com o modelo-base no mesmo conjunto reservado.

### Assistente clínico

- Aplicar guardrails antes de qualquer chamada ao modelo.
- Cada resposta precisa informar fontes identificáveis, não apenas categorias.
- A decisão de revisão humana deve ser registrada e verificável; uma flag isolada não é aprovação humana.
- Estado de workflow não pode vazar entre pacientes ou execuções.

## Testes e documentação

- Teste o menor escopo que demonstra o critério de aceite da tarefa.
- Mocke rede e LLM em testes de software; execute modelos reais apenas nas tarefas de avaliação.
- `scripts/demo.py` é demonstração, não substituto de testes com asserts.
- Atualize README e plano quando uma tarefa modificar fluxo, comando, artefato ou status. O relatório técnico histórico será revisto somente em `ENT-01`.

## Comandos

Partindo de `FIAP/`:

```powershell
& .\.venv\Scripts\python.exe -B fase3/src/medical_assistant/evaluation/benchmark_ollama.py --help
& .\.venv\Scripts\python.exe -m pytest fase3/tests
```

Executar benchmark ou treinamento pode criar arquivos, consumir disco e chamar Ollama. Verifique a tarefa, modelos e diretório de saída antes de iniciar.
