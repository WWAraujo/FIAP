# Avaliação de modelos

Este pacote mede inferência de modelos via Ollama para reduzir a lista de candidatos. Ele não treina modelos e não comprova, sozinho, segurança clínica, qualidade clínica ou elegibilidade para fine-tuning.

Use-o em **EVAL-01** e **EVAL-02** do [plano da fase](../../../../docs/PLANO_EXECUCAO.md). A comparação de modelo-base e adapter foi realizada em **EVAL-03** e refinada nos experimentos FT-10/FT-11; resultados atuais no [relatório técnico](../../../../docs/RELATORIO_TECNICO.md). O benchmark Ollama abaixo é uma ferramenta histórica de comparação, não o modelo em uso no chat.

## O que existe hoje

- `benchmark_ollama.py`: chama `/api/chat`, mede latência/tokens e grava CSV/JSON com
  configuração, identidade/digest do modelo, ambiente, respostas e falhas.
- `clinical_cases.jsonl`: sete casos sintéticos, incluindo cenários de segurança.
- `evaluate_results.py`: gera resumo Markdown a partir do CSV.
- `../../../artifacts/benchmarks/`: destino das saídas do benchmark.

O benchmark envia o contexto escrito nos próprios casos diretamente ao Ollama. Ele não executa RAG, SQLite, LangChain, LangGraph nem os guardrails da aplicação. Os checks automáticos exigem resposta não vazia, termos obrigatórios (ou alternativas explícitas declaradas no caso) e ausência de termos proibidos, mas continuam sendo triagem lexical; não use seu resultado como certificação de segurança.

## Consultar a CLI

Na raiz `FIAP/`:

```powershell
& .\fase3\llm\.venv\Scripts\python.exe -B fase3/llm/src/finetuning/evaluation/benchmark_ollama.py --help
& .\fase3\llm\.venv\Scripts\python.exe -B fase3/llm/src/finetuning/evaluation/evaluate_results.py --help
```

Os defaults atuais são `gemma3:4b`, `qwen3:8b`, `qwen2.5:7b-instruct` e `qwen3.8:27b`, com três repetições. Para sete casos, isso produz até 21 linhas por modelo. `qwen3.8:27b` é uma referência de inferência, não um candidato para fine-tuning local em 16 GB de VRAM. Repetições não representam novos casos clínicos.

## Executar deliberadamente

O comando abaixo pode chamar Ollama, baixar os modelos informados e criar resultados em `fase3/llm/artifacts/benchmarks/`.

```powershell
& .\fase3\llm\.venv\Scripts\python.exe fase3/llm/src/finetuning/evaluation/benchmark_ollama.py `
  --models gemma3:latest qwen_iq4_xs:latest `
  --repetitions 3 `
  --no-auto-setup
```

`--no-auto-setup` desativa instalação/inicialização automática do Ollama, mas não impede download de modelos ausentes. `--skip-evaluation` só pula a geração do resumo Markdown; não pula as chamadas ao modelo. Escolha os modelos explicitamente e confira espaço em disco antes de executar. Use a tag mostrada por `ollama list` (por exemplo, `biomistral_q8_0:latest`); `:local` é uma tag literal, não um sinônimo de modelo instalado.

## Como decidir o próximo modelo para treino

Após corrigir o benchmark, registre uma tabela de decisão que inclua:

1. qualidade, segurança e fontes, revisadas por humano nos casos reservados;
2. modelo-base, revisão, licença e formato dos pesos;
3. tokenizer/template e compatibilidade com PEFT/LoRA;
4. memória e tempo necessários para treino e recarga;
5. latência e throughput como restrições operacionais.

O candidato escolhido nesta etapa segue para um ensaio LoRA. O adapter só é aprovado após comparação controlada contra seu próprio modelo-base.

## Evidências esperadas

Cada execução preserva CSV, JSON de ambiente/parâmetros, respostas e uma decisão associada ao `run_id`. Falhas de download e execução também ficam registradas no CSV e JSON. Não publique respostas contendo dados pessoais.
