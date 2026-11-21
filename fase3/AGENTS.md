# Desenvolvimento e documentação — Fase 3

## Fonte de verdade e escopo

Leia `PROPOSTA.md`, `README.md`, `docs/PLANO_EXECUCAO.md` e o README do pacote afetado antes de alterar código. A proposta contém as cinco páginas do enunciado; não é relatório de implementação. Siga a autorização vigente do usuário: uma solicitação explícita para concluir várias tarefas autoriza executá-las, uma de cada vez, sem pedir confirmação repetida.

Implemente somente o necessário à tarefa. Preserve mudanças locais, banco, pesos, corpora e evidências. Não execute treinos, downloads grandes ou testes reservados como rotina de desenvolvimento. Este projeto é um protótipo acadêmico, sem validação clínica ou autenticação profissional.

## Responsabilidades do código

- `src/medical_assistant/application/`: contratos, API, orquestração do assistente, grafo e serviço de workflow.
- `clinical/`: leitura SQLite, contexto, recuperação de fatos e seed sintético. `tools/`: catálogo e cálculo de IMC. O chat ainda reconhece essas rotas por regras; não declarar agente de tools integrado.
- `safety/`: guardas, auditoria e revisão persistente. Nunca gerar aprovação humana automaticamente. A identidade do revisor é declarada pelo usuário neste protótipo.
- `interfaces/`: Streamlit e CLI. Reutilizar contratos/serviços; não duplicar regras ou gravar formatos paralelos de auditoria.
- `llm/src/finetuning/`: preparação, treino QLoRA e avaliação. `llm/src/rag/`: backend FT-11, recuperação BM25, cadeia documental e cadeia factual de prontuário.
- `tests/` usa SDK API; `llm/tests/` usa SDK LLM. Consulte o ambiente configurado no PyCharm antes de executar Python. Dependências dos dois SDKs têm versões diferentes deliberadamente.

## Instruções para os arquivos Markdown

| Arquivo | Responsabilidade e quando atualizar |
|---|---|
| `README.md` | Entrada do projeto: instalação, execução, funcionalidades, limitações e links. Atualizar ao mudar fluxo, comando ou dependência. |
| `PROPOSTA.md` | Enunciado original, sem marcar requisitos como concluídos dentro dele. Preservar texto institucional. |
| `docs/PLANO_EXECUCAO.md` | Tarefas, status, ambiente, comandos, resultados e artefatos. Marcar em andamento antes e concluir apenas com evidência. Entradas antigas são históricas. |
| `docs/RELATORIO_TECNICO.md` | Arquitetura efetiva, diagrama LangChain/LangGraph, dados, fine-tuning, avaliação, fontes e limitações. Atualizar após validação de mudanças relevantes. |
| `docs/GUIA_VALIDACAO.md` | Casos reproduzíveis, resultados esperados, testes e procedimento manual de revisão. Não declarar testes executados sem registro. |
| `docs/ROTEIRO_VIDEO.md` | Plano para apresentação de até 15 minutos; distinguir roteiro de vídeo realmente gravado. |
| `docs/RESUMO_ENTREGA_GRUPO.md` | Descrição da entrega para o grupo: funcionalidades, pastas/arquivos, testes, reprodução e pendências. Atualizar quando mudar o escopo apresentado. |
| `docs/REVISAO_DOCUMENTACAO.md` | Inventário dos `.md` próprios, classificação ativo/histórico/gerado e links da revisão. |
| `docs/PLANO_RAG.md` | Contratos e preservação do piloto documental. Nova versão exige corpus/índice/avaliação novos, sem sobrescrever a versão aceita. |
| `llm/README.md` | SDK LLM, base/adapter, treino, reprodução, curadoria e resultados. |
| READMEs em `application/`, `tools/`, `llm/src/`, `llm/data/` | Contratos e comandos locais do pacote. Verificar links relativos e consistência com o README principal. |
| `llm/artifacts/**/review.md`, relatórios por run_id e model cards gerados | Evidências históricas: preservar bytes, datas, falhas e decisões. Não reescrever uma reprovação como aprovação atual. Registrar seu caráter histórico no índice documental e no README dos artefatos. |
| Revisões/snapshots anteriores em `docs/` | Identificar como históricos e apontar para o relatório atual; não apresentar contagens/passos antigos como estado vigente. |

Ao mudar comportamento, atualize somente os documentos afetados. Verifique comandos, links locais, nomes de pacotes e ausência de alegações antigas como “RAG substitui fine-tuning”, “100% seguro/LGPD” ou “LangGraph” apenas por nome de classe. Um teste passado não demonstra cobertura universal ou aprovação clínica. Não replique métricas detalhadas no AGENTS da raiz: direcione ao relatório técnico da fase.

## Integridade e qualidade

Os módulos listados em `llm/data/rag_index_v3_final/readiness.json` estão congelados. Não editar hashes para contornar falhas. A cadeia factual de paciente e o runtime são extensões separadas; não herdam automaticamente as métricas do piloto documental. Novos dados/documentos exigem versão e avaliação novas.

Use dados sintéticos/anonimizados. Mantenha pesos, ambientes, SQLite, logs e consultas operacionais fora do Git. O repositório acadêmico deve conter código de treino, exemplos/corpora permitidos, configuração, testes, relatório e evidências selecionadas — apenas o runtime é insuficiente para a proposta. Preserve origem, licença, versão, split, seed, hashes e identidade dos adapters.

Use mocks de rede/modelo em testes de software. Teste o escopo alterado, paciente/execução isolados, fontes, falhas e revisão explícita. Uma decisão simulada em teste não é aprovação clínica. Não reabra um teste reservado já consumido. Registre inferência real em run_id novo; nunca normalize saídas para fabricar aprovação.

## Comandos na raiz FIAP

```bash
fase3/.venv/bin/python -B -m pytest fase3/tests -q -p no:cacheprovider
fase3/llm/.venv/bin/python -B -m pytest fase3/llm/tests -q -p no:cacheprovider
```

O README informa os comandos para iniciar os serviços e reproduzir o treinamento. As bibliotecas de terceiros em `.venv/` e documentos do modelo-base não são documentação própria a reescrever.
