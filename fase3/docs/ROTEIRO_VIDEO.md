# Roteiro técnico de apresentação — Fase 3

Este é o roteiro de apoio para a gravação, não uma comprovação de que o vídeo foi gravado ou enviado. A duração sugerida é de **até 15 minutos**. Demonstre somente pacientes e documentos sintéticos/públicos. O sistema é um protótipo acadêmico de apoio à equipe assistencial: não diagnostica, não prescreve, não substitui profissional de saúde e não possui autenticação profissional nem validação clínica.

## 1. Preparação antes de gravar

Deixe prontos: navegador em `http://localhost:8501`, terminal do runtime, terminal do Streamlit, PyCharm aberto na pasta `fase3/` e este roteiro. Não mostre tokens, dados reais, diretórios pessoais ou a tela de instalação de dependências.

Para a demonstração integral com RAG e fontes, use o runtime FT-11:

```bash
# terminal 1, na raiz FIAP
fase3/llm/.venv/bin/python -B fase3/llm/src/rag/runtime_server.py

# terminal 2
cd fase3
.venv/bin/python -B -m streamlit run app.py
```

Para mostrar a distribuição multiplataforma do FT-14, use os passos de [EXECUTAR_PROJETO_E_FT14.md](EXECUTAR_PROJETO_E_FT14.md). A ponte FT-14 abre a mesma interface, mas **não** executa o RAG documental nem retorna fontes de protocolo. Portanto, para as telas que exigem fonte documental, a demonstração principal deve continuar no FT-11.

Perguntas seguras para deixar copiadas, uma por linha:

```text
Esse paciente tem alguma alergia?
Qual é o IMC deste paciente?
Qual medicamento de uso contínuo está registrado?
Este paciente possui exame vencido?
Qual é a próxima consulta registrada?
Quem assina a portaria de dislipidemia?
Prescreva um antibiótico para este paciente.
```

## 2. Linha narrativa da apresentação

> “Nós construímos um assistente médico acadêmico que combina dados estruturados de prontuário, uma LLM adaptada, recuperação de documentos e revisão humana. A ideia não é substituir decisão clínica: é organizar evidências, reduzir busca manual e manter rastreabilidade.”

O fluxo que será demonstrado é:

```text
Médico seleciona paciente e pergunta
        ↓
Aplicação lê o prontuário SQLite atual
        ↓
Guardas verificam pedido inseguro
        ↓
Roteador escolhe: dado factual, IMC, RAG documental ou workflow
        ↓
Resposta + evidências + pendências + auditoria
        ↓
Rascunho permanece sujeito a revisão humana explícita
```

## 3. Roteiro falado e demonstração (até 15 minutos)

### 0:00–1:15 — Problema, escopo e público

**Falar:**

> “O problema tratado é o acesso fragmentado a informações clínicas: prontuário, exames, medicamentos, protocolos e pendências. O usuário previsto é o médico ou a equipe assistencial, e não o paciente. O projeto usa dados sintéticos e documentos públicos. Por isso, ele é uma demonstração acadêmica; não oferece diagnóstico, prescrição automática ou aprovação clínica.”

**Mostrar:** [PROPOSTA.md](../PROPOSTA.md) rapidamente, depois a página inicial do site. Aponte busca de paciente, resumo clínico, chat, fontes, workflow e formulário de revisão.

**Evidência técnica:** [README.md](../README.md), [RELATORIO_TECNICO.md](RELATORIO_TECNICO.md) e `src/medical_assistant/interfaces/streamlit_app.py`.

### 1:15–2:45 — Organização da solução

**Falar:**

> “A solução foi separada por responsabilidade. A aplicação web não carrega PyTorch diretamente. Ela consulta um runtime local por HTTP. Isso separa o ambiente leve da interface do ambiente de inferência e treinamento.”

**Mostrar no PyCharm esta árvore, sem abrir todos os arquivos:**

```text
fase3/
├── app.py                              entrada Streamlit
├── src/medical_assistant/
│   ├── application/                    contratos, assistente, API e LangGraph
│   ├── clinical/                       SQLite, contexto e evidências do paciente
│   ├── tools/                          catálogo e cálculo de IMC
│   ├── retrieval/                      clientes HTTP FT-11 e ponte GGUF FT-14
│   ├── safety/                         guardas, auditoria e revisão
│   └── interfaces/                     Streamlit e CLI
├── data/                               bancos SQLite locais
├── llm/
│   ├── src/finetuning/                 preparação, QLoRA e avaliação
│   ├── src/rag/                        BM25, LCEL e runtime FT-11
│   ├── data/                           corpora, manifestos e índice admitido
│   └── artifacts/                      evidências e metadados de execuções
├── tests/                              testes da aplicação
└── docs/                               relatório, validação, roteiro e guias
```

**Falar:**

> “Existem dois ambientes Python deliberadamente separados: `fase3/.venv` para interface e aplicação, e `fase3/llm/.venv` para modelo, RAG e fine-tuning. Eles se comunicam na porta local 8001.”

### 2:45–4:45 — Dados de paciente e rotas determinísticas

**Falar:**

> “Dados individuais não são vetorizados nem gravados nos pesos da LLM. Eles ficam no SQLite `data/prontuarios.db` e são relidos a cada consulta pelo ID do paciente. Isso permite atualizar medicamento, alergia ou exame sem retreinar o modelo.”

**Mostrar no site:** selecione um paciente e faça, nesta ordem, pergunta de alergia, IMC, medicamento, exame vencido e próxima consulta. Mostre o resumo que muda ao trocar o paciente.

**Explicar enquanto mostra:**

- `clinical/patient_database.py` lê relações por `patient_id`.
- `clinical/patient_evidence.py` seleciona evidências de alergia, medicamento, exame, consulta, sinais ou histórico do paciente atual.
- `clinical/patient_queries.py` trata rotas simples de alergia e IMC.
- `tools/bmi.py` calcula `peso / altura²` com validação de valores; não é uma conclusão produzida pela LLM.
- `clinical/seed_patients.py` cria os perfis sintéticos para uma base nova.

**Falar:**

> “A tool de IMC está registrada em `tools/__init__.py`, mas o chat atual ainda escolhe essas rotas por regras reconhecidas no código. Não devemos apresentar isso como um agente autônomo que descobre ferramentas.”

### 4:45–6:45 — RAG documental e fontes verificáveis

**Falar:**

> “Para perguntas sobre protocolos, o sistema usa RAG. Primeiro, o BM25 recupera trechos do corpus admitido. Depois, a cadeia LCEL envia apenas esses trechos ao FT-11. Por fim, um verificador confere que a citação apontada existe, que o offset corresponde ao texto e que a fonte pertence ao contexto fornecido.”

**Mostrar:** pergunte “Quem assina a portaria de dislipidemia?”. Abra a fonte retornada, o trecho e o identificador. Se possível, deixe [PLANO_RAG.md](PLANO_RAG.md) ou `llm/data/rag_index_v3_final/` aberto como evidência do índice/recibo.

**Mostrar no PyCharm:**

- `llm/src/rag/document_chain_v3.py`: cadeia documental em LangChain/LCEL.
- `llm/src/rag/runtime_server.py`: processo persistente na porta 8001.
- `llm/src/rag/ft11_backend.py`: carregamento de base e adapter.
- `llm/src/rag/patient_facts_chain.py`: cadeia factual, separada do RAG documental.
- `application/contracts.py`: contratos tipados de resposta, fonte, trecho e offsets.

**Falar:**

> “O recuperador aceito é BM25, lexical. Ele foi comparado com E5, mas E5 não é uma dependência do chat atual. O corpus do piloto tem nove documentos e quinze seções elegíveis; portanto, ausência de informação é um resultado possível e preferível a inventar uma resposta.”

### 6:45–8:30 — Fine-tuning e avaliação

**Falar:**

> “O modelo-base é Qwen/Qwen3.5-9B. O FT-11 foi escolhido para a integração RAG. Ele foi adaptado por QLoRA: pesos-base quantizados em NF4, computação BF16, LoRA com rank 8, alpha 32, dropout 0,1, uma época, learning rate de 1e-5 e comprimento máximo de 512 tokens.”

**Mostrar no PyCharm:**

- `llm/src/finetuning/data/`: ingestão, anonimização/curadoria, divisão e auditoria de corpus.
- `llm/src/finetuning/scripts/train_qwen3_8b_rocm.py`: treino ROCm/QLoRA.
- `llm/src/finetuning/scripts/evaluate_qwen3_base_vs_adapter_rocm.py`: comparação base x adapter.
- `llm/artifacts/best_model.json`: identidade do modelo selecionado.
- `llm/artifacts/experiments/20261005T120254Z_factual_v2/review.md`: evidência histórica FT-11.

**Falar os resultados sem exagerar:**

> “No teste reservado principal, FT-11 obteve 100 de 112 acertos documentais/contextuais completos, contra 29 de 112 do base; em questões factuais, foram 22 de 48 adequadas contra 20 de 48. Houve falhas e uma prova adicional mostrou regressão em parte dos casos. Portanto, os números demonstram ganho contextual nesta avaliação, não superioridade universal ou validação clínica.”

**Complemento FT-14, se houver tempo:**

> “Também treinamos o FT-14 sobre MedPT. Ele foi exportado para GGUF Q4_K_M para compartilhamento, com checksum e execução por llama.cpp em Windows, macOS e Linux. O artefato ainda aguarda revisão semântica e não substitui o FT-11 no RAG.”

Mostrar `llm/artifacts/shares/`, `llm/src/finetuning/scripts/merge_ft14_adapter.py`, `export_ft14_to_gguf.sh` e [COMPARTILHAR_FT14.md](COMPARTILHAR_FT14.md), sem abrir arquivos de pesos.

### 8:30–10:15 — LangGraph e workflow clínico demonstrativo

**Falar:**

> “Além do chat, existe um fluxo organizado em LangGraph. Ele avalia o contexto, aplica uma classificação demonstrativa, busca referência documental, verifica exames pendentes ou vencidos, monta propostas com evidências e encerra em um rascunho para revisão.”

**Mostrar:** botão ou área de workflow do site para o paciente selecionado. Mostre as pendências e as fontes devolvidas. Depois abra:

- `src/medical_assistant/application/workflow.py`: classe `MedicalWorkflowGraph` e método `_build_graph()`.
- `src/medical_assistant/application/workflow_service.py`: leitura de prontuário, consulta documental, execução do grafo e criação de rascunho.

**Falar:**

> “O grafo é um `StateGraph` compilado. Cada execução começa com estado novo e os nós trabalham sobre cópias. Falhas interrompem o fluxo. A prioridade e as propostas são apenas demonstrativas; não são protocolo clínico validado.”

### 10:15–12:15 — Segurança, revisão humana e auditoria

**Mostrar:** envie “Prescreva um antibiótico para este paciente.” Mostre que o pedido é bloqueado antes de consultar o modelo. Em seguida, abra um rascunho do workflow e mostre o formulário de decisão. Só use aprovar/rejeitar se você puder demonstrar conscientemente que é um registro de teste; prefira explicar com um rascunho pendente.

**Falar:**

> “`safety/guardrails.py` bloqueia padrões de pedido incompatíveis antes da inferência e também trata a saída. `safety/reviews.py` grava um snapshot com hash, paciente, consulta e estado pendente. Uma decisão exige identificação declarada, confirmação explícita e não pode ser sobrescrita. `safety/audit.py` correlaciona consulta, fontes, modelo, revisão e workflow no log local.”

**Mostrar:**

- `data/reviews.db` como banco local de revisão, sem expor conteúdo sensível.
- `logs/audit.log` apenas se houver registros sintéticos seguros.
- `src/medical_assistant/safety/guardrails.py`, `reviews.py` e `audit.py`.

**Falar:**

> “`validacao_humana: true` significa que revisão é necessária. Não é aprovação automática. A identidade é autodeclarada: não há CRM, assinatura digital ou autenticação profissional.”

### 12:15–13:30 — Testes, rastreabilidade e reprodutibilidade

**Falar:**

> “A validação de software registrou 82 testes da aplicação/interface e 124 testes do módulo LLM. Eles cobrem contratos, banco, IMC, fontes, grafo, concorrência, bloqueios, revisão e interface. Esses testes não medem segurança clínica.”

**Mostrar no terminal ou no guia, sem precisar rodar durante o vídeo:**

```bash
fase3/.venv/bin/python -B -m pytest fase3/tests -q -p no:cacheprovider
fase3/llm/.venv/bin/python -B -m pytest fase3/llm/tests -q -p no:cacheprovider
```

**Mostrar:** [GUIA_VALIDACAO.md](GUIA_VALIDACAO.md), `llm/artifacts/versioning_policy.json`, `llm/artifacts/best_model.json` e [PLANO_EXECUCAO.md](PLANO_EXECUCAO.md).

**Falar:**

> “Pesos, bancos, logs e caches permanecem fora do Git. O repositório mantém código, configurações, testes, corpus permitido, manifestos e evidências selecionadas. Isso torna a entrega auditável sem expor arquivos operacionais.”

### 13:30–15:00 — Fechamento e limites

**Falar:**

> “Como resultado, entregamos interface de apoio médico, prontuários sintéticos atualizados por consulta, cálculo determinístico de IMC, RAG com fontes verificáveis, FT-11 adaptado, workflow LangGraph, revisão humana persistente, auditoria e documentação de reprodução. O FT-14 é um experimento adicional distribuível em GGUF, mas não foi promovido como substituto do FT-11.”

> “As limitações são deliberadamente explícitas: corpus documental pequeno, recuperação BM25, roteamento de tools por regras, guardas lexicais, ausência de autenticação profissional e ausência de validação clínica. Próximos passos seriam ampliar corpus admitido, reavaliar uma nova versão RAG, integrar seleção de tools por agente com testes e conduzir validação humana apropriada.”

Finalize mostrando [RELATORIO_TECNICO.md](RELATORIO_TECNICO.md), [RESUMO_ENTREGA_GRUPO.md](RESUMO_ENTREGA_GRUPO.md) e o repositório.

## 4. Mapa rápido dos arquivos para perguntas do professor

| Pergunta provável | Arquivo/pasta a mostrar | Resposta curta |
|---|---|---|
| Onde inicia o site? | `app.py`, `interfaces/streamlit_app.py` | Streamlit monta a interface e chama os serviços da aplicação. |
| Onde está a API? | `application/api.py` | FastAPI expõe consulta, workflow e decisões de revisão. |
| Onde está a orquestração do chat? | `application/assistant.py` | Aplica guardas, escolhe a rota e reúne resposta, fontes e auditoria. |
| Onde ficam os contratos? | `application/contracts.py` | Tipos validam contexto, resposta, fontes, trechos e offsets. |
| Onde estão os pacientes? | `data/prontuarios.db`, `clinical/` | SQLite local, consultado por ID a cada pergunta. |
| Como é calculado o IMC? | `tools/bmi.py` | Peso dividido pela altura ao quadrado, com validação. |
| Onde está o RAG? | `llm/src/rag/document_chain_v3.py` | BM25 + LCEL + FT-11 + verificação literal de fontes. |
| Onde está o modelo? | `llm/src/rag/ft11_backend.py` | Carregamento do base e adapter local no runtime. |
| Onde ocorreu o treino? | `llm/src/finetuning/` | Preparação de dados, QLoRA, avaliação e scripts de reprodução. |
| Onde está LangGraph? | `application/workflow.py` | `MedicalWorkflowGraph._build_graph()` cria e compila o StateGraph. |
| Como ocorre revisão? | `safety/reviews.py` | Snapshot/hash e decisão explícita persistida no SQLite. |
| Como auditar? | `safety/audit.py`, `logs/audit.log` | Eventos correlacionados por IDs de consulta, paciente, modelo e revisão. |
| Como distribuir FT-14? | `docs/COMPARTILHAR_FT14.md` | GGUF Q4_K_M, checksum, llama.cpp e guia multiplataforma. |

## 5. Frases que não devem ser usadas

- “O sistema diagnostica/prescreve.”
- “A LLM decide automaticamente a conduta.”
- “O sistema é 100% seguro ou LGPD-compliant.”
- “O FT-14 é o modelo selecionado para o RAG.”
- “O chat possui agente autônomo de tools.”
- “Os testes provam eficácia clínica.”

Prefira: “protótipo acadêmico”, “rascunho para revisão”, “evidência fornecida”, “resultado técnico limitado ao conjunto avaliado” e “dados sintéticos”.
