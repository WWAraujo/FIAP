# Aplicação e contratos

`assistant.py` valida entrada, registra consulta, aplica guardas e consulta alergia/IMC ou o cliente FT-11. `contracts.py` preserva fontes e exige offsets literais; fontes de outro paciente são recusadas. O histórico de chat serve à interface, não é memória clínica ilimitada enviada ao modelo.

`workflow.py` compila e invoca um `StateGraph` real com nós de avaliação, prioridade, referência documental, exames, proposta e conclusão. Os nós trabalham com cópias independentes; falha encerra sem afirmar conclusão. `workflow_service.py` obtém dados do SQLite, consulta documentos com FT-11, invoca o grafo, registra fontes e cria revisão pendente. Exames pendentes/vencidos são os do banco; ausência de evidência não autoriza conduta inventada. Regras de triagem são demonstrativas, não validadas clinicamente.

## API local

Da raiz FIAP:

```bash
fase3/.venv/bin/python -B -m uvicorn medical_assistant.application.api:app --app-dir fase3/src --host 127.0.0.1 --port 8000
```

| Rota | Entrada/uso |
|---|---|
| `POST /assistant/query` | `query`, `patient_id` opcional. Resposta, fontes, modelo/run_id e revisão pendente. |
| `POST /workflows` | `patient_id` e `query`. Executa grafo e retorna rascunho, fontes, decisões e `review_id`. |
| `GET /reviews/{review_id}` | Consulta snapshot, hash, status e eventual decisão. |
| `POST /reviews/{review_id}/decision` | `patient_id` (ou null em conversa sem paciente), `reviewer`, `decision` (`approved`/`rejected`), `notes` opcional e `confirmed_human: true`. |
| `POST /query` | Rota legada Ollama: `query` e `mode` (`standard`/`fast`). Não é o chat FT-11. |

A decisão de revisão é única e vinculada ao snapshot/paciente; não altera o rascunho. Sem identificação, confirmação ou evidências válidas não há aprovação. A interface permite decisão explícita; a aplicação não escolhe aprovação pelo profissional. A identidade é declarada, sem autenticação ou conferência de registro profissional. O endpoint é para protótipo local e não deve ser exposto como sistema clínico autenticado.

A rota Ollama preserva a política de `configs/config.yaml`, conferindo nome/digest antes/depois da geração, sem fallback. `OLLAMA_BASE_URL` altera o destino. FT-11 usa `FT11_RUNTIME_URL`, padrão localhost:8001; erros de inferência são explícitos. Fontes ausentes representam pendência, não sucesso clínico.

## Persistência e auditoria

`data/prontuarios.db` armazena pacientes sintéticos. `data/reviews.db` armazena snapshots e decisões; `REVIEW_DB_PATH` permite outro destino (usado para isolar testes). `logs/audit.log` registra consultas, respostas completas, fontes, modelo, eventos do workflow e revisão com hash e identificadores correlacionados. Registros do modelo ficam em `llm/artifacts/experiments/*_app01_*/`.

Dados e logs são locais e ignorados. Não enviar prontuários reais ou decisões simuladas como aprovação de médicos. Confira o [guia de validação](../../../docs/GUIA_VALIDACAO.md) e o [README principal](../../../README.md).
