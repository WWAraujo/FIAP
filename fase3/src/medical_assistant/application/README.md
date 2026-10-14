# Aplicação

O chat Streamlit passa a pergunta original e o contexto do paciente separadamente para `process_query`. O histórico compõe o prompt/contexto auditado, mas não é concatenado à pergunta usada para selecionar a resposta simulada. Isso evita acionar a resposta de hipertensão apenas por essa condição constar no prontuário; não adiciona inferência real nem compreensão conversacional ao protótipo.

Os controles compartilhados ficam em `medical_assistant.safety`: `guardrails.py` contém `SecuritySystem` e `audit.py` contém `AuditLogger`. Assistente e API importam essas classes; a extração preserva os filtros lexicais e o formato/destino da auditoria, sem implementar a validação integrada prevista em SAFE-01.

`api.py` implementa APP-00: inferência Ollama com seleção controlada por `configs/config.yaml`. O cliente envia somente `query` e, opcionalmente, `mode` (`standard` ou `fast`); campos adicionais são rejeitados. `standard` usa `model.base_model` (Qwen configurado) e `fast` usa `model.alternative_model` (Gemma configurado). `OLLAMA_BASE_URL` pode substituir o endereço do servidor.

Partindo da raiz, com o ambiente API instalado:

```bash
fase3/.venv/bin/python -m uvicorn medical_assistant.application.api:app --app-dir fase3/src --host 127.0.0.1 --port 8000
curl -X POST http://127.0.0.1:8000/query -H 'Content-Type: application/json' \
  -d '{"query":"Quais são as limitações deste assistente?","mode":"standard"}'
```

O servidor Ollama e os dois modelos configurados precisam existir no host. Esta API não baixa modelos e não troca automaticamente de modelo quando há falha. A resposta e `fase3/logs/audit.log` registram modo, nome e digest. A identidade vem do inventário [`/api/tags`](https://docs.ollama.com/api/tags), conferido antes e depois da geração; o nome retornado por [`/api/generate`](https://docs.ollama.com/api/generate) também precisa corresponder. Alteração detectada, resposta vazia/incompleta, modelo ausente ou erro do servidor retornam HTTP 503 com `model_unavailable`. Isso é rastreabilidade do servidor local, não atestado criptográfico dos pesos executados.

Entradas bloqueadas pelos filtros existentes não chamam Ollama: o nome e o digest ficam nulos porque não houve modelo utilizado. Erros registram o modelo solicitado e identidade utilizada não confirmada. As respostas requerem revisão; não representam aprovação humana.

**Limite atual:** os nomes Ollama desta política são os configurados historicamente; não carregam automaticamente o adapter FT-11. APP-01 ainda deve conectar o FT-11 real, a cadeia documental RAG e o prontuário, com fontes verificáveis e contratos compartilhados. Aqui `sources` é vazio; não há consulta ao banco nem alegação de cobertura clínica. CLI/Streamlit continuam usando o protótipo `assistant.py`, com respostas simuladas; `workflow.py` continua sequencial até APP-02. Os filtros existentes não constituem validação clínica nem segurança completa; SAFE-01 avalia o fluxo integrado.
