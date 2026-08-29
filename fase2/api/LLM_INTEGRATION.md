# Integração com LLM para Interpretação em Linguagem Natural

## Visão Geral

A API integra um **LLM** para gerar interpretações contextualizadas das
predições de hipertensão, transformando uma probabilidade crua em uma
análise legível para não-especialistas. Dois provedores são suportados,
escolhidos em tempo de execução:

- **Google Gemini** — via API paga (com tier gratuito), qualidade mais
  alta e mais rápido
- **Ollama** — modelo rodando localmente, sem custo e sem depender de
  internet/créditos, mais lento e dependente do hardware local

### Exemplo de Resposta

**Antes (sem LLM):**
```json
{
  "classe_prevista": 1,
  "descricao": "Com indicativo de hipertensão",
  "probabilidade_hipertensao": 0.75,
  "probabilidade_percentual": 75.0
}
```

**Depois (com LLM):**
```json
{
  "classe_prevista": 1,
  "descricao": "Com indicativo de hipertensão",
  "probabilidade_hipertensao": 0.75,
  "probabilidade_percentual": 75.0,
  "interpretacao_llm": "Com base nos dados fornecidos, existe um risco elevado de hipertensão (75%). O excesso de peso e a inatividade física são fatores de risco identificados. Recomenda-se consulta médica para confirmação diagnóstica e acompanhamento preventivo.",
  "duracao_llm_segundos": 2.1,
  "llm_habilitado": true
}
```

`llm_habilitado` reflete se **essa chamada específica** gerou
interpretação — não se um provedor em particular está configurado, já
que qualquer um dos dois pode ter sido usado.

---

## Como Configurar

### Opção A — Google Gemini

1. Acesse https://aistudio.google.com/app/apikey
2. Clique em "Create API Key" e copie o valor
3. No `.env`:
   ```env
   GOOGLE_API_KEY=sua_chave_aqui
   GOOGLE_MODEL=gemini-3.7-flash
   ```

### Opção B — Ollama local (sem custo)

1. Instale o [Ollama](https://ollama.com)
2. Baixe um modelo: `ollama pull qwen3.5`
3. Confirme que está rodando: `ollama list`
4. No `.env`:
   ```env
   OLLAMA_URL=http://host.docker.internal:11434
   OLLAMA_MODEL=qwen3.5
   ```
   `host.docker.internal` é necessário porque a API roda em Docker —
   de dentro do container, `localhost` aponta para o próprio container,
   não para o Ollama rodando no host. Rodando a API fora do Docker, use
   `http://localhost:11434`.

### Escolhendo o provedor padrão

```env
LLM_PROVIDER=google   # ou "ollama"
```

Usado sempre que uma requisição a `/prever` não especifica o provedor
explicitamente.

### Docker Compose

As variáveis do `.env` **não chegam automaticamente** ao container —
precisam estar listadas em `environment:` no `docker-compose.yml`:
```yaml
environment:
  - LLM_PROVIDER=${LLM_PROVIDER:-google}
  - GOOGLE_API_KEY=${GOOGLE_API_KEY:-}
  - GOOGLE_MODEL=${GOOGLE_MODEL:-gemini-3.7-flash}
  - OLLAMA_URL=${OLLAMA_URL:-http://host.docker.internal:11434}
  - OLLAMA_MODEL=${OLLAMA_MODEL:-qwen3.5}
  - TIMEOUT_LLM=${TIMEOUT_LLM:-300}
```
Depois de editar, é preciso recriar o container (variáveis de ambiente
só são lidas na criação):
```bash
docker compose up -d --build --force-recreate
```

### Iniciar a API (fora do Docker)

```bash
uvicorn app.api_modelo:app --reload                          # desenvolvimento
uvicorn app.api_modelo:app --host 0.0.0.0 --port 8000         # produção
```

---

## Escolhendo o provedor por requisição

Além do padrão configurado no `.env`, cada chamada a `/prever` pode
escolher o provedor e o modelo via query params, sobrescrevendo o
padrão:

```bash
curl -X POST "http://localhost:8000/prever?provedor=ollama&modelo=qwen3.5" \
  -H "Content-Type: application/json" \
  -d @exemplo_entrada_api.json
```

Útil para comparar qualidade/latência entre provedores sem reiniciar a
API, ou para usar o Ollama pontualmente quando o Google estiver sem
créditos.

---

## Comportamento

### Quando o LLM gera interpretação com sucesso

- `interpretacao_llm` traz o texto (já sem marcações Markdown — ver
  seção "Formatação da resposta" abaixo)
- `llm_habilitado: true`
- Log: `Interpretação LLM gerada com sucesso`, com `provedor`, `modelo`
  e `duracao_segundos`

### Quando o LLM falha ou está indisponível

- `interpretacao_llm` vem `null`, `llm_habilitado: false`
- A predição continua normal — o LLM nunca bloqueia a resposta da API
- Log: `Erro ao gerar interpretação com LLM`, com `tipo_erro` e `erro`
- Casos comuns: `GOOGLE_API_KEY` vazia (provedor `google`), Ollama fora
  do ar (provedor `ollama`), timeout, provedor desconhecido

---

## Formatação da resposta (remoção de Markdown)

O prompt instrui o modelo a responder em texto simples, mas LLMs nem
sempre obedecem — é comum a resposta vir com `**negrito**`, `*itálico*`
ou marcadores de lista. Como o formulário exibe texto puro (não
renderiza Markdown), a resposta passa por uma limpeza automática antes
de ser devolvida (`_remover_markdown` em `llm_interpreter.py`), que
remove esses símbolos independente do provedor usado.

## Decodificação das variáveis no prompt

As variáveis de entrada chegam ao modelo já codificadas (`0`/`1`,
faixas numeradas) — é assim que o RandomForest foi treinado. Para o
LLM, esses códigos são traduzidos para texto legível antes de montar o
prompt (`VARIAVEIS_NOMES_CLINICOS` para os nomes, `VALORES_CLINICOS`
para os valores — ex.: `dislip: 1` vira `Dislipidemia: Sim`), usando o
dicionário oficial do VIGITEL como referência. Essa tradução é só para
o texto enviado ao LLM — os valores usados na predição do modelo não
mudam.

---

## Troubleshooting

### Google: `GOOGLE_API_KEY não configurada`

O provedor `google` requer chave; sem ela a chamada é recusada antes
de sair da API (não chega a bater na rede). Configure `GOOGLE_API_KEY`
ou troque para `provedor=ollama`.

### Google: erro 500 / resposta vazia ou truncada

Verifique se `GOOGLE_MODEL` está correto e se a chave ainda tem
créditos/quota disponível. O endpoint usado
(`/v1beta/interactions`) tem um formato de resposta diferente do
endpoint clássico `generateContent` — se `GOOGLE_MODEL` apontar para
um modelo que não usa esse endpoint, a chamada falha.

### Ollama: `Erro de conexão` / timeout

- Confirme que o Ollama está rodando: `ollama list`
- Se a API roda em Docker, `OLLAMA_URL` precisa usar
  `host.docker.internal`, não `localhost`
- Teste a partir de dentro do container:
  ```bash
  docker compose exec api python -c "import requests; print(requests.get('http://host.docker.internal:11434/api/tags').json())"
  ```

### `LLM_PROVIDER` (ou qualquer variável) sempre volta `None`/padrão

A variável está no `.env` mas não está listada em `environment:` no
`docker-compose.yml` — o Compose não repassa `.env` automaticamente
para o container, só o que está explicitamente declarado. Adicione a
variável na lista e recrie o container com `--force-recreate`.

### Interpretação vazia ou muito curta

- Ajuste o prompt em `llm_interpreter.py`, função `_construir_prompt`
- Para o Google, aumente `TIMEOUT_LLM` se a resposta estiver sendo
  cortada por lentidão de rede
- Para o Ollama, modelos menores tendem a gerar respostas mais curtas
  — considere um modelo maior se a qualidade for insuficiente

---

## Custos

| Provedor | Custo |
|---|---|
| **Google Gemini** | Tier gratuito com cota limitada; acima disso, cobrado por uso — confirme o plano/cota atual em https://aistudio.google.com |
| **Ollama** | Gratuito, roda no seu hardware — custo é o de infraestrutura local (CPU/GPU/RAM), sem chamadas de rede externas |

Ollama é a opção recomendada para desenvolvimento/testes intensivos ou
quando não há créditos disponíveis no Google.

---

## Exemplo de Uso

### Via cURL

```bash
curl -X POST "http://localhost:8000/prever" \
  -H "Content-Type: application/json" \
  -d @modelo_api/exemplo_entrada_api.json
```

Escolhendo o provedor explicitamente:
```bash
curl -X POST "http://localhost:8000/prever?provedor=ollama&modelo=qwen3.5" \
  -H "Content-Type: application/json" \
  -d @modelo_api/exemplo_entrada_api.json
```

### Via Python

```python
import requests

url = "http://localhost:8000/prever"

# As 20 variáveis esperadas pelo modelo (ver GET /variaveis)
dados = {
    "diab": 0,
    "iddpapa": 3,
    "imc": 27.5,
    "excpeso": 1,
    "imc_i": 3,
    "iddpapa_old": 1,
    "excpeso_i": 1,
    "iddmamo": None,
    "af": 0,
    "exfuma": 0,
    "obesid": 0,
    "obesid_i": 0,
    "ind_med_db": 0,
    "med_db": 0,
    "atiocu": 0,
    "af3dominios_insu_2023": 1,
    "dislip": 0,
    "af3dominios_2023": 0,
    "inativo_2023": 1,
    "saruim": 0,
}

response = requests.post(url, json=dados, params={"provedor": "ollama"})
resultado = response.json()

print(f"Predição: {resultado['classe_prevista']}")
print(f"Probabilidade: {resultado['probabilidade_percentual']}%")
print(f"Interpretação: {resultado['interpretacao_llm']}")
```

### Script de teste dedicado

```bash
python scripts/testar_llm_integration.py
```
Importa `gerar_interpretacao_llm` diretamente (não precisa da API
rodando) e testa 3 cenários de risco contra o provedor configurado em
`LLM_PROVIDER`.

---

## Observações Técnicas

### Privacidade

- No provedor `google`, as variáveis de entrada (já decodificadas em
  texto) são enviadas à API do Google para gerar a interpretação
- No provedor `ollama`, os dados não saem da máquina — toda a inferência
  é local
- Nunca commite `GOOGLE_API_KEY` no repositório (mantenha `.env` fora
  do controle de versão)

### Performance

- Predição do modelo: ~50ms, local, independe do provedor de LLM
- Google Gemini: tipicamente 2-5s (rede + geração)
- Ollama: variável, depende do hardware local e do tamanho do modelo —
  pode ser mais lento que o Google em máquinas sem GPU dedicada

### Escalabilidade

- Em produção com múltiplas réplicas da API, todas compartilham o
  mesmo `.env`/config — não há isolamento de provedor por réplica
- Ollama rodando fora do container Docker da API não escala
  automaticamente com as réplicas da API (é um serviço único); considere
  isso ao decidir Ollama vs. Google para produção com autoscaling

---

## Próximos Passos

1. ✅ Interpretação via Google Gemini
2. ✅ Fallback local via Ollama, selecionável por variável de ambiente
   ou por requisição
3. ⏳ Cache de interpretações para entradas repetidas/similares
4. ⏳ Métricas Prometheus específicas por provedor de LLM
5. ⏳ Fila/processamento assíncrono para não bloquear a resposta em
   picos de latência do LLM

---

## Referências

- [Google AI Studio — API Keys](https://aistudio.google.com/app/apikey)
- [Documentação Ollama](https://ollama.com)
- [Ollama API — /api/generate](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion)