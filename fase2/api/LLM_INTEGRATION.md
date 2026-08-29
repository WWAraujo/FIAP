# Integração com LLM para Interpretação em Linguagem Natural

## Visão Geral

A API agora integra um **LLM (Large Language Model) gratuito** do Hugging Face para gerar interpretações contextualizadas das predições de hipertensão. Isso transforma a resposta da API de uma simples probabilidade em uma análise compreensível para não-especialistas.

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
  "interpretacao_llm": "Com base nos dados fornecidos, existe um risco elevado de hipertensão (75%). Fatores como pressão arterial elevada e circunferência da cintura aumentada são indicativos importantes. Recomenda-se consulta médica para confirmação diagnóstica e possível início de tratamento.",
  "llm_habilitado": true
}
```

---

## Como Configurar

### Pré-requisitos

1. **Conta no Hugging Face** (gratuita)
   - Acesse: https://huggingface.co
   - Crie uma conta (se ainda não tiver)

2. **Token de API do Hugging Face**
   - Vá para: https://huggingface.co/settings/tokens
   - Clique em "New token"
   - Selecione "Read" como tipo de acesso
   - Copie o token (começa com `hf_`)

### Passos de Configuração

#### 1. Instale as dependências
```bash
pip install -r requirements.txt
```

#### 2. Configure as variáveis de ambiente

**Opção A: Arquivo `.env`**
```bash
cp .env.example .env
# Edite o arquivo .env e adicione seu token
# HF_API_TOKEN=hf_seu_token_aqui
```

**Opção B: Variáveis de ambiente do sistema**
```bash
export HF_API_TOKEN="hf_seu_token_aqui"
export HF_MODEL_URL="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
export TIMEOUT_LLM=10
```

**Opção C: Docker/Docker Compose**
Adicione no `docker-compose.yml`:
```yaml
environment:
  - HF_API_TOKEN=hf_seu_token_aqui
  - HF_MODEL_URL=https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1
  - TIMEOUT_LLM=10
```

#### 3. Inicie a API
```bash
# Desenvolvimento
uvicorn app.api_modelo:app --reload

# Produção
uvicorn app.api_modelo:app --host 0.0.0.0 --port 8000
```

---

## Modelos Disponíveis

O Hugging Face oferece vários modelos **gratuitos** de linguagem. Escolha um conforme sua necessidade:

| Modelo | URL | Tamanho | Velocidade | Qualidade |
|--------|-----|--------|-----------|-----------|
| **Mistral-7B** (padrão) | `https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1` | 7B | ⭐⭐⭐⭐ Muito rápido | ⭐⭐⭐⭐ Excelente |
| Zephyr-7B | `https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta` | 7B | ⭐⭐⭐⭐ Muito rápido | ⭐⭐⭐⭐ Excelente |
| Llama-2-7B | `https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf` | 7B | ⭐⭐⭐ Rápido | ⭐⭐⭐⭐ Bom |
| Mistral-Medium | `https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1` | 7B | ⭐⭐⭐ Rápido | ⭐⭐⭐⭐ Excelente |

**Recomendação:** Use **Mistral-7B** por padrão — oferece o melhor balanço entre velocidade e qualidade.

---

## Comportamento

### Quando o LLM está ativado

- A resposta inclui o campo `interpretacao_llm` com a análise do modelo
- A resposta inclui `llm_habilitado: true`
- Se houver timeout ou erro na chamada ao LLM, a interpretação será `null` mas a predição continua normal
- Logs são registrados com a duração da chamada ao LLM

### Quando o LLM está desativado

- Se `HF_API_TOKEN` não estiver configurado:
  - `interpretacao_llm` será `null`
  - `llm_habilitado: false`
  - A API continua funcionando normalmente
- Útil para desenvolvimento local ou quando não há quota disponível

---

## Troubleshooting

### Problema: "Modelo carregando ou indisponível" (HTTP 503)

**Causa:** O modelo está sendo carregado pela primeira vez no servidor Hugging Face.

**Solução:**
- Espere 30-60 segundos
- Tente a requisição novamente
- A chamada será bem-sucedida na segunda tentativa

### Problema: "Authorization failed" (HTTP 401)

**Causa:** Token inválido ou não configurado.

**Solução:**
1. Verifique se o token está correto em `HF_API_TOKEN`
2. Certifique-se de que está usando um token com permissão de "Read"
3. Regenere o token em https://huggingface.co/settings/tokens se necessário

### Problema: Timeout na chamada ao LLM

**Causa:** A rede está lenta ou o servidor Hugging Face está congestionado.

**Solução:**
- Aumente o valor de `TIMEOUT_LLM` (padrão: 10 segundos)
- A interpretação não bloqueará a predição — ela retornará `null`
- A API continua respondendo normalmente

### Problema: Interpretação vazia ou muito curta

**Causa:** O modelo retornou uma resposta inadequada.

**Solução:**
- Ajuste o prompt em `llm_interpreter.py` na função `_construir_prompt()`
- Experimente com diferentes valores de `temperature` (0.5-0.8)
- Aumente `max_new_tokens` para respostas mais longas

---

## Custos

✅ **100% Gratuito!**

- Hugging Face Inference API é gratuita para modelos públicos
- Sem limite de requisições (apenas rate-limiting justo)
- Sem necessidade de cartão de crédito
- Ideal para prototipagem e educação

---

## Exemplo de Uso

### Via cURL

```bash
curl -X POST "http://localhost:8000/prever" \
  -H "Content-Type: application/json" \
  -d '{
    "idade": 45,
    "sexo": "M",
    "pressao_sistolica": 150,
    "pressao_diastolica": 95,
    "imc": 28.5,
    "cintura": 95,
    "frequencia_cardiaca": 75,
    "glicose": 110,
    "colesterol": 240,
    "hdl": 35,
    "ldl": 160,
    "triglicerides": 200,
    "fumante": 1,
    "consumo_alcool": 1,
    "atividade_fisica": 0,
    "estresse": 3,
    "diabetes": 0,
    "medicamentos_hipertensao": 0
  }'
```

### Via Python

```python
import requests

url = "http://localhost:8000/prever"

dados = {
    "idade": 45,
    "sexo": "M",
    "pressao_sistolica": 150,
    "pressao_diastolica": 95,
    "imc": 28.5,
    # ... restante das variáveis
}

response = requests.post(url, json=dados)
resultado = response.json()

print(f"Predição: {resultado['classe_prevista']}")
print(f"Probabilidade: {resultado['probabilidade_percentual']}%")
print(f"Interpretação: {resultado['interpretacao_llm']}")
```

---

## Observações Técnicas

### Privacidade

- As variáveis de entrada são enviadas ao Hugging Face para gerar a interpretação
- Use `HF_API_TOKEN` com cuidado e não a exponha em repositórios públicos
- Considere usar um servidor Hugging Face privado em produção se houver preocupações

### Performance

- Tempo de resposta típico: 2-5 segundos (inclui latência de rede)
- A predição é feita localmente (~50ms)
- A geração de texto via LLM é o gargalo principal (~2-4 segundos)
- Implementar cache de respostas pode melhorar performance

### Escalabilidade

- Em produção com múltiplas réplicas, considere:
  - Usar um servidor Hugging Face dedicado
  - Implementar fila/async para processar interpretações em background
  - Cache de interpretações para pacientes similares

---

## Próximos Passos

1. ✅ Implementar interpretações via LLM
2. ⏳ Adicionar cache de respostas
3. ⏳ Usar filas (Celery/RabbitMQ) para processar interpretações async
4. ⏳ Criar métricas Prometheus específicas para LLM
5. ⏳ Adicionar modelos menores/mais rápidos como fallback

---

## Referências

- [Hugging Face Inference API](https://huggingface.co/inference-api)
- [Modelos disponíveis](https://huggingface.co/models)
- [Documentação de autenticação](https://huggingface.co/docs/api-inference/quicktour)
- [Limite de rate limiting](https://huggingface.co/docs/api-inference/rate-limits)
