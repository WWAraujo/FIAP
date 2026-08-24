# Scripts de Teste da API

## Teste de Integração com LLM

O script `testar_llm_integration.py` executa um conjunto completo de testes na API de hipertensão com integração ao LLM.

### Pré-requisitos

1. **API rodando localmente ou remotamente**
   ```bash
   cd fase2/api
   docker-compose up
   # ou
   uvicorn src/app/api_modelo:app --reload
   ```

2. **Variáveis de ambiente configuradas**
   ```bash
   export HF_API_TOKEN="hf_seu_token_aqui"
   ```

3. **Dependências Python**
   ```bash
   pip install requests
   ```

### Uso

#### Teste contra API local (padrão)
```bash
python testar_llm_integration.py
```

#### Teste contra API remota
```bash
python testar_llm_integration.py --url http://api.exemplo.com:8000
```

#### Teste com timeout customizado
```bash
python testar_llm_integration.py --timeout 60
```

### O que o script testa

1. **Health Check**
   - Verifica se a API está disponível
   - Exibe informações do modelo carregado

2. **Predição com ALTO RISCO**
   - Simula um paciente com múltiplos fatores de risco
   - Verifica interpretação do LLM

3. **Predição com BAIXO RISCO**
   - Simula um paciente com fatores protetores
   - Verifica qualidade da interpretação para baixo risco

4. **Predição com RISCO INTERMEDIÁRIO**
   - Simula um paciente com perfil misto
   - Testa casos intermediários

### Saída esperada

```
======================================================================
TESTE DE INTEGRAÇÃO COM LLM - API DE HIPERTENSÃO
======================================================================

🔍 Verificando conexão com API...
   URL: http://localhost:8000
✅ API está saudável
   Modelo: modelo_genetico_vencedor
   Versão: 2.0.0
   Uptime: 45 segundos

...

🤖 INTERPRETAÇÃO DO LLM:
   Com base nos dados fornecidos, existe um risco elevado de
   hipertensão (85%). Fatores críticos incluem pressão arterial
   elevada, obesidade central e dislipidemia. Recomenda-se
   consulta médica urgente para confirmação diagnóstica.

======================================================================
TESTES CONCLUÍDOS
======================================================================
```

### Interpretação dos resultados

#### ✅ Sucesso

```
✅ API está saudável
🤖 INTERPRETAÇÃO DO LLM:
   [texto gerado pelo modelo]
```

**O quê fazer:**
- Verifique se as interpretações fazem sentido clínico
- Ajuste o prompt em `llm_interpreter.py` se necessário

#### ⚠️ LLM desabilitado

```
⚠️ LLM NÃO HABILITADO
   Configure HF_API_TOKEN para usar a interpretação em linguagem natural
```

**O quê fazer:**
1. Obtenha um token em https://huggingface.co/settings/tokens
2. Configure a variável de ambiente:
   ```bash
   export HF_API_TOKEN="hf_seu_token_aqui"
   ```
3. Reinicie a API

#### ⚠️ Modelo carregando (503)

```
⚠️ LLM habilitado mas não gerou interpretação (pode estar carregando)
```

**O quê fazer:**
- Aguarde 30-60 segundos (primeira execução carrega o modelo)
- Tente novamente

#### ❌ Erro de conexão

```
❌ API indisponível. Encerrando.
```

**O quê fazer:**
1. Verifique se a API está rodando:
   ```bash
   curl http://localhost:8000/health
   ```
2. Verifique se a porta está correta:
   ```bash
   lsof -i :8000  # macOS/Linux
   netstat -ano | findstr :8000  # Windows
   ```

#### ❌ Timeout

```
❌ Timeout na requisição (>30 segundos)
```

**O quê fazer:**
- Aumente o timeout:
  ```bash
  python testar_llm_integration.py --timeout 60
  ```
- Verifique conexão de rede
- Verifique se o servidor Hugging Face está respondendo

### Monitoramento em tempo real

Durante os testes, você pode monitorar a API em outro terminal:

```bash
# Ver logs da API
tail -f api/logs/api.log

# Ver métricas Prometheus
curl http://localhost:9090

# Ver dashboard Grafana
open http://localhost:3000
```

### Troubleshooting

#### Problema: "Authorization failed" (401)

Token inválido ou não configurado.

```bash
# Verifique se o token está correto
echo $HF_API_TOKEN

# Regenere em: https://huggingface.co/settings/tokens
export HF_API_TOKEN="hf_seu_novo_token"
```

#### Problema: Interpretação muito curta

O modelo não gerou resposta completa.

**Solução:** Ajuste o prompt em `../src/app/llm_interpreter.py`:

```python
def _construir_prompt(...):
    prompt = f"""[Aumente max_new_tokens no payload]
    "parameters": {
        "max_new_tokens": 300,  # Era 200
        ...
    }
    """
```

#### Problema: Muita latência

Requisições levando mais de 5 segundos.

**Causas:**
- Rede lenta
- Servidor Hugging Face congestionado
- Modelo grande sendo carregado

**Soluções:**
- Use um modelo menor: `mistralai/Mistral-7B-v0.1`
- Implemente cache de respostas
- Use processamento async

### Próximos testes

Após executar com sucesso:

1. **Teste de carga**
   ```bash
   # Requisições concorrentes
   ab -n 100 -c 10 http://localhost:8000/health
   ```

2. **Teste de estresse**
   - Aumentar número de réplicas no docker-compose
   - Monitorar métricas no Grafana

3. **Teste de qualidade**
   - Ajustar prompt conforme feedback clínico
   - Comparar interpretações com diagnósticos reais

---

## Outros Scripts

Consulte o README principal da API para informações sobre:
- Scripts de treinamento
- Scripts de avaliação
- Scripts de deploy
