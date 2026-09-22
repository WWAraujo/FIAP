# Docker - Fase 3 Tech Challenge

## Visão Geral

Este documento descreve como usar Docker para conteinerizar e executar o Assistente Médico da Fase 3 Tech Challenge.

## Arquitetura Docker

### Por que usar Docker?

1. **Isolamento de Dependências:** Separa dependências complexas (como bindings C++ do llama-cpp-python)
2. **Reprodutibilidade:** Garante que o projeto rode em qualquer máquina
3. **Persistência de Dados:** Mapeia volumes locais para não perder dados
4. **Fácil Deploy:** Simplifica deployment em produção

## Estrutura de Volumes

O `docker-compose.yml` mapeia os seguintes volumes locais:

```yaml
volumes:
  - ./data:/app/data              # Banco SQLite e dados sintéticos
  - ./models:/app/models          # Modelos GGUF e configurações
  - ./logs:/app/logs              # Logs de auditoria
  - ./reports:/app/reports       # Relatórios de benchmark
  - ./configs:/app/configs       # Configurações
```

## Como Executar

### Pré-requisitos

1. **Docker Desktop** instalado (Windows/Mac) ou **Docker Engine** (Linux)
2. **Ollama** rodando no host (para comunicação com LLM)

### Passo 1: Iniciar Ollama (se necessário)

```bash
# Em terminal separado
ollama serve
```

### Passo 2: Construir e Iniciar Contêiner

```bash
cd fase3
docker-compose up -d --build
```

### Passo 3: Acessar Interface Web

A interface web estará disponível em: **http://localhost:8501**

## Comandos Docker Úteis

### Verificar Status dos Contêineres

```bash
docker-compose ps
```

### Ver Logs do Contêiner

```bash
docker-compose logs -f medical-assistant
```

### Parar Contêiner

```bash
docker-compose down
```

### Reiniciar Contêiner

```bash
docker-compose restart
```

### Acessar Shell do Contêiner

```bash
docker-compose exec medical-assistant bash
```

## Configuração Ollama com Docker

### Opção 1: Ollama no Host (Padrão - Recomendado)

**Configuração atual:** Ollama roda no host e o contêiner se comunica via `host.docker.internal`:

```yaml
environment:
  - OLLAMA_BASE_URL=http://host.docker.internal:11434
```

**Pré-requisitos:**
- Ollama instalado e rodando no host: `ollama serve`
- Docker Desktop (Windows/Mac) ou Linux com configuração adequada

**Vantagens:**
- ✅ Simples configuração
- ✅ Ollama pode usar GPU do host facilmente
- ✅ Compartilha modelos entre projetos
- ✅ Menor overhead de recursos

**Como usar:**
```bash
# Terminal 1: Iniciar Ollama no host
ollama serve

# Terminal 2: Iniciar contêiner
cd fase3
docker-compose up -d --build
```



## Resolução de Problemas

### Erro: "Cannot connect to Ollama"

1. Verifique se Ollama está rodando: `ollama serve`
2. Verifique se a porta 11434 está acessível
3. Teste comunicação: `curl http://localhost:11434/api/tags`

### Erro: "Port 8501 already in use"

1. Pare outros serviços usando a porta 8501
2. Ou altere a porta mapeada no `docker-compose.yml`:
   ```yaml
   ports:
     - "8502:8501"  # Mapeia 8502 no host para 8501 no contêiner
   ```

### Erro: "Volume not found"

Os volumes são criados automaticamente pelo Docker Compose. Se necessário, limpe volumes:

```bash
docker-compose down -v
docker-compose up -d --build
```

### Erro: "Permission denied" (Linux)

Se tiver problemas de permissão com volumes no Linux:

```bash
# Adicionar usuário ao grupo docker
sudo usermod -aG docker $USER

# Logout e login novamente
```

## Deploy em Produção

Para produção, considere:

1. **Alterar porta exposta:** Não usar 8501 em produção
2. **HTTPS:** Use reverse proxy (nginx/apache) com SSL
3. **Autenticação:** Adicionar sistema de login
4. **Monitoramento:** Adicionar logs centralizados
5. **Backup:** Implementar backup dos volumes data/ e models/

## Exemplo de docker-compose.yml para Produção

```yaml
version: '3.8'

services:
  medical-assistant:
    build: .
    container_name: fiap_fase3_medical_assistant_prod
    ports:
      - "8080:8501"  # Porta HTTP padrão
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
      - ./reports:/app/reports
      - ./configs:/app/configs
    environment:
      - PYTHONUNBUFFERED=1
      - OLLAMA_BASE_URL=http://ollama:11434
      - ENVIRONMENT=production
    restart: always
    networks:
      - medical_network
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama:latest
    container_name: fiap_fase3_ollama
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    restart: always
    networks:
      - medical_network

volumes:
  ollama_data:

networks:
  medical_network:
    driver: bridge
```

## Recursos Adicionais

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Streamlit Deployment](https://docs.streamlit.io/deploy)