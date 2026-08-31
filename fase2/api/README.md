# API de Hipertensão (v2.0.0) — deploy do modelo genético (Fase 2)

Stack de deploy do modelo vencedor do algoritmo genético (`fase2/`):
RandomForest com hiperparâmetros otimizados, servido via FastAPI, escalado
horizontalmente com `docker-compose` (nginx + autoscaler baseado em CPU) e
com monitoramento completo (Prometheus + Grafana + cAdvisor).

Arquitetura completa, decisões e trade-offs: [ARCHITECTURE.md](./ARCHITECTURE.md).

> O código desta API parte da mesma base da API original da Fase 1
> (`fase1/`, que permanece inalterada como baseline histórico), mas roda
> de forma independente aqui, apontando para o modelo otimizado pelo
> algoritmo genético.

## Pré Requisito: Criar a chave de autenticação da API Gemini
## Criar sua [Appkey](https://ai.google.dev/gemini-api/docs/api-key?hl=pt-br)
---

## 1. Passo a passo: clone e setup do zero

### Pré-requisitos

- **Docker** e **Docker Compose** (Docker Desktop já traz os dois — `docker compose version` deve funcionar)
- **git**
- ~300 MB de espaço livre em disco

### Passo 1 — Clonar o repositório

```bash
git clone https://github.com/WWAraujo/FIAP.git
cd FIAP
```

O modelo em produção (`fase2/api/modelo_api/modelo_genetico_vencedor.joblib`,
~70 MB) já vem versionado no repositório — nenhum download manual é
necessário. Ele é uma versão podada do vencedor do algoritmo genético
(400 → 50 árvores, mesma acurácia dentro da margem de ruído estatístico;
detalhes em [modelo_api/README.md](./modelo_api/README.md#poda-do-modelo)
e em `poda_pos_treino` dentro de [modelo_api/metadata_modelo_api.json](./modelo_api/metadata_modelo_api.json)).

Confira que o arquivo veio no clone:
```bash
ls -lh fase2/api/modelo_api/modelo_genetico_vencedor.joblib
```

> Só é preciso gerar o modelo do zero se você quiser retreinar/reotimizar
> (não para simplesmente rodar a API). Nesse caso, rode o pipeline da
> Fase 2 (`cd fase2 && python tech_challenge_fase2.py` — requer a base
> `vigitel-2024.csv` em `shared/data/`, peça o link ao time) e substitua
> `fase2/api/modelo_api/modelo_genetico_vencedor.joblib` pelo resultado.

### Passo 2 — Subir a stack completa

```bash
cd fase2/api
docker compose up -d --build
```

Isso builda a imagem da API e sobe 7 containers: `api` (2 réplicas),
`nginx`, `autoscaler`, `prometheus`, `cadvisor`, `grafana`. O primeiro
build demora alguns minutos (baixa as imagens base e instala dependências);
as próximas subidas usam cache e são rápidas.

### Passo 3 — Verificar que subiu tudo certo

```bash
docker compose ps
```
Todos os serviços devem aparecer `Up`, com `api` e `cadvisor` marcados
`(healthy)` após ~30-40s.

```bash
curl http://localhost:8000/health
```
Deve retornar `{"status": "ok", "modelo_carregado": true, ...}`.

Teste uma predição de exemplo:
```bash
curl -s http://localhost:8000/prever \
  -H "Content-Type: application/json" \
  -d @modelo_api/exemplo_entrada_api.json
```

### Passo 4 — Acessar as interfaces

| Interface | URL | Observação |
|---|---|---|
| Formulário web / API | http://localhost:8000 | Ponto de entrada público (via nginx) |
| Documentação da API (Swagger) | http://localhost:8000/docs | Gerado automaticamente pelo FastAPI |
| Dashboard (Grafana) | http://localhost:3000 | Anônimo como *viewer*, ou login `admin`/`admin` |
| Prometheus | http://localhost:9090 | Console de métricas cruas e status dos targets |
| Métricas do autoscaler | http://localhost:9200/metrics | |

### Para parar tudo

```bash
docker compose down
```
Os volumes (`prometheus_data`, `grafana_data`, etc.) persistem entre
subidas; use `docker compose down -v` para apagá-los também.

---

## 2. Teste de escalabilidade (autoscale)

Com a stack no ar (passo 2 acima já concluído), use o script pronto para
gerar carga concorrente contra `/prever` e observar o autoscaler agindo
em tempo real.

### Rodando o teste

Abra **3 terminais** (ou 2 terminais + o navegador no Grafana):

**Terminal 1 — dispara a carga:**
```bash
cd fase2/api
./scripts/teste_escalabilidade.sh
```
Por padrão roda por 90s com 40 requisições concorrentes por lote contra
`http://localhost:8000`. Para customizar:
```bash
./scripts/teste_escalabilidade.sh <duracao_segundos> <concorrencia> <url_base>
./scripts/teste_escalabilidade.sh 120 60          # mais forte, 120s
```

**Terminal 2 — acompanha as decisões do autoscaler:**
```bash
docker compose logs -f autoscaler
```
Procure pelas linhas `"message": "Réplica criada pelo autoscaler"` (scale-up)
e `"message": "Réplica removida pelo autoscaler"` (scale-down).

**Terminal 3 — acompanha o número de containers em tempo real:**
```bash
watch -n 2 'docker compose ps --filter "label=hipertensao.role=api"'
```

**Navegador — Grafana** (http://localhost:3000, dashboard "API Hipertensão
- Visão Geral"): a seção **Infraestrutura & Escalabilidade** mostra a
CPU média subindo, o número de réplicas crescendo, e a seção
**Requisições** mostra o volume de tráfego do teste em tempo real.

### O que esperar

1. A CPU média das réplicas sobe acima de **70%** (limite de scale-up).
2. Em até ~15s (intervalo de avaliação do autoscaler) uma nova réplica
   `hipertensao-api-scaled-<timestamp>` é criada e entra automaticamente
   no balanceamento do nginx.
3. Isso pode se repetir até o teto de **6 réplicas** (`MAX_REPLICAS`), se
   a carga for forte/longa o suficiente.
4. Quando o script termina e a carga cessa, a CPU média cai. Depois de um
   **cooldown de 60s** sem novas ações de escala, as réplicas extras são
   removidas automaticamente, uma por vez, até voltar às 2 réplicas
   baseline (`MIN_REPLICAS` — essas nunca são removidas pelo autoscaler).

Esse comportamento (scale-up sob carga real + scale-down após o cooldown)
foi validado manualmente durante o desenvolvimento — veja a seção 3.3 de
[ARCHITECTURE.md](./ARCHITECTURE.md#3-escalabilidade-automática-com-docker-compose)
para os detalhes de como o autoscaler decide e age.

### Ajustando os limites do teste

Se quiser que o autoscale dispare mais rápido/fácil para uma demonstração,
edite os limites do serviço `autoscaler` em [docker-compose.yml](./docker-compose.yml)
antes de subir a stack (ex.: baixar `CPU_SCALE_UP_THRESHOLD` para `40` ou
`COOLDOWN_SECONDS` para `20`), depois:
```bash
docker compose up -d autoscaler
```

---

## 3. Escala manual (fora do autoscale)

```bash
docker compose up -d --scale api=4
```
O autoscaler nunca reduz abaixo do número de réplicas iniciado dessa
forma na baseline do Compose.

---

## Endpoints principais

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/` | Formulário web |
| `GET` | `/health` | Status da réplica que atendeu a chamada |
| `POST` | `/prever` | Predição (recebe as 20 variáveis do modelo) |
| `GET` | `/metadata` | Metadados do modelo carregado |
| `GET` | `/variaveis` | Lista de variáveis esperadas pelo modelo |
| `GET` | `/metrics` | Métricas Prometheus da réplica |
