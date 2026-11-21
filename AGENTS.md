# AGENTS.md - Guia para Desenvolvimento com Devin

## Regra principal

Implemente SOMENTE o que foi explicitamente solicitado pelo usuário.

O objetivo é realizar a menor alteração possível para atender ao requisito.

## Escopo

- Não implemente funcionalidades não solicitadas.
- Não faça refatorações que não sejam necessárias para a tarefa.
- Não altere arquivos que não sejam necessários.
- Não renomeie classes, métodos, variáveis ou arquivos sem necessidade.
- Não altere arquitetura ou estrutura do projeto sem solicitação explícita.
- Não atualize dependências sem necessidade direta para a tarefa.
- Não adicione bibliotecas ou frameworks sem necessidade direta.
- Não faça melhorias de código apenas porque encontrou uma oportunidade.
- Não corrija outros bugs encontrados incidentalmente.
- Não altere comportamento existente que não esteja relacionado à tarefa.

## Testes

- Execute apenas os testes necessários para validar a alteração.
- Não crie uma grande suíte de testes para uma alteração simples.
- Não altere testes existentes apenas para fazê-los passar, exceto quando o comportamento esperado realmente mudou.
- Se encontrar uma falha não relacionada à tarefa, não tente corrigi-la.

## Documentação

Não crie ou atualize documentação, comentários, README, changelog ou exemplos,
a menos que isso seja necessário para a tarefa solicitada.

## Formato da implementação

Prefira:
1. menor número possível de arquivos alterados;
2. menor quantidade possível de código novo;
3. reutilização do código existente;
4. solução simples e direta.

Evite:
- abstrações prematuras;
- generalizações para casos futuros;
- refatorações oportunistas;
- mudanças arquiteturais;
- otimizações não solicitadas.

## Antes de alterar o código

Primeiro entenda o código existente e identifique o menor ponto de alteração capaz de resolver o problema.

Não implemente nada adicional apenas porque parece uma melhoria.

## Critério de conclusão

A tarefa está concluída quando o requisito solicitado estiver funcionando
e tiver sido validado.

Depois disso, PARE.

Não procure novas melhorias para implementar.

## Quando encontrar algo adicional

Se encontrar uma melhoria, bug, refatoração ou possível problema fora do escopo:

- não implemente;
- mencione brevemente no final;
- deixe para uma tarefa futura.

## Regra de ouro

"Não faça o que pode ser feito.
Faça apenas o que precisa ser feito."

---

## Visão Geral do Projeto

Projeto FIAP Tech Challenge organizado em fases independentes: triagem de hipertensão com machine learning (Fase 1), otimização genética (Fase 2) e protótipo de assistente médico com LLM (Fase 3).

**Contexto:** Este é um projeto da Pós-Tech em IA para Desenvolvedores da FIAP (Faculdade de Informática e Administração Paulista).

## Estrutura de Diretórios

```
FIAP/
├── fase1/                    # API de Triagem (baseline)
│   ├── src/techchallenge_fase1/
│   │   ├── api_modelo.py     # API FastAPI principal
│   │   └── formulario.html   # Interface web
│   ├── modelo_api/           # Modelo treinado e metadados
│   ├── data/                 # Dados de treinamento (vigitel.csv, dicionário, variáveis)
│   ├── requirements.txt
│   └── Dockerfile
│
├── fase2/                    # Otimização Genética + API escalável
│   ├── tech_challenge_fase2.py  # Programa principal
│   ├── src/
│   │   ├── genetic/          # Algoritmo genético (chromosome, fitness, optimizer)
│   │   ├── data.py           # Processamento de dados
│   │   ├── experiment.py     # Configuração de experimentos
│   │   ├── modeling.py       # Modelagem ML
│   │   └── reporting.py      # Geração de relatórios
│   ├── configs/              # Configurações de experimentos (a, b, c)
│   ├── data/                 # Dados de treinamento da fase2
│   ├── tests/                # Testes unitários
│   ├── scripts/              # Scripts executáveis
│   ├── api/                  # API v2.0.0 escalável
│   │   ├── src/app/api_modelo.py
│   │   ├── docker-compose.yml
│   │   ├── nginx/            # Load balancer
│   │   ├── autoscaler/       # Autoscaler CPU-based
│   │   └── monitoring/       # Prometheus + Grafana
│   └── requirements.txt
│
├── resultados/               # Outputs dos experimentos
└── fase3/                    # Assistente médico LLM: protótipo acadêmico
    ├── PROPOSTA.md           # Requisitos do Tech Challenge
    ├── README.md            # Objetivos, estrutura, execução e estado atual
    ├── AGENTS.md            # Orientações específicas desta fase
    ├── data/                # Prontuários e revisões SQLite locais
    ├── llm/                 # Dados, treino, RAG, pesos e evidências
    ├── src/medical_assistant/ # Aplicação, contexto, tools, segurança e interfaces
    ├── tests/               # Testes da aplicação e interface
    └── docs/
        └── PLANO_EXECUCAO.md # Tarefas, dependências, critérios e evidências
```

## Como Executar

### Fase 1: API Baseline
```bash
cd fase1
pip install -r requirements.txt
python src/techchallenge_fase1/api_modelo.py
# Acesse http://localhost:8000
```

### Fase 2: Otimização Genética
```bash
cd fase2
pip install -r requirements.txt
python tech_challenge_fase2.py
```

### Fase 2: Testes
```bash
cd fase2
python -m pytest tests/
python scripts/smoke_test.py
```

### Fase 2: API Escalável (Docker Compose)
```bash
cd fase2/api
docker compose up -d --build
# API: http://localhost:8000
# Grafana: http://localhost:3000
```

### Fase 3: Assistente Médico LLM

Consulte os comandos e as limitações atuais em [fase3/README.md](fase3/README.md).
Antes de alterar essa fase, leia [fase3/AGENTS.md](fase3/AGENTS.md) e selecione a tarefa no [plano de execução](fase3/docs/PLANO_EXECUCAO.md).

## Dependências Principais

### Fase 1
- fastapi==0.111.0
- uvicorn==0.30.1
- scikit-learn==1.5.0
- pandas
- joblib

### Fase 2
- scikit-learn>=1.5,<2
- pandas>=2.2,<3
- numpy>=1.26,<3
- matplotlib>=3.8,<4
- joblib>=1.4,<2

### Fase 2 API
- fastapi==0.111.0
- uvicorn==0.30.1
- scikit-learn==1.5.0
- prometheus-client==0.20.0

### Fase 3

Consulte [fase3/requirements.txt](fase3/requirements.txt) e as instruções de ambiente no [README da fase](fase3/README.md). Uma dependência declarada não comprova sua integração no código.

## Fluxo de Dados

### Fase 1 e 2
1. Fase 1: Dados de treinamento em `fase1/data/` (vigitel.csv, dicionário, variáveis)
2. Fase 1: Modelo Random Forest treinado salvo em `fase1/modelo_api/modelo_hipertensao_api.joblib`
3. Fase 2: Lê metadados do modelo da Fase 1 + configurações de `fase2/configs/`
4. Fase 2: Executa 3 configurações do algoritmo genético
5. Fase 2: Gera modelo_genetico_vencedor.joblib
6. Fase 2 API: Copia modelo vencedor para deploy escalável

### Fase 3

O fluxo implementado inclui preparação/avaliação e fine-tuning FT-11, integrado ao assistente com LangChain/LCEL, LangGraph, contexto SQLite, fontes, revisão persistente e auditoria. A implementação atual e as etapas pendentes estão discriminadas no [README](fase3/README.md) e no [plano](fase3/docs/PLANO_EXECUCAO.md).

## Comandos Úteis

### Ver estrutura do projeto
```powershell
Get-ChildItem -Recurse -Depth 2
```

### Rodar testes específicos
```bash
cd fase2
python -m pytest tests/test_genetic.py
python -m pytest tests/test_data.py
python -m pytest tests/test_integration.py
```

### Ver resultados dos experimentos
```powershell
Get-ChildItem fase2/resultados
```

## Padrões e Convenções

### Fases 1 e 2
- Python como linguagem principal
- FastAPI para APIs
- scikit-learn para ML
- Estrutura modular com separação de responsabilidades
- Configurações externalizadas em JSON
- Testes unitários em Fase 2
- Docker para deploy da API escalável
- Logging estruturado e monitoramento (Prometheus/Grafana)

### Fase 3

As responsabilidades dos pacotes, os critérios de validação e as regras de atualização do progresso ficam em [fase3/AGENTS.md](fase3/AGENTS.md). Preserve a distinção entre estrutura planejada, código existente e comportamento comprovado.

## Arquivos Importantes

### Fase 1
- `fase1/modelo_api/metadata_modelo_api.json` - Metadados do modelo baseline

### Fase 2
- `fase2/api/modelo_api/metadata_modelo_api.json` - Metadados do modelo otimizado
- `fase2/configs/experimento_*.json` - Configurações do algoritmo genético
- `fase2/api/ARCHITECTURE.md` - Arquitetura da API escalável

### Fase 3

- [fase3/PROPOSTA.md](fase3/PROPOSTA.md) - Enunciado e requisitos acadêmicos
- [fase3/README.md](fase3/README.md) - Objetivos, estrutura e estado atual
- [fase3/AGENTS.md](fase3/AGENTS.md) - Orientações para desenvolvimento da fase
- [fase3/docs/PLANO_EXECUCAO.md](fase3/docs/PLANO_EXECUCAO.md) - Ordem das tarefas, critérios de aceite e registro de progresso

### Documentação Geral
- `STRUCTURE.md` - Documentação detalhada da estrutura

## Ambientes Virtuais

- `.venv/` já existe na raiz do projeto
- Cada fase tem seu próprio requirements.txt
- Use `pip install -r requirements.txt` em cada fase conforme necessário

## Integração entre Fases

### Fase 1 (API de Triagem)
- Localização: `fase1/`
- Modelo baseline: Random Forest para triagem de hipertensão
- Integração: A Fase 3 pode usar os mesmos dados Vigitel como contexto adicional

### Fase 2 (Otimização Genética)
- Localização: `fase2/`
- Modelo otimizado: Algoritmo genético para hiperparâmetros
- Integração: A Fase 3 pode usar os metadados do modelo otimizado para contextualização

### Fase 3 (Assistente Médico LLM)
- Localização: `fase3/`
- Esta fase é independente mas complementar
- Objetivo: assistente médico acadêmico com modelo adaptado e fluxos assistidos
- Integração com resultados das fases anteriores é uma possibilidade futura, não uma dependência da entrega atual

## Observações Importantes

### Fases 1 e 2
- Fase 1 é o baseline histórico e permanece inalterada
- Fase 2 é independente e gera modelo otimizado
- Fase 2 API é um deploy escalável do modelo vencedor
- Cada fase tem seus próprios dados e configurações

### Fase 3

- Protótipo acadêmico com integração validada nos casos registrados; isso não comprova prontidão para produção.
- Benchmark de inferência é uma etapa da seleção; a escolha para treino também exige verificar licença, pesos treináveis e recursos disponíveis.
- Recuperação de contexto (RAG), configuração de treinamento e respostas simuladas não comprovam fine-tuning executado.
- Segurança, fontes e auditoria precisam ser validadas no fluxo real de resposta. Não declarar conformidade ou segurança total com base apenas em filtros ou anonimização.
- O estado detalhado e as evidências devem ser mantidos nos documentos locais da Fase 3, evitando repetir resultados nesta raiz.

## Solução de Problemas

### Fase 1 e 2
- Verificar dependências instaladas com `pip list`
- Verificar portas disponíveis (8000, 3000)
- Verificar logs de aplicação

### Fase 3

Consulte as limitações no [README](fase3/README.md) e registre a evidência na tarefa correspondente do [plano](fase3/docs/PLANO_EXECUCAO.md). Uma demonstração que imprime mensagens não substitui os critérios de aceite da tarefa.

## Desenvolvimento

### Fase 1
```bash
cd fase1
pip install -r requirements.txt
python src/techchallenge_fase1/api_modelo.py
```

### Fase 2
```bash
cd fase2
pip install -r requirements.txt
python tech_challenge_fase2.py
python -m pytest tests/
```

### Fase 3

Execute uma tarefa por vez seguindo [fase3/AGENTS.md](fase3/AGENTS.md) e [fase3/docs/PLANO_EXECUCAO.md](fase3/docs/PLANO_EXECUCAO.md). Conclua cada tarefa com validação e evidências. Uma solicitação explícita do usuário para concluir várias etapas autoriza executá-las em sequência. As responsabilidades de cada arquivo Markdown e regras para relatórios históricos estão em [fase3/AGENTS.md](fase3/AGENTS.md).

## Contato e Suporte

### Fases 1 e 2
- Verificar logs de aplicação
- Revisar configurações em JSON
- Consultar documentação específica de cada fase

### Fase 3

Use o [README da fase](fase3/README.md) para orientação e o [plano de execução](fase3/docs/PLANO_EXECUCAO.md) para pendências, decisões e evidências. Relatórios históricos devem ser confrontados com os artefatos e validações registrados.

## Status do Projeto

### Fase 1
**Status:** ✅ COMPLETO
- API de triagem funcional
- Modelo Random Forest baseline
- Documentação completa

### Fase 2
**Status:** ✅ COMPLETO
- Algoritmo genético implementado
- API escalável com Docker
- Monitoramento Prometheus/Grafana
- Testes unitários

### Fase 3

**Status:** PROTÓTIPO ACADÊMICO — desenvolvimento do fluxo implementado; entrega humana pendente.

Há FT-11 treinado com evidências, integração LangChain/LCEL com documentos e fatos do prontuário, SQLite, interface e LangGraph compilado, com revisão de rascunhos persistida por decisão explícita. É um protótipo acadêmico sem autenticação profissional nem validação clínica; vídeo/apresentação dependem de participação humana. Consulte o [README](fase3/README.md) para o inventário atual e o [plano de execução](fase3/docs/PLANO_EXECUCAO.md) para tarefas concluídas e pendentes.
