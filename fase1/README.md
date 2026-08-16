# Tech Challenge - Pós-Tech IA para Devs | Fase 1

## 📁 Estrutura do Repositório

Este projeto faz parte de um repositório maior que contém **Fase 1 e Fase 2**. A estrutura completa está documentada em [../../STRUCTURE.md](../../STRUCTURE.md).

Localização: `./fase1/` na raiz do repositório FIAP.

---

## 🏥 Cenário e Objetivo do Projeto
Este projeto foi desenvolvido como atividade obrigatória para a Fase 1 do Tech Challenge. O objetivo é implementar um sistema inteligente de suporte ao diagnóstico clínico focado em Machine Learning para um grande hospital universitário. 

A solução realiza o processamento de dados médicos para triagem automatizada, permitindo acelerar a análise inicial de exames e apoiar as decisões médicas para otimizar o tempo dos profissionais, garantindo que o(a) médico(a) sempre tenha a palavra final no diagnóstico.

---

## 📊 Base de Dados e Estratégia "Anticola" (Vazamento de Dados)
O modelo utiliza dados históricos da base pública **Vigitel** (Vigilância de Fatores de Risco e Proteção para Doenças Crônicas por Inquérito Telefônico, Ministério da Saúde).

* **Target (Variável Alvo):** `hart` (Diagnóstico médico prévio de hipertensão).
* **Tratamento de Data Leakage:** Durante os ciclos iniciais de desenvolvimento, foi detectado um vazamento de dados clássico (*data leakage*): variáveis como `trat_med_has` (tratamento médico para hipertensão) e `med_has` (uso de remédios para pressão) entregavam a resposta ao modelo antes da hora. O pipeline conta com uma rotina automatizada de auditoria que expurga essas variáveis "intrusas", forçando os modelos a aprenderem padrões clínicos e epidemiológicos reais (como idade, IMC, tabagismo, sedentarismo e hábitos alimentares).

> ⚠️ **Nota sobre o Dataset:** Devido às restrições de tamanho de arquivos do GitHub (limite de 100MB), o arquivo bruto `vigitel.csv` está listado no `.gitignore`. O download do arquivo deve ser feito através do link externo disponibilizado pelo grupo e colado na raiz do diretório antes da execução.
> 🔗 [Insira aqui o link do seu Google Drive/OneDrive onde salvou o vigitel.csv]


---

## 📊 Base de Dados e Estratégia "Anticola" (Vazamento de Dados)

[... Conteúdo atual que o Wallace escreveu sobre o tratamento do Vigitel e o expurgo das colunas de vazamento ...]


---


## 🔬 Experimentos Avançados e Cenários de Simulação

Para além do pipeline tradicional, o repositório conta com uma estrutura de **Laboratório de Dados (Sandbox)** desenvolvida para simular diferentes tomadas de decisão estratégica para o Hospital Universitário. O código foi desacoplado para permitir a execução de três cenários práticos independentes no notebook `tech_challenge.ipynb`:

### 🧪 1. Cópia Segura de Dados e Inserção de Novas Variáveis (Bloco 3.5)
Criamos uma esteira customizada que extrai cópias isoladas das bases de treino e teste através do Pandas para não corromper o histórico do grupo. A partir disso, geramos **4 novos indicadores cruzados** para dar mais contexto clínico ao modelo:
* **Risco Combinado Peso-Idade:** Interação não-linear multiplicando o IMC pela idade do paciente, diferenciando o impacto do peso de acordo com a faixa etária.
* **Score de Síndrome Metabólica:** Indicador cumulativo de co-morbidades, somando o histórico de Diabetes, Colesterol e Excesso de Peso do paciente.
* **Balanço Nutricional:** Indicador contínuo que calcula o saldo real entre hábitos alimentares protetores (frutas, hortaliças) e fatores de risco (doces, ultraprocessados).
* **Índice de Estresse/Sono:** Mapeamento do eixo psicológico e sistema nervoso, cruzando o histórico de depressão com distúrbios de sono crônicos.

### 🗺️ 2. Cenários de Negócio Disponíveis para Execução

A arquitetura do notebook permite alternar a ordem de execução das células para responder a diferentes dores de negócio da instituição de saúde:

| Cenário | Foco Estratégico | Configuração Técnica | Impacto Prático no Hospital |
| :--- | :--- | :--- | :--- |
| **Cenário A: Baseline** | **Segurança Clínica** | Modelos puros, peso `balanced`, corte padrão em 50% | **Recall de 69%**. Prioriza a captação massiva de doentes na triagem para que ninguém saia sem diagnóstico. |
| **Cenário B: Alta Precisão** | **Eficiência Médica** | Novas variáveis, sem peso de classe, corte elevado em 60% | **Precisão de 74%**. Elimina radicalmente os alarmes falsos, otimizando o tempo dos médicos e braçadeiras de triagem. |
| **Cenário C: Impacto Puro** | **Validação Científica** | Novas variáveis, peso `balanced`, corte padrão em 50% | **Evolução do F1-Score**. Isola o teste para provar o ganho estatístico que as novas variáveis trouxeram sozinhas para a Random Forest. |

> 🚀 **Nota de Execução:** No topo do notebook, inserimos um cabeçalho técnico indicando a sequência exata de cliques nas células para carregar e simular cada cenário de forma rápida e independente, economizando tempo de processamento de máquina.

---

## 🛠️ Tecnologias e Dependências
O ecossistema do projeto foi construído utilizando:
* **Python 3.10+**
* **Pandas & NumPy:** Manipulação e filtragem otimizada de dados.
* **Scikit-Learn:** Engenharia de pipelines de pré-processamento de dados em Python, tuning de hiperparâmetros e modelagem preditiva.
* **Matplotlib & Seaborn:** Geração automática de artefatos visuais, distribuições relevantes e curvas de avaliação.
* **Joblib:** Serialização e salvamento físico dos modelos treinados para reuso em produção.

---

## 🚀 Como Executar o Projeto

Deploy do projeto:

### Pré-requisitos:
Ambiente Docker e git.

1. Realizer o clone do projeto.

         git clone https://github.com/MattManhaes/pos-tech-ia-tech-challenge-fase1.git

2. No diretório pos-tech-ia-tech-challenge-fase1, fazer o build do projeto:
   
         docker build -t api-hipertensao-fastapi:latest .

3. Execute o container para rodar o pipeline de Machine Learning:
   
         docker run -d -p 8000:8000 --name meu_modelo_fastapi api-hipertensao-fastapi:latest

---

## 📈 Artefatos e Resultados Gerados
Sempre que o pipeline conclui sua execução, ele gera uma subpasta exclusiva estruturada por data e hora dentro do diretório `resultados_sem_vazamento/`. Lá você encontrará automaticamente:
* **`comparacao_modelos_cross_validation_sem_vazamento.csv`:** Tabela comparativa com as médias estáveis de cada métrica avaliada (accuracy, recall, F1-score).
* **`matriz_confusao_[MODELO].png`:** Matrizes de confusão com rótulos clínicos claros indicando acertos e erros de triagem.
* **`curva_roc_comparativa.png` e `curva_precisao_recall_comparativa.png`:** Gráficos de performance comparando o comportamento da Regressão Logística, Random Forest e Árvore de Decisão.
* **`top20_feature_importance_random_forest_sem_vazamento.png`:** Gráfico mapeando a interpretação dos resultados por relevância das variáveis (feature importance).
* **`modelos_salvos/`:** Os arquivos físicos treinados prontos para deploy clínico imediato.