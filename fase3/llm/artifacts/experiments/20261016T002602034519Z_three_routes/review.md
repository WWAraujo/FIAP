# Demonstração de três caminhos — FT-11

Execução única em 16/10/2026 UTC (15/10 local), sem ajuste de prompt ou reparo de respostas. Protocolo em `protocol.json`, identidade em `identity.json`, resultados brutos nos três JSON e resumo em `summary.json`.

| Caminho | Pergunta | Resultado |
|---|---|---|
| Tool | Confira o índice de massa corporal da paciente selecionada. | Selecionou `calcular_imc` sem argumentos. Adaptador de execução chamou o método existente com sim_001: 28,20 kg/m² e fonte SQLite. Aprovado. |
| Conhecimento | Como é herdada a deficiência congênita de plasminogênio? Responda em uma frase, sem avaliar uma pessoa. | Respondeu sem tool/RAG: “A deficiência congênita de plasminogênio é herdada de forma autossômica recessiva.” Conteúdo esperado; reprovado no contrato: `status="ok"` e `dados` texto em vez de objeto. |
| RAG | Quem assinou a portaria de dislipidemia da base documental? | Selecionou `buscar_documentos`; cadeia v3 retornou Francisco de Assis Figueiredo e Denizar Vianna com fonte documental e offsets literais válidos. Aprovado. |

Três escolhas de caminho adequadas, **2/3 aprovados no contrato completo**. O prompt pediu cinco chaves, mas não especificou todos os valores permitidos de status nem a estrutura interna de dados; o resultado não isola falha do modelo de insuficiência dessa instrução. Não houve normalização para transformar a resposta em aprovação.

A pergunta educativa trata de um exemplo presente no treino (`F2-GHR-0000237-4`), sem enviar a resposta de referência ao modelo. Isso não prova origem exclusiva no fine-tuning. A escolha da tool foi livre, mas a execução usa um adaptador de teste para o método existente; não é agente integrado ao chat. O RAG inclui a seleção FT-11 e o verificador-base já existentes na cadeia v3.

Condição de avanço 3/3 não atendida: nenhum pacote novo de tools nem alteração do chat nesta tarefa. A avaliação anterior permanece preservada. Execução funcional acadêmica, sem validação clínica ou prova de generalização.
