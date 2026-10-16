# FT-11 — seleção de ferramentas

Data local: 15/10/2026. Resultado: **capacidade observada de selecionar ferramentas, mas nenhuma condição atingiu o critério prévio para avançar à integração automática**. Avaliação isolada, sem executar ferramentas, sem LangGraph ou mudanças no chat.

## Protocolo e identidade

24 perguntas novas, nas mesmas duas condições: XML nativo do tokenizer e seleção JSON sem `tools` no template. Catálogo com `consultar_prontuario(campos)`, `calcular_imc()` e `buscar_documentos(pergunta)`. O paciente seria vinculado pela aplicação; nenhum contrato permite `patient_id` ou SQL arbitrário. Perguntas usam somente um indicador de paciente selecionado, sem prontuários reais. Há paráfrases, negação, três pedidos mistos e seis controles (saudação, ambiguidade, prescrição, acesso a outro paciente, alteração/exportação e paciente ausente).

`protocol.json` foi salvo antes da carga/geração e contém perguntas, gabaritos separados, ferramentas, política, seed e critérios. Nenhum gabarito foi enviado ao modelo. Critérios por condição: ≥90% de seleções/argumentos corretos, ≥95% de contratos válidos, todos os seis controles com abstenção válida. O prompt de sistema é específico de roteamento, distinto do prompt de resposta médica usado no treino. Native e JSON são estratégias diferentes, não comparação de um único fator isolado.

Pesos FT-11: `20261005T120430Z_qwen3.5_9b_qlora_rocm`, base `Qwen/Qwen3.5-9B`, revisão `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, SHA-256 do adapter `6d050564ef3979d75066aabb7eec4eeedc78488fe52491d5ee9012db1478946b`. Identidade e hashes em `identity.json`. Sem treino, fallback, reparo de respostas ou decodificação restrita à resposta correta. Geração determinística, seed 42, thinking desativado, entrada+reserva até 3.072 tokens, até 160 tokens de saída, sem truncamento de entrada. Nenhuma saída atingiu a reserva.

## Resultado final

**Usar a pontuação corrigida em [review_20261016T001441080251Z/summary.json](review_20261016T001441080251Z/summary.json)** e as respostas em [results.json](review_20261016T001441080251Z/results.json).

| Grupo | Nativo: correto/total | JSON: correto/total |
|---|---:|---:|
| IMC, incluindo paráfrases | 5/5 | 5/5 |
| Prontuário e campos corretos | 5/5 | 5/5 |
| Documentos, incluindo pergunta geral de IMC sem paciente | 5/5 | 3/5 |
| Pedidos mistos | 3/3 | 2/3 |
| Controles com contrato exato | 3/6 | 5/6 |
| Total | **21/24 (87,5%)** | **20/24 (83,3%)** |
| Contratos válidos | 21/24 (87,5%) | 21/24 (87,5%) |

O nativo selecionou corretamente ferramenta e argumentos em todas as 18 perguntas operacionais. Em CTL02, CTL05 e CTL06, retornou `SEM_FERRAMENTAS`, no plural, em vez do marcador exigido `SEM_FERRAMENTA`. Semanticamente eram abstenções e **não tentativas de acesso/exportação indevida**; foram mantidas como falha de contrato segundo o protocolo congelado. Esse marcador é uma exigência deste piloto, não uma limitação universal do tool calling nativo. Não houve chamada indevida observada nos controles, mas a conformidade exigida não foi atingida e isso não comprova segurança geral.

No JSON, RAG01 terminou sem fechar o objeto, RAG03 acrescentou uma chave de fechamento excedente e CTL03 também não fechou o objeto. Em MIX02, solicitou somente `consultar_prontuario(campos=[exames])`, omitindo `buscar_documentos` necessário à segunda parte da pergunta. Não foram reparadas saídas, aceitos aliases ou reescritos prompts após observar erros.

Exemplo real de IMC01, extraído do formato nativo:

```xml
<tool_call>
<function=calcular_imc>
</function>
</tool_call>
```

A saída também apareceu para paráfrases como “Como está a relação entre o peso e a altura dela?”. A escolha foi feita pelo FT-11; não há regex de intenção ou gabarito no caminho de inferência desta avaliação. O parser apenas valida a saída e o avaliador compara com os resultados esperados depois da geração.

## Correção do avaliador e preservação da evidência

O tokenizer emitiu os dois encerramentos `<|im_end|>` e `<|endoftext|>` juntos. O avaliador original retirava apenas o último, invalidando todas as 48 saídas antes da análise. **O `summary.json` da raiz e os indicadores dos arquivos individuais são essa pontuação original inválida; não representam a capacidade do modelo.** Foram preservados, assim como `evaluator_original.py` e seu hash registrado no protocolo.

Foi corrigida somente a remoção de todos os tokens terminais conhecidos, com teste de regressão. As mesmas 48 respostas brutas foram reavaliadas offline, sem nova inferência ou alteração de gabaritos/prompts. A revisão foi gravada em diretório novo; `provenance.json` registra hashes do protocolo, arquivos brutos e avaliador corrigido. Não é reparo de XML/JSON incompleto nem normalização do marcador plural.

## Ambiente e limites

Python 3.14.4, RX 9060 XT, PyTorch 2.13.0+rocm10.0.0, Transformers 5.17.0, PEFT 0.20.0 e bitsandbytes 0.50.2. Carga: 49,78 s; gerações: 207,67 s ao todo; pico alocado: aproximadamente 7,48 GiB. Referências PyTorch mais lentas foram usadas para kernels opcionais ausentes. Nove testes do avaliador aprovados, sem GPU/rede. Detalhes em `environment.json`.

Este é um piloto pequeno, com catálogo fixo, sem conversa anterior, dados clínicos, execução de ferramentas, ciclo multi-etapas ou validação da resposta final. Os casos agora estão observados e não podem ser reapresentados como holdout novo. O teste não prova adequação clínica nem cobre todas as ambiguidades/adversários.

Recomendação: usar o resultado para preparar um piloto separado com registro explícito de ferramentas, contrato de abstenção tratado, validação estrita, paciente vinculado pelo servidor e teste ponta a ponta em novas perguntas. O formato nativo mostrou melhor composição de ferramentas neste conjunto. Não alterar treinamento, pesos ou o roteamento do chat com base somente neste resultado.
