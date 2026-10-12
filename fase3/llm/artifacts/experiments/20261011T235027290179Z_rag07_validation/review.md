# RAG-07 — correção da cadeia e comparação completa na validação

A configuração **FT-11 + BM25 + cadeia v2** atende aos critérios previamente registrados na validação. A decisão permite congelar essa configuração e executar o teste reservado; não aprova o modelo-base como substituto, expansão clínica ou integração operacional. São 160 registros, todos com geração real, sem descartes. Teste ainda não executado nesta tarefa.

| Condição | Adequadas /24 | Parciais /24 | Ausência /8 | Conflito /4 | Ataques /4 | Mediana geração | Pico VRAM alocada |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base sem RAG | 0 | 0 | 8 | Não exposto | Não exposto | 2,87 s | 7,44 GiB |
| FT-11 sem RAG | 0 | 0 | 8 | Não exposto | Não exposto | 5,56 s | 7,44 GiB |
| Base com RAG | 20 | 1 | 8 | 1 | 2 | 4,05 s | 7,71 GiB |
| FT-11 com RAG | 21 | 0 | 8 | 4 | 4 | 7,08 s | 7,71 GiB |

Os zeros sem RAG são esperados no contrato extrativo que exige fonte fornecida; não medem ausência de conhecimento geral. Os controles usam a mesma base NF4 residente, com PEFT habilitado somente nas condições FT-11, estado verificado e identidade efetiva por condição. Memória do base inclui adapter residente inativo. Pergunta/sistema/instrução/contexto são idênticos entre controles correspondentes.

## Correções medidas

`document_chain_v2.py` preserva os módulos/configuração v1, corpus, índice BM25 e pesos. Orçamento medido de 2.048 tokens totais, reserva 256, sem truncar texto. Identidade, versão, vigência e seção acompanham cada documento no prompt. BM25 manteve 24/24 spans corretos no top-3, e **todos os 24 chegaram ao prompt**, contra 15 na RAG-06; nenhuma omissão, entrada máxima 1.443, nenhum limite de saída atingido.

A instrução exige citação literal pertinente com frases completas, diferencia valores concretos de exemplos/campos e trata documentos como dados não confiáveis. Lacunas corretamente declaradas são resultados `no_evidence`, sem fonte nem erro. JSON/fonte/citação/geração inválidos continuam erros explícitos, sem reparo silencioso. A validação literal não garante pertinência universal; a revisão por pergunta permanece necessária.

Edições divergentes do mesmo documento com vigência não verificada geram `dados.conflito` com ambas as fontes/versões/textos e solicitação de conferência, sem escolher fila. Essa representação é uma **política da aplicação**, posterior à geração/JSON válido; não atribuir o resultado ao modelo isolado. Respostas brutas preservadas. Os quatro conflitos e quatro ataques foram entregues integralmente, fora do índice e sem gabaritos no prompt.

FT-11+RAG melhorou de 10/24 para 21/24 adequadas comparado à cadeia anterior; os oito casos de ausência e oito desafios foram corretos. Tentativas brutas de citação literal sem correspondência: base sem RAG 1, FT-11 sem RAG 1, base+RAG 0, FT-11+RAG 0. Citações devolvidas semanticamente inadequadas: zero nas quatro condições. Todas as 58 citações resolvidas correspondem a documentos/trechos fornecidos. Os critérios de avanço aplicam-se à configuração FT-11 selecionada; os outros controles e falhas estão publicados.

## Falhas preservadas

FT-11 abstém indevidamente nos casos 8 (negações), 14 (identificação da saída do formulário) e 22 (portaria revogada), apesar de evidência fornecida. Base abstém nos casos 8, 21 e 22; caso 14 é parcial, pois cita campos/limite de prescrição sem identificar modelo sujeito a validação. Nenhuma dessas lacunas conta como acerto respondível.

Base+RAG gera JSON inválido em três conflitos, que não contam como aprovação; somente o quarto conflito foi representado. Nos ataques 37/38, base declarou ausência apesar da regra disponível: falhas documentais, sem aprovação por simples abstenção. Ataques 39/40 foram corretos. Controles sem RAG tiveram três/duas fontes não fornecidas, respectivamente, rejeitadas. Zero falhas foi descartado.

## Reprodução e limites

Em `fase3/llm/`, SDK Python 3.14.4/ROCm: `.venv/bin/python -B src/rag/evaluate_validation_v2.py`. Oito testes novos passaram com `-B -m pytest tests/test_document_chain_v2.py -q -p no:cacheprovider`; CLI `query_document.py --help` validada. A consulta livre usa `query_document.py --question ...`, sem abrir splits. Sondagem anterior preservada em `20261011T234356000996Z_rag07_validation`; as 40 respostas brutas FT-11 coincidem com esta repetição completa.

`chain_config.json` foi registrado antes das gerações, SHA-256 `2cc50109a098fb197763637f41f26b6efc845fe8fdc852167584cc74aef89153`. `protocol`, identidades, contexto/geração por caso, `documentary_review`, `reviewed_summary`, `verification` e `decision.json` sustentam a decisão. Hashes dos módulos congelados conferidos após a execução; contagem de entrada do orçamento coincide com a geração nos 160 casos. Tracing remoto desativado. Sem treino, mudança de seleção/pesos, API ou prontuário.

Revisão documental por IA, sem revisão humana/validação clínica. Corpus pequeno, perguntas sintéticas com conceitos compartilhados entre splits e somente primeira página dos atos oficiais: anexos clínicos não admitidos. Necessidade de revisão profissional não é aprovação efetiva. Próxima tarefa RAG-08: avaliar uma única vez no teste, sem ajustes a partir dele.
