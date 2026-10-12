# RAG-09 — validação final da cadeia v3, 12/10/2026

160 registros novos, quatro condições, orçamento 3.072/reserva 256, mesmos pesos FT-11/base e mesmo corpus/índice. Todos os critérios congelados passaram; `decision.json` autoriza somente o reservado v4.

| Condição | Completas respondíveis | Parciais | Ausências | Conflitos | Instruções hostis |
|---|---:|---:|---:|---:|---:|
| FT-11 + RAG | 22/24 | 2 | 8/8 | 4/4 | 4/4 |
| Base + RAG | 22/24 | 2 | 8/8 | 4/4 | 4/4 |
| FT-11 sem RAG | 0/24 | 0 | 8/8 | 0/4 | 0/4 |
| Base sem RAG | 0/24 | 0 | 8/8 | 0/4 | 0/4 |

As 160 respostas e fontes recém-geradas são iguais às já revisadas da validação anterior `20261012T122108946656Z_rag09_validation`. Anotações semânticas reutilizadas somente após conferir essa igualdade, sem reutilizar gerações: casos 14 e 18 permanecem parciais, e 19–24 citam parágrafos pertinentes de páginas inteiras usadas como referência. Respostas/citações/erros e revisão por caso permanecem disponíveis.

Nenhuma omissão, erro, fonte inexistente ou citação final sem suporte; todos os 24 spans e todos os desafios com RAG foram integralmente entregues. `integrity_verification.json` confere módulos atuais/snapshots, índice, identidade dos pesos, offsets, estados PEFT e orçamento. Foram 160 gerações de seleção e 92 de conferência de suporte. Entrada máxima 1.784, pico VRAM alocada 7,91 GiB. Medianas por pergunta, sem carga do modelo: FT+RAG 8,31 s, base+RAG 5,73 s, FT sem RAG 5,43 s, base sem RAG 2,80 s. 38 testes de software aprovados no SDK Python 3.14.4/ROCm.

A primeira avaliação reservada da v3 reprovou por uma omissão de distrator em 2.048 tokens; sua decisão negativa e dados consumidos estão preservados em `20261012T170701149180Z_rag09_test`. O orçamento novo foi medido sobre todos os 455 trios elegíveis, com pergunta genérica fora da avaliação, máximo de 2.560 tokens de entrada. A validação intermediária `20261012T172949190191Z_rag09_validation` foi interrompida/preservada para corrigir rejeição de documento suplementar legítimo reproduzida por teste de software com nova pergunta de validação. Fontes catalogadas de outro protocolo continuam rejeitadas; suplementos explícitos exigem resolução literal e conferência semântica. Critérios de aceite não foram relaxados.

FT-11/base usam seus próprios estados na seleção; a conferência de suporte é comum na mesma base quantizada sem adapter. As regras de conflitos/diretivas e a decodificação JSON são partes do pipeline. O empate não demonstra superioridade do FT-11 nem resistência do modelo isolado. Zero sem RAG decorre do modo extrativo sem fontes fornecidas, sem medir conhecimento geral.

Novo reservado v4 fixado antes da revisão do orçamento: 40 paráfrases dos mesmos conceitos/corpus e famílias de desafios herdadas. Não demonstra independência temática ou clínica. Os testes RAG-02 e reservado v3 consumidos não foram reutilizados. Corpus oficial limitado às páginas de portaria admitidas, sem anexos clínicos. Revisão documental por IA, sem aprovação profissional. Pesos/seleção e decisões anteriores preservados; API/prontuário/LangGraph fora desta preparação.

Comando a partir de `fase3/llm/`: `.venv/bin/python -B src/rag/evaluate_v3.py --conditions ft11_with_rag base_with_rag ft11_without_rag base_without_rag`. O gate do teste exige revisão/decisão positivas, configuração e hashes iguais antes de abrir o conjunto reservado; registros e decisões usam criação exclusiva.
