# RAG-09 — primeiro teste da v3, 12/10/2026

160 registros completos, quatro condições, mesmos pesos/configuração da validação anterior. FT-11+RAG e base+RAG: 23/24 completas, uma parcial (14), 8/8 ausências, 4/4 conflitos e 4/4 instruções hostis. Todos os 24 spans esperados entregues; nenhum erro, fonte inexistente ou citação final sem suporte. Sem RAG: 0/24 completas no modo extrativo sem documentos, 8/8 ausências; não mede conhecimento geral.

**Aceite reprovado:** uma omissão por orçamento em ambas as condições com RAG, caso 32. O terceiro candidato, página de dislipidemia, não cabe nos 2.048 tokens após duas FAQs. É um distrator para vacinação, e a lacuna foi respondida corretamente; ainda assim o gate congelado exige zero omissões e não será relaxado após o teste. `decision.json` mantém `rag_preparation_accepted: false`. Não foi criado recibo de prontidão nem liberada a CLI.

Medianas totais por pergunta, excluindo carga: base sem RAG 2,77 s; FT-11 sem RAG 5,24 s; base+RAG 5,66 s; FT-11+RAG 8,05 s. Pico VRAM alocada 7,91 GiB com RAG. Anotações semânticas, respostas brutas, código/configuração congelados, resumo/revisão e decisão preservados. O caso 14 cita uma regra verdadeira sobre campos null e não prescrição, mas omite identificação como modelo sujeito a validação.

Os controles com RAG usam um verificador comum na base sem adapter; empataram. Ganhos pertencem ao pipeline completo, incluindo decodificação restrita e regras da aplicação. A conferência de conflitos é política da aplicação; revisão documental por IA não equivale a aprovação clínica. Os quatro ataques reservados reutilizam famílias da validação e as perguntas identificam explicitamente a instrução indevida. Este é um holdout de paráfrases, sem independência temática/clínica.

O conjunto `data/rag_reserved_v3` está consumido. Próxima revisão corrigirá somente o orçamento, medirá todos os trios de seções elegíveis com o tokenizer real e revalidará antes de abrir outro conjunto reservado. Este resultado não será sobrescrito nem reapresentado como teste independente.
