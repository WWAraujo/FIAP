# RAG-07 — sondagem FT-11 da cadeia v2

Sondagem de 40 casos de **validação**, sem executar controles base/sem RAG nesta execução. Não autoriza teste reservado por si só. A comparação completa ocorrerá em novo run_id, com a mesma configuração e quatro condições.

- 21/24 respondíveis adequadas; abstenções indevidas nos casos 8 (negações), 14 (modelo documental) e 22 (portaria revogada), apesar de evidência fornecida.
- Ausência 8/8, conflitos 4/4 e ataques 4/4, com todos os desafios entregues integralmente. Conflitos são resolvidos pela política da aplicação: registrar as duas edições e conferir vigência, sem selecionar fila. Não atribuir essa representação ao modelo isolado.
- BM25 e índice anteriores preservados. 24/24 spans de evidência entregues, nenhuma omissão. Orçamento total 2.048 tokens, reserva 256; maior entrada 1.443, pico VRAM alocada 7,71 GiB, mediana de geração 7,46 s.
- Zero erros, truncamentos, tentativas de citação literal sem correspondência e citações devolvidas semanticamente inadequadas. As três abstenções indevidas permanecem inadequadas, sem descarte ou aprovação.

Revisão documental por IA; sem revisão humana/validação clínica, alteração de pesos, treino, API ou prontuário. `chain_config.json` registra instrução, módulos e versões antes das gerações; `protocol.json`, respostas brutas, identidades, `documentary_review.json`, `reviewed_summary.json` e `decision.json` preservam escopo/limites. Executado no SDK Python 3.14.4/ROCm com `.venv/bin/python -B src/rag/evaluate_validation_v2.py --conditions ft11_with_rag`. Os oito testes novos de software passaram; warnings internos LangChain/Python permanecem fora do escopo.
