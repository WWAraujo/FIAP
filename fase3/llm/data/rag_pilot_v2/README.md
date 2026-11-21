> Registro da preparação documental RAG-02 (10/10/2026). Expressões como “futura avaliação” e “nenhuma resposta gerada” descrevem essa etapa histórica. A cadeia v3 foi avaliada depois e está integrada; veja [README RAG](../../src/rag/README.md) e [relatório técnico](../../../docs/RELATORIO_TECNICO.md). Não sobrescrever snapshots nem testes consumidos.

# Corpus documental e avaliação — RAG-02

Versão congelada em 10/10/2026, preservando `rag_pilot_v1`. `catalog.json` contém o conteúdo admitido; `manifest.json` contém IDs e hashes; `sources.json` registra os downloads. Dez documentos/versões, nove elegíveis: seis procedimentos sintéticos vigentes, uma versão substituída e três atos oficiais. São 17 seções, 15 elegíveis.

## Admissão das fontes

Os três PDFs completos estão em `snapshots/` e a extração integral em `extracted/`, ambos locais e ignorados pelo Git. **Somente a primeira página de cada PDF foi admitida**, com texto integral da portaria, espaços e quebras preservados conforme `pdftotext -layout` 26.01.0. A revisão documental por IA conferiu identificação, decisão, revogação, assinaturas e entrada em vigor contra o texto publicado no portal. Não houve validação clínica. Os anexos clínicos, tabelas, unidades e negações ainda não revisados permanecem fora do corpus elegível; não afirmar cobertura de diagnóstico ou tratamento por estes PCDTs.

| Snapshot | Páginas | Ato / emissão | Publicação |
|---|---:|---|---|
| Hipertensão | 97 | SECTICS/MS 49, 23/07/2025 | 24/07/2025 |
| Diabete tipo 2 | 80 | SCTIE/MS 13, 21/02/2026 | 24/02/2026 |
| Dislipidemia | 31 | Conjunta 8, 30/07/2019 | 07/08/2019 |

Edições/publicações conferidas no [catálogo oficial da Conitec](https://www.gov.br/conitec/pt-br/assuntos/avaliacao-de-tecnologias-em-saude/protocolos-clinicos-e-diretrizes-terapeuticas/pcdt) em 10/10/2026. A [portaria de diabetes](https://www.gov.br/saude/pt-br/assuntos/pcdt/d/diabete-melito-tipo-2.pdf/view) revoga expressamente a portaria 7/2024: não é apenas recomendação de atualização. A [página de dislipidemia](https://www.gov.br/saude/pt-br/assuntos/pcdt/d/dislipidemia/view) informa anexo alterado em 01/04/2022; sua atualização de página em 2025 não é nova portaria. A consulta pública 94/2026 não foi tratada como aprovação. `current_in_pilot` identifica o snapshot selecionado nessa data, sem promessa de vigência permanente.

O conteúdo admitido é texto de atos oficiais, conforme [Lei 9.610/1998, art. 8º, IV](https://www.planalto.gov.br/ccivil_03/leis/l9610.htm). Atribuição ao Ministério da Saúde/Conitec e URLs mantidas. Esta decisão não atribui a mesma condição a imagens, tabelas e demais materiais dos anexos. O aviso CC BY-ND 3.0 do portal está registrado; não foram feitas traduções ou adaptações clínicas.

## Avaliação reservada

`evaluation/validation.jsonl` e `evaluation/test.jsonl` têm 40 perguntas cada: 24 respondíveis (18 sintéticas e seis sobre atos oficiais), oito sem evidência, quatro conflitos de edição e quatro instruções indevidas em documentos. `evaluation/evaluation_spec.json` preserva os critérios da RAG-01 e congela os hashes do manifesto e de ambos os splits antes de recuperação ou geração.

Cada pergunta tem ID, comportamento esperado, documento/versão/seção, página e trecho literal de referência quando há evidência. Os gabaritos foram revisados documentalmente por IA; sem revisão humana. Os splits compartilham corpus e conceitos, com perguntas distintas: não são holdout por tema nem avaliação clínica independente. A verificação de não reutilização detecta cópia literal nos JSONL anteriores, não equivalência semântica.

Indexar **somente seções elegíveis do manifesto**. Perguntas, gabaritos, especificações e `context_fixture` nunca entram no índice. Os casos especiais fornecem, na futura avaliação da resposta, documentos adversariais identificados: duas edições contraditórias sem vigência conhecida ou texto com instrução maliciosa. Eles avaliam o comportamento diante desse contexto, separado da métrica de recuperação dos 24 casos respondíveis. A fonte limpa de referência acompanha os casos adversariais; registrar também os IDs dos fixtures realmente fornecidos. Nenhuma resposta do modelo foi gerada.

## Reproduzir os downloads locais

Partindo deste diretório, em pastas `snapshots/` e `extracted/` novas/vazias, com `curl` e Poppler instalados:

```bash
mkdir -p snapshots extracted
curl -fL --retry 2 'https://www.gov.br/saude/pt-br/assuntos/pcdt/h/hipertensao-arterial-sistemica.pdf/@@download/file' -o snapshots/has.pdf
curl -fL --retry 2 'https://www.gov.br/saude/pt-br/assuntos/pcdt/d/diabete-melito-tipo-2.pdf/@@download/file' -o snapshots/dm2.pdf
curl -fL --retry 2 'https://www.gov.br/saude/pt-br/assuntos/pcdt/d/dislipidemia/@@download/file' -o snapshots/dislipidemia.pdf
sha256sum -c SHA256SUMS
```

Se algum hash divergir, interromper: o portal pode ter substituído o arquivo. Recuperar o snapshot original de uma cópia preservada ou preparar outra versão com nova revisão; não atualizar silenciosamente os hashes/gabaritos desta versão.

Após hashes aprovados:

```bash
pdftotext -layout snapshots/has.pdf extracted/has.txt
pdftotext -layout snapshots/dm2.pdf extracted/dm2.txt
pdftotext -layout snapshots/dislipidemia.pdf extracted/dislipidemia.txt
```

Hashes de extração em `sources.json`; versões diferentes do Poppler podem alterar espaços. O catálogo já preserva os trechos admitidos e permite conferir o corpus sem baixar PDFs. Nos testes, arquivos locais existentes têm hashes/página conferidos; ausentes não são baixados automaticamente.

Partindo de `fase3/llm/`, com o SDK configurado:

```bash
.venv/bin/python -B src/rag/prepare_manifest.py --catalog data/rag_pilot_v2/catalog.json --output data/rag_pilot_v2/manifest.json --check
.venv/bin/python -B -m pytest tests/test_rag_manifest.py tests/test_rag_corpus.py -q -p no:cacheprovider
```

RAG-03 executada em 11/10/2026: suporte efetivo no top-3 BM25 24/24, E5/CPU 23/24; BM25 congelado para o piloto. [Resultados e limites](../../artifacts/experiments/20261011T212901816585Z_rag03/review.md). Próxima etapa: comprovar acesso ao FT-11 antes da cadeia documental. Os anexos clínicos exigem uma admissão futura em outra versão.
