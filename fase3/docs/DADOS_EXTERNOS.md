# Dados externos — MedQuAD

O corpus externo da Fase 3 é uma amostra de 1.000 pares QA em inglês do [MedQuAD oficial](https://github.com/abachaa/MedQuAD), publicado sob [CC BY 4.0](https://github.com/abachaa/MedQuAD/blob/master/LICENSE.txt). A fonte é composta por pares de perguntas e respostas de sites do NIH; ela não representa protocolos internos brasileiros nem validação clínica local.

## Reprodução

Partindo de `FIAP/`, baixe a mesma revisão e gere a amostra com seed fixa:

```powershell
& .\.venv\Scripts\python.exe -B fase3/src/medical_assistant/data/ingest_medquad.py `
  --revision 577bd37b96c02d1833b2c9eed2de9f96964e96cb `
  --sample-size 1000 `
  --seed 42

& .\.venv\Scripts\python.exe -B fase3/src/medical_assistant/data/split_dataset.py `
  --seed 42 `
  --external-input fase3/data/external/medquad/sample.jsonl `
  --external-manifest fase3/data/external/medquad/manifest.json `
  --external-limit 300
```

Os arquivos baixados ficam em `data/external/medquad/` e permanecem ignorados pelo Git. O manifesto local registra revisão, SHA-256 do arquivo baixado, licença, idioma, seed e contagens.

## Uso e limites

- Atribua MedQuAD e CC BY 4.0 no relatório e no manifesto do treinamento.
- O conteúdo é majoritariamente em inglês. A DAT-05 cria 200 exemplos comportamentais sintéticos em português e limita o MedQuAD a 300 pares na composição, registrando as proporções no manifesto de splits.
- MedQuAD amplia diversidade de QA, mas não autoriza alegar segurança clínica, aderência a protocolos hospitalares ou aptidão para uso médico real.
- PubMedQA permanece candidato de **avaliação**, não entra neste corpus de fine-tuning porque sua tarefa é classificar respostas como `yes/no/maybe` a partir de abstracts.
