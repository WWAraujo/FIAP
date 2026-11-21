# Dados externos — versão portuguesa

> Registro histórico da tentativa MedPT rejeitada. Para reproduzir o treino selecionado FT-10/FT-11, use os corpora versionados e os comandos atuais do [README LLM](../README.md).

O corpus externo da segunda versão é uma amostra determinística de 5.000 pares em português brasileiro do [MedPT](https://huggingface.co/datasets/AKCIT/MedPT), sob [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). A fonte contém perguntas e respostas históricas de interações paciente-profissional. Ela não substitui protocolos internos, nem validação clínica local.

## Reprodução

Partindo de `fase3/llm/`, resolva a revisão uma vez, baixe o snapshot e gere a amostra com seed fixa:

```bash
.venv/bin/python src/finetuning/data/ingest_medpt.py \
  --revision 24aeac298da0f1b46e6e7360eeeb0d36e711779d \
  --sample-size 5000 \
  --seed 42

.venv/bin/python src/finetuning/scripts/train_qwen3_8b_rocm.py --rebuild-data --validate-only
```

Os arquivos baixados ficam em `data/external/medpt/` e permanecem ignorados pelo Git. O manifesto local registra a revisão SHA, hashes dos Parquet, hash da amostra, licença, idioma, seed, filtros e contagens. O corpus resultante terá cerca de 4.000 pares externos no treino e 500 em cada split de validação e teste, além dos exemplos sintéticos em português.

## Uso e limites

- Atribua MedPT e CC BY 4.0 no relatório e no manifesto do treinamento.
- O filtro remove os tipos explicitamente rotulados como diagnóstico ou tratamento e deduplica perguntas normalizadas. Os 225 exemplos sintéticos de segurança, protocolos e encaminhamento continuam na composição.
- A fonte amplia QA em português, mas não autoriza alegar segurança clínica, aderência a protocolos hospitalares ou aptidão para uso médico real.
- O MedQuAD não entra nesta versão porque é em inglês. Ele pode ser mantido como avaliação fora de distribuição, caso a pergunta e a resposta esperada sejam tratadas em inglês.

## Avaliação externa — nunca usada no treino

| Fonte | Uso previsto | Regra |
|---|---|---|
| [HealthQA-BR](https://huggingface.co/datasets/Larxel/healthqa-br) | Questões brasileiras de múltipla escolha para medir conhecimento | Não incluir em `data/splits/`; manter respostas e gabaritos somente para avaliação. |
| [DrBodeBench](https://huggingface.co/datasets/recogna-nlp/drbodebench) | Benchmark em português baseado em Revalida e FUVEST | Não incluir em `data/splits/`; definir métrica de múltipla escolha antes de baixar. |

## Fontes candidatas para RAG futuro

Os [PCDTs do Ministério da Saúde](https://dadosabertos.saude.gov.br/dataset/pcdt-protocolos-clinicos-e-diretrizes-terapeuticas) são fontes candidatas para recuperação em tempo de resposta. Antes da ingestão, registrar por documento a versão, a data de acesso, a URL e os termos de uso; o portal informa CC BY-ND 3.0. Esses documentos não fazem parte do fine-tuning desta versão.
