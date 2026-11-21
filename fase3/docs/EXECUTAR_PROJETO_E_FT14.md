# Executar o projeto e o modelo FT-14

Este guia separa duas coisas que têm requisitos diferentes:

| Objetivo | Modelo | Sistemas | Situação |
|---|---|---|---|
| Testar o FT-14 compartilhado | GGUF Q4_K_M | Windows, macOS e Linux | Disponível pelo llama.cpp. |
| Rodar o site completo da Fase 3 | FT-11 com RAG | Linux + AMD/ROCm | Fluxo atualmente integrado e validado. |
| Usar FT-14 dentro do site | FT-14 GGUF | Windows, macOS e Linux | Ponte llama.cpp disponível, sem RAG documental. |

O FT-14 não deve ser colocado no lugar do FT-11 em `best_model.json`: o runtime FT-11 confere hashes e a configuração RAG. A ponte FT-14 é iniciada separadamente na mesma porta e mantém o contrato usado pelo site.

## 1. Executar o FT-14 em qualquer sistema

### Requisitos mínimos práticos

- Windows 10/11 64 bits, macOS Intel/Apple Silicon ou Linux 64 bits.
- Pelo menos 16 GB de RAM recomendados.
- Cerca de 7 GB livres para modelo, prompt e binário.
- Os quatro arquivos compartilhados: GGUF, `.sha256`, prompt e README.
- Baixar os arquivos no https://drive.google.com/drive/folders/1WSLm9vIEQDMthPerf1nauMcAyNNgtNbd?usp=sharing

### Passos

1. Baixe os arquivos compartilhados para uma mesma pasta.
2. Confira o arquivo:

   ```bash
   sha256sum -c FT14_MedPT_20261020-Q4_K_M.gguf.sha256
   ```

   No Windows, use `certutil -hashfile FT14_MedPT_20261020-Q4_K_M.gguf SHA256` e compare o resultado com o `.sha256`.

3. Baixe e extraia o binário correspondente ao sistema na [release b12138 do llama.cpp](https://github.com/ggml-org/llama.cpp/releases/tag/b12138): `win-cpu-x64`, `macos-arm64`, `macos-x64` ou `ubuntu-x64`.
4. Execute na pasta em que estão o binário, o modelo e o prompt:

   ```bash
   ./llama-cli -m ./FT14_MedPT_20261020-Q4_K_M.gguf \
     --system-prompt-file ./system_prompt_ft14_medpt.txt \
     -p "Responda somente OK." -n 16 --reasoning off --single-turn
   ```

   No Windows, troque `./llama-cli` por `.\llama-cli.exe` e use PowerShell.

O resultado esperado é `OK`. Esse passo executa somente o FT-14; ele não abre o site, RAG, prontuários ou LangGraph.

## 2. Rodar o site com FT-14 GGUF

Além dos quatro arquivos compartilhados e do llama.cpp, precisa de Python 3.14 e do projeto FIAP. Na pasta `fase3`:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
```

Em três terminais, a partir da pasta que contém o GGUF:

```bash
./llama-server -m ./FT14_MedPT_20261020-Q4_K_M.gguf --port 8080 --reasoning off
```

Na raiz do repositório:

```bash
fase3/.venv/bin/python -B -m medical_assistant.retrieval.gguf_runtime_server \
  --system-prompt-file /caminho/system_prompt_ft14_medpt.txt
```

E, por fim:

```bash
cd fase3
.venv/bin/python -B -m streamlit run app.py
```

Abra `http://localhost:8501`. A interface, prontuários, IMC, revisões e workflow continuam disponíveis. Consultas ao FT-14 usam o prontuário como contexto, mas não usam a cadeia RAG documental nem devolvem fontes de protocolo; a resposta sempre informa essa limitação.

## 3. Rodar o site completo FT-11/RAG

O site completo usa FT-11, não o FT-14. Para isso, o colega precisa de:

- Linux, Python 3.14.4, GPU AMD com ROCm e 16 GB de VRAM na configuração validada;
- uma cópia do repositório;
- a base `Qwen/Qwen3.5-9B` com aproximadamente 19 GB;
- o adapter FT-11 com seu `manifest.json`, em `llm/artifacts/adapters/20261005T120430Z_qwen3.5_9b_qlora_rocm/`.

Depois de instalar os dois ambientes descritos no [README principal](../README.md), inicie em terminais separados, na raiz FIAP:

```bash
fase3/llm/.venv/bin/python -B fase3/llm/src/rag/runtime_server.py
```

Espere a mensagem `FT-11/RAG pronto`. Em outro terminal:

```bash
cd fase3
.venv/bin/python -B -m streamlit run app.py
```

Abra `http://localhost:8501`.

## 4. Limitação FT-14

A ponte aceita GGUF/llama.cpp e mantém o contrato do site, mas RAG documental com fontes ainda é exclusivo do runtime FT-11. Integrar BM25, fontes verificáveis e o verificador documental ao FT-14 exige uma avaliação nova.

Use o [guia de compartilhamento](COMPARTILHAR_FT14.md) para distribuição do modelo e o [README](../README.md) para a aplicação integrada.
