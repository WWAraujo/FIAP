# Compartilhar o FT-14 com o grupo

Use a versão GGUF para Windows, macOS e Linux. Envie pelo Google Drive estes arquivos de `fase3/llm/artifacts/shares/`:

- `FT14_MedPT_20261020-Q4_K_M.gguf` — modelo portátil, 5,3 GB;
- `FT14_MedPT_20261020-Q4_K_M.gguf.sha256` — checksum;
- `system_prompt_ft14_medpt.txt` — prompt usado no treino;
- `README_FT14_MedPT_GGUF.md` — resumo de uso.

O GGUF já contém a base Qwen3.5-9B com o adapter FT-14 mesclado e quantizado. Não é necessário baixar a base de 19 GB nem instalar Python, ROCm ou o repositório FIAP.

O FT-14 continua candidato experimental: a quantização Q4_K_M não é bit a bit igual ao adapter original e ainda requer revisão semântica. Não substitui o FT-11 da aplicação nem deve ser usado para cuidado clínico.

## Conferir o download

```bash
sha256sum -c FT14_MedPT_20261020-Q4_K_M.gguf.sha256
```

O resultado esperado é `OK`. No Windows, use `certutil -hashfile FT14_MedPT_20261020-Q4_K_M.gguf SHA256` e compare com o `.sha256`.

## Executar com llama.cpp

O método validado foi llama.cpp b12138, em CPU/Linux. Baixe o binário do seu sistema na [página de releases](https://github.com/ggml-org/llama.cpp/releases/tag/b12138), extraia-o e deixe o executável na pasta dos arquivos recebidos.

### Windows x64

Baixe `llama-b12138-bin-win-cpu-x64.zip` e execute no PowerShell:

```powershell
.\llama-cli.exe -m .\FT14_MedPT_20261020-Q4_K_M.gguf `
  --system-prompt-file .\system_prompt_ft14_medpt.txt `
  -p "Responda somente OK." -n 16 --reasoning off --single-turn
```

### macOS

Baixe `llama-b12138-bin-macos-arm64.tar.gz` para Apple Silicon ou `macos-x64` para Intel:

```bash
./llama-cli -m ./FT14_MedPT_20261020-Q4_K_M.gguf \
  --system-prompt-file ./system_prompt_ft14_medpt.txt \
  -p "Responda somente OK." -n 16 --reasoning off --single-turn
```

### Linux

Baixe `llama-b12138-bin-ubuntu-x64.tar.gz` e execute o mesmo comando acima. O resultado esperado em todos os sistemas é `OK`.

Em CPU, o modelo funciona, mas pode ser lento e exige memória suficiente para carregar 5,3 GB. A execução em Windows/macOS é suportada pelo formato GGUF e pelos binários fornecidos pelo llama.cpp; foi validada localmente somente em Linux/CPU.

## Versão original Linux/AMD

`FT14_MedPT_20261020.tar.zst` contém o adapter original de 102 MB. Ele exige Linux, GPU AMD/ROCm, Python 3.14 e a base Qwen de 19 GB. Para demonstração entre sistemas, use o GGUF.

## Recriar a exportação

[export_ft14_to_gguf.sh](../llm/src/finetuning/scripts/export_ft14_to_gguf.sh) reproduz a mesclagem e quantização nesta máquina. Ele cria temporários grandes, mas não altera FT-11, FT-14 original ou a seleção da aplicação.
