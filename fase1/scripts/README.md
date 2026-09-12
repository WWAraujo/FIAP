# Scripts de Utilidade - Fase 1

Este diretório contém scripts utilitários para a Fase 1 do projeto.

## Scripts Disponíveis

### download_vigitel.py

Script para download automático dos dados do Vigitel 2006-2024 do Ministério da Saúde.

**Fonte:** https://svs.aids.gov.br/daent/cgdnt/vigitel/

**Funcionalidades:**
- Busca automaticamente os links de download mais recentes na página oficial
- Baixa a base de dados Vigitel 2006-2024 (formato CSV - ZIP)
- Baixa o dicionário de dados (Excel)
- Extrai automaticamente os arquivos ZIP
- Renomeia o CSV extraído para um nome padrão
- Verifica se os arquivos já existem para evitar downloads duplicados

**Uso:**
```bash
cd fase1/scripts
python download_vigitel.py
```

**Requisitos:**
- Python 3.7+
- Bibliotecas: `requests`, `beautifulsoup4`

**Saída:**
Os arquivos são salvos em `fase1/data/`:
- `vigitel.csv` - Base de dados completa (967.5 MB)
- `dicionario-vigitel-2006-2024.xlsx` - Dicionário de variáveis
- `vigitel_2006_2024_csv.zip` - Arquivo ZIP original (mantido para backup)

**Notas:**
- O script usa web scraping para encontrar os links mais recentes, caso o Ministério da Saúde mude os URLs
- Se o web scraping falhar, ele usa URLs fallback
- Os arquivos grandes são automaticamente ignorados pelo .gitignore
- O download pode levar alguns minutos dependendo da conexão (arquivo ~83 MB)
