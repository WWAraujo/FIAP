#!/usr/bin/env python3
"""
Script para download automático dos dados do Vigitel 2006-2024.

Fonte: https://svs.aids.gov.br/daent/cgdnt/vigitel/

Este script baixa:
- Base de dados Vigitel 2006-2024 (formato CSV - ZIP)
- Dicionário de dados (Excel)

Uso:
    python download_vigitel.py
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
from urllib.parse import urljoin
import time
import re
from bs4 import BeautifulSoup

# URLs oficiais do Vigitel (Ministério da Saúde)
BASE_URL = "https://svs.aids.gov.br/daent/cgdnt/vigitel/"

# URLs específicas (serão atualizadas se os links diretos mudarem)
# Nota: Os links podem mudar, então o script tenta encontrar os links na página
VIGITEL_CSV_ZIP = "https://svs.aids.gov.br/images/stories/Vigitel/arquivos_completos/vigitel_2006_2024_csv.zip"
VIGITEL_DICT_XLSX = "https://svs.aids.gov.br/images/stories/Vigitel/arquivos_completos/dicionario_vigitel_2006_2024.xlsx"

# Diretório de destino
DATA_DIR = Path(__file__).parent.parent / "data"


def find_download_links():
    """
    Tenta encontrar os links de download na página do Vigitel.
    
    Returns:
        tuple: (csv_zip_url, dict_xlsx_url) ou (None, None) se não encontrar
    """
    print(f"\n🔍 Buscando links de download em {BASE_URL}...")
    
    try:
        response = requests.get(BASE_URL, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Procura por links específicos
        csv_zip_url = None
        dict_xlsx_url = None
        
        for link in soup.find_all('a', href=True):
            href = link['href'].lower()
            text = link.get_text().lower()
            
            # Procura link do CSV completo 2006-2024
            if 'vigitel-2006-2024-peso-rake-csv.zip' in href:
                if href.startswith('http'):
                    csv_zip_url = link['href']
                else:
                    csv_zip_url = urljoin(BASE_URL, link['href'])
                print(f"   📎 Link CSV completo encontrado: {csv_zip_url}")
                break  # Usa o primeiro link correto encontrado
            
            # Fallback: procura qualquer link com "2006-2024" e "csv"
            if '2006-2024' in href and 'csv' in href and not csv_zip_url:
                if href.startswith('http'):
                    csv_zip_url = link['href']
                else:
                    csv_zip_url = urljoin(BASE_URL, link['href'])
                print(f"   📎 Link CSV alternativo encontrado: {csv_zip_url}")
            
            # Procura link do dicionário 2006-2024
            if 'dicionario-vigitel-2006-2024' in href:
                if href.startswith('http'):
                    dict_xlsx_url = link['href']
                else:
                    dict_xlsx_url = urljoin(BASE_URL, link['href'])
                print(f"   📎 Link Dicionário encontrado: {dict_xlsx_url}")
        
        return csv_zip_url, dict_xlsx_url
        
    except Exception as e:
        print(f"   ⚠️  Erro ao buscar links: {e}")
        return None, None


def download_file(url: str, destination: Path, description: str) -> bool:
    """
    Baixa um arquivo de uma URL com barra de progresso.
    
    Args:
        url: URL do arquivo
        destination: Caminho de destino
        description: Descrição do arquivo para logging
        
    Returns:
        True se sucesso, False caso contrário
    """
    print(f"\n📥 Baixando {description}...")
    print(f"   URL: {url}")
    print(f"   Destino: {destination}")
    
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r   Progresso: {percent:.1f}%", end='', flush=True)
        
        print(f"\n   ✅ Download concluído: {destination.stat().st_size / (1024*1024):.1f} MB")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\n   ❌ Erro no download: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """
    Extrai um arquivo ZIP.
    
    Args:
        zip_path: Caminho do arquivo ZIP
        extract_to: Diretório de extração
        
    Returns:
        True se sucesso, False caso contrário
    """
    print(f"\n📦 Extraindo {zip_path.name}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"   ✅ Extração concluída em {extract_to}")
            print(f"   Arquivos extraídos: {len(zip_ref.namelist())}")
        return True
    except zipfile.BadZipFile as e:
        print(f"   ❌ Erro na extração: {e}")
        return False


def main():
    """Função principal."""
    print("=" * 60)
    print("📊 Download dos Dados do Vigitel 2006-2024")
    print("=" * 60)
    print(f"Fonte: {BASE_URL}")
    print(f"Diretório de destino: {DATA_DIR}")
    
    # Criar diretório de dados
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Tenta encontrar links dinamicamente
    csv_zip_url, dict_xlsx_url = find_download_links()
    
    # Usa os links encontrados ou os fallbacks
    csv_zip_url = csv_zip_url or VIGITEL_CSV_ZIP
    dict_xlsx_url = dict_xlsx_url or VIGITEL_DICT_XLSX
    
    # Download do dicionário (Excel)
    dict_path = DATA_DIR / "dicionario-vigitel-2006-2024.xlsx"
    if dict_path.exists():
        print(f"\n⏭️  Dicionário já existe: {dict_path}")
    else:
        if download_file(dict_xlsx_url, dict_path, "Dicionário de dados"):
            print(f"   ✅ Dicionário salvo")
        else:
            print(f"   ⚠️  Falha no download do dicionário")
    
    # Download da base CSV (ZIP)
    zip_path = DATA_DIR / "vigitel_2006_2024_csv.zip"
    if zip_path.exists():
        print(f"\n⏭️  Arquivo ZIP já existe: {zip_path}")
    else:
        if download_file(csv_zip_url, zip_path, "Base de dados Vigitel (ZIP)"):
            print(f"   ✅ ZIP salvo")
        else:
            print(f"   ⚠️  Falha no download do ZIP")
            print(f"\n💡 Tente baixar manualmente em: {BASE_URL}")
            return 1
    
    # Extrair ZIP
    if zip_path.exists():
        # Verificar se já existe o CSV extraído
        csv_files = list(DATA_DIR.glob("vigitel*.csv"))
        if csv_files:
            print(f"\n⏭️  Arquivo CSV já existe: {csv_files[0]}")
        else:
            if extract_zip(zip_path, DATA_DIR):
                # Renomear o arquivo CSV extraído para um nome padrão
                extracted_csvs = list(DATA_DIR.glob("*.csv"))
                if extracted_csvs:
                    csv_path = extracted_csvs[0]
                    standard_name = DATA_DIR / "vigitel.csv"
                    if csv_path != standard_name:
                        csv_path.rename(standard_name)
                        print(f"   ✅ CSV renomeado para: {standard_name}")
            else:
                print(f"   ⚠️  Falha na extração")
                return 1
    
    # Resumo final
    print("\n" + "=" * 60)
    print("📋 Resumo dos arquivos em fase1/data/:")
    print("=" * 60)
    
    for file in sorted(DATA_DIR.iterdir()):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  • {file.name:40} ({size_mb:.1f} MB)")
    
    print("\n✅ Processo concluído!")
    print(f"\n💡 Os dados estão prontos para uso em: {DATA_DIR}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
