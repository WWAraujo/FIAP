# Usa uma imagem oficial do Python estável e leve
FROM python:3.10-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de dependências primeiro para otimizar o cache do Docker
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o conteúdo do projeto para dentro do container
COPY . .

# Expõe a porta padrão do FastAPI/Uvicorn (geralmente 8000)
EXPOSE 8000

# Seta PYTHONPATH para permitir imports relativos
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Healthcheck usado pelo Docker/Compose para saber se o container está
# pronto para receber tráfego (nginx e orquestração de escala dependem disso).
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

# Roda uvicorn com PYTHONPATH configurado para imports relativos funcionarem
CMD ["uvicorn", "app.api_modelo:app", "--host", "0.0.0.0", "--port", "8000"]
