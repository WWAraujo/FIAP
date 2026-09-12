# Usa uma imagem oficial do Python estável e leve
FROM python:3.10-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de dependências primeiro para otimizar o cache do Docker
COPY fase1/requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia dados de treinamento da fase1
COPY fase1/data/ ./data/

# Copia código da API
COPY fase1/src/ ./src/

# Copia modelo treinado da fase1
COPY fase1/modelo_api/ ./modelo_api/

# Expõe a porta padrão do FastAPI/Uvicorn (geralmente 8000)
EXPOSE 8000

# Comando para rodar o Uvicorn
CMD ["uvicorn", "src.techchallenge_fase1.api_modelo:app", "--host", "0.0.0.0", "--port", "8000"]