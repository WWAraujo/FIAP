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

# Comando para rodar o Uvicorn
# Substitua "api_modelo" pelo nome do seu arquivo .py (sem a extensão)
# E "app" pelo nome da variável da sua instância FastAPI (ex: app = FastAPI())
CMD ["uvicorn", "api_modelo:app", "--host", "0.0.0.0", "--port", "8000"]