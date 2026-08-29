#!/bin/bash

# Verifica se Ollama está instalado
if ! command -v ollama &> /dev/null
then
    echo "Ollama não encontrado. Instalando..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Inicia o servidor Ollama em background
echo "Iniciando servidor Ollama..."
ollama serve &

# Aguarda alguns segundos
sleep 5

# Verifica se o modelo está instalado
if ! ollama list | grep -q "qwen3.5"; then
    echo "Modelo qwen3.5 não encontrado. Baixando..."
    ollama pull qwen3.5
else
    echo "Modelo qwen3.5 já está disponível."
fi

echo "Ollama está rodando e pronto para uso."

read -p "Pressione ENTER para sair..."



#Basta salvar o script como .sh, dar permissão (chmod +x start-ollama.sh) e rodar (./start-ollama.sh).