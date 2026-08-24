"""
Módulo para interpretação em linguagem natural das predições usando LLM.

Chama a API do Google Gemini (endpoint "interactions") para gerar
interpretações contextualizadas das predições de hipertensão.
"""

import os
import re
import time
from typing import Any, Dict, Optional

import requests

from .logging_config import log_evento

# ============================================================
# CONFIGURAÇÃO
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GOOGLE_MODEL = os.environ.get("GOOGLE_MODEL", "gemini-3.7-flash")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
TIMEOUT_LLM = int(os.environ.get("TIMEOUT_LLM", "20"))

PREVIEW_MAX_CHARS = 200

LLM_HABILITADO = bool(GOOGLE_API_KEY.strip())

if LLM_HABILITADO:
    log_evento("LLM inicializado", provedor="google_gemini", modelo=GOOGLE_MODEL)
else:
    log_evento("LLM desabilitado: GOOGLE_API_KEY não configurada", nivel="warning")


# ============================================================
# INTERPRETAÇÃO COM LLM
# ============================================================

def gerar_interpretacao_llm(
    probabilidade: float,
    classe_prevista: int,
    variaveis_entrada: Dict[str, Any],
    variaveis_nomes_clinicos: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Gera uma interpretação em linguagem natural da predição usando o LLM."""

    if not LLM_HABILITADO:
        log_evento("Chamada ao LLM ignorada: LLM desabilitado", nivel="debug")
        return None

    contexto_variaveis = _formatar_variaveis_contexto(variaveis_entrada, variaveis_nomes_clinicos)
    prompt = _construir_prompt(probabilidade, classe_prevista, contexto_variaveis)

    log_evento(
        "Iniciando chamada ao LLM",
        provedor="google_gemini",
        modelo=GOOGLE_MODEL,
        tamanho_prompt_caracteres=len(prompt),
    )

    inicio = time.perf_counter()
    try:
        resposta = _chamar_gemini_api(prompt)
        duracao = round(time.perf_counter() - inicio, 3)
        log_evento(
            "Interpretação LLM gerada com sucesso",
            modelo=GOOGLE_MODEL,
            duracao_segundos=duracao,
            tamanho_resposta_caracteres=len(resposta),
            resposta_preview=_truncar(resposta),
        )
        return resposta

    except Exception as erro:
        duracao = round(time.perf_counter() - inicio, 3)
        log_evento(
            "Erro ao gerar interpretação com LLM",
            nivel="error",
            modelo=GOOGLE_MODEL,
            duracao_segundos=duracao,
            tipo_erro=type(erro).__name__,
            erro=str(erro),
        )
        return None


def _truncar(texto: str, limite: int = PREVIEW_MAX_CHARS) -> str:
    return texto if len(texto) <= limite else texto[:limite] + "..."


def _formatar_variaveis_contexto(
    variaveis_entrada: Dict[str, Any],
    nomes_clinicos: Optional[Dict[str, str]] = None,
) -> str:
    """Formata as variáveis em texto legível, ignorando valores None.

    Os valores chegam codificados (0/1, faixas numeradas) porque é assim
    que o modelo foi treinado — aqui só a versão para o LLM é decodificada
    para texto, o valor original enviado ao modelo não muda.
    """
    nomes_clinicos = nomes_clinicos or {}
    linhas = []
    for nome, valor in variaveis_entrada.items():
        if valor is None:
            continue
        nome_exibicao = nomes_clinicos.get(nome, nome)
        valor_exibicao = _decodificar_valor(nome, valor)
        linhas.append(f"- {nome_exibicao}: {valor_exibicao}")
    return "\n".join(linhas) if linhas else "Nenhuma variável fornecida"


def _decodificar_valor(nome_variavel: str, valor: Any) -> Any:
    """Traduz o código bruto (0/1, faixa numerada) para texto, usando
    VALORES_CLINICOS. Sem mapeamento para a variável (ex.: imc, que já é
    um número legível), devolve o valor original sem alteração."""
    mapa = VALORES_CLINICOS.get(nome_variavel)
    if not mapa:
        return valor
    try:
        codigo = int(valor)
    except (TypeError, ValueError):
        return valor
    return mapa.get(codigo, valor)


def _construir_prompt(probabilidade: float, classe_prevista: int, contexto_variaveis: str) -> str:
    """Monta o prompt enviado ao LLM."""
    risco_percentual = round(probabilidade * 100, 2)
    classificacao = "elevado risco de hipertensão" if classe_prevista == 1 else "baixo risco de hipertensão"

    return f"""Você é um assistente médico especializado em análise de hipertensão.
Baseado nos seguintes dados clínicos, forneça uma interpretação breve e clara da predição.

Dados do Paciente:
{contexto_variaveis}

Resultado da Predição:
- Classificação: {classificacao}
- Probabilidade de hipertensão: {risco_percentual}%

Sua Tarefa:
1. Resuma a predição em 2-3 frases
2. Destaque os fatores de risco identificados (se houver)
3. Forneça uma recomendação breve e prática

Responda em texto simples, sem usar Markdown (nada de **, *, #, listas com marcadores
ou qualquer outra formatação) — apenas frases corridas, em linguagem acessível e
adequada para comunicação direta com pacientes."""


def _chamar_gemini_api(prompt: str) -> str:
    """
    Chama a API Gemini (endpoint /v1beta/interactions) e devolve o texto
    da resposta. Levanta exceção em qualquer falha — quem chamou decide
    o que logar (ver gerar_interpretacao_llm).
    """
    resposta = requests.post(
        GEMINI_URL,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GOOGLE_API_KEY,
        },
        json={"model": GOOGLE_MODEL, "input": prompt},
        timeout=TIMEOUT_LLM,
    )
    resposta.raise_for_status()
    dados = resposta.json()

    if dados.get("status") != "completed":
        raise RuntimeError(f"Interação não concluída: status={dados.get('status')}")

    # A resposta vem em "steps"; o texto final está no step do tipo
    # "model_output", dentro de content[].text (ignora steps de "thought").
    for step in dados.get("steps", []):
        if step.get("type") != "model_output":
            continue
        for parte in step.get("content", []):
            if parte.get("type") == "text" and parte.get("text"):
                return _remover_markdown(parte["text"].strip())

    raise RuntimeError("Resposta do Gemini sem texto em model_output")


def _remover_markdown(texto: str) -> str:
    """Remove marcações Markdown que o modelo às vezes usa mesmo quando
    instruído a não usar (negrito, itálico, cabeçalhos, marcadores de
    lista) — o formulário exibe texto puro, não HTML/Markdown renderizado.
    """
    texto = re.sub(r"\*\*(.+?)\*\*", r"\1", texto)  # **negrito**
    texto = re.sub(r"(?<!\w)\*(?!\s)(.+?)\*(?!\w)", r"\1", texto)  # *itálico*
    texto = re.sub(r"^\s*[\*\-]\s+", "", texto, flags=re.MULTILINE)  # * item / - item
    texto = re.sub(r"^#+\s*", "", texto, flags=re.MULTILINE)  # # Título
    return texto.strip()


# ============================================================
# MAPEAMENTO DE NOMES CLÍNICOS
# ============================================================
# Nomes legíveis para os códigos de variável do VIGITEL usados pelo
# modelo (ver metadata_modelo_api.json > variaveis_entrada), confirmados
# contra o dicionário oficial "dicionario-vigitel-2006-2024.xlsx"
# (aba Indicadores_Vigitel — são variáveis derivadas, não perguntas
# cruas do questionário). "af" ficou com o rótulo original do
# dicionário ("Tipo atv. fis. no tempo livre") por ser uma variável
# categórica (0/1/2), não um indicador sim/não como as demais.

VARIAVEIS_NOMES_CLINICOS = {
    "diab": "Diagnóstico médico de diabetes",
    "iddpapa": "Faixa de idade alvo para Papanicolau",
    "imc": "Índice de Massa Corporal (IMC)",
    "excpeso": "Excesso de peso",
    "imc_i": "IMC (com imputações)",
    "iddpapa_old": "Faixa de idade alvo para Papanicolau (critério anterior)",
    "excpeso_i": "Excesso de peso (com imputações)",
    "iddmamo": "Faixa de idade alvo para mamografia",
    "af": "Tipo de atividade física no tempo livre (lazer)",
    "exfuma": "Ex-fumante",
    "obesid": "Obesidade",
    "obesid_i": "Obesidade (com imputações)",
    "ind_med_db": "Indicação médica de diabetes",
    "med_db": "Uso de medicamento para diabetes",
    "atiocu": "Atividade física no trabalho",
    "af3dominios_insu_2023": "Atividade física insuficiente (≤150min/sem, 3 domínios, 2023)",
    "dislip": "Dislipidemia",
    "af3dominios_2023": "Atividade física suficiente (≥150min/sem, 3 domínios, 2023)",
    "inativo_2023": "Inatividade física (2023)",
    "saruim": "Autoavaliação ruim da condição de saúde",
}


# ============================================================
# DECODIFICAÇÃO DE VALORES
# ============================================================
# Traduz os códigos numéricos (0/1, faixas) para texto, também extraído
# do dicionário oficial do VIGITEL. Variáveis ausentes daqui (imc, imc_i)
# já são números legíveis e não precisam de tradução.

_SIM_NAO = {0: "Não", 1: "Sim"}

VALORES_CLINICOS: Dict[str, Dict[int, str]] = {
    "diab": _SIM_NAO,
    "excpeso": _SIM_NAO,
    "excpeso_i": _SIM_NAO,
    "exfuma": _SIM_NAO,
    "obesid": _SIM_NAO,
    "obesid_i": _SIM_NAO,
    "ind_med_db": _SIM_NAO,
    "med_db": _SIM_NAO,
    "atiocu": _SIM_NAO,
    "af3dominios_insu_2023": _SIM_NAO,
    "dislip": _SIM_NAO,
    "af3dominios_2023": _SIM_NAO,
    "inativo_2023": _SIM_NAO,
    "saruim": _SIM_NAO,
    "iddpapa": {1: "25 a 34 anos", 2: "35 a 44 anos", 3: "45 a 54 anos", 4: "55 a 64 anos"},
    "iddpapa_old": {1: "25 a 34 anos", 2: "35 a 44 anos", 3: "45 a 54 anos", 4: "55 a 59 anos"},
    "iddmamo": {1: "50 a 59 anos", 2: "60 a 69 anos"},
    "af": {
        0: "sem prática relatada de atividade física no lazer",
        1: "pratica atividade física moderada/vigorosa no lazer",
        2: "pratica atividade física leve no lazer",
    },
}