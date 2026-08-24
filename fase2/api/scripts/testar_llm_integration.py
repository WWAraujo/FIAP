#!/usr/bin/env python3
"""
Script para testar a integração com o Google Gemini.

Importa as funções reais de app/llm_interpreter.py (em vez de duplicar a
lógica de chamada), garantindo que o teste exercite exatamente o mesmo
código usado em produção pela API.

Uso:
    cd fase2/api/src
    python ../scripts/testar_llm_integration.py
"""

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# O .env fica no mesmo nível da pasta scripts/ (ex.: fase2/api/.env).
# Precisa ser carregado ANTES de importar app.llm_interpreter, porque
# esse módulo lê GOOGLE_API_KEY assim que é importado (no nível do
# módulo) — carregar o .env depois não teria efeito.
PASTA_API = Path(__file__).resolve().parent.parent
load_dotenv(PASTA_API / ".env")

# Localiza a pasta src/ (onde fica o pacote "app") e adiciona ao path,
# do mesmo jeito que tech_challenge_fase2.py faz.
PASTA_SRC = PASTA_API / "src"
sys.path.insert(0, str(PASTA_SRC))

from app.llm_interpreter import (  # noqa: E402
    gerar_interpretacao_llm,
    VARIAVEIS_NOMES_CLINICOS,
    LLM_HABILITADO,
    GOOGLE_MODEL,
)


# ============================================================
# CENÁRIOS DE TESTE
# ============================================================
# As 20 variáveis reais do modelo (ver metadata_modelo_api.json),
# com valores plausíveis para cada perfil de risco.

CENARIO_ALTO_RISCO = {
    "diab": 1,
    "iddpapa": 3,
    "imc": 32.0,
    "excpeso": 1,
    "imc_i": 3,
    "iddpapa_old": 3,
    "excpeso_i": 1,
    "iddmamo": 2,
    "af": 0,
    "exfuma": 1,
    "obesid": 1,
    "obesid_i": 1,
    "ind_med_db": 1,
    "med_db": 1,
    "atiocu": 0,
    "af3dominios_insu_2023": 1,
    "dislip": 1,
    "af3dominios_2023": 0,
    "inativo_2023": 1,
    "saruim": 1,
}

CENARIO_BAIXO_RISCO = {
    "diab": 0,
    "iddpapa": 1,
    "imc": 22.0,
    "excpeso": 0,
    "imc_i": 1,
    "iddpapa_old": 1,
    "excpeso_i": 0,
    "iddmamo": None,
    "af": 1,
    "exfuma": 0,
    "obesid": 0,
    "obesid_i": 0,
    "ind_med_db": 0,
    "med_db": 0,
    "atiocu": 1,
    "af3dominios_insu_2023": 0,
    "dislip": 0,
    "af3dominios_2023": 1,
    "inativo_2023": 0,
    "saruim": 0,
}

CENARIO_INTERMEDIARIO = {
    "diab": 0,
    "iddpapa": 2,
    "imc": 27.0,
    "excpeso": 1,
    "imc_i": 2,
    "iddpapa_old": 2,
    "excpeso_i": 1,
    "iddmamo": None,
    "af": 2,
    "exfuma": 0,
    "obesid": 0,
    "obesid_i": 0,
    "ind_med_db": 0,
    "med_db": 0,
    "atiocu": 0,
    "af3dominios_insu_2023": 1,
    "dislip": 0,
    "af3dominios_2023": 0,
    "inativo_2023": 1,
    "saruim": 0,
}


# ============================================================
# EXIBIÇÃO DO RESULTADO
# ============================================================

def exibir_resultado(
    numero: int,
    titulo: str,
    probabilidade: float,
    classe_prevista: int,
    interpretacao: str | None,
    duracao: float,
) -> None:
    risco_percentual = round(probabilidade * 100, 2)
    classificacao = "elevado risco de hipertensão" if classe_prevista == 1 else "baixo risco de hipertensão"

    print(f"\n{'=' * 70}")
    print(f"TESTE #{numero}: {titulo}")
    print(f"{'=' * 70}")
    print(f"\nPREDIÇÃO: {classificacao} ({risco_percentual}%)")
    print(f"TEMPO: {duracao:.2f}s")

    if interpretacao:
        print("\nINTERPRETAÇÃO DO LLM:")
        print("-" * 70)
        print(interpretacao)
        print("-" * 70)
    else:
        print("\n⚠️  Falha ao gerar interpretação (ver logs em app.llm_interpreter para detalhes)")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("TESTE DE INTEGRAÇÃO COM GOOGLE GEMINI")
    print("=" * 70)
    print(f"\nModelo configurado: {GOOGLE_MODEL}")
    print(f"LLM habilitado: {LLM_HABILITADO}")

    if not LLM_HABILITADO:
        print("\n❌ GOOGLE_API_KEY não configurada. Configure o .env antes de testar.\n")
        return

    cenarios = [
        ("Paciente com ALTO RISCO", 0.85, 1, CENARIO_ALTO_RISCO),
        ("Paciente com BAIXO RISCO", 0.15, 0, CENARIO_BAIXO_RISCO),
        ("Paciente com RISCO INTERMEDIÁRIO", 0.52, 1, CENARIO_INTERMEDIARIO),
    ]

    resultados = []
    for numero, (titulo, probabilidade, classe_prevista, variaveis) in enumerate(cenarios, start=1):
        inicio = time.time()
        interpretacao = gerar_interpretacao_llm(
            probabilidade=probabilidade,
            classe_prevista=classe_prevista,
            variaveis_entrada=variaveis,
            variaveis_nomes_clinicos=VARIAVEIS_NOMES_CLINICOS,
        )
        duracao = time.time() - inicio
        resultados.append((interpretacao is not None, duracao))
        exibir_resultado(numero, titulo, probabilidade, classe_prevista, interpretacao, duracao)

    print("\n" + "=" * 70)
    print("TESTES CONCLUÍDOS")
    print("=" * 70)
    sucessos = sum(1 for sucesso, _ in resultados if sucesso)
    tempo_total = sum(duracao for _, duracao in resultados)
    print(f"\n{sucessos}/{len(resultados)} interpretações geradas com sucesso")
    print(f"Tempo total: {tempo_total:.2f}s")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Teste interrompido pelo usuário")