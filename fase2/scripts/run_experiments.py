"""Compatibilidade com o comando usado na primeira versão da Fase 2.

O programa principal agora fica na raiz, seguindo o padrão visual da Fase 1.
Este arquivo apenas encaminha a execução e os argumentos recebidos.
"""

from __future__ import annotations

# ============================================================
# IMPORTAÇÕES
# ============================================================

import runpy
from pathlib import Path


# ============================================================
# ENCAMINHAMENTO PARA O PROGRAMA PRINCIPAL
# ============================================================

PASTA_PROJETO = Path(__file__).resolve().parents[1]
PROGRAMA_PRINCIPAL = PASTA_PROJETO / "tech_challenge_fase2.py"


if __name__ == "__main__":
    runpy.run_path(str(PROGRAMA_PRINCIPAL), run_name="__main__")
