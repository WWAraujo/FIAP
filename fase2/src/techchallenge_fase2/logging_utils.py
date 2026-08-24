from __future__ import annotations

# ============================================================
# IMPORTAÇÕES
# ============================================================

from datetime import datetime


# ============================================================
# FUNÇÕES DE LOG
# ============================================================

def log_etapa(mensagem: str) -> None:
    """Exibe o início de uma etapa importante da execução."""
    agora = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{agora}] {mensagem}", flush=True)


def log_info(mensagem: str) -> None:
    """Exibe uma informação de acompanhamento da execução."""
    agora = datetime.now().strftime("%H:%M:%S")
    print(f"[{agora}] {mensagem}", flush=True)


def imprimir_linha() -> None:
    """Imprime o separador visual usado nos programas da Fase 1."""
    print("=" * 80, flush=True)

