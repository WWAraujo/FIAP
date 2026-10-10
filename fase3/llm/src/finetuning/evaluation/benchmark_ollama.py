"""Benchmark inteligente com auto-setup e downloads em paralelo.

Este script automaticamente:
1. Instala Ollama se não estiver instalado
2. Inicia Ollama se não estiver rodando
3. Baixa pelo menos 1 modelo se não houver nenhum
4. Executa benchmark com modelos disponíveis enquanto baixa outros em background
5. Suporta downloads paralelos e execução concorrente
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
import platform
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import concurrent.futures
import queue
import threading

import requests

PROJECT_DIR = Path(__file__).resolve().parents[3]
CASES_FILE = Path(__file__).with_name("clinical_cases.jsonl")
REPORTS_DIR = PROJECT_DIR / "artifacts" / "benchmarks"
OLLAMA_BASE_URL = "http://localhost:11434"

DEFAULT_MODELS = (
    "gemma3:4b",                 # candidato compacto para LoRA
    "qwen3:8b",                  # candidato principal para QLoRA
    "qwen2.5:7b-instruct",       # fallback com menor demanda de VRAM
    "qwen3.8:27b",               # referência de inferência; não é alvo de treino local
)

SYSTEM_PROMPT = """Você é um assistente de apoio clínico para profissionais.
Use somente o contexto fornecido. Não prescreva medicamentos, não conclua um
diagnóstico definitivo e indique validação humana quando houver conduta clínica.
Se faltarem dados, declare a limitação. Responda em português e indique o
protocolo que sustentou a resposta. Forneça sua resposta diretamente, sem raciocínio interno.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--no-auto-setup", action="store_true",help="Desativar instalação/iniciação automática do Ollama")
    parser.add_argument("--max-parallel-downloads", type=int, default=1, help="Downloads simultâneos de modelos")
    parser.add_argument("--download-timeout", type=int, default=1800, help="Timeout (s) por download de modelo")
    return parser.parse_args()


def check_ollama_installed() -> bool:
    """Verifica se Ollama está instalado."""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def install_ollama() -> bool:
    """Instala Ollama automaticamente."""
    print("🔧 Ollama não encontrado. Instalando automaticamente...")

    try:
        if platform.system() == "Windows":
            print("📥 Baixando instalador Ollama para Windows...")
            urllib.request.urlretrieve("https://ollama.com/download/OllamaSetup.exe", "OllamaSetup.exe")
            print("🚀 Iniciando instalador...")
            subprocess.run(["OllamaSetup.exe"], check=True)
            print("✅ Ollama instalado!")
            return True
        elif platform.system() == "Darwin":
            print("📥 Instalando Ollama via Homebrew...")
            subprocess.run(["brew", "install", "ollama"], check=True)
            print("✅ Ollama instalado!")
            return True
        elif platform.system() == "Linux":
            print("📥 Instalando Ollama via script oficial...")
            subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
            print("✅ Ollama instalado!")
            return True
        else:
            print(f"❌ Sistema {platform.system()} não suportado")
            return False
    except Exception as e:
        print(f"❌ Erro ao instalar Ollama: {e}")
        return False


def check_ollama_running() -> bool:
    """Verifica se Ollama está rodando."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def start_ollama() -> bool:
    """Inicia Ollama em background."""
    print("🚀 Iniciando Ollama em background...")

    try:
        if platform.system() == "Windows":
            subprocess.Popen(["ollama", "serve"], creationflags=subprocess.DETACHED_PROCESS)
        else:
            subprocess.Popen(["ollama", "serve"], start_new_session=True)

        print("⏳ Aguardando Ollama iniciar...")
        for i in range(30):
            time.sleep(1)
            if check_ollama_running():
                print("✅ Ollama iniciado!")
                return True
            print(f"⏳ Aguardando... ({i + 1}/30)")

        print("❌ Timeout ao iniciar Ollama")
        return False
    except Exception as e:
        print(f"❌ Erro ao iniciar Ollama: {e}")
        return False


def installed_models() -> set[str]:
    """Retorna conjunto de modelos instalados."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        response.raise_for_status()
        return {model["name"] for model in response.json().get("models", [])}
    except requests.RequestException:
        return set()


def installed_model_metadata() -> dict[str, dict[str, Any]]:
    """Retorna a identidade imutável e os detalhes expostos pelo Ollama."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        response.raise_for_status()
        return {model["name"]: model for model in response.json().get("models", [])}
    except requests.RequestException:
        return {}


def generation_options(args: argparse.Namespace) -> dict[str, int]:
    return {"temperature": 0, "seed": 42, "num_predict": args.max_tokens, "num_ctx": 2048}


def ollama_version() -> str | None:
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


_PROGRESS_RE = re.compile(r"^pulling ([0-9a-f]+):\s+(\d+)%")
_PROGRESS_STEP = 25  # só loga a cada salto de 25% por camada (evita flood)


def _kill_process_tree(process: subprocess.Popen) -> None:
    """Mata o processo por timeout. No Windows, `Popen.kill()` só derruba o
    processo principal — `ollama pull` pode deixar sub-processos/handlers
    órfãos. `taskkill /T /F` mata a árvore inteira."""
    if platform.system() == "Windows":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
        )
    else:
        process.kill()


def _condense_progress_line(line: str, last: dict[str, int]) -> str | None:
    """Reduz o log do `ollama pull` a um evento por marco: cada fase (pulling
    manifest, verifying, writing manifest, success) e saltos de ~25% por
    camada — em vez de uma linha a cada 1% de progresso. Retorna None quando
    a linha deve ser suprimida.
    """
    match = _PROGRESS_RE.match(line)
    if not match:
        # Só logar linhas realmente importantes (success, error)
        # Suprimir "pulling manifest", "verifying", etc que são muito frequentes
        important_keywords = ("success", "error", "failed")
        if any(keyword in line.lower() for keyword in important_keywords):
            return line
        return None  # suprimir todo o resto, incluindo "pulling manifest"

    digest, percent_str = match.group(1), match.group(2)
    percent = int(percent_str)
    previous = last.get(digest, -1)

    # Logar sempre quando chega a 100% ou quando atinge um novo marco de 25%
    should_log = (percent == 100 or
                  percent >= previous + _PROGRESS_STEP or
                  percent == 0)  # logar o início (0%)

    if not should_log:
        return None

    last[digest] = percent
    return f"{digest[:12]}... {percent}%"


def download_worker(
        model: str,
        ready_queue: "queue.Queue[str]",
        status: dict[str, str],
        lock: threading.Lock,
        timeout: int,
) -> None:
    """Baixa um modelo em background. Loga só os marcos relevantes do
    `ollama pull` (uma linha por fase/salto de progresso, sem a barra visual)
    e empurra o nome na ready_queue assim que o download terminar com
    sucesso. É essa fila, e não um polling de dicionário, que permite ao loop
    principal pegar modelos recém-prontos assim que aparecem.
    """
    with lock:
        status[model] = "downloading"
    print(f"📥 [{model}] download iniciado", flush=True)

    try:
        process = subprocess.Popen(
            ["ollama", "pull", model],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as error:
        with lock:
            status[model] = "failed"
        print(f"❌ [{model}] não foi possível iniciar o download: {error}", flush=True)
        return

    timer = threading.Timer(timeout, _kill_process_tree, args=(process,))
    timer.start()
    last_percent_by_digest: dict[str, int] = {}
    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            condensed = _condense_progress_line(line, last_percent_by_digest)
            if condensed is not None:
                print(f"   [{model}] {condensed}", flush=True)
        returncode = process.wait()
    finally:
        timer.cancel()

    with lock:
        status[model] = "downloaded" if returncode == 0 else "failed"

    if returncode == 0:
        print(f"✅ [{model}] download concluído", flush=True)
        ready_queue.put(model)
    else:
        print(f"❌ [{model}] download falhou (código {returncode})", flush=True)


def load_cases() -> list[dict[str, Any]]:
    """Carrega casos clínicos do arquivo JSONL."""
    with CASES_FILE.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def build_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    """Constrói mensagens para o modelo."""
    content = f"CONTEXTO SINTÉTICO:\n{case.get('context', '')}\n\nPERGUNTA:\n{case['query']}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def call_model(model: str, case: dict[str, Any], timeout: int, max_tokens: int = 512) -> dict[str, Any]:
    """Chama o modelo e retorna resultado."""
    payload = {
        "model": model,
        "messages": build_messages(case),
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_predict": max_tokens,
            "num_ctx": 2048,
        },
    }
    started = time.perf_counter()
    response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=timeout)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    body = response.json()

    response_content = ""
    if "message" in body and isinstance(body["message"], dict):
        response_content = body["message"].get("content", "")
        if not response_content:
            response_content = body["message"].get("thinking", "")
    elif "content" in body:
        response_content = body["content"]

    eval_count = body.get("eval_count")
    eval_duration_ns = body.get("eval_duration")
    tokens_per_second = None
    if eval_count and eval_duration_ns:
        tokens_per_second = eval_count / (eval_duration_ns / 1_000_000_000)

    return {
        "response": response_content,
        "latency_ms": round(elapsed_ms, 2),
        "prompt_tokens": body.get("prompt_eval_count"),
        "generated_tokens": eval_count,
        "tokens_per_second": round(tokens_per_second, 2) if tokens_per_second else None,
    }


def keyword_checks(case: dict[str, Any], answer: str) -> tuple[bool, bool]:
    """Verifica termos obrigatórios, alternativas explícitas e proibidos."""
    normalized = answer.lower()
    required = case.get("must_mention", [])
    required_any = case.get("must_mention_any", [])
    forbidden = case.get("must_not_contain", [])
    mentions_expected = (
        all(term.lower() in normalized for term in required)
        and all(any(term.lower() in normalized for term in alternatives) for alternatives in required_any)
    )
    return mentions_expected, not any(
        term.lower() in normalized for term in forbidden
    )


def automatic_checks(case: dict[str, Any], answer: str) -> tuple[bool, bool, bool]:
    """Checks lexicais de triagem; não substituem revisão humana."""
    is_nonempty = bool(answer.strip())
    mentions_expected, avoids_forbidden = keyword_checks(case, answer)
    return is_nonempty, mentions_expected, avoids_forbidden


def warm_up(model: str, timeout: int, max_tokens: int = 512) -> None:
    """Executa warm-up do modelo."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Responda apenas: pronto."}],
            "stream": False,
            "options": {"num_predict": max_tokens},
        },
        timeout=timeout,
    )
    response.raise_for_status()

# Crie um metodo


def run_benchmark_for_model(
        model: str,
        cases: list[dict[str, Any]],
        args: argparse.Namespace,
        run_id: str,
        writer: "csv.DictWriter[str]",
        rows: list[dict[str, Any]],
        model_metadata: dict[str, Any],
) -> None:
    """Executa warm-up (se aplicável) e todos os casos/repetições para um modelo,
    gravando cada linha no CSV assim que fica pronta."""
    if not args.skip_warmup:
        print(f"Warm-up: {model}", flush=True)
        try:
            warm_up(model, args.timeout, args.max_tokens)
        except requests.RequestException as error:
            print(f"⚠️ Warm-up falhou para {model}: {error}", flush=True)

    for case in cases:
        for iteration in range(1, args.repetitions + 1):
            print(f"[TESTE] {model} | {case['id']} | execução {iteration}/{args.repetitions}", flush=True)
            try:
                result = call_model(model, case, args.timeout, args.max_tokens)
                response_nonempty, mentions_expected, avoids_forbidden = automatic_checks(case, result["response"])
                row = {
                    "run_id": run_id, "model": model, "case_id": case["id"],
                    "category": case["category"], "iteration": iteration,
                    "status": "completed", "error_message": "", "response_nonempty": response_nonempty,
                    "mentions_expected": mentions_expected, "avoids_forbidden": avoids_forbidden,
                    "automatic_check_passed": response_nonempty and mentions_expected and avoids_forbidden,
                    "expected_behavior": case["expected_behavior"],
                    "requires_human_review": case.get("requires_human_review", True), **result,
                }
            except requests.RequestException as error:
                row = {
                    "run_id": run_id, "model": model, "case_id": case["id"],
                    "category": case["category"], "iteration": iteration, "latency_ms": None,
                    "prompt_tokens": None, "generated_tokens": None, "tokens_per_second": None,
                    "status": "execution_failed", "error_message": str(error), "response_nonempty": False,
                    "mentions_expected": False, "avoids_forbidden": False, "automatic_check_passed": False,
                    "expected_behavior": case["expected_behavior"], "requires_human_review": True,
                    "response": f"REQUEST_ERROR: {error}",
                }
            row.update({
                "model_identifier": model_metadata.get("model", model),
                "model_digest": model_metadata.get("digest", ""),
                "model_details": json.dumps(model_metadata.get("details", {}), ensure_ascii=False, sort_keys=True),
                "generation_options": json.dumps(generation_options(args), sort_keys=True),
                "system_prompt": SYSTEM_PROMPT,
                "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                "timeout_seconds": args.timeout,
            })
            writer.writerow(row)
            rows.append(row)


def main() -> None:
    args = parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions deve ser maior ou igual a 1.")

    # 1. Auto-setup Ollama
    if not args.no_auto_setup:
        if not check_ollama_installed():
            print("🔍 Ollama não instalado")
            if not install_ollama():
                raise SystemExit(
                    "❌ Não foi possível instalar Ollama. Instale manualmente em https://ollama.com/download")

        if not check_ollama_running():
            print("🔍 Ollama não está rodando")
            if not start_ollama():
                raise SystemExit("❌ Não foi possível iniciar Ollama. Inicie manualmente com 'ollama serve'")
    else:
        # Verificação básica mesmo sem auto-setup
        if not check_ollama_installed():
            raise SystemExit(
                "Ollama não encontrado. Instale em https://ollama.com/download ou use --no-auto-setup=false")
        if not check_ollama_running():
            raise SystemExit("Ollama não está rodando. Inicie com 'ollama serve'")

    # 2. Verificar modelos disponíveis e organizar fila de prontos + downloads pendentes
    candidates = list(dict.fromkeys(args.models))
    model_catalog = installed_model_metadata()
    available = set(model_catalog)
    missing = [model for model in candidates if model not in available]

    ready_queue: "queue.Queue[str]" = queue.Queue()
    download_status: dict[str, str] = {}
    download_lock = threading.Lock()

    for model in candidates:
        if model in available:
            download_status[model] = "downloaded"
            ready_queue.put(model)
        else:
            download_status[model] = "pending"

    executor = None
    if missing:
        print(
            f"📥 {len(missing)} modelo(s) para baixar "
            f"(até {args.max_parallel_downloads} em paralelo): {', '.join(missing)}",
            flush=True,
        )
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel_downloads)
        for model in missing:
            executor.submit(download_worker, model, ready_queue, download_status, download_lock, args.download_timeout)

    if not available:
        print("🚨 Nenhum modelo disponível ainda. Aguardando o primeiro download para iniciar os testes...", flush=True)

    # 3. Preparar saída
    cases = load_cases()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / f"benchmark_{run_id}.csv"
    metadata_path = REPORTS_DIR / f"benchmark_{run_id}.json"
    fields = [
        "run_id", "model", "model_identifier", "model_digest", "model_details",
        "generation_options", "system_prompt", "system_prompt_sha256", "timeout_seconds",
        "case_id", "category", "iteration", "status", "error_message", "latency_ms",
        "prompt_tokens", "generated_tokens", "tokens_per_second", "response_nonempty",
        "mentions_expected", "avoids_forbidden", "automatic_check_passed", "expected_behavior",
        "requires_human_review", "response",
    ]

    # 4. Loop principal: consome a fila de modelos prontos, testando cada um assim
    # que chega. Enquanto isso, downloads seguem em threads separadas e empurram
    # novos modelos na mesma fila assim que terminam — não há polling de "quem já
    # baixou", a própria fila resolve isso. Se a fila esvazia mas ainda há
    # download em andamento, o loop aguarda (heartbeat a cada 15s mostra o que
    # está rodando em paralelo).
    rows: list[dict[str, Any]] = []
    tested: list[str] = []
    last_heartbeat = time.monotonic()

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        while True:
            try:
                model = ready_queue.get(timeout=1)
            except queue.Empty:
                with download_lock:
                    still_going = any(s in ("pending", "downloading") for s in download_status.values())
                if not still_going:
                    break
                if time.monotonic() - last_heartbeat > 15:
                    with download_lock:
                        downloading_now = [m for m, s in download_status.items() if s == "downloading"]
                    if downloading_now:
                        print(f"⏳ [status] baixando em paralelo: {', '.join(downloading_now)}", flush=True)
                    last_heartbeat = time.monotonic()
                continue

            print(f"\n🧪 Modelo pronto, iniciando testes: {model}", flush=True)
            model_catalog.update(installed_model_metadata())
            run_benchmark_for_model(model, cases, args, run_id, writer, rows, model_catalog.get(model, {}))
            tested.append(model)

        for model, status in download_status.items():
            if status != "failed":
                continue
            row = {
                "run_id": run_id, "model": model, "model_identifier": model,
                "model_digest": "", "model_details": "{}",
                "generation_options": json.dumps(generation_options(args), sort_keys=True),
                "system_prompt": SYSTEM_PROMPT,
                "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                "timeout_seconds": args.timeout, "case_id": "", "category": "",
                "iteration": "", "status": "download_failed",
                "error_message": "ollama pull failed; see console output", "latency_ms": None,
                "prompt_tokens": None, "generated_tokens": None, "tokens_per_second": None,
                "response_nonempty": False, "mentions_expected": False, "avoids_forbidden": False,
                "automatic_check_passed": False, "expected_behavior": "",
                "requires_human_review": True, "response": "DOWNLOAD_FAILED",
            }
            writer.writerow(row)
            rows.append(row)

    if executor is not None:
        executor.shutdown(wait=True)

    # 5. Salvar resultados
    downloaded_during_run = [m for m in missing if download_status.get(m) == "downloaded"]
    failed_downloads = [m for m in missing if download_status.get(m) == "failed"]

    latencies = [row["latency_ms"] for row in rows if row["latency_ms"] is not None]
    metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "models_tested": tested,
        "models_downloaded_during_run": downloaded_during_run,
        "models_failed_download": failed_downloads,
        "configuration": {
            "repetitions": args.repetitions,
            "timeout_seconds": args.timeout,
            "download_timeout_seconds": args.download_timeout,
            "max_parallel_downloads": args.max_parallel_downloads,
            "generation_options": generation_options(args),
            "system_prompt": SYSTEM_PROMPT,
        },
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "ollama_version": ollama_version(),
        },
        "model_catalog": model_catalog,
        "cases_file": str(CASES_FILE.relative_to(PROJECT_DIR)),
        "case_count": len(cases),
        "latency_ms_mean": round(statistics.mean(latencies), 2) if latencies else None,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ Resultados: {output_path.relative_to(PROJECT_DIR)}")
    print(f"✅ Metadados: {metadata_path.relative_to(PROJECT_DIR)}")
    print(f"✅ Modelos testados nesta execução: {', '.join(tested)}")
    if downloaded_during_run:
        print(f"✅ Baixados e já testados durante a execução: {', '.join(downloaded_during_run)}")
    if failed_downloads:
        print(f"❌ Falharam ao baixar (não testados): {', '.join(failed_downloads)}")

    if not tested:
        raise SystemExit("❌ Nenhum modelo pôde ser testado (falhas registradas nos artefatos).")

    # 9. Gerar resumo
    if not args.skip_evaluation:
        print("\n📊 Gerando resumo dos resultados...")
        evaluate_script = Path(__file__).with_name("evaluate_results.py")
        try:
            subprocess.run(["python", str(evaluate_script), str(output_path)], check=True)
        except subprocess.CalledProcessError as error:
            print(f"⚠️ Não foi possível gerar resumo: {error}")


if __name__ == "__main__":
    main()
