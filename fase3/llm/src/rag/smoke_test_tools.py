"""Três perguntas: tool de IMC, conhecimento sem consulta e RAG com fonte."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from medical_assistant.application.contracts import AssistantRequest, PatientContext
from medical_assistant.clinical.patient_database import PatientDatabase
from medical_assistant.clinical.patient_queries import bmi_query
from document_chain_v3 import FT11BackendV3, DocumentChainV3
from document_chain import DocumentaryAnswer
from evaluate_tool_selection import TOOLS, parse_response, strip_termination
from ft11_backend import ROOT, sha
from query_document import check_readiness

SYSTEM = """Você é um assistente médico acadêmico com ferramentas. Para dados individuais e cálculos
do paciente selecionado, solicite a ferramenta apropriada no formato nativo. A aplicação vincula o
paciente; não forneça outro ID. Para perguntas sobre protocolos ou documentos da base, chame
buscar_documentos. Para pergunta educativa geral sem pedido de documento, responda diretamente
com seu conhecimento em exatamente cinco chaves JSON: status, dados (resposta), pendencias,
fonte (null pois nenhuma fonte foi fornecida) e validacao_humana (true). Não chame ferramentas
nesse caso. Não invente valores, não diagnostique nem prescreva. Para ausência de ferramenta,
quando for necessário esclarecer ou recusar, retorne SEM_FERRAMENTA ou SEM_FERRAMENTAS."""

CASES = [
    {"id": "tool", "question": "Confira o índice de massa corporal da paciente selecionada.",
     "patient_selected": True, "expected": "calcular_imc; executar com perfil sintético de 74 kg/1,62 m e obter 28,20 kg/m²"},
    {"id": "knowledge", "question": "Como é herdada a deficiência congênita de plasminogênio? Responda em uma frase, sem avaliar uma pessoa.",
     "patient_selected": False, "expected": "Resposta educativa em JSON sem ferramenta/fonte: herança autossômica recessiva",
     "training_reference": "factual_v2/data/splits/train.jsonl: F2-GHR-0000237-4"},
    {"id": "rag", "question": "Quem assinou a portaria de dislipidemia da base documental?",
     "patient_selected": False, "expected": "buscar_documentos e obter Francisco de Assis Figueiredo e Denizar Vianna com fonte e offsets"},
]


def main():
    output = ROOT / "artifacts/experiments" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_three_routes")
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save("protocol.json", {"cases": CASES, "system": SYSTEM, "tools": TOOLS, "acceptance": "3/3",
        "max_new_tokens": 192, "max_total_tokens": 3072, "seed": 42, "do_sample": False,
        "script_sha256": sha(Path(__file__)), "knowledge_reference_not_sent_to_model": True})
    print(f"Artefatos: {output}", flush=True)
    try:
        receipt = check_readiness()
        backend = FT11BackendV3()
        for key, expected in receipt["model_identity"].items():
            if backend.identity[key] != expected:
                raise ValueError(f"Modelo diferente da seleção: {key}")
        save("identity.json", backend.identity)
        backend.torch.manual_seed(42)
        results = []
        for case in CASES:
            row = {"id": case["id"], "question": case["question"], "passed": False}
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content":
                json.dumps({"patient_selected": case["patient_selected"], "question": case["question"]}, ensure_ascii=False)}]
            row["rendered_prompt"] = backend.tokenizer.apply_chat_template(messages, tools=TOOLS,
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = backend.tokenizer.apply_chat_template(messages, tools=TOOLS, tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True)
            count = inputs["input_ids"].shape[-1]
            if count + 192 > 3072:
                raise ValueError("Orçamento excedido, sem truncamento")
            started = time.perf_counter()
            with backend.torch.inference_mode():
                generated = backend.model.generate(**inputs.to(backend.device), max_new_tokens=192,
                    do_sample=False, pad_token_id=backend.tokenizer.pad_token_id)
            backend.torch.cuda.synchronize()
            tokens = generated[0][count:]
            raw = backend.tokenizer.decode(tokens, skip_special_tokens=False)
            row.update(raw_response=raw, response=strip_termination(raw), input_tokens=count,
                output_tokens=len(tokens), latency_seconds=time.perf_counter() - started)
            try:
                if len(tokens) >= 192:
                    raise ValueError("Saída atingiu limite")
                if case["id"] == "knowledge":
                    answer = DocumentaryAnswer.model_validate(json.loads(row["response"])).model_dump()
                    text = str(answer["dados"].get("resposta", "")).lower()
                    row.update(answer=answer, tools_called=[], rag_used=False)
                    row["passed"] = (answer["status"] == "informativo" and answer["fonte"] is None and
                        answer["validacao_humana"] is True and "autoss" in text and "recessiv" in text)
                else:
                    calls = parse_response(row["response"], "native")
                    row["calls"] = calls
                    expected_name = "calcular_imc" if case["id"] == "tool" else "buscar_documentos"
                    if len(calls) != 1 or calls[0]["name"] != expected_name:
                        raise ValueError("Ferramenta incorreta")
                    if case["id"] == "tool":
                        db = PatientDatabase()
                        try:
                            patient_id = 'paciente_011' if db.get_patient_info('paciente_011') else 'sim_001'
                            context = PatientContext.from_database(db, patient_id)
                        finally:
                            db.close()
                        # Adaptador somente de execução: a seleção acima foi feita pelo modelo.
                        result = bmi_query(AssistantRequest(query="Qual o IMC?", patient_context=context), "smoke_tool")
                        row["execution"] = result.model_dump()
                        row["passed"] = "28,20 kg/m²" in result.response and bool(result.sources)
                    else:
                        state = DocumentChainV3(backend).invoke(calls[0]["arguments"]["pergunta"])
                        row["execution"] = state
                        text = json.dumps(state.get("answer"), ensure_ascii=False)
                        sources = state.get("sources", [])
                        row["passed"] = (not state.get("error") and "FRANCISCO DE ASSIS FIGUEIREDO" in text and
                            "DENIZAR VIANNA" in text and bool(sources) and all(
                                s["text"][s["char_start"]:s["char_end"]] == s["quoted_text"] for s in sources))
            except (ValueError, TypeError, KeyError, RuntimeError) as error:
                row["error"] = str(error)
            results.append(row)
            save(case["id"] + ".json", row)
            print(f"{case['id']}: passed={row['passed']}", flush=True)
        summary = {"passed": sum(r["passed"] for r in results), "count": 3,
            "approved_for_tools_package": all(r["passed"] for r in results),
            "clinical_validation": False, "generalization_proven": False,
            "gpu": backend.torch.cuda.get_device_name(),
            "peak_allocated_vram_bytes": backend.torch.cuda.max_memory_allocated()}
        save("summary.json", summary)
        print(json.dumps(summary), flush=True)
    except Exception as error:
        save("error.json", {"type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
