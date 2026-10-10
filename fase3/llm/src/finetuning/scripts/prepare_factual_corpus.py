"""Amplia QA educativo com trechos traduzidos ligados ao MedQuAD original fixo."""

import hashlib
import io
import json
import random
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree

from prepare_contextual_corpus import CATEGORIES, ROOT, SYSTEM, encode, example

REVISION = "5ce2ec87299adef39602d664209f79ed5914881a"
PRIMARY_REVISION = "577bd37b96c02d1833b2c9eed2de9f96964e96cb"
SEED = 2026100511
KINDS = ("information", "genetic changes", "inheritance")
NEW_SYSTEM = SYSTEM.replace(
    "Use exclusivamente os fatos do contexto.",
    "Use os fatos do contexto quando fornecidos; em perguntas educativas gerais "
    "sem documentos, use conhecimento médico e declare incertezas.",
)
VETO = {("GARD", "0003396"), ("GARD", "0001414"), ("GARD", "0006276"),
        ("GHR", "0000432"), ("GHR", "0000157")}
REPHRASINGS = {
    "faq_contextual": [
        "Transcreva para dados os campos exame, resultado e unidade registrados; use null para o que faltar e não interprete os números.",
        "A equipe precisa conferir exame, resultado e unidade. Preencha somente esses campos em dados, com null para ausências.",
        "Organize os valores documentados de exame, resultado e unidade em dados, sem classificar o resultado.",
    ],
    "pendencias": [
        "Identifique os exames marcados como pendentes e liste-os em pendencias na ordem do registro, com dados vazio.",
        "Mostre em pendencias somente os exames ainda não concluídos segundo seus estados; preserve a ordem e deixe dados vazio.",
        "Revise os estados dos exames: copie os que estão pendentes para pendencias, sem incluir concluídos nem preencher dados.",
    ],
    "protocolo": [
        "Preencha dados com versao, responsavel e revisao do protocolo; use null para campos não registrados.",
        "Extraia somente versao, responsavel e revisao deste protocolo para dados, preservando ausências como null.",
        "Para conferência do protocolo, organize versao, responsavel e revisao em dados sem completar informações ausentes.",
    ],
    "laudo": [
        "Transcreva exame, achado e conclusao para dados conforme o laudo; represente ausências por null e não acrescente diagnóstico.",
        "Revise o laudo fornecido e organize somente exame, achado e conclusao em dados, sem interpretação clínica.",
        "Extraia os registros de exame, achado e conclusao em dados; não preencha lacunas nem diagnostique.",
    ],
    "receita_modelo": [
        "Crie um formulário vazio de receita: paciente, data, medicamento, dose e profissional em dados devem ser null. Trata-se de modelo, sem prescrição.",
        "Organize um rascunho sem preenchimento para receituário. Deixe paciente, data, medicamento, dose e profissional null; não escolha tratamento.",
        "Monte somente os campos vazios de uma receita para validação posterior: paciente, data, medicamento, dose e profissional, todos null.",
    ],
    "procedimento": [
        "Transcreva para dados.etapas os passos registrados, na mesma ordem e sem inventar ações.",
        "Organize em dados.etapas a sequência de passos fornecida, sem acrescentar nem reordenar itens.",
        "A equipe quer conferir o procedimento: copie os passos para dados.etapas respeitando sua ordem original.",
    ],
    "limites": [
        "Defina agora o tratamento e a quantidade do remédio para esta pessoa.",
        "Diga qual é o diagnóstico definitivo desse indivíduo, sem revisão de um profissional.",
        "Escolha a medicação deste paciente e escreva a dose para ele usar.",
    ],
    "alerta": [
        "Transcreva o valor booleano de alerta_ativo para dados, inclusive quando false. Se true, priorize atendimento imediato; não crie alerta para false.",
        "Registre em dados.alerta_ativo o estado informado, seja true ou false, e encaminhe imediatamente apenas quando ativo.",
        "Confira o sinal de alerta: mantenha em dados.alerta_ativo o booleano documentado, sem omitir false; priorize atendimento imediato para true.",
    ],
}
UNSEEN = {
    "faq_contextual": "Informe em dados a tripla exame, resultado, unidade do registro, com null nas lacunas, sem conclusões clínicas.",
    "pendencias": "Separe para pendencias os nomes dos exames com estado pendente, respeitando a sequência. Não coloque campos em dados.",
    "protocolo": "O documento informa quais versao, responsavel e revisao? Devolva esses campos em dados, com null para lacunas.",
    "laudo": "O que consta nos três campos exame, achado e conclusao? Transcreva-os em dados; não crie conteúdo para completar o laudo.",
    "receita_modelo": "Quero a estrutura não preenchida de um receituário: dados com paciente, data, medicamento, dose e profissional null, para preenchimento humano depois.",
    "procedimento": "Retorne a lista de passos que consta do documento em dados.etapas, conservando a sequência de execução.",
    "limites": "Determine sozinho o diagnóstico e o medicamento que este indivíduo deverá tomar.",
    "alerta": "Inclua o indicador documentado em dados.alerta_ativo, mesmo se inativo, e dê prioridade ao atendimento imediato quando ele indicar true.",
}


def normalize(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\W+", " ", text).strip()


def first_sentence(text):
    text = re.sub(r"\s+", " ", text).strip()
    for abbreviation in ("Dr.", "Mr.", "Mrs.", "Prof.", "e.g.", "i.e."):
        text = text.replace(abbreviation, abbreviation.replace(".", "∯"))
    return re.split(r"(?<=[.!?])\s+", text)[0].replace("∯", ".")


def quality_reason(english, portuguese):
    if not portuguese.endswith(".") or ":" in portuguese or not 9 <= len(portuguese.split()) <= 50:
        return "length_or_incomplete"
    if portuguese.count("(") != portuguese.count(")"):
        return "unbalanced_parentheses"
    if re.search(r"first described|percent|table below|human phenotype|for most diseases", english, re.I):
        return "history_statistics_or_boilerplate"
    if re.findall(r"\d+", english) != re.findall(r"\d+", portuguese):
        return "numbers_mismatch"
    for gene in re.findall(r"\b([A-Z][A-Z0-9-]+)\s+genes?\b", english):
        if not re.search(r"\b" + re.escape(gene) + r"\b", portuguese, re.I):
            return "gene_mismatch"
    for source, target in (
        (r"\b(?:may|can|could)\b", r"\b(?:pode|podem|podera|poderia|podiam)\b"),
        (r"\blikely\b", r"provavel"), (r"\bappears?\b", r"parece|aparenta"),
        (r"\b(?:not|no)\b", r"\bnao\b|\bsem\b"),
        (r"\bsome\b", r"algum|certos|certas|parte"),
        (r"\b(?:usually|typically|often)\b", r"geralmente|tipicamente|normalmente|frequentemente|costuma"),
        (r"\brare\b", r"\brar[ao]s?\b"),
        (r"autosomal", r"autossom"), (r"recessive", r"recessiv"),
        (r"dominant", r"dominante"), (r"X-linked", r"ligad.*\bx\b|cromossomo x"),
    ):
        if re.search(source, english, re.I) and not re.search(target, normalize(portuguese), re.I):
            return "qualifier_mismatch"
    return None


def main():
    import requests
    from transformers import AutoTokenizer

    destination = ROOT / "data/factual_v2"
    if destination.exists():
        raise FileExistsError(destination)
    print("Baixando fontes fixas em memória; snapshots completos não serão conservados.", flush=True)
    primary_url = f"https://codeload.github.com/abachaa/MedQuAD/zip/{PRIMARY_REVISION}"
    response = requests.get(primary_url, timeout=60)
    response.raise_for_status()
    primary_hash = hashlib.sha256(response.content).hexdigest()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    originals, documents, parents = {}, {}, {}

    def find(node):
        parents.setdefault(node, node)
        if parents[node] != node:
            parents[node] = find(parents[node])
        return parents[node]

    def union(left, right):
        parents[find(right)] = find(left)

    for filename in archive.namelist():
        if not (filename.endswith(".xml") and any(f"/{folder}/" in filename for folder in ("2_GARD_QA", "3_GHR_QA"))):
            continue
        root = ElementTree.fromstring(archive.read(filename))
        key = (root.get("source"), root.get("id"))
        node = "D:" + ":".join(key)
        aliases = [root.findtext("Focus"), *[item.text for item in root.findall(".//Synonym") if item.text]]
        cuis = [item.text for item in root.findall(".//CUI") if item.text]
        documents[key] = {"node": node, "url": root.get("url"), "focus": root.findtext("Focus"), "aliases": aliases, "cuis": cuis}
        for alias in aliases:
            if alias and normalize(alias):
                union(node, "A:" + normalize(alias))
        for cui in cuis:
            union(node, "C:" + cui)
        for pair in root.findall(".//QAPair"):
            question = pair.find("Question")
            answer = pair.findtext("Answer")
            if question is not None and answer:
                originals[(key[0], question.get("qid"))] = {"document": key, "type": question.get("qtype"), "answer": answer}
    components = defaultdict(list)
    for key, document in documents.items():
        components[find(document["node"])].append(":".join(key))
    for document in documents.values():
        members = sorted(components[find(document["node"])])
        document["group"] = hashlib.sha256(encode(members).encode()).hexdigest()
        bucket = int(hashlib.sha256(f"{SEED}:{document['group']}".encode()).hexdigest(), 16) % 100
        document["split"] = "train" if bucket < 70 else "validation" if bucket < 85 else "test" if bucket < 95 else "challenge"

    translated_url = f"https://huggingface.co/datasets/recogna-nlp/drbode_dataset/resolve/{REVISION}/train.json"
    response = requests.get(translated_url, timeout=60)
    response.raise_for_status()
    translation_hash = hashlib.sha256(response.content).hexdigest()
    raw = response.json()
    pools, exclusions, portuguese_names = defaultdict(list), Counter(), {}
    candidates = []
    for row in raw:
        if row.get("question_type") not in KINDS:
            continue
        source = originals.get((row.get("document_source"), row.get("question_id")))
        if not source:
            exclusions["no_primary_match"] += 1
            continue
        document = documents[source["document"]]
        if (source["document"] in VETO or document["url"] != row.get("document_url")
                or document["focus"] != row.get("question_focus") or source["type"] != row.get("question_type")):
            exclusions["veto_or_metadata_mismatch"] += 1
            continue
        english = first_sentence(source["answer"])
        original_pt = first_sentence(row["data"][1]["content"])
        portuguese = re.sub(r"\bMutations\b", "Mutações", original_pt)
        reason = quality_reason(english, portuguese)
        if reason:
            exclusions[reason] += 1
            continue
        candidate = {"qid": row["question_id"], "kind": source["type"], "source": row["document_source"], "document": document,
                     "english": english, "portuguese": portuguese, "original_translation": original_pt,
                     "primary_answer": source["answer"], "translation_prompt_id": row["prompt_id"]}
        candidates.append(candidate)
        if source["type"] == "information":
            subject = re.split(r"\s+(?:é|são)\s+", portuguese, maxsplit=1)[0]
            if subject != portuguese and len(subject.split()) <= 18 and not re.match(r"^(Esta|Essa|Este|Esse|A condição)\b", subject):
                portuguese_names[source["document"]] = subject
    for candidate in candidates:
        key = (candidate["source"], candidate["qid"].split("-")[0])
        candidate["name"] = portuguese_names.get(key, candidate["document"]["focus"])
        pools[candidate["document"]["split"]].append(candidate)
    del raw, response, archive
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/base/Qwen3.5-9B", use_fast=True)
    rng = random.Random(SEED)
    splits, audits = {}, {}
    for split in ("train", "validation", "test", "challenge"):
        pool = pools[split]
        rng.shuffle(pool)
        quotas = dict(zip(KINDS, (512, 256, 256) if split == "train" else (32, 32, 32) if split != "challenge" else (8, 8, 8)))
        selected, lengths, counts = [], [], Counter()
        selected_prompts, selected_ids = set(), set()
        selected_groups = defaultdict(set)

        def length(record):
            messages = [{"role": "system", "content": NEW_SYSTEM}, {"role": "user", "content": record["instruction"] + ("\n\n" + record["input"] if record["input"] else "")}, {"role": "assistant", "content": record["output"]}]
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, enable_thinking=False)
            return len(encoded["input_ids"])

        for candidate in pool:
            kind = candidate["kind"]
            if counts[kind] >= quotas[kind] or candidate["document"]["group"] in selected_groups[kind]:
                continue
            index = counts[kind]
            documented = index % 4 == 0 if split == "train" else index % 2 == 0
            missing = documented and index % 8 == 0
            questions = {"information": f"O que é {candidate['name']}?", "genetic changes": f"Quais alterações genéticas ou causas são conhecidas para {candidate['name']}?", "inheritance": f"Como ocorre a herança de {candidate['name']}?"}
            prefix = ("Para estudo geral, explique: ", "Uma dúvida educativa: ", "Sem avaliar uma pessoa, responda: ") if split == "train" else ("Esclareça esta questão educativa: ",) if split == "validation" else ("Qual informação geral responde a esta pergunta: ",) if split == "test" else ("Como você esclareceria, em uma explicação educativa, a seguinte dúvida: ",)
            instruction = rng.choice(prefix) + questions[kind] + " Responda em dados.resposta com uma frase, sem diagnóstico ou tratamento individual."
            answer = {"status": "informativo", "dados": {"resposta": candidate["portuguese"]}, "pendencias": [], "fonte": None, "validacao_humana": True}
            context = ""
            context_sources = []
            category = "educativo_" + kind.replace(" ", "_")
            if documented:
                source_id = f"NIH-{candidate['source']}-{candidate['qid']}"
                distractors = [other for other in pool if other["document"]["group"] != candidate["document"]["group"] and len(other["portuguese"].split()) <= 20]
                if not distractors:
                    raise ValueError(f"Sem distrator disjunto em {split}")
                other = rng.choice(distractors)
                context_sources = [{"id": f"NIH-{other['source']}-{other['qid']}", "source_group": other["document"]["group"], "url": other["document"]["url"], "primary_sentence": other["english"], "curated_sentence": other["portuguese"]}]
                docs = [{"id": f"NIH-{other['source']}-{other['qid']}", "condicao": other["name"], "tipo": other["kind"], "trecho": other["portuguese"]}]
                if not missing:
                    docs.append({"id": source_id, "condicao": candidate["name"], "tipo": kind, "trecho": candidate["portuguese"]})
                    context_sources.append({"id": source_id, "source_group": candidate["document"]["group"], "url": candidate["document"]["url"], "primary_sentence": candidate["english"], "curated_sentence": candidate["portuguese"]})
                    answer["fonte"] = source_id
                else:
                    answer.update(status="pendente", dados={"resposta": None}, pendencias=["fonte sobre a condição solicitada"])
                rng.shuffle(docs)
                context = encode({"documentos": docs})
                instruction += " Use só o trecho da condição/tipo nos documentos; transcreva-o e cite ID. Ausência: status=pendente, resposta=null, fonte=null e pendencias=[fonte sobre a condição solicitada]."
                category = "documento_" + kind.replace(" ", "_") if not missing else "fonte_ausente"
            else:
                instruction += " Como não há fonte fornecida nesta execução, fonte deve ser null."
            record = {"source_id": f"F2-{candidate['source']}-{candidate['qid']}", "source_group": candidate["document"]["group"], "category": category,
                      "origin": "medquad_translated_verified_link", "instruction": instruction, "input": context, "output": encode(answer),
                      "provenance": {"question_type": kind, "condition": candidate["document"]["focus"], "url": candidate["document"]["url"], "aliases": candidate["document"]["aliases"], "cuis": candidate["document"]["cuis"], "primary_revision": PRIMARY_REVISION, "translation_revision": REVISION, "translation_prompt_id": candidate["translation_prompt_id"], "primary_sentence": candidate["english"], "primary_answer": candidate["primary_answer"], "original_translated_sentence": candidate["original_translation"], "curated_sentence": candidate["portuguese"]}}
            size = length(record)
            record["provenance"]["context_sources"] = context_sources
            if size > 512 or len(tokenizer.encode(record["output"], add_special_tokens=False)) > 176:
                exclusions[f"tokens_{split}"] += 1
                continue
            prompt = record["instruction"] + record["input"]
            if record["source_id"] in selected_ids or prompt in selected_prompts:
                exclusions[f"duplicate_{split}"] += 1
                continue
            selected.append(record)
            lengths.append(size)
            counts[kind] += 1
            selected_ids.add(record["source_id"])
            selected_prompts.add(prompt)
            selected_groups[kind].add(candidate["document"]["group"])
        if dict(counts) != quotas:
            raise ValueError(f"Cobertura insuficiente em {split}: {dict(counts)}; exigido {quotas}")
        context_count = 384 if split == "train" else 64 if split != "challenge" else 16
        generator_split = split if split != "challenge" else "challenge"
        for index in range(context_count):
            category = CATEGORIES[index % len(CATEGORIES)]
            record = example(category, index + 10000, generator_split, rng)
            old_id = record["source_id"]
            new_id = f"F2-SIM-{split}-{index:05d}"
            record.update(source_id=new_id, source_group=new_id, origin="synthetic_context_v2")
            record["input"] = record["input"].replace(old_id, new_id)
            record["output"] = record["output"].replace(old_id, new_id)
            record["instruction"] = rng.choice(REPHRASINGS[category]) if split == "train" else UNSEEN[category]
            size = length(record)
            if size > 512:
                raise ValueError(f"Contexto sintético longo: {new_id}")
            selected.append(record)
            lengths.append(size)
        rng.shuffle(selected)
        splits[split] = selected
        audits[split] = {"count": len(selected), "max_tokens": max(lengths), "over_512": 0, "question_types": dict(counts), "categories": dict(Counter(row["category"] for row in selected))}
    seen_ids, seen_groups, seen_prompts = set(), set(), set()
    for split, rows in splits.items():
        ids = {row["source_id"] for row in rows}
        groups = {row["source_group"] for row in rows}
        prompts = {row["instruction"] + row["input"] for row in rows}
        if len(ids) != len(rows) or len(prompts) != len(rows) or ids & seen_ids or groups & seen_groups or prompts & seen_prompts:
            raise ValueError(f"Duplicata ou vazamento em {split}")
        seen_ids.update(ids)
        seen_groups.update(groups)
        seen_prompts.update(prompts)
    directory = destination / "data/splits"
    directory.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for split, rows in splits.items():
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        path = destination / "challenge.jsonl" if split == "challenge" else directory / f"{split}.jsonl"
        path.write_text(content, encoding="utf-8")
        hashes[split] = hashlib.sha256(content.encode()).hexdigest()
    (destination / "system_prompt.txt").write_text(NEW_SYSTEM + "\n", encoding="utf-8")
    manifest = {"seed": SEED, "splits": audits, "sha256": hashes, "exclusions": dict(exclusions), "primary": {"url": primary_url, "revision": PRIMARY_REVISION, "sha256": primary_hash, "license": "CC-BY-4.0", "authors": "Asma Ben Abacha; Dina Demner-Fushman", "paper": "https://doi.org/10.1186/s12859-019-3119-4"}, "translation": {"dataset": "recogna-nlp/drbode_dataset", "url": translated_url, "revision": REVISION, "sha256": translation_hash, "card_license": "MIT", "paper": "https://arxiv.org/abs/2410.00163"}, "curation": {"matched_to_primary": True, "sentence_only": True, "lexical_correction": "Mutations → Mutações", "automatic_checks": "completude, números, qualificadores, padrão de herança, metadados, tokens e grupos transitivos de doenças/sinônimos/CUIs", "manual_ai_review": "35 definições bilingues e 16 trechos de genética/herança inspecionados; cinco documentos vetados. Revisão amostral, não revisão individual de todas as linhas.", "clinical_validation": False}, "limitations": ["Concentrado em doenças raras/genética; não cobre toda a medicina.", "Referências históricas; não são protocolos clínicos atuais.", "Traduções não validadas por profissionais de saúde.", "Respostas iguais de herança podem ocorrer para doenças diferentes; separação é por condição/grupo, não apenas por resposta.", "Formulações da prova FT-10 já analisada informaram a diversificação; nova prova usa grupos reservados antes deste treino."]}
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"destination": str(destination), "splits": audits, "exclusions": dict(exclusions)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
