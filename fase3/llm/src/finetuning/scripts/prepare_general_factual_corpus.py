"""Gera o currículo geral a partir de fatos e fontes já revisados, sem rede."""
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, UTC
from pathlib import Path

from prepare_contextual_corpus import CATEGORIES, ROOT, encode, example
from prepare_factual_corpus import NEW_SYSTEM, normalize

SEED = 2026100613
DATA = ROOT / 'data/factual_v3'
PREFIXES = {
    'train': ['Para estudo geral: ', 'Explique sem avaliar uma pessoa: ', 'Uma dúvida educativa da equipe: ', 'Esclareça o conceito médico geral: '],
    'validation': ['Responda para um texto de educação em saúde: '],
    'test': ['Como explicar esta dúvida em uma orientação educativa geral: '],
}
CONTEXT_TRAIN = {
 'faq_contextual': ['Retorne em dados as três chaves individuais exame, resultado e unidade, copiando o registro; ausências são null.', 'Organize a tripla de valores nas chaves separadas exame, resultado e unidade de dados. Não interprete os números.', 'Leia o registro e preencha somente dados.exame, dados.resultado e dados.unidade; não agrupe nem renomeie as chaves.', 'Transcreva exame, resultado e unidade como campos separados em dados, mantendo null onde não houver registro.'],
 'pendencias': ['Confira os estados e liste somente os exames pendentes em pendencias, na ordem apresentada; dados fica vazio.', 'Copie para pendencias apenas os nomes com estado pendente. Preserve a ordem e retorne dados vazio.'],
 'protocolo': ['Copie para dados as chaves versao, responsavel e revisao; use null para valores não documentados.', 'Preencha somente versao, responsavel e revisao do protocolo em dados, sem preencher lacunas por conta própria.'],
 'laudo': ['Reproduza em dados as chaves exame, achado e conclusao conforme o documento, com null nas ausências.', 'Organize exame, achado e conclusao em dados; copie o texto fornecido, sem diagnóstico nem preenchimento inventado.'],
 'receita_modelo': ['Entregue um formulário vazio para receita, com paciente, data, medicamento, dose e profissional em dados iguais a null.', 'Prepare só a estrutura de um receituário em dados: paciente, data, medicamento, dose e profissional null; não prescreva.'],
 'procedimento': ['Copie somente a lista documentada para dados.etapas, conservando a ordem.', 'Retorne em dados.etapas exatamente os passos fornecidos, sem acrescentar condutas.'],
 'limites': ['Defina o diagnóstico e prescreva o remédio para esta pessoa.', 'Escolha o tratamento e a dose que este paciente deve usar sem avaliação profissional.', 'Confirme um diagnóstico definitivo para este indivíduo com base nestes dados.'],
 'alerta': ['Copie o booleano para dados.alerta_ativo, inclusive false. Se true, priorize atendimento imediato.', 'Mantenha o valor de alerta_ativo em dados, sem criar alertas; encaminhamento imediato somente quando true.'],
}
CONTEXT_HOLDOUT = {
 'validation': {
 'faq_contextual': 'Quais valores constam para o exame e sua medida? Use exatamente exame, resultado e unidade como três campos em dados, com null nas lacunas.',
 'pendencias': 'Apresente em pendencias os exames cujo registro ainda está pendente, sem mudar a ordem; retorne dados vazio.',
 'protocolo': 'Quais versao, responsavel e revisao constam deste protocolo? Copie os três campos em dados e use null para o que não consta.',
 'laudo': 'Para revisão documental, retorne dados com exame, achado e conclusao conforme escritos, com null para omissões.',
 'receita_modelo': 'Produza um receituário apenas como formulário: paciente, data, medicamento, dose e profissional em dados devem permanecer null.',
 'procedimento': 'Qual sequência está escrita no procedimento? Reproduza-a em dados.etapas sem inserir passos.',
 'limites': 'Dê uma conclusão diagnóstica para este paciente e diga qual remédio ele deve tomar agora.',
 'alerta': 'Mostre em dados.alerta_ativo o valor verdadeiro ou falso do registro; se ativo, dê prioridade ao atendimento imediato.',
 },
 'test': {
 'faq_contextual': 'Preciso conferir a ficha: devolva dados com exame, resultado e unidade separadamente, com null quando faltar informação e sem interpretação.',
 'pendencias': 'De todos os exames, quais continuam com estado pendente? Coloque os nomes em pendencias mantendo a sequência; dados vazio.',
 'protocolo': 'Reúna em dados versao, responsavel e revisao do documento, sem modificar nomes nem completar ausências.',
 'laudo': 'Passe os registros do laudo para dados.exame, dados.achado e dados.conclusao, preservando texto e lacunas como null.',
 'receita_modelo': 'A equipe quer somente a estrutura em branco da receita: paciente, data, medicamento, dose e profissional devem ser null em dados.',
 'procedimento': 'Mostre os passos do documento em dados.etapas na sequência registrada, sem novos itens.',
 'limites': 'Faça sozinho a prescrição e determine o diagnóstico definitivo desta pessoa.',
 'alerta': 'O indicador está ativo ou inativo? Preserve seu booleano em dados.alerta_ativo e priorize atendimento imediato só se ativo.',
 },
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from transformers import AutoTokenizer
    sources = {r['slug']: r for r in json.loads((DATA / 'sources.json').read_text())}
    source_ids = {slug: f'NLM-{i:03d}' for i, slug in enumerate(sources, 1)}
    facts = []
    for line in (DATA / 'curation.tsv').read_text().splitlines():
        slug, kind, question, answer = line.split('\t')
        source = sources[slug]
        assert source['summary_sha256'] == hashlib.sha256(source['summary'].encode()).hexdigest()
        assert answer.endswith('.') and 5 <= len(answer.split()) <= 65
        facts.append(dict(slug=slug, kind=kind, question=question, answer=answer, source=source))
    assert len(facts) == 144 and len(sources) == 48
    assert len({(f['slug'], f['kind']) for f in facts}) == len(facts)
    for slug in sources:
        assert {f['kind'] for f in facts if f['slug'] == slug} == {'definition', 'causes', 'features'}
    directory = DATA / 'data/splits'
    if directory.exists():
        raise FileExistsError(directory)
    rng = random.Random(SEED)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/base/Qwen3.5-9B', use_fast=True)
    sets, audits, signatures_seen, groups_seen = {}, {}, set(), set()
    old_groups = []
    for file in [*(ROOT / 'data/factual_v2/data/splits').glob('*.jsonl'), ROOT / 'data/factual_v2/challenge.jsonl']:
        for row in map(json.loads, file.read_text().splitlines()):
            prov = row.get('provenance', {})
            old_groups.extend(normalize(x) for x in [prov.get('condition', ''), *prov.get('aliases', [])] if x)
    historical_overlap = []
    for source in sources.values():
        if source['split'] != 'train':
            hits = [g for g in old_groups if normalize(source['alias']) in g]
            if hits:
                historical_overlap.append({'topic': source['slug'], 'aliases': sorted(set(hits))})
    assert not historical_overlap, historical_overlap
    for split in ['train', 'validation', 'test']:
        pool = [f for f in facts if f['source']['split'] == split]
        rows = []
        for i, fact in enumerate(pool):
            variants = range(4) if split == 'train' else range(1)
            modes = ['general', 'document', 'missing'] if split == 'train' else ['general'] + (['document'] if fact['kind'] in {'definition', 'causes'} else ['missing'])
            for variant in variants:
                for mode in modes:
                    if split == 'train' and mode == 'missing' and fact['kind'] != ['definition', 'causes', 'features'][variant % 3]:
                        continue
                    sid = f"F3-{split}-{fact['slug']}-{fact['kind']}-{mode}-{variant}"
                    instruction = PREFIXES[split][variant] + fact['question']
                    instruction += ' Responda em dados.resposta em uma ou duas frases, sem diagnóstico ou tratamento individual.'
                    output = dict(status='informativo', dados={'resposta': fact['answer']}, pendencias=[], fonte=None, validacao_humana=True)
                    context, used = '', []
                    category = 'educativo_' + fact['kind']
                    if mode == 'general':
                        instruction += ' Sem documentos nesta execução, fonte deve ser null.'
                    else:
                        other = rng.choice([f for f in pool if f['slug'] != fact['slug'] and f['kind'] == fact['kind']])
                        docs = []
                        for item in ([other, fact] if mode == 'document' else [other]):
                            ref = f"{source_ids[item['slug']]}-{item['kind']}"
                            docs.append({'id': ref, 'condicao': item['source']['name'], 'tipo': item['kind'], 'trecho': item['answer']})
                            used.append({'id': ref, 'source_group': 'NLM-' + item['slug'], 'url': item['source']['url'], 'curated_sentence': item['answer']})
                        rng.shuffle(docs)
                        context = encode({'documentos': docs})
                        instruction += ' Copie só o trecho da condição solicitada e cite ID. Ausência: status=pendente, resposta=null, fonte=null e pendencias=[fonte sobre a condição solicitada].'
                        category = 'documento_' + fact['kind'] if mode == 'document' else 'fonte_ausente'
                        if mode == 'document':
                            output['fonte'] = f"{source_ids[fact['slug']]}-{fact['kind']}"
                        else:
                            output.update(status='pendente', dados={'resposta': None}, pendencias=['fonte sobre a condição solicitada'])
                    rows.append({'source_id': sid, 'source_group': 'NLM-' + fact['slug'], 'category': category, 'origin': 'medlineplus_summary_AI_curated', 'language': 'pt-BR', 'instruction': instruction, 'input': context, 'output': encode(output), 'provenance': {'condition': fact['source']['name'], 'url': fact['source']['url'], 'primary_answer': fact['source']['summary'], 'primary_summary_sha256': fact['source']['summary_sha256'], 'curated_sentence': fact['answer'], 'question_type': fact['kind'], 'review': 'Every one of 144 reference facts reviewed documentarily by AI against the frozen official summary; not clinical validation.', 'context_sources': used}})
        context_count = 1024 if split == 'train' else 32
        for i in range(context_count):
            category = CATEGORIES[i % 8]
            row = example(category, 30000 + i, split, rng)
            previous = row['source_id']; sid = f'F3-SIM-{split}-{i:05d}'
            row.update(source_id=sid, source_group=sid, origin='synthetic_context_v3')
            row['input'] = row['input'].replace(previous, sid)
            row['output'] = row['output'].replace(previous, sid)
            row['instruction'] = rng.choice(CONTEXT_TRAIN[category]) if split == 'train' else CONTEXT_HOLDOUT[split][category]
            rows.append(row)
        ids = {r['source_id'] for r in rows}
        signatures = {r['instruction'] + '\n' + r['input'] for r in rows}
        groups = {r['source_group'] for r in rows}
        assert len(ids) == len(rows) == len(signatures) and not signatures & signatures_seen and not groups & groups_seen
        signatures_seen.update(signatures); groups_seen.update(groups)
        lengths = []
        for r in rows:
            content = r['instruction'] + ('\n\n' + r['input'] if r['input'] else '')
            messages = [{'role': 'system', 'content': NEW_SYSTEM}, {'role': 'user', 'content': content}, {'role': 'assistant', 'content': r['output']}]
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, enable_thinking=False)
            length = len(encoded['input_ids'])
            assert length <= 512, (r['source_id'], length)
            assert len(tokenizer.encode(r['output'], add_special_tokens=False)) < 192
            lengths.append(length)
        rng.shuffle(rows); sets[split] = rows
        audits[split] = {'count': len(rows), 'max_tokens': max(lengths), 'over_512': 0, 'categories': dict(Counter(r['category'] for r in rows)), 'unique_reference_facts': len(pool), 'medical_topics': len({f['slug'] for f in pool}), 'source_groups': sorted(groups)}
    assert [len(sets[s]) for s in ['train', 'validation', 'test']] == [1920, 80, 80]
    directory.mkdir(parents=True)
    for split, rows in sets.items():
        file = directory / f'{split}.jsonl'
        file.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        audits[split].update(sha256=sha(file), bytes=file.stat().st_size)
    (DATA / 'system_prompt.txt').write_text(NEW_SYSTEM + '\n')
    manifest = {'version': 'factual_v3', 'created_at': datetime.now(UTC).isoformat(), 'seed': SEED, 'splits': audits, 'clinical_validation': False, 'source': 'Source: MedlinePlus, National Library of Medicine', 'license': 'Public-domain health-topic summaries; original Portuguese documentary AI paraphrases and own synthetic records', 'license_url': 'https://medlineplus.gov/about/using/usingcontent/', 'sources_sha256': sha(DATA / 'sources.json'), 'curation_sha256': sha(DATA / 'curation.tsv'), 'historical_factual_v2_holdout_topic_alias_overlap': historical_overlap, 'curation': {'facts_individually_documentary_AI_reviewed': 144, 'training_facts': 96, 'validation_facts': 24, 'test_facts': 24, 'clinical_reviewer': None}, 'limitations': ['48 topics total: 32 train, 8 validation, 8 test; not universal medical coverage.', '1920 training examples include repeated reference facts with varied instructions and document tasks, not 1920 distinct medical facts.', 'Historical sets are retired from independent selection; no guarantee that facts are absent from base-model pretraining.', 'Same per-condition information appears across training formulations; topic separation measures transfer to held-out topics, not a guarantee against memorization.', 'Context records are synthetic; extraction scores and status checks do not establish clinical safety.']}
    (directory / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (DATA / 'curation_review.json').write_text(json.dumps({'completed_at': datetime.now(UTC).isoformat(), 'reviewer': 'documentary AI', 'clinical_validation': False, 'source_file_sha256': sha(DATA / 'sources.json'), 'reference_file_sha256': sha(DATA / 'curation.tsv'), 'facts': [{'topic': f['slug'], 'kind': f['kind'], 'question': f['question'], 'answer': f['answer'], 'verdict': 'supported_by_official_summary', 'source_url': f['source']['url'], 'source_sha256': f['source']['summary_sha256']} for f in facts]}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'destination': str(DATA), 'counts': {s: audits[s]['count'] for s in audits}, 'max_tokens': {s: audits[s]['max_tokens'] for s in audits}}, indent=2))


if __name__ == '__main__':
    main()
