"""Prepara MedPT para continuar FT-11, sem substituir corpora históricos."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from finetuning.data.ingest_medpt import normalize, record_from_row, file_sha256

CONTACT = re.compile(r'https?://|www\.|[\w.+-]+@[\w.-]+\.\w+|instagram|whatsapp|doctoralia|\bcrm\b|\b(?:dr|dra)\.\s|\b\d{3}\.\d{3}\.\d{3}-\d{2}\b|\(\d{2}\)\s*\d{4,5}[- ]?\d{4}', re.I)
DIRECTIVE = re.compile(r'\b(?:tome|aplique|prescrevo|prescreva|receite|inicie|suspenda)\b|\b(?:deve|pode|posso)\s+(?:tomar|usar|aumentar|reduzir)\b', re.I)


def example(row):
    record = record_from_row(row)
    if record is None:
        return None, 'empty_or_excluded_type'
    question, answer = record['instruction'], record['output']
    if CONTACT.search(question+'\n'+answer):
        return None, 'contact_or_promotion'
    if DIRECTIVE.search(normalize(question+' '+answer)):
        return None, 'direct_medication_instruction'
    if len(answer.split()) < 8 or len(set(normalize(answer).split())) < 6:
        return None, 'short_or_repetitive'
    words=normalize(answer).split()
    trigrams=[tuple(words[i:i+3]) for i in range(len(words)-2)]
    if trigrams and max(Counter(trigrams).values()) > 3:
        return None, 'repeated_trigram'
    record.update(source_group='MEDPT-condition:'+normalize(record.get('condition') or question),
        output=json.dumps({'status':'informativo','dados':{'resposta':answer},'pendencias':[],
                           'fonte':None,'validacao_humana':True},ensure_ascii=False),
        license='not_declared_in_downloaded_card',
        provenance={'dataset':'AKCIT/MedPT','original_id':row['id'],
                    'transformation':'Original answer wrapped in FT-11 JSON; no medical rewriting.',
                    'clinical_validation':False})
    return record, None


def partition(group, seed):
    bucket=int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:8],16)%100
    return 'train' if bucket<80 else 'validation' if bucket<90 else 'test'


def full_tokens(tokenizer, row, system):
    encoded=tokenizer.apply_chat_template([
        {'role':'system','content':system},
        {'role':'user','content':row['instruction']+ ('\n\n'+row['input'] if row['input'] else '')},
        {'role':'assistant','content':row['output']}],tokenize=True,enable_thinking=False,return_dict=True)
    return len(encoded['input_ids'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sample-size',type=int,default=0,help='0 (padrão) usa todos os exemplos aproveitáveis; valor positivo limita a amostra.')
    parser.add_argument('--seed',type=int,default=2026101914)
    args=parser.parse_args()
    if args.sample_size<0:
        parser.error('sample-size deve ser >= 0')
    if args.output.exists():
        raise FileExistsError('Corpus existente não será sobrescrito.')
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    system=(ROOT/'data/factual_v2/system_prompt.txt').read_text()
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/base/Qwen3.5-9B',use_fast=True,local_files_only=True)
    excluded_ids=set()
    # Somente IDs dos conjuntos antigos, para não treinar em suas referências.
    for name in ('validation','test'):
        path=ROOT/f'data/splits/{name}.jsonl'
        if path.exists():
            excluded_ids.update(json.loads(line)['source_id'] for line in path.read_text().splitlines() if line.strip())
    counts=Counter(); seen=set(); candidates=[]
    parquet=sorted(args.snapshot.rglob('*.parquet'))
    if not parquet:
        raise FileNotFoundError('Snapshot sem Parquet.')
    for path in parquet:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
            for raw in batch.to_pylist():
                counts['source_rows']+=1
                record,reason=example(raw)
                if reason:
                    counts[reason]+=1;continue
                if record['source_id'] in excluded_ids:
                    counts['historical_heldout_id']+=1;continue
                question=normalize(record['instruction'])
                if question in seen:
                    counts['duplicate_question']+=1;continue
                seen.add(question)
                candidates.append(record)
    random.Random(args.seed).shuffle(candidates)
    quota={'train':args.sample_size*8//10,'validation':args.sample_size//10}
    quota['test']=args.sample_size-quota['train']-quota['validation']
    splits={name:[] for name in quota}; maxima=Counter()
    for i,record in enumerate(candidates):
        split=partition(record['source_group'],args.seed)
        if args.sample_size and len(splits[split])>=quota[split]:continue
        count=full_tokens(tokenizer,record,system)
        if count>512:
            counts['over_512_selected_candidates']+=1;continue
        splits[split].append(record);maxima[split]=max(maxima[split],count)
        if args.sample_size and all(len(splits[k])==quota[k] for k in quota):break
    if any(not rows for rows in splits.values()):
        raise ValueError('Split vazio após os filtros.')
    if args.sample_size and any(len(splits[k])!=quota[k] for k in quota):
        raise ValueError('Não há exemplos suficientes para as cotas solicitadas.')
    medpt_counts={k:len(v) for k,v in splits.items()}
    # Replay somente dos exemplos sintéticos do treino anterior, nunca dos seus testes.
    replay=[json.loads(line) for line in (ROOT/'data/factual_v2/data/splits/train.jsonl').read_text().splitlines()]
    replay=[r for r in replay if r.get('origin')=='synthetic_context_v2']
    splits['train'].extend(replay)
    for left,right in (('train','validation'),('train','test'),('validation','test')):
        assert not {r['source_group'] for r in splits[left]} & {r['source_group'] for r in splits[right]}
        assert not {normalize(r['instruction']) for r in splits[left]} & {normalize(r['instruction']) for r in splits[right]}
    for name in splits:
        random.Random(f'{args.seed}:{name}').shuffle(splits[name])
    args.output.mkdir(parents=True)
    splitdir=args.output/'data/splits';splitdir.mkdir(parents=True)
    for name,rows in splits.items():
        (splitdir/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (args.output/'system_prompt.txt').write_text(system)
    validation=sorted(splits['validation'],key=lambda r:r['source_id'])
    random.Random(args.seed).shuffle(validation)
    (args.output/'evaluation_medpt.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in validation[:30]))
    previous=[json.loads(line) for line in (ROOT/'data/factual_v2/data/splits/validation.jsonl').read_text().splitlines()]
    previous=[r for r in previous if r.get('origin')=='synthetic_context_v2']
    (args.output/'regression_context.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in previous))
    manifest={'dataset':'AKCIT/MedPT','revision':args.snapshot.name.removeprefix('MedPT-'),
        'parquet_sha256':{str(p.relative_to(args.snapshot)):file_sha256(p) for p in parquet},
        'seed':args.seed,'sample_size_requested':args.sample_size,'counts':dict(counts),
        'eligible_before_length_check':len(candidates),'medpt_splits':medpt_counts,'synthetic_train_replay':len(replay),
        'splits':{k:{'count':len(v),'sha256':file_sha256(splitdir/f'{k}.jsonl'),
                     'category_counts':dict(Counter(r['category'] for r in v)),'max_tokens_medpt':maxima[k]} for k,v in splits.items()},
        'license':'not_declared_in_downloaded_card','clinical_validation':False,
        'limitations':['Filtros técnicos e agrupamento por condição não são curadoria clínica.',
            'Respostas históricas podem conter erros, omissões e orientações inadequadas não detectadas.',
            'Mesmo snapshot MedPT já foi utilizado em experimentos anteriores; não é corpus novo independente.',
            'Agrupamento usa nomes normalizados de condição; não garante ausência de sinônimos entre splits.',
            'Somente validação usada para comparar candidato; teste novo permanece fechado.']}
    (splitdir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
