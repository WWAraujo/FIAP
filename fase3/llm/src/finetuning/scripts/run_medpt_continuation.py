"""Supervisiona uma continuação MedPT, preservando FT-11 e registrando falhas."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[3]
SCRIPTS=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str)+'\n')
    temporary.replace(path)


def contract_metrics(path,dataset):
    references={r['source_id']:r for r in dataset}
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if {r['source_id'] for r in rows}!=set(references) or len(rows)!=len(references):
        raise ValueError('Avaliação incompleta ou IDs duplicados.')
    counts=Counter(count=len(rows));details=[]
    for row in rows:
        ref=references[row['source_id']]
        try:
            actual=json.loads(row['response']);expected=json.loads(ref['output'])
        except (ValueError,TypeError):
            actual=None;expected=json.loads(ref['output'])
        checks={'nonempty':bool(row['response'].strip()),'token_limit':row['hit_token_limit'],
            'json_object':isinstance(actual,dict),
            'five_keys':isinstance(actual,dict) and set(actual)==set(expected),
            'human_validation':isinstance(actual,dict) and actual.get('validacao_humana') is True,
            'exact_reference':actual==expected}
        for name,result in checks.items():counts[name]+=int(result)
        kind='medpt' if ref.get('origin')=='medpt' else 'context_regression'
        counts[kind+'_count']+=1;counts[kind+'_exact']+=int(actual==expected)
        details.append({'source_id':row['source_id'],'category':ref['category'],'checks':checks})
    return {'counts':dict(counts),'details':details,'clinical_validation':False,
        'semantic_review':'pending; exact reference match does not establish medical correctness'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--experiment',type=Path,required=True)
    args=parser.parse_args()
    corpus=args.corpus.resolve();out=args.experiment.resolve();out.mkdir(parents=True,exist_ok=False)
    selected=ROOT/'artifacts/best_model.json';selection=json.loads(selected.read_text())
    parent=Path(selection['adapter']);base=Path(selection['base'])
    if sha(parent/'adapter_model.safetensors')!=selection['adapter_weights_sha256']:
        raise ValueError('Pesos de origem divergem da seleção.')
    original_selection=sha(selected)
    manifest=corpus/'data/splits/manifest.json';data=json.loads(manifest.read_text())
    if data['dataset']!='AKCIT/MedPT' or any(data['splits'][s]['sha256']!=sha(corpus/f'data/splits/{s}.jsonl') for s in ('train','validation','test')):
        raise ValueError('Corpus/manifesto inconsistente.')
    protocol={'task':'FT-14','created_at':datetime.now(timezone.utc).isoformat(),
        'supervisor_pid':os.getpid(),'python':sys.executable,'source_selection_sha256':original_selection,
        'parent_adapter':str(parent),'parent_weights_sha256':selection['adapter_weights_sha256'],
        'corpus_manifest_sha256':sha(manifest),'split_counts':{s:data['splits'][s]['count'] for s in data['splits']},
        'training':{'epochs':1,'learning_rate':5e-6,'max_length':512,'seed':42,
                    'batch_size':1,'gradient_accumulation':8,'initial_adapter':'FT-11'},
        'automatic_promotion':False,'clinical_validation':False,
        'evaluation':'30 MedPT validation rows plus 16 already observed contextual validation regressions; test not generated.',
        'eligibility':'No automatic adoption. Completed training and technical checks require separate semantic review.',
        'commands':[]}
    evaluation=[json.loads(line) for line in (corpus/'evaluation_medpt.jsonl').read_text().splitlines()]
    context=[json.loads(line) for line in (corpus/'regression_context.jsonl').read_text().splitlines()]
    for category in sorted({r['category'] for r in context}):
        evaluation.extend(sorted([r for r in context if r['category']==category],key=lambda r:r['source_id'])[:2])
    cases=out/'validation_cases.jsonl'
    cases.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in evaluation))
    protocol['evaluation_cases_sha256']=sha(cases)
    save(out/'protocol.json',protocol)
    state={'status':'running','stage':'preflight','pid':os.getpid()}
    def run(stage,arguments):
        state.update(stage=stage,status='running',started_at=datetime.now(timezone.utc).isoformat())
        command=[sys.executable,'-B',*map(str,arguments)]
        protocol['commands'].append({'stage':stage,'argv':command});save(out/'protocol.json',protocol)
        env={**os.environ,'PYTHONUNBUFFERED':'1','PYTORCH_ALLOC_CONF':'expandable_segments:True','TORCH_BLAS_PREFER_HIPBLASLT':'0'}
        with (out/f'{stage}.log').open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=process.pid;save(out/'status.json',state)
            code=process.wait()
        state['exit_code']=code;save(out/'status.json',state)
        if code:raise RuntimeError(f'{stage} terminou com código {code}; consulte o log.')
    common=[SCRIPTS/'train_qwen3_8b_rocm.py','--project-dir',corpus,'--model-id','Qwen/Qwen3.5-9B',
        '--model-dir',base,'--initial-adapter',parent,'--system-prompt-file',corpus/'system_prompt.txt',
        '--frozen-embeddings-bf16','--max-length','512','--epochs','1','--learning-rate','5e-6','--seed','42']
    try:
        run('preflight',[*common,'--output-dir',out/'probe','--validate-only'])
        run('training',[*common,'--output-dir',out/'training'])
        trained=list((out/'training').glob('*/manifest.json'))
        if len(trained)!=1:raise ValueError('Manifesto do novo treino não identificado.')
        adapter=trained[0].parent/'adapter'
        for label,path in (('parent',parent),('candidate',adapter)):
            run('evaluation_'+label,[SCRIPTS/'evaluate_qwen3_base_vs_adapter_rocm.py','--adapter-only',
                '--adapter-dir',path,'--model-dir',base,'--test-file',cases,
                '--system-prompt-file',corpus/'system_prompt.txt','--output-dir',out/('evaluation_'+label),
                '--limit','99999','--max-new-tokens','384','--seed','42'])
        metrics={}
        for label in ('parent','candidate'):
            responses=list((out/('evaluation_'+label)).glob('*/adapter.jsonl'))
            if len(responses)!=1:raise ValueError('Arquivo de respostas ausente/ambíguo.')
            metrics[label]=contract_metrics(responses[0],evaluation)
        assert sha(selected)==original_selection,'Seleção original alterada por outro processo.'
        assert sha(parent/'adapter_model.safetensors')==selection['adapter_weights_sha256'],'FT-11 original alterado.'
        save(out/'comparison.json',{'systems':metrics,'candidate':str(adapter),
            'candidate_sha256':sha(adapter/'adapter_model.safetensors'),'selected_model_unchanged':True,
            'promotion':False,'clinical_validation':False,'semantic_review_required':True})
        state.update(status='completed_awaiting_semantic_review',stage='finished',candidate=str(adapter),
                     completed_at=datetime.now(timezone.utc).isoformat())
        save(out/'status.json',state)
        print(json.dumps(state),flush=True)
    except BaseException as error:
        state.update(status='failed',error=str(error),traceback=traceback.format_exc(),
                     selected_model_unchanged=sha(selected)==original_selection)
        save(out/'status.json',state)
        raise


if __name__=='__main__':
    main()
