import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/finetuning/scripts'))
from prepare_medpt_continuation import example, partition, full_tokens


def row(**changes):
    return {'id':7,'question':'Qual é a função de um registro de acompanhamento?',
            'answer':'O registro permite acompanhar informações documentadas ao longo do tempo e deve ser conferido pela equipe responsável.',
            'question_type':'Outros','condition':'Condição de exemplo','medical_specialty':'Clínica',**changes}


def test_original_answer_is_preserved_in_contract_with_provenance():
    original=row();record,reason=example(original)
    assert reason is None
    answer=json.loads(record['output'])
    assert answer['dados']['resposta']==original['answer']
    assert answer['fonte'] is None and answer['validacao_humana'] is True
    assert record['provenance']['original_id']==7
    assert record['license']=='not_declared_in_downloaded_card'


def test_direct_instructions_contacts_and_excluded_types_are_filtered():
    for changes in ({'question_type':'Tratamento'},{'answer':'Tome o medicamento indicado hoje e mantenha o uso por alguns dias.'},
                    {'answer':'Encontre as informações completas no site https://example.test e consulte nossa equipe responsável.'}):
        record,reason=example(row(**changes))
        assert record is None and reason


def test_same_normalized_condition_has_same_partition():
    first,_=example(row(condition='Condição de exemplo'))
    second,_=example(row(id=8,condition='condicao de exemplo'))
    assert first['source_group']==second['source_group']
    assert partition(first['source_group'],42)==partition(second['source_group'],42)


def test_token_count_uses_input_ids_not_encoding_keys():
    from unittest.mock import Mock
    tokenizer=Mock()
    tokenizer.apply_chat_template.return_value={'input_ids':list(range(513)),'attention_mask':[1]*513}
    record,_=example(row())
    assert full_tokens(tokenizer,record,'system')==513


def test_evaluation_counts_contract_failures_without_claiming_semantic_accuracy(tmp_path):
    from run_medpt_continuation import contract_metrics
    record,_=example(row());path=tmp_path/'responses.jsonl'
    path.write_text(json.dumps({'source_id':record['source_id'],'response':record['output'],'hit_token_limit':False})+'\n')
    result=contract_metrics(path,[record])
    assert result['counts']['medpt_exact']==1 and result['clinical_validation'] is False
    path.write_text(json.dumps({'source_id':record['source_id'],'response':'invalid','hit_token_limit':True})+'\n')
    result=contract_metrics(path,[record])
    assert result['counts']['json_object']==0 and result['counts']['token_limit']==1


def test_incomplete_comparison_is_rejected(tmp_path):
    import pytest
    from run_medpt_continuation import contract_metrics
    record,_=example(row());path=tmp_path/'responses.jsonl';path.write_text('')
    with pytest.raises(ValueError,match='incompleta'):
        contract_metrics(path,[record])
