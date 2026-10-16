from unittest.mock import Mock
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/rag'))
from patient_facts_chain import PatientFactsChain


def evidence():
    return {'records':[{'id':1,'medication_name':'Metformina'}], 'pending':'Sem registro.',
            'sources':[{'id':'sqlite:medications:1','text':'Medicamento: Metformina','quoted_text':'Medicamento: Metformina','char_start':0,'char_end':22}]}


def test_model_selects_supplied_patient_facts_without_inventing_response():
    backend=Mock()
    def generate(prompt,**options):
        assert 'Metformina' in prompt and 'P1' in prompt
        return {'response':json.dumps(options['allowed_json'][0]),'output_limit_reached':False}
    backend.generate.side_effect=generate
    result=PatientFactsChain(backend,evidence()).invoke('Qual medicamento está registrado?')
    assert result['answer']['dados']['trecho']=='Medicamento: Metformina'
    assert result['answer']['fonte']=='sqlite:medications:1'
    assert result['sources'] and backend.generate.call_count==1


def test_missing_records_do_not_call_model_and_invalid_generation_is_explicit():
    backend=Mock()
    data=evidence();data['sources']=[]
    result=PatientFactsChain(backend,data).invoke('Qual medicamento?')
    assert result['answer']['status']=='pendente' and result['sources']==[]
    backend.generate.assert_not_called()
    backend.generate.return_value={'response':'{"status":"informativo","dados":{"valor":"inventado"}}','output_limit_reached':False}
    with pytest.raises(ValueError,match='fora dos contratos'):
        PatientFactsChain(backend,evidence()).invoke('Qual medicamento?')
