# Ferramenta de IMC

`bmi.py` contém `calcular_imc(context, query_id)` e `CALCULAR_IMC_SCHEMA`. `__init__.py` expõe `TOOLS` e `execute_tool`. Há somente a ferramenta IMC; o executor rejeita nomes desconhecidos e argumentos adicionais. Paciente, medidas e query_id são vinculados pela aplicação.

```python
from medical_assistant.tools import TOOLS, execute_tool

result = execute_tool(
    'calcular_imc', {},
    patient_context=request.patient_context,
    query_id=query_id,
)
```

O cálculo usa peso e altura da mesma medição mais recente, valida valores/unidades e retorna fórmula, classificação por idade, fonte SQLite e necessidade de revisão. Não combina datas para preencher medidas ausentes nem produz diagnóstico. Critérios demonstrativos registrados em `bmi.py`; avaliar condições particulares com profissional.

O chat reconhece perguntas de IMC em `clinical/patient_queries.py` e delega ao executor. A escolha automática de tools pela LLM foi avaliada, mas não foi ligada ao chat. Perguntas adicionais sobre registros usam `patient_evidence.py` e a cadeia factual FT-11; isso não acrescenta ferramentas ao catálogo.

As respostas do chat recebem revisão pendente, com decisão manual registrada quando aplicável. Ver [contratos da aplicação](../application/README.md) e [testes de validação](../../../docs/GUIA_VALIDACAO.md).
