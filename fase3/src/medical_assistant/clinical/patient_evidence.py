"""Recupera fatos do contexto estruturado, sem inferir conduta clínica."""
from datetime import date, datetime
import re
import unicodedata

from medical_assistant.application.contracts import Source


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text.lower()) if unicodedata.category(c) != 'Mn')


def patient_evidence(request):
    question = normalized(request.query)
    if re.search(r'protocolo|diretriz|documento|tratamento|conduta|diagnost|sintomas|o que e |como tratar', question):
        return None
    definitions = [
        ('medicamentos', r'medicament|remedio|uso continuo', 'medications',
         [('medication_name','Medicamento'),('dosage','Dose registrada'),('frequency','Frequência'),('start_date','Início'),('end_date','Fim')]),
        ('exames', r'exames?|resultados?', 'exams',
         [('exam_type','Exame'),('exam_date','Data'),('status','Situação'),('result','Resultado registrado'),('doctor_notes','Observações')]),
        ('consultas', r'consultas?|agendament|retorno', 'consultations',
         [('consultation_date','Data'),('doctor','Profissional/serviço'),('chief_complaint','Motivo registrado'),('follow_up','Retorno registrado')]),
        ('sinais_vitais', r'peso|altura|pressao|sinais vitais|saturacao|frequencia cardiaca|temperatura', 'vital_signs',
         [('recorded_at','Registro'),('blood_pressure','Pressão registrada'),('heart_rate','Frequência cardíaca registrada'),('temperature','Temperatura registrada'),('oxygen_saturation','Saturação registrada'),('weight','Peso (kg)'),('height','Altura (m)')]),
        ('historico', r'historico|condicoes registradas|doencas registradas', 'medical_history',
         [('condition','Condição registrada'),('status','Situação'),('diagnosis_date','Data registrada'),('notes','Observações')]),
        ('idade', r'idade|quantos anos', 'patients', [('age','Idade cadastrada')]),
    ]
    selected = next((d for d in definitions if re.search(d[1], question)), None)
    if selected is None:
        return None
    field, _, table, columns = selected
    context = request.patient_context
    result = {'field': field, 'sources': [], 'records': [], 'pending': 'Selecione um paciente para consultar o prontuário.'}
    if context is None:
        return result
    records = ([{'id':context.patient_id, 'age':context.idade}] if field == 'idade'
               else list(getattr(context, field)))
    if field == 'sinais_vitais':
        records = records[:1]
    if field == 'exames' and re.search(r'vencid|pendent', question):
        statuses = {s for token,s in [('vencid','Vencido'),('pendent','Pendente')] if token in question}
        records = [r for r in records if r.get('status') in statuses]
    if field == 'consultas' and re.search(r'proxim|futur|agend', question):
        records = sorted([r for r in records if str(r.get('consultation_date') or '')[:10] >= date.today().isoformat()],
                         key=lambda r: r['consultation_date'])[:1]
    if field == 'idade' and context.idade is None:
        records = []
    result['pending'] = f'Não há registros correspondentes a {field} no contexto atual fornecido. Isso não confirma ausência clínica.'
    for row in records:
        quote = '; '.join(f'{label}: {row[key]}' for key,label in columns if row.get(key) is not None)
        if not quote or row.get('id') is None:
            continue
        source = Source(source_type='prontuario', id=f"sqlite:{table}:{row['id']}",
            document_id=f'prontuario:{context.patient_id}', title=f'Prontuário {context.patient_id} — {field}',
            version=str(row.get('recorded_at') or row.get('exam_date') or row.get('consultation_date') or datetime.now().isoformat()),
            section_id=table, section=field, text=quote, quoted_text=quote, char_start=0, char_end=len(quote),
            evidence_id=f"{table}:{row['id']}")
        result['sources'].append(source.model_dump())
        result['records'].append(row)
    return result
