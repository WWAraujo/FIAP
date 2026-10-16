"""Conecta prontuário, documentos, grafo e revisão registrada."""
from uuid import uuid4

from medical_assistant.application.assistant import MedicalAssistantLangChain
from medical_assistant.application.contracts import AssistantRequest, PatientContext
from medical_assistant.application.workflow import MedicalWorkflowGraph
from medical_assistant.clinical.patient_database import PatientDatabase
from medical_assistant.clinical.patient_evidence import patient_evidence
from medical_assistant.safety.audit import AuditLogger
from medical_assistant.safety.guardrails import SecuritySystem
from medical_assistant.safety.reviews import ReviewStore


def start_workflow(patient_id, query, *, assistant=None):
    query_id = 'workflow_'+uuid4().hex
    audit = AuditLogger()
    audit.log_event('workflow_started',query_id=query_id,patient_id=patient_id,query=query)
    guard = SecuritySystem().check_query(query)
    if not guard['allowed']:
        audit.log_event('workflow_blocked',query_id=query_id,patient_id=patient_id,reason=guard['reason'])
        return {'completed':False,'security_blocked':True,'error':guard['message'],'query_id':query_id}
    db = PatientDatabase()
    try:
        summary = db.get_patient_summary(patient_id)
        if 'error' in summary:
            raise ValueError(summary['error'])
        context = PatientContext.from_database(db,patient_id)
        context.exames = db.get_exams(patient_id)
    finally:
        db.close()
    sources = []
    for question in ('medicamentos','exames','sinais vitais','histórico'):
        evidence = patient_evidence(AssistantRequest(query=question,patient_context=context))
        sources.extend(evidence['sources'])
    engine = assistant or MedicalAssistantLangChain()
    documents = engine.process_query('Nos documentos da base, qual regra se aplica ao caso: '+query,context)
    sources.extend(documents['sources'])
    sources = list({s['id']:s for s in sources}.values())
    patient_data = {**summary['patient_info'], 'medical_history':context.historico,
        'vital_signs':context.sinais_vitais[0] if context.sinais_vitais else {},
        'exams':context.exames, 'documentary_response':documents, 'record_sources':sources,
        'completed_exams':[e['exam_type'] for e in context.exames if e.get('result') and e['status'] in ('Normal','Atenção','Concluído')]}
    result = MedicalWorkflowGraph().execute_full_workflow(patient_data,query)
    result.update(query_id=query_id,sources=sources,model_usage=documents.get('model_usage'),
                  review_eligible=bool(sources) and not result.get('error') and not documents.get('security_blocked'))
    review_id = ReviewStore().create(result,patient_id=patient_id,query_id=query_id)
    result.update(review_id=review_id,review_status='pending')
    audit.log_event('workflow_draft',query_id=query_id,patient_id=patient_id,
                   review_id=review_id,review_status='pending',result=result)
    return result
