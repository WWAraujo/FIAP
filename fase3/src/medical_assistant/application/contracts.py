from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PatientContext(Contract):
    patient_id: str = Field(min_length=1)
    idade: int | None = None
    genero: str | None = None
    historico: list[dict[str, Any]] = Field(default_factory=list)
    sinais_vitais: list[dict[str, Any]] = Field(default_factory=list)
    exames: list[dict[str, Any]] = Field(default_factory=list)
    medicamentos: list[dict[str, Any]] = Field(default_factory=list)
    consultas: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def from_database(cls, db, patient_id):
        summary = db.get_patient_summary(patient_id)
        if "error" in summary:
            raise ValueError(summary["error"])
        return cls(patient_id=patient_id, idade=summary["patient_info"]["age"],
                   genero=summary["patient_info"]["gender"], historico=summary["medical_history"],
                   sinais_vitais=summary["recent_vital_signs"], exames=summary["recent_exams"],
                   medicamentos=summary["current_medications"], consultas=summary["recent_consultations"])


class AssistantRequest(Contract):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=8000)
    patient_context: PatientContext | None = None


class Source(Contract):
    source_type: Literal["documento", "prontuario"] = "documento"
    id: str
    document_id: str
    title: str
    version: str
    url: str | None = None
    page: int | None = None
    section_id: str
    section: str
    text: str
    quoted_text: str
    char_start: int
    char_end: int
    evidence_id: str

    @model_validator(mode='after')
    def literal_evidence(self):
        if not 0 <= self.char_start < self.char_end <= len(self.text) or self.text[self.char_start:self.char_end] != self.quoted_text:
            raise ValueError('Fonte sem trecho literal/offsets válidos.')
        return self


class DocumentaryAnswer(Contract):
    status: Literal["informativo", "pendente", "alerta", "modelo", "recusado"]
    dados: dict[str, Any]
    pendencias: list[str]
    fonte: str | None
    validacao_humana: Literal[True]


class RuntimeResponse(Contract):
    answer: DocumentaryAnswer
    sources: list[Source]
    error: dict[str, Any] | None = None
    model_usage: dict[str, Any]
    patient_context_used: bool
    run_id: str

    @model_validator(mode='after')
    def supplied_source(self):
        if self.answer.fonte is not None and self.answer.fonte not in {s.id for s in self.sources}:
            raise ValueError('A fonte da resposta não foi fornecida.')
        if self.answer.status == 'informativo' and not self.sources:
            raise ValueError('Resposta informativa sem evidência verificável.')
        return self


class AssistantResponse(Contract):
    response: str
    query_id: str
    sources: list[Source] = Field(default_factory=list)
    requires_validation: Literal[True] = True
    security_blocked: bool = False
    model_usage: dict[str, Any] | None = None
    patient_context_used: bool = False
    run_id: str | None = None
    review_id: str | None = None
    review_status: Literal['pending', 'approved', 'rejected'] | None = None
