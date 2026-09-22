"""Testes da curadoria e anonimização da Fase 3."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medical_assistant.data.preprocessing import MedicalDataPreprocessor
from medical_assistant.data.split_dataset import build_manifest, split_examples
from medical_assistant.data.ingest_medquad import extract_records, select_sample
from medical_assistant.data.generate_synthetic import build_dataset


def test_anonymization_removes_pii_and_preserves_clinical_language():
    text = """Paciente: Maria da Silva
CPF: 123.456.789-09
E-mail: maria@example.com
Telefone: (11) 99999-1234
Data de nascimento: 01/02/1980
Médico solicitante: Dr. João da Silva
Paciente com pressão arterial elevada e hemoglobina glicada pendente."""

    anonymized = MedicalDataPreprocessor().anonymize_text(text)

    assert "Maria da Silva" not in anonymized
    assert "123.456.789-09" not in anonymized
    assert "maria@example.com" not in anonymized
    assert "99999-1234" not in anonymized
    assert "01/02/1980" not in anonymized
    assert "João da Silva" not in anonymized
    assert "pressão arterial elevada" in anonymized
    assert "hemoglobina glicada pendente" in anonymized


def test_curate_dataset_keeps_raw_input_and_all_content_types():
    raw_dataset = {
        "protocolos": [{"id": "P1", "titulo": "Protocolo de teste", "conteudo": "Paciente deve aguardar 5 minutos."}],
        "perguntas_frequentes": [{"id": "F1", "pergunta": "Como medir?", "resposta": "Com cuidado."}],
        "modelos_laudos": [{"id": "L1", "tipo": "Laudo", "template": "Paciente: {nome_paciente}"}],
        "receitas_modelos": [{"id": "R1", "tipo": "Receita", "template": "Paciente: {nome_paciente}"}],
        "procedimentos_internos": [{"id": "PR1", "nome": "Aferição", "passos": ["Medir pressão."]}],
    }

    preprocessor = MedicalDataPreprocessor()
    curated = preprocessor.curate_dataset(raw_dataset)
    examples = preprocessor.format_for_fine_tuning(curated)

    assert raw_dataset["protocolos"][0]["conteudo"] == "Paciente deve aguardar 5 minutos."
    assert curated["dataset_metadata"]["contains_real_patient_data"] is False
    assert curated["dataset_metadata"]["record_counts"]["receitas_modelos"] == 1
    assert {example["category"] for example in examples} == {
        "protocolo",
        "faq",
        "modelo_laudo",
        "modelo_receita",
        "procedimento",
    }


def test_split_is_deterministic_has_no_leakage_and_covers_all_types():
    categories = ["protocolo", "faq", "modelo_laudo", "modelo_receita", "procedimento"]
    examples = [
        {"source_id": f"{category}-{index}", "category": category, "instruction": "x", "input": "", "output": "y"}
        for category in categories
        for index in range(5)
    ]

    first_run = split_examples(examples, seed=42)
    second_run = split_examples(examples, seed=42)
    manifest = build_manifest(first_run, {"record_counts": {category: 5 for category in categories}}, seed=42)

    assert first_run == second_run
    assert {example["source_id"] for example in first_run["train"]}.isdisjoint(
        example["source_id"] for example in first_run["validation"]
    )
    assert {example["source_id"] for example in first_run["train"]}.isdisjoint(
        example["source_id"] for example in first_run["test"]
    )
    assert {example["category"] for example in first_run["test"]} == set(categories)
    assert manifest["sparse_categories"] == []


def test_medquad_xml_extraction_and_sampling_are_deterministic(tmp_path):
    xml_path = tmp_path / "sample.xml"
    xml_path.write_text(
        """<QAPairs>
        <QAPair><Question>What is hypertension?</Question><Answer>A chronic condition.</Answer></QAPair>
        <QAPair><Question>What is hypertension?</Question><Answer>Duplicated question.</Answer></QAPair>
        <QAPair><Question>What is diabetes?</Question><Answer>A metabolic condition.</Answer></QAPair>
        </QAPairs>""",
        encoding="utf-8",
    )

    records = extract_records(tmp_path)
    first_sample = select_sample(records, sample_size=2, seed=42)
    second_sample = select_sample(records, sample_size=2, seed=42)

    assert len(records) == 3
    assert len(first_sample) == 2
    assert first_sample == second_sample
    assert {record["category"] for record in first_sample} == {"external_medquad_qa"}
    assert all(record["license"] == "CC-BY-4.0" for record in first_sample)


def test_behavioral_examples_are_grouped_and_cover_five_safe_behaviors():
    dataset = build_dataset()
    examples = dataset["exemplos_assistenciais"]
    formatted = MedicalDataPreprocessor().format_for_fine_tuning(dataset)
    splits = split_examples(formatted, seed=42)

    assert len(examples) == 200
    assert {example["category"] for example in examples} == {
        "recusa_prescricao",
        "recusa_diagnostico",
        "dados_insuficientes",
        "alerta_urgencia",
        "fonte_e_validacao",
    }
    all_groups = [example["source_group"] for records in splits.values() for example in records]
    assert len(all_groups) > len(set(all_groups))
    assert all(
        sum(group in {example["source_group"] for example in records} for records in splits.values()) == 1
        for group in set(all_groups)
    )
