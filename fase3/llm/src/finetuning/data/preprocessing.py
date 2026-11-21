"""Preprocessing, anonimização e curadoria do dataset sintético da Fase 3."""

from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalDataPreprocessor:
    """Anonimiza PII por contexto e prepara exemplos para treinamento futuro."""

    CONTENT_KEYS = (
        "protocolos",
        "perguntas_frequentes",
        "modelos_laudos",
        "receitas_modelos",
        "procedimentos_internos",
        "exemplos_assistenciais",
    )

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.anonymization_patterns = self._load_anonymization_patterns()

    @staticmethod
    def _load_anonymization_patterns() -> list[tuple[re.Pattern[str], str]]:
        """Retorna padrões de PII que não confundem linguagem clínica com nomes."""
        return [
            (re.compile(r"\b\d{3}[.\s]?\d{3}[.\s]?\d{3}-?\d{2}\b"), "[CPF_REMOVIDO]"),
            (re.compile(r"\b\d{1,2}[.]?\d{3}[.]?\d{3}-?[A-Za-z0-9]\b"), "[RG_REMOVIDO]"),
            (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL_REMOVIDO]"),
            (re.compile(r"(?:\+55\s*)?(?:\(?\d{2}\)?\s*)?9?\d{4}-?\d{4}\b"), "[TELEFONE_REMOVIDO]"),
            (re.compile(r"\bCRM[-/]?\s*\d{4,6}\b", re.IGNORECASE), "[CRM_REMOVIDO]"),
        ]

    @staticmethod
    def _replace_labeled_value(match: re.Match[str], placeholder: str) -> str:
        """Preserva o rótulo e placeholders de template, removendo somente valores reais."""
        value = match.group("value").strip()
        if "{" in value or "}" in value:
            return match.group(0)
        return f"{match.group('label')}{placeholder}"

    def anonymize_text(self, text: str) -> str:
        """Remove PII conhecida sem aplicar regex genérico a frases clínicas."""
        if not text:
            return text

        anonymized = text
        for pattern, placeholder in self.anonymization_patterns:
            anonymized = pattern.sub(placeholder, anonymized)

        labeled_patterns = [
            (
                re.compile(r"(?im)^(?P<label>\s*(?:paciente|nome do paciente)\s*:\s*)(?P<value>[^\n]+)$"),
                "[NOME_REMOVIDO]",
            ),
            (
                re.compile(r"(?im)^(?P<label>\s*(?:m[eé]dico(?:\s+solicitante)?|profissional)\s*:\s*)(?P<value>[^\n]+)$"),
                "[NOME_REMOVIDO]",
            ),
            (
                re.compile(r"(?im)^(?P<label>\s*endere[cç]o\s*:\s*)(?P<value>[^\n]+)$"),
                "[ENDERECO_REMOVIDO]",
            ),
            (
                re.compile(r"(?im)(?P<label>\bdata\s+de\s+nascimento\s*:\s*)(?P<value>\d{2}/\d{2}/\d{4})"),
                "[DATA_NASCIMENTO_REMOVIDA]",
            ),
        ]
        for pattern, placeholder in labeled_patterns:
            anonymized = pattern.sub(
                lambda match: self._replace_labeled_value(match, placeholder), anonymized
            )

        doctor_name = re.compile(
            r"\b(?:Dr|Dra)\.?\s+[A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ]+(?:\s+(?:de|da|do|dos|das))?\s+[A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ]+"
        )
        return doctor_name.sub("[NOME_REMOVIDO]", anonymized)

    def _anonymize_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.anonymize_text(value)
        if isinstance(value, list):
            return [self._anonymize_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self._anonymize_value(item) for key, item in value.items()}
        return value

    def anonymize_dataset(self, dataset: dict[str, Any]) -> dict[str, Any]:
        """Retorna uma cópia profunda anonimizando valores textuais do dataset."""
        anonymized_dataset = self._anonymize_value(copy.deepcopy(dataset))
        logger.info("Dataset anonimizado sem alterar a fonte em memória")
        return anonymized_dataset

    def curate_dataset(self, dataset: dict[str, Any]) -> dict[str, Any]:
        """Acrescenta proveniência e finalidade à cópia anonimizada do dataset."""
        curated_dataset = self.anonymize_dataset(dataset)
        record_counts = {
            content_key: len(curated_dataset.get(content_key, []))
            for content_key in self.CONTENT_KEYS
        }
        curated_dataset["dataset_metadata"] = {
            "schema_version": "1.0",
            "source": "dataset sintético gerado localmente",
            "source_version": "1.0",
            "purpose": "demonstração acadêmica de curadoria, RAG e fine-tuning",
            "contains_real_patient_data": False,
            "record_counts": record_counts,
        }
        return curated_dataset

    def format_for_fine_tuning(self, dataset: dict[str, Any]) -> list[dict[str, str]]:
        """Converte todos os tipos de conteúdo em exemplos de instrução/resposta."""
        fine_tuning_data: list[dict[str, str]] = []

        def add_example(
            source_id: str,
            category: str,
            instruction: str,
            output: str,
            source_group: str | None = None,
        ) -> None:
            fine_tuning_data.append(
                {
                    "source_id": source_id,
                    "source_group": source_group or source_id,
                    "category": category,
                    "origin": "synthetic",
                    "language": "pt-BR",
                    "instruction": instruction,
                    "input": "",
                    "output": output.strip(),
                }
            )

        for protocol in dataset.get("protocolos", []):
            add_example(protocol["id"], "protocolo", f"Explique o protocolo: {protocol['titulo']}", protocol["conteudo"])
        for faq in dataset.get("perguntas_frequentes", []):
            add_example(faq["id"], "faq", faq["pergunta"], faq["resposta"])
        for laudo in dataset.get("modelos_laudos", []):
            add_example(laudo["id"], "modelo_laudo", f"Como preencher um {laudo['tipo']}?", f"Template para {laudo['tipo']}:\n{laudo['template']}")
        for receita in dataset.get("receitas_modelos", []):
            add_example(
                receita["id"],
                "modelo_receita",
                f"Quais campos compõem um modelo de {receita['tipo']}?",
                "Este é um modelo documental. A emissão e qualquer definição de medicamento ou dose exigem profissional habilitado e validação humana.\n"
                f"Template:\n{receita['template']}",
            )
        for procedure in dataset.get("procedimentos_internos", []):
            steps = "\n".join(f"{index + 1}. {step}" for index, step in enumerate(procedure["passos"]))
            add_example(procedure["id"], "procedimento", f"Descreva o procedimento: {procedure['nome']}", steps)
        for example in dataset.get("exemplos_assistenciais", []):
            add_example(
                example["id"],
                example["category"],
                example["instruction"],
                f"{example['response']}\n\nFonte: {example['source_reference']}",
                example.get("source_group"),
            )

        logger.info("Formatados %s exemplos para fine-tuning", len(fine_tuning_data))
        return fine_tuning_data

def main() -> None:
    """Executa a curadoria e gera saídas derivadas locais."""
    project_dir = Path(__file__).resolve().parents[3]
    config = {
        "raw_data_path": project_dir / "data/raw/dataset_medico.json",
        "processed_data_path": project_dir / "data/curated",
    }
    with config["raw_data_path"].open(encoding="utf-8") as raw_file:
        raw_dataset = json.load(raw_file)

    preprocessor = MedicalDataPreprocessor(config)
    curated_dataset = preprocessor.curate_dataset(raw_dataset)
    curated_path = config["processed_data_path"] / "dataset_medico_anonimizado.json"
    curated_path.parent.mkdir(parents=True, exist_ok=True)
    with curated_path.open("w", encoding="utf-8") as curated_file:
        json.dump(curated_dataset, curated_file, ensure_ascii=False, indent=2)

    logger.info("Pipeline de preprocessing concluído; execute split_dataset.py para criar os splits.")


if __name__ == "__main__":
    main()
