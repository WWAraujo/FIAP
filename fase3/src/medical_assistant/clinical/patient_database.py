"""
Sistema de Consultas a Base de Dados Estruturadas (Prontuários)
Implementa interface para consultar e contextualizar informações de pacientes
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, date

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatientDatabase:
    """Interface para base de dados de prontuários médicos"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            base_dir = Path(__file__).resolve().parents[3]
            db_path = base_dir / "data" / "prontuarios.db"

        self.db_path = db_path
        self.conn = None
        self._initialize_database()

    def _initialize_database(self):
        """Inicializa banco de dados SQLite"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()
        self._insert_sample_data()

        logger.info(f"Banco de dados inicializado: {self.db_path}")

    def _create_tables(self):
        """Cria tabelas do banco de dados"""
        cursor = self.conn.cursor()

        # Tabela de pacientes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                birth_date TEXT,
                contact TEXT,
                address TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # Tabela de histórico médico
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS medical_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                condition TEXT,
                diagnosis_date TEXT,
                status TEXT,
                notes TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
        """)

        # Tabela de sinais vitais
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vital_signs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                blood_pressure TEXT,
                heart_rate INTEGER,
                temperature REAL,
                oxygen_saturation INTEGER,
                weight REAL,
                height REAL,
                recorded_at TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
        """)

        # Tabela de exames
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                exam_type TEXT,
                exam_date TEXT,
                result TEXT,
                status TEXT,
                doctor_notes TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
        """)

        # Tabela de medicamentos
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS medications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                medication_name TEXT,
                dosage TEXT,
                frequency TEXT,
                start_date TEXT,
                end_date TEXT,
                prescribed_by TEXT,
                active INTEGER DEFAULT 1,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
        """)

        # Tabela de consultas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consultations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                consultation_date TEXT,
                doctor TEXT,
                chief_complaint TEXT,
                diagnosis TEXT,
                treatment TEXT,
                follow_up TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
        """)

        self.conn.commit()

    def _insert_sample_data(self):
        """Insere dados de exemplo para teste"""
        cursor = self.conn.cursor()

        # Verificar se já existe dados
        cursor.execute("SELECT COUNT(*) FROM patients")
        if cursor.fetchone()[0] > 0:
            return

        logger.info("Inserindo dados de exemplo...")

        # Paciente de exemplo
        patient_data = {
            "id": "patient_001",
            "name": "João Silva",
            "age": 45,
            "gender": "M",
            "birth_date": "1979-05-15",
            "contact": "(11) 98765-4321",
            "address": "Rua das Flores, 123 - São Paulo",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        cursor.execute("""
            INSERT INTO patients (id, name, age, gender, birth_date, contact, address, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, tuple(patient_data.values()))

        # Histórico médico
        medical_history = [
            ("patient_001", "Hipertensão Arterial", "2020-03-10", "Ativo", "Controlada com medicação"),
            ("patient_001", "Diabetes Tipo 2", "2018-07-22", "Ativo", "Em acompanhamento"),
            ("patient_001", "Dislipidemia", "2019-11-05", "Controlado", "Colesterol elevado")
        ]

        for history in medical_history:
            cursor.execute("""
                INSERT INTO medical_history (patient_id, condition, diagnosis_date, status, notes)
                VALUES (?, ?, ?, ?, ?)
            """, history)

        # Sinais vitais recentes
        vital_signs = [
            ("patient_001", "145/95", 88, 36.5, 98, 78.5, 1.75, datetime.now().isoformat()),
            ("patient_001", "140/90", 85, 36.6, 98, 78.3, 1.75, "2024-09-01T10:00:00"),
            ("patient_001", "138/88", 82, 36.4, 99, 78.0, 1.75, "2024-08-15T09:30:00")
        ]

        for signs in vital_signs:
            cursor.execute("""
                INSERT INTO vital_signs (patient_id, blood_pressure, heart_rate, temperature, oxygen_saturation, weight, height, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, signs)

        # Exames
        exams = [
            ("patient_001", "Hemoglobina Glicada", "2024-09-05", "7.2%", "Normal", "Glicemia controlada"),
            ("patient_001", "Eletrocardiograma", "2024-08-20", "Ritmo sinusal normal", "Normal", "Sem alterações"),
            ("patient_001", "Perfil Lipídico", "2024-08-25", "LDL elevado", "Atenção", "Iniciar estatinas")
        ]

        for exam in exams:
            cursor.execute("""
                INSERT INTO exams (patient_id, exam_type, exam_date, result, status, doctor_notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, exam)

        # Medicamentos
        medications = [
            ("patient_001", "Losartana", "50mg", "1x ao dia", "2020-03-10", None, "Dr. Silva", 1),
            ("patient_001", "Metformina", "850mg", "2x ao dia", "2018-07-22", None, "Dr. Silva", 1),
            ("patient_001", "Sinvastatina", "20mg", "1x à noite", "2019-11-05", None, "Dr. Silva", 1)
        ]

        for med in medications:
            cursor.execute("""
                INSERT INTO medications (patient_id, medication_name, dosage, frequency, start_date, end_date, prescribed_by, active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, med)

        # Consultas recentes
        consultations = [
            ("patient_001", "2024-09-10", "Dr. Silva", "Pressão elevada", "Hipertensão descontrolada", "Ajustar medicação", "2024-10-10"),
            ("patient_001", "2024-08-15", "Dr. Silva", "Check-up rotineiro", "Diabetes controlada", "Manter tratamento", "2024-11-15")
        ]

        for consultation in consultations:
            cursor.execute("""
                INSERT INTO consultations (patient_id, consultation_date, doctor, chief_complaint, diagnosis, treatment, follow_up)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, consultation)

        self.conn.commit()
        logger.info("Dados de exemplo inseridos com sucesso")

    def get_patient_info(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retorna informações básicas do paciente"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
        row = cursor.fetchone()

        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None

    def get_medical_history(self, patient_id: str) -> List[Dict[str, Any]]:
        """Retorna histórico médico do paciente"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM medical_history WHERE patient_id = ?", (patient_id,))
        rows = cursor.fetchall()

        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def get_vital_signs(self, patient_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retorna sinais vitais recentes do paciente"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM vital_signs
            WHERE patient_id = ?
            ORDER BY recorded_at DESC
            LIMIT ?
        """, (patient_id, limit))

        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def get_exams(self, patient_id: str, status: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """Retorna exames do paciente"""
        cursor = self.conn.cursor()

        if status:
            query = """
                SELECT * FROM exams
                WHERE patient_id = ? AND status = ?
                ORDER BY exam_date DESC
            """
            params = (patient_id, status)
        else:
            query = """
                SELECT * FROM exams
                WHERE patient_id = ?
                ORDER BY exam_date DESC
            """
            params = (patient_id,)

        if limit:
            query += " LIMIT ?"
            params = params + (limit,)

        cursor.execute(query, params)

        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def get_medications(self, patient_id: str, active_only: bool = True) -> List[Dict[str, Any]]:
        """Retorna medicamentos do paciente"""
        cursor = self.conn.cursor()

        if active_only:
            cursor.execute("""
                SELECT * FROM medications
                WHERE patient_id = ? AND active = 1
                ORDER BY start_date DESC
            """, (patient_id,))
        else:
            cursor.execute("""
                SELECT * FROM medications
                WHERE patient_id = ?
                ORDER BY start_date DESC
            """, (patient_id,))

        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def get_consultations(self, patient_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retorna consultas recentes do paciente"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM consultations
            WHERE patient_id = ?
            ORDER BY consultation_date DESC
            LIMIT ?
        """, (patient_id, limit))

        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def get_patient_summary(self, patient_id: str) -> Dict[str, Any]:
        """Retorna resumo completo do paciente para contexto do assistente"""
        patient_info = self.get_patient_info(patient_id)
        if not patient_info:
            return {"error": "Paciente não encontrado"}

        return {
            "patient_info": patient_info,
            "medical_history": self.get_medical_history(patient_id),
            "recent_vital_signs": self.get_vital_signs(patient_id, limit=3),
            "recent_exams": self.get_exams(patient_id, limit=5),
            "current_medications": self.get_medications(patient_id, active_only=True),
            "recent_consultations": self.get_consultations(patient_id, limit=3)
        }

    def search_patients(self, search_term: str) -> List[Dict[str, Any]]:
        """Busca pacientes por nome ou ID"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM patients
            WHERE name LIKE ? OR id LIKE ?
        """, (f"%{search_term}%", f"%{search_term}%"))

        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def add_vital_signs(self, patient_id: str, vital_signs: Dict[str, Any]) -> bool:
        """Adiciona novos sinais vitais"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO vital_signs
                (patient_id, blood_pressure, heart_rate, temperature, oxygen_saturation, weight, height, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                patient_id,
                vital_signs.get("blood_pressure"),
                vital_signs.get("heart_rate"),
                vital_signs.get("temperature"),
                vital_signs.get("oxygen_saturation"),
                vital_signs.get("weight"),
                vital_signs.get("height"),
                datetime.now().isoformat()
            ))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Erro ao adicionar sinais vitais: {e}")
            return False

    def close(self):
        """Fecha conexão com banco de dados"""
        if self.conn:
            self.conn.close()


class PatientContextBuilder:
    """Constrói contexto de paciente para o assistente médico"""

    def __init__(self, db: PatientDatabase):
        self.db = db

    def build_context_for_assistant(self, patient_id: str) -> str:
        """Constrói contexto formatado para o assistente médico"""
        summary = self.db.get_patient_summary(patient_id)

        if "error" in summary:
            return f"Erro: {summary['error']}"

        patient_info = summary["patient_info"]

        context = f"""
CONTEXTO DO PACIENTE:
ID: {patient_info['id']}
Nome: {patient_info['name']}
Idade: {patient_info['age']} anos
Gênero: {patient_info['gender']}

HISTÓRICO MÉDICO:
"""
        for condition in summary["medical_history"]:
            context += f"- {condition['condition']}: {condition['status']} (desde {condition['diagnosis_date']})\n"

        context += "\nSINAIS VITAIS RECENTES:\n"
        for vital in summary["recent_vital_signs"]:
            context += f"- {vital['recorded_at']}: PA {vital['blood_pressure']}, FC {vital['heart_rate']}, Temp {vital['temperature']}°C\n"

        context += "\nMEDICAÇÕES ATIVAS:\n"
        for med in summary["current_medications"]:
            context += f"- {med['medication_name']} {med['dosage']} ({med['frequency']})\n"

        context += "\nEXAMES RECENTES:\n"
        for exam in summary["recent_exams"]:
            context += f"- {exam['exam_type']} ({exam['exam_date']}): {exam['result']} - {exam['status']}\n"

        return context


def main():
    """Função principal para testar o sistema de banco de dados"""

    # Criar banco de dados
    db = PatientDatabase()

    # Testar consultas
    print("="*60)
    print("SISTEMA DE PRONTUÁRIOS MÉDICOS")
    print("="*60)

    # Buscar paciente
    patient_id = "patient_001"
    print(f"\nInformações do paciente {patient_id}:")
    patient_info = db.get_patient_info(patient_id)
    print(json.dumps(patient_info, indent=2, ensure_ascii=False))

    # Histórico médico
    print(f"\nHistórico médico:")
    medical_history = db.get_medical_history(patient_id)
    for condition in medical_history:
        print(f"- {condition['condition']}: {condition['status']}")

    # Sinais vitais
    print(f"\nSinais vitais recentes:")
    vital_signs = db.get_vital_signs(patient_id)
    for vital in vital_signs:
        print(f"- {vital['recorded_at']}: PA {vital['blood_pressure']}, FC {vital['heart_rate']}")

    # Medicamentos
    print(f"\nMedicações ativas:")
    medications = db.get_medications(patient_id)
    for med in medications:
        print(f"- {med['medication_name']} {med['dosage']} ({med['frequency']})")

    # Construir contexto para assistente
    print("\n" + "="*60)
    print("CONTEXTO PARA ASSISTENTE MÉDICO")
    print("="*60)

    context_builder = PatientContextBuilder(db)
    context = context_builder.build_context_for_assistant(patient_id)
    print(context)

    # Testar busca
    print("\n" + "="*60)
    print("BUSCA DE PACIENTES")
    print("="*60)

    search_results = db.search_patients("João")
    for patient in search_results:
        print(f"- {patient['id']}: {patient['name']}")

    db.close()


if __name__ == "__main__":
    main()
