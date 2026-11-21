from datetime import date

from medical_assistant.application.contracts import PatientContext
from medical_assistant.clinical.patient_database import PatientDatabase
from medical_assistant.clinical.seed_patients import seed_patients


def test_ten_distinct_profiles_have_dates_medication_allergy_and_context(tmp_path):
    db = PatientDatabase(tmp_path / "patients.db")
    today = date(2026, 10, 14)
    try:
        inserted = seed_patients(db, today)
        assert inserted == [f"sim_{i:03d}" for i in range(1, 11)]
        contexts = [PatientContext.from_database(db, patient_id) for patient_id in inserted]
        assert len({db.get_patient_info(patient_id)["name"] for patient_id in inserted}) == 10
        assert len({context.idade for context in contexts}) == 10
        assert all(context.historico and context.sinais_vitais and context.exames and context.consultas
                   for context in contexts)
        assert any(context.medicamentos for context in contexts)
        assert any(not context.medicamentos for context in contexts)
        assert {exam["status"] for context in contexts for exam in context.exames} == {
            "Vencido", "Pendente", "Normal", "Atenção"}
        for context in contexts:
            for exam in context.exames:
                if exam["status"] == "Vencido":
                    assert date.fromisoformat(exam["exam_date"]) < today
                    assert "prazo administrativo" in exam["doctor_notes"].lower()
                elif exam["status"] == "Pendente":
                    assert date.fromisoformat(exam["exam_date"]) > today
        assert any(date.fromisoformat(c["consultation_date"]) > today
                   for context in contexts for c in context.consultas)
        assert any("reagendamento pendente" in c["follow_up"]
                   for context in contexts for c in context.consultas)
        assert contexts[0].historico[1]["condition"] == "Alergia a dipirona"
        assert contexts[8].historico[1]["condition"] == "Alergias não informadas"
        assert contexts[9].medicamentos[0]["frequency"] == "Uso contínuo"
    finally:
        db.close()


def test_seeding_twice_preserves_existing_patients_and_manual_edits(tmp_path):
    db = PatientDatabase(tmp_path / "patients.db")
    try:
        original = db.get_patient_summary("patient_001")
        seed_patients(db, date(2026, 10, 14))
        before = {table: db.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                  for table in ("patients", "medical_history", "vital_signs", "exams", "medications", "consultations")}
        db.conn.execute("UPDATE patients SET name = ? WHERE id = ?", ("Nome ajustado", "sim_001"))
        db.conn.commit()
        assert seed_patients(db, date(2026, 11, 14)) == []
        assert db.get_patient_info("sim_001")["name"] == "Nome ajustado"
        assert db.get_patient_summary("patient_001") == original
        assert all(db.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == count
                   for table, count in before.items())
    finally:
        db.close()


def test_seeding_after_known_id_migration_does_not_duplicate_profiles(tmp_path):
    db = PatientDatabase(tmp_path/'patients.db')
    try:
        seed_patients(db)
        with db.conn:
            for n in range(1,11):
                old=f'sim_{n:03d}'
                new='paciente_011' if n==1 else f'paciente_{n:03d}'
                for table in ('medical_history','vital_signs','exams','medications','consultations'):
                    db.conn.execute(f'UPDATE {table} SET patient_id=? WHERE patient_id=?',(new,old))
                db.conn.execute('UPDATE patients SET id=? WHERE id=?',(new,old))
        assert seed_patients(db)==[]
        assert len(db.search_patients(''))==11
        assert not db.conn.execute('PRAGMA foreign_key_check').fetchall()
    finally:
        db.close()
