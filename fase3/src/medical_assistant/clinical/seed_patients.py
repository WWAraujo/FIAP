from datetime import date, datetime, timedelta

from medical_assistant.clinical.patient_database import PatientDatabase


PROFILES = [
    {"name": "Ana — sintética", "age": 62, "gender": "F", "condition": "Hipertensão arterial",
     "note": "Retorno para acompanhamento de pressão; registro domiciliar incompleto.",
     "allergy": "Alergia a dipirona", "reaction": "Urticária relatada na simulação.",
     "medications": ["Losartana"], "exam": "Creatinina", "status": "Vencido", "next_days": 7,
     "specialty": "Clínica médica", "pressure": "152/94", "pulse": 82, "weight": 74, "height": 1.62},
    {"name": "Bruno — sintético", "age": 54, "gender": "M", "condition": "Diabetes tipo 2",
     "note": "Relata dificuldade de manter os registros de glicemia.",
     "allergy": "Alergia a penicilina", "reaction": "Exantema relatado na simulação.",
     "medications": ["Metformina"], "exam": "Hemoglobina glicada", "status": "Pendente", "next_days": 14,
     "specialty": "Endocrinologia", "pressure": "128/82", "pulse": 76, "weight": 89, "height": 1.75},
    {"name": "Carla — sintética", "age": 38, "gender": "F", "condition": "Asma",
     "note": "Relata tosse noturna intermitente; avaliação ambulatorial agendada.",
     "allergy": "Alergia a látex", "reaction": "Dermatite de contato relatada na simulação.",
     "medications": ["Budesonida inalatória"], "exam": "Espirometria", "status": "Pendente", "next_days": 3,
     "specialty": "Pneumologia", "pressure": "112/72", "pulse": 78, "weight": 61, "height": 1.65},
    {"name": "Daniel — sintético", "age": 47, "gender": "M", "condition": "Dislipidemia",
     "note": "Exames realizados; retorno para revisão documental dos resultados.",
     "allergy": "Sem alergias conhecidas", "reaction": "Negadas no cadastro sintético; não equivale a investigação.",
     "medications": ["Sinvastatina"], "exam": "Perfil lipídico", "status": "Normal", "next_days": 21,
     "specialty": "Clínica médica", "pressure": "124/78", "pulse": 70, "weight": 83, "height": 1.78},
    {"name": "Elisa — sintética", "age": 71, "gender": "F", "condition": "Hipotireoidismo",
     "note": "Retorno anterior não comparecido; ainda sem novo agendamento.",
     "allergy": "Alergia a contraste iodado", "reaction": "Urticária em exposição anterior fictícia.",
     "medications": ["Levotiroxina"], "exam": "TSH", "status": "Vencido", "next_days": None,
     "specialty": "Endocrinologia", "pressure": "132/80", "pulse": 68, "weight": 66, "height": 1.57},
    {"name": "Felipe — sintético", "age": 29, "gender": "M", "condition": "Rinite alérgica",
     "note": "Acompanhamento de sintomas sazonais; sem medicação contínua cadastrada.",
     "allergy": "Alergia a ácaros", "reaction": "Espirros e coriza relatados na simulação.",
     "medications": [], "exam": "Avaliação alérgica", "status": "Pendente", "next_days": 30,
     "specialty": "Alergologia", "pressure": "118/76", "pulse": 72, "weight": 72, "height": 1.80},
    {"name": "Gabriela — sintética", "age": 43, "gender": "F", "condition": "Anemia em investigação",
     "note": "Hemograma com alteração fictícia; retorno já agendado para revisão.",
     "allergy": "Alergia a sulfonamidas", "reaction": "Exantema relatado na simulação.",
     "medications": ["Sulfato ferroso"], "exam": "Hemograma", "status": "Atenção", "next_days": 2,
     "specialty": "Clínica médica", "pressure": "110/70", "pulse": 88, "weight": 58, "height": 1.63},
    {"name": "Henrique — sintético", "age": 68, "gender": "M", "condition": "Doença renal crônica em acompanhamento",
     "note": "Necessita revisar exames pendentes na próxima consulta; sem ajuste terapêutico simulado.",
     "allergy": "Alergia a ibuprofeno", "reaction": "Urticária relatada na simulação.",
     "medications": ["Anlodipino"], "exam": "Potássio sérico", "status": "Vencido", "next_days": 5,
     "specialty": "Nefrologia", "pressure": "144/86", "pulse": 74, "weight": 77, "height": 1.70},
    {"name": "Isabela — sintética", "age": 35, "gender": "F", "condition": "Enxaqueca em acompanhamento",
     "note": "Diário de sintomas preenchido parcialmente; sem medicação contínua cadastrada.",
     "allergy": "Alergias não informadas", "reaction": "Campo ainda não investigado; não interpretar como ausência de alergia.",
     "medications": [], "exam": "Avaliação neurológica", "status": "Pendente", "next_days": 12,
     "specialty": "Neurologia", "pressure": "116/74", "pulse": 75, "weight": 64, "height": 1.68},
    {"name": "Joaquim — sintético", "age": 76, "gender": "M", "condition": "Hipertensão e diabetes tipo 2",
     "note": "Retorno em atraso; familiar relata dificuldade de organizar os medicamentos.",
     "allergy": "Alergia a adesivo hospitalar", "reaction": "Dermatite de contato relatada na simulação.",
     "medications": ["Losartana", "Metformina"], "exam": "Hemoglobina glicada", "status": "Vencido", "next_days": None,
     "specialty": "Geriatria", "pressure": "148/88", "pulse": 80, "weight": 79, "height": 1.69},
]


def seed_patients(db, reference_date=None):
    reference_date = reference_date or date.today()
    def day(offset):
        return (reference_date + timedelta(days=offset)).isoformat()

    inserted = []
    with db.conn:
        for number, profile in enumerate(PROFILES, 1):
            patient_id = f"sim_{number:03d}"
            renamed_id = 'paciente_011' if number == 1 else f'paciente_{number:03d}'
            if db.get_patient_info(patient_id) is not None or db.get_patient_info(renamed_id) is not None:
                continue
            now = datetime.now().isoformat()
            db.conn.execute(
                "INSERT INTO patients (id, name, age, gender, birth_date, contact, address, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (patient_id, profile["name"], profile["age"], profile["gender"],
                 date(reference_date.year - profile["age"], 1, 1).isoformat(),
                 "Sem contato real — simulação", "Cadastro fictício", now, now))
            db.conn.executemany(
                "INSERT INTO medical_history (patient_id, condition, diagnosis_date, status, notes) VALUES (?, ?, ?, ?, ?)",
                [(patient_id, profile["condition"], day(-730), "Ativo", profile["note"]),
                 (patient_id, profile["allergy"], day(-180), "Registrado", profile["reaction"])])
            db.conn.execute(
                "INSERT INTO vital_signs (patient_id, blood_pressure, heart_rate, temperature, oxygen_saturation, "
                "weight, height, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (patient_id, profile["pressure"], profile["pulse"], 36.5, 98,
                 profile["weight"], profile["height"], day(-1) + "T09:00:00"))
            status = profile["status"]
            exam_day = day(-30 if status == "Vencido" else 4 if status == "Pendente" else -7)
            result = "Não realizado" if status in ("Vencido", "Pendente") else "Resultado fictício para revisão profissional"
            notes = (f"Prazo administrativo para realização expirado em {exam_day}; solicitação não realizada."
                     if status == "Vencido" else "Agendamento futuro; sem resultado." if status == "Pendente"
                     else "Exame realizado na simulação; sem validade clínica.")
            db.conn.execute(
                "INSERT INTO exams (patient_id, exam_type, exam_date, result, status, doctor_notes) VALUES (?, ?, ?, ?, ?, ?)",
                (patient_id, profile["exam"], exam_day, result, status, notes))
            for medication in profile["medications"]:
                db.conn.execute(
                    "INSERT INTO medications (patient_id, medication_name, dosage, frequency, start_date, "
                    "end_date, prescribed_by, active) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (patient_id, medication, "Conforme receita fictícia", "Uso contínuo", day(-180),
                     None, "Profissional fictício — simulação", 1))
            next_day = day(profile["next_days"]) if profile["next_days"] is not None else None
            db.conn.execute(
                "INSERT INTO consultations (patient_id, consultation_date, doctor, chief_complaint, diagnosis, "
                "treatment, follow_up) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (patient_id, day(-60), profile["specialty"] + " — simulação", profile["note"], profile["condition"],
                 "Acompanhamento fictício; sem nova prescrição.", next_day or f"Retorno vencido em {day(-10)}; reagendamento pendente."))
            if next_day:
                db.conn.execute(
                    "INSERT INTO consultations (patient_id, consultation_date, doctor, chief_complaint, diagnosis, "
                    "treatment, follow_up) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (patient_id, next_day, profile["specialty"] + " — simulação", "Consulta agendada: " + profile["condition"],
                     "Não avaliado — consulta futura", "Não realizado — consulta futura", "Aguardando atendimento"))
            inserted.append(patient_id)
    return inserted


def main():
    db = PatientDatabase()
    try:
        inserted = seed_patients(db)
        print(f"Pacientes sintéticos adicionados: {len(inserted)}. Banco: {db.db_path}")
        for patient in db.search_patients("sim_"):
            print(f"{patient['id']}: {patient['name']} ({patient['age']} anos)")
    finally:
        db.close()


if __name__ == "__main__":
    main()
