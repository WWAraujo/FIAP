from concurrent.futures import ThreadPoolExecutor

from langgraph.graph.state import CompiledStateGraph

from medical_assistant.application.workflow import MedicalWorkflowGraph


def patient(identifier, pressure="120/75", exams=None):
    return {"id": identifier, "age": 45, "medical_history": [],
            "vital_signs": {"blood_pressure": pressure, "heart_rate": 80,
                            "temperature": 36.5, "oxygen_saturation": 98},
            "completed_exams": exams or []}


def test_compiled_graph_reads_patient_and_exams_without_shared_state():
    workflow = MedicalWorkflowGraph()
    assert isinstance(workflow.graph, CompiledStateGraph)
    first = workflow.execute_full_workflow(patient("a", "145/95"), "pressão")
    second = workflow.execute_full_workflow(patient("b", exams=["Pressão arterial", "Eletrocardiograma", "Exames de sangue"]), "pressão")
    assert first["completed"] and second["completed"]
    assert first["patient_data"]["id"] == "a" and second["patient_data"]["id"] == "b"
    assert len(first["alerts"]) == 2 and not second["alerts"]
    assert first["decisions"][0]["priority"] == "alta"
    assert first["decisions"] is not second["decisions"]


def test_same_graph_can_run_concurrently_without_patient_leaks():
    workflow = MedicalWorkflowGraph()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda n: workflow.execute_full_workflow(patient(str(n)), "rotina"), range(2)))
    assert [r["patient_data"]["id"] for r in results] == ["0", "1"]
    assert all(len(r["decisions"]) == 4 for r in results)


def test_failed_step_stops_graph_without_claiming_completion():
    result = MedicalWorkflowGraph().execute_full_workflow({"vital_signs": {"heart_rate": "invalid"}}, "rotina")
    assert result["error"] and not result["completed"]
    assert result["current_step"] == "failed" and not result["recommendations"]
