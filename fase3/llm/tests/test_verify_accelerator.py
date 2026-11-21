import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "src/finetuning/scripts"
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    spec = importlib.util.spec_from_file_location("verify_accelerator", SCRIPTS / "verify_accelerator.py")
    accelerator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(accelerator)


class AcceleratorTests(unittest.TestCase):
    def test_missing_torch_records_failure(self):
        with patch.dict(sys.modules, {"torch": None}):
            report = accelerator.probe_accelerator()
        self.assertFalse(report["passed"])
        self.assertIn("error", report)

    def test_cpu_cannot_pass(self):
        torch = SimpleNamespace(
            __version__="test+cpu", version=SimpleNamespace(cuda=None, hip=None),
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        with patch.dict(sys.modules, {"torch": torch}):
            report = accelerator.probe_accelerator()
        self.assertFalse(report["passed"])
        self.assertFalse(report["cuda_available"])

    def test_failure_preserves_separate_reports_and_returns_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(sys, "argv", ["verify_accelerator", "--output-dir", directory]),
                patch.object(accelerator, "probe_accelerator", return_value={"passed": False}),
                patch.object(accelerator, "command_output", return_value={"error": "unavailable"}),
                patch.object(accelerator, "installed_versions", return_value={}),
            ):
                for _ in range(2):
                    with self.assertRaises(SystemExit):
                        accelerator.main()
            reports = list(Path(directory).glob("org04_*.json"))
            self.assertEqual(len(reports), 2)
            for path in reports:
                report = json.loads(path.read_text())
                self.assertFalse(report["org04_complete"])
                self.assertFalse(report["accelerator"]["passed"])


if __name__ == "__main__":
    unittest.main()
