import pytest


@pytest.fixture(autouse=True)
def isolated_reviews(monkeypatch, tmp_path):
    monkeypatch.setenv('REVIEW_DB_PATH', str(tmp_path/'reviews.db'))
    from medical_assistant.safety.audit import AuditLogger
    def init(self):
        self.log_dir = tmp_path
        self.audit_file = tmp_path/'audit.log'
    monkeypatch.setattr(AuditLogger,'__init__',init)
