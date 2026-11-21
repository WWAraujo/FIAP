"""Revisões de rascunhos imutáveis; identidade declarada no protótipo local."""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from uuid import uuid4


def digest(payload):
    return hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class ReviewStore:
    def __init__(self, path=None):
        self.path = Path(path or os.environ.get('REVIEW_DB_PATH') or Path(__file__).resolve().parents[3]/'data/reviews.db')

    def _connect(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        db = sqlite3.connect(self.path,timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('CREATE TABLE IF NOT EXISTS reviews (id TEXT PRIMARY KEY, patient_id TEXT, query_id TEXT NOT NULL, payload TEXT NOT NULL, digest TEXT NOT NULL, status TEXT NOT NULL, reviewer TEXT, notes TEXT, created_at TEXT NOT NULL, reviewed_at TEXT)')
        return db

    def create(self, payload, *, patient_id, query_id):
        identifier = 'review_'+uuid4().hex
        with closing(self._connect()) as db, db:
            db.execute('INSERT INTO reviews (id,patient_id,query_id,payload,digest,status,created_at) VALUES (?,?,?,?,?,?,?)',
                (identifier,patient_id,query_id,json.dumps(payload,ensure_ascii=False),digest(payload),'pending',datetime.now(timezone.utc).isoformat()))
        return identifier

    def get(self, identifier):
        with closing(self._connect()) as db:
            row = db.execute('SELECT * FROM reviews WHERE id=?',(identifier,)).fetchone()
        if row is None:
            raise ValueError('Revisão não encontrada.')
        result = dict(row)
        result['payload'] = json.loads(result['payload'])
        if digest(result['payload']) != result['digest']:
            raise ValueError('Rascunho alterado; revisão recusada.')
        return result

    def decide(self, identifier, *, patient_id, reviewer, decision, notes='', confirmed_human=False):
        if confirmed_human is not True or decision not in ('approved','rejected') or not reviewer.strip() or len(reviewer)>120 or len(notes)>2000:
            raise ValueError('Informe responsável e decisão válidos.')
        current = self.get(identifier)
        if current['patient_id'] != patient_id:
            raise ValueError('A revisão pertence a outro paciente.')
        if decision=='approved' and not current['payload'].get('review_eligible'):
            raise ValueError('Rascunho bloqueado ou sem evidências; aprovação indisponível.')
        sources = current['payload'].get('sources',[])
        if decision=='approved' and (not sources or any(s['text'][s['char_start']:s['char_end']] != s['quoted_text'] for s in sources)):
            raise ValueError('Fontes ausentes ou inconsistentes.')
        with closing(self._connect()) as db, db:
            count = db.execute('UPDATE reviews SET status=?,reviewer=?,notes=?,reviewed_at=? WHERE id=? AND status=? AND digest=?',
                (decision,reviewer.strip(),notes.strip(),datetime.now(timezone.utc).isoformat(),identifier,'pending',current['digest'])).rowcount
            if count != 1:
                raise ValueError('Esta revisão já foi decidida ou alterada.')
        result = self.get(identifier)
        from medical_assistant.safety.audit import AuditLogger
        AuditLogger().log_event('human_review',query_id=result['query_id'],patient_id=patient_id,
            review_id=identifier,reviewer=reviewer.strip(),decision=decision,notes=notes.strip(),payload_sha256=result['digest'])
        return result
