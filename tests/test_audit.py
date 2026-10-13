import pytest
from app.services.audit import (
    init_audit_db,
    record_audit_entry,
    get_recent_audit_logs,
    hash_question,
)


def test_hash_question_deterministic():
    q1 = "How many sick leaves do I get?"
    q2 = "How many sick leaves do I get?"
    assert hash_question(q1) == hash_question(q2)
    assert len(hash_question(q1)) == 16


def test_record_audit_entry_and_retrieve(tmp_path, monkeypatch):
    test_db = tmp_path / "test_audit.db"
    monkeypatch.setattr("app.services.audit.settings.audit_db_path", str(test_db))

    init_audit_db()

    record_audit_entry(
        request_id="req_test_123",
        question="What is the maternity leave policy for full-time employees?",
        stages=["Router -> KB", "Pinecone retrieval -> 4 chunks", "KB grade -> GOOD"],
        source_used="private_kb",
        kb_grade="GOOD",
        web_grade=None,
        retry_count=0,
        duration_seconds=0.452,
        final_outcome="private_kb",
    )

    logs = get_recent_audit_logs(limit=10)
    assert len(logs) == 1
    log = logs[0]
    assert log["request_id"] == "req_test_123"
    assert log["source_used"] == "private_kb"
    assert log["kb_grade"] == "GOOD"
    assert log["web_grade"] is None
    assert log["retry_count"] == 0
    assert "Router -> KB" in log["stages_json"]
    assert log["final_outcome"] == "private_kb"
