"""
Tests for the snip-first flow additions: audit service (SQLite), status
computation, /api/audit-log, and /api/snip-extract (native vector path).

Audit DB is redirected to a temp file via LEDGERLENS_AUDIT_DB so tests never
touch the developer's real %LOCALAPPDATA% store.
"""
import sqlite3

import pytest

from backend.services import audit_service


@pytest.fixture(autouse=True)
def isolated_audit_db(tmp_path, monkeypatch):
    db_path = tmp_path / "audit_test.db"
    monkeypatch.setenv("LEDGERLENS_AUDIT_DB", str(db_path))
    yield db_path


class TestComputeStatus:
    def test_match_when_values_agree(self):
        assert audit_service.compute_status("date", "2024-01-05", "2024-01-05", 0.99) == "MATCH"

    def test_match_text_normalized(self):
        assert audit_service.compute_status("description", "OFFICE  Supplies", "office supplies", 0.99) == "MATCH"

    def test_conflict_when_values_disagree(self):
        assert audit_service.compute_status("date", "2024-01-05", "2024-01-06", 0.99) == "CONFLICT"

    def test_different_when_no_prior_value(self):
        assert audit_service.compute_status("description", "", "ACME CORP", 0.99) == "DIFFERENT"
        assert audit_service.compute_status("description", None, "ACME CORP", 0.99) == "DIFFERENT"

    def test_different_when_prior_amount_is_zero_placeholder(self):
        assert audit_service.compute_status("amount", 0.0, "2500.00", 0.99) == "DIFFERENT"

    def test_low_confidence_takes_priority_over_conflict(self):
        assert audit_service.compute_status("date", "2024-01-05", "totally wrong", 0.40) == "LOW_CONFIDENCE"

    def test_high_fractional_confidence_is_not_low(self):
        assert audit_service.compute_status("date", "2024-01-05", "2024-01-05", 0.99) == "MATCH"

    def test_low_confidence_threshold_is_configurable(self, monkeypatch):
        monkeypatch.setattr(audit_service, "LOW_CONFIDENCE_THRESHOLD", 0.30)
        assert audit_service.compute_status("date", "2024-01-05", "2024-01-06", 0.40) == "CONFLICT"

    def test_amount_tolerance(self):
        assert audit_service.compute_status("amount", 150.0, "150.004", 0.99) == "MATCH"
        assert audit_service.compute_status("amount", 150.0, "151.00", 0.99) == "CONFLICT"

    def test_none_confidence_skips_low_confidence_check(self):
        assert audit_service.compute_status("date", "", "2024-01-05", None) == "DIFFERENT"


class TestLogChange:
    def test_row_persists_with_all_columns(self):
        row = audit_service.log_change(
            session_id="s1",
            transaction_index=2,
            field_name="amount",
            old_value="",
            new_value="2500.00",
            page=1,
            bbox={"x": 10, "y": 20, "width": 30, "height": 40, "page": 1},
            source="manual_snip",
            status="DIFFERENT",
            confidence=0.97,
        )
        assert row["id"] is not None
        assert row["timestamp"] is not None

        conn = sqlite3.connect(str(audit_service.resolve_db_path()))
        conn.row_factory = sqlite3.Row
        try:
            saved = conn.execute("SELECT * FROM audit_log WHERE id = ?", (row["id"],)).fetchone()
        finally:
            conn.close()

        assert saved["session_id"] == "s1"
        assert saved["transaction_index"] == 2
        assert saved["field_name"] == "amount"
        assert saved["new_value"] == "2500.00"
        assert saved["source"] == "manual_snip"
        assert saved["status"] == "DIFFERENT"
        assert abs(saved["confidence"] - 0.97) < 1e-9
        assert '"page": 1' in saved["bbox"]

    def test_invalid_source_rejected(self):
        with pytest.raises(ValueError):
            audit_service.log_change(
                session_id="s1", transaction_index=0, field_name="date",
                old_value="", new_value="x", page=1, bbox=None,
                source="hacker", status="MATCH", confidence=None,
            )

    def test_exact_schema_columns(self):
        audit_service.log_change(
            session_id="s", transaction_index=0, field_name="date",
            old_value="a", new_value="b", page=1, bbox=None,
            source="auto", status="MATCH", confidence=1.0,
        )
        conn = sqlite3.connect(str(audit_service.resolve_db_path()))
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(audit_log)").fetchall()]
        finally:
            conn.close()
        assert cols == [
            "id", "timestamp", "session_id", "transaction_index", "field_name",
            "old_value", "new_value", "page", "bbox", "source", "status", "confidence",
        ]


from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def _upload_vector_statement() -> dict:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    text = (
        "FIRST GLOBAL BANK - STATEMENT\n"
        "Opening Balance: $5,000.00\n\n"
        "Date Description Amount Balance\n"
        "2024-01-05 DEPOSIT CLIENT PAYMENT $2,500.00 $7,500.00\n"
        "2024-01-12 OFFICE SUPPLIES PURCHASE -$150.00 $7,350.00\n\n"
        "Closing Balance: $7,350.00\n"
    )
    page.insert_text((50, 70), text, fontsize=12)
    pdf = doc.tobytes()
    doc.close()

    res = client.post("/api/upload", files={"file": ("stmt.pdf", pdf, "application/pdf")})
    assert res.status_code == 200
    return res.json()


class TestSnipExtractEndpoint:
    def test_vector_page_uses_native_extraction(self):
        payload = _upload_vector_statement()
        # Snip a wide band covering the transaction rows (300 DPI space).
        scale = 300.0 / 72.0
        bbox = {
            "x": int(40 * scale),
            "y": int(100 * scale),
            "width": int(480 * scale),
            "height": int(50 * scale),
            "page": 1,
        }
        res = client.post("/api/snip-extract", json={
            "session_id": payload["session_id"],
            "page_index": 0,
            "bbox": bbox,
            "field_hint": "all",
        })
        assert res.status_code == 200
        data = res.json()
        assert data["extraction_source"] == "digital_native"
        assert abs(data["confidence"] - 0.99) < 1e-9
        assert "2024" in data["cleaned_text"]
        assert data["parsed_amount"] is not None

    def test_unknown_session_returns_empty(self):
        res = client.post("/api/snip-extract", json={
            "session_id": "no-such-session",
            "page_index": 0,
            "bbox": {"x": 0, "y": 0, "width": 100, "height": 50, "page": 1},
        })
        assert res.status_code == 200
        assert res.json()["cleaned_text"] == ""


def _audit_row_count() -> int:
    # _connect() lazily creates the schema on first use
    conn = audit_service._connect()
    try:
        return conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
    finally:
        conn.close()


class TestAuditLogEndpoint:
    def test_dry_run_computes_without_persisting(self):
        payload = _upload_vector_statement()
        before = _audit_row_count()

        res = client.post("/api/audit-log", json={
            "session_id": payload["session_id"],
            "transaction_index": 0,
            "field_name": "date",
            "old_value": "",
            "new_value": "2030-12-31",
            "page": 1,
            "bbox": {"x": 100, "y": 100, "width": 300, "height": 40, "page": 1},
            "source": "manual_snip",
            "confidence": 0.99,
            "dry_run": True,
        })
        assert res.status_code == 200
        body = res.json()
        assert body["id"] is None          # nothing persisted
        assert body["status"] == "CONFLICT"  # baseline 2024-01-05 disagrees

        after = _audit_row_count()
        assert after == before

    def test_server_uses_session_baseline_not_client_old_value(self):
        payload = _upload_vector_statement()
        # Client claims old value was empty -> naive logic would say DIFFERENT;
        # server must consult the stored pipeline value instead -> CONFLICT.
        res = client.post("/api/audit-log", json={
            "session_id": payload["session_id"],
            "transaction_index": 0,
            "field_name": "date",
            "old_value": "",
            "new_value": "1999-01-01",
            "source": "manual_snip",
            "confidence": 0.99,
            "dry_run": True,
        })
        assert res.json()["status"] == "CONFLICT"

    def test_persisted_log_has_id_and_timestamp(self):
        payload = _upload_vector_statement()
        res = client.post("/api/audit-log", json={
            "session_id": payload["session_id"],
            "transaction_index": 0,
            "field_name": "date",
            "old_value": "",
            "new_value": "2024-01-05",
            "page": 1,
            "bbox": None,
            "source": "manual_snip",
            "confidence": 0.99,
            "dry_run": False,
        })
        assert res.status_code == 200
        body = res.json()
        assert body["id"] is not None
        assert body["timestamp"] is not None
        assert body["status"] == "MATCH"  # agrees with the pipeline baseline

    def test_invalid_field_name_rejected(self):
        res = client.post("/api/audit-log", json={
            "session_id": "s", "transaction_index": 0,
            "field_name": "evil_attr", "new_value": "x",
        })
        assert res.status_code == 400

    def test_low_confidence_flagged_server_side(self):
        payload = _upload_vector_statement()
        res = client.post("/api/audit-log", json={
            "session_id": payload["session_id"],
            "transaction_index": 0,
            "field_name": "description",
            "old_value": "whatever",
            "new_value": "garble",
            "source": "manual_snip",
            "confidence": 0.42,
            "dry_run": True,
        })
        assert res.json()["status"] == "LOW_CONFIDENCE"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
