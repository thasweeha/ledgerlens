"""
Audit Service: SQLite-backed change log for extraction corrections.

One row is appended per field change when a user applies a manual snip
(or any other source). The decision status (MATCH / DIFFERENT /
LOW_CONFIDENCE / CONFLICT) is computed SERVER-SIDE at log time -- the
frontend is never trusted to classify its own edits.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Configuration constants (single source of truth; override via env vars)
# Confidence values are FRACTIONS in [0, 1] (TrOCR mean token probability).
# ---------------------------------------------------------------------------
LOW_CONFIDENCE_THRESHOLD = 0.60     # (= 60%) below this -> LOW_CONFIDENCE
AMOUNT_MATCH_TOLERANCE = 0.005      # amounts equal within half a cent -> MATCH

AUDIT_DB_ENV = "LEDGERLENS_AUDIT_DB"

STATUS_MATCH = "MATCH"
STATUS_DIFFERENT = "DIFFERENT"
STATUS_LOW_CONFIDENCE = "LOW_CONFIDENCE"
STATUS_CONFLICT = "CONFLICT"
VALID_SOURCES = ("auto", "manual_snip")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT    NOT NULL,
    session_id        TEXT    NOT NULL,
    transaction_index INTEGER NOT NULL,
    field_name        TEXT    NOT NULL,
    old_value         TEXT,
    new_value         TEXT,
    page              INTEGER,
    bbox              TEXT,
    source            TEXT    NOT NULL CHECK (source IN ('auto', 'manual_snip')),
    status            TEXT    NOT NULL CHECK (status IN ('MATCH', 'DIFFERENT', 'LOW_CONFIDENCE', 'CONFLICT')),
    confidence        REAL
);
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_log(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_txn ON audit_log(session_id, transaction_index);
"""

# Reentrant: log_change holds it while _connect() acquires it for schema init.
_write_lock = threading.RLock()
_initialized: Dict[str, bool] = {}


def resolve_db_path() -> Path:
    """DB location: env override > %LOCALAPPDATA%/LedgerLens > ~/.ledgerlens."""
    override = os.environ.get(AUDIT_DB_ENV)
    if override:
        return Path(override)

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "LedgerLens" / "audit.db"

    return Path.home() / ".ledgerlens" / "audit.db"


def _connect() -> sqlite3.Connection:
    db_path = resolve_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    if not _initialized.get(str(db_path)):
        with _write_lock:
            if not _initialized.get(str(db_path)):
                conn.executescript(_SCHEMA)
                conn.commit()
                _initialized[str(db_path)] = True
    return conn


def _normalize_text(value: Any) -> str:
    """Whitespace-collapsed, case-insensitive comparison form."""
    return " ".join(str(value if value is not None else "").split()).casefold()


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def values_agree(field_name: str, old_value: Any, new_value: Any) -> bool:
    """Field-aware equality used by the status classifier."""
    if field_name in ("amount", "balance"):
        a, b = _as_float(old_value), _as_float(new_value)
        if a is None or b is None:
            return False
        return abs(a - b) <= AMOUNT_MATCH_TOLERANCE
    return _normalize_text(old_value) == _normalize_text(new_value)


def has_prior_value(value: Any) -> bool:
    """True when an automatic value existed and was non-empty."""
    if value is None:
        return False
    if isinstance(value, float) and value == 0.0:
        # Zero is treated as "nothing extracted" for auto-filled numeric cells;
        # the UI initializes missing amounts to 0.0.
        return False
    return str(value).strip() != ""


def compute_status(
    field_name: str,
    old_value: Any,
    new_value: Any,
    confidence: Optional[float],
) -> str:
    """
    Decision matrix (evaluated BEFORE the value is written):

      LOW_CONFIDENCE : recognizer confidence below threshold (flag, don't trust)
      DIFFERENT      : no prior automatic value existed / it was empty
      MATCH          : snipped value agrees with the existing automatic value
      CONFLICT       : snipped value disagrees with the automatic pipeline value
    """
    conf = _as_float(confidence)
    if conf is not None and conf < LOW_CONFIDENCE_THRESHOLD:
        return STATUS_LOW_CONFIDENCE

    if not has_prior_value(old_value):
        return STATUS_DIFFERENT

    if values_agree(field_name, old_value, new_value):
        return STATUS_MATCH

    return STATUS_CONFLICT


def log_change(
    session_id: str,
    transaction_index: int,
    field_name: str,
    old_value: Any,
    new_value: Any,
    page: Optional[int],
    bbox: Optional[Dict[str, Any]],
    source: str,
    status: str,
    confidence: Optional[float],
) -> Dict[str, Any]:
    """Persists one audit row and returns it (including id + timestamp)."""
    if source not in VALID_SOURCES:
        raise ValueError(f"source must be one of {VALID_SOURCES}, got {source!r}")

    timestamp = datetime.now(timezone.utc).isoformat()
    bbox_json = json.dumps(bbox) if bbox else None
    old_str = "" if old_value is None else str(old_value)
    new_str = "" if new_value is None else str(new_value)

    with _write_lock:
        conn = _connect()
        try:
            cursor = conn.execute(
                """
                INSERT INTO audit_log (
                    timestamp, session_id, transaction_index, field_name,
                    old_value, new_value, page, bbox, source, status, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    session_id,
                    int(transaction_index),
                    field_name,
                    old_str,
                    new_str,
                    int(page) if page is not None else None,
                    bbox_json,
                    source,
                    status,
                    float(confidence) if confidence is not None else None,
                ),
            )
            conn.commit()
            row_id = cursor.lastrowid
        finally:
            conn.close()

    return {
        "id": row_id,
        "timestamp": timestamp,
        "session_id": session_id,
        "transaction_index": int(transaction_index),
        "field_name": field_name,
        "old_value": old_str,
        "new_value": new_str,
        "page": int(page) if page is not None else None,
        "bbox": bbox,
        "source": source,
        "status": status,
        "confidence": float(confidence) if confidence is not None else None,
    }
