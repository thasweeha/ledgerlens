"""Unified statement processing engine (spec §1-§6).

Routes every page, extracts tokens via the right pipeline, classifies the
statement, parses the ledger through column anchoring, reconciles balances,
and emits the unified JSON payload.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

import pymupdf as fitz

from pipeline.classifier import classify_statement
from pipeline.columns import parse_rows
from pipeline.digital import extract_page_tokens as extract_digital_tokens
from pipeline.money import decimal_to_float, parse_amount_decimal
from pipeline.reconciler import (
    LedgerEntry,
    extract_continuity_checkpoints,
    reconcile_document,
)
from pipeline.router import PageMode, route_document
from pipeline.summary import extract_statement_summary
from service.schemas import CellProvenance, StatementPayload

ENGINE_VERSION = "ledgerlens-unified-2.0"


def _load_ocr(model_name: str | None = None):
    from pipeline.ocr_trocr import DEFAULT_MODEL, TrOCRRecognizer

    return TrOCRRecognizer(model_name or DEFAULT_MODEL)


def _cell_provenance(cell: dict, file_name: str) -> CellProvenance:
    tokens = cell.get("tokens", [])
    confidence = min((t.confidence for t in tokens), default=None)
    return CellProvenance(
        text=cell.get("text", ""),
        page=cell.get("page"),
        mode=cell.get("mode"),
        bbox=list(cell["bbox"]) if cell.get("bbox") else None,
        confidence=round(confidence, 4) if confidence is not None else None,
    )


def process_statement(
    path: str | Path,
    dpi: int | None = None,
    tolerance: float = 0.01,
    model_path: str | Path | None = None,
) -> dict:
    """Run the full hybrid pipeline over one statement PDF.

    dpi=None lets raster pages render at their embedded image's native
    resolution."""
    path = Path(path)
    plan = route_document(path)
    if plan["error"]:
        raise RuntimeError(f"routing failed: {plan['error']}")

    file_name = path.name
    pages_tokens: Dict[int, list] = {}
    raster_diagnostics: Dict[int, dict] = {}
    ocr = None

    doc = fitz.open(path)
    total_pages = len(plan["modes"])
    try:
        for page_index, mode_value in enumerate(plan["modes"]):
            if mode_value == PageMode.DIGITAL_VECTOR.value:
                print(f"[ledgerlens] page {page_index + 1}/{total_pages}: digital vector", flush=True)
                pages_tokens[page_index] = extract_digital_tokens(doc[page_index], page_index)
            else:
                if ocr is None:
                    print("[ledgerlens] loading OCR model ...", flush=True)
                    ocr = _load_ocr(model_path)
                from pipeline.raster import extract_page_tokens as extract_raster_tokens

                print(
                    f"[ledgerlens] page {page_index + 1}/{total_pages}: OCR (raster)",
                    flush=True,
                )
                result = extract_raster_tokens(path, page_index, ocr, dpi=dpi)
                pages_tokens[page_index] = result.pop("tokens")
                raster_diagnostics[page_index] = result
                print(f"[ledgerlens] page {page_index + 1}/{total_pages} done", flush=True)
    finally:
        doc.close()

    classification = classify_statement(pages_tokens)
    parsed = parse_rows(pages_tokens)
    summary = extract_statement_summary(pages_tokens)

    rows = parsed["transactions"]
    checkpoints = extract_continuity_checkpoints(rows)
    entries: List[LedgerEntry] = [
        LedgerEntry(
            date=row.get("date", ""),
            description=row.get("description", ""),
            amount=row.get("amount"),
            balance=row.get("balance"),
            check_number=row.get("check_number"),
            page=int(row.get("page") or 0),
            is_balance_forward=bool(row.get("is_balance_forward")),
        )
        for row in rows
    ]

    opening = summary["opening_balance"]
    opening_source = "summary_band"
    if opening is None:
        if checkpoints and checkpoints[0].get("balance") is not None:
            opening = checkpoints[0]["balance"]
            opening_source = "balance_forward_row"
        elif entries and entries[0].balance is not None and entries[0].amount is not None:
            opening = entries[0].balance - entries[0].amount
            opening_source = "ledger_derived"

    closing = summary["closing_balance"]
    closing_source = "summary_band"
    if closing is None:
        trailing = [e for e in entries if e.balance is not None and not e.is_balance_forward]
        if trailing:
            closing = trailing[-1].balance
            closing_source = "ledger_derived"

    recon, segments = reconcile_document(
        entries,
        classification["statement_type"],
        opening,
        closing,
        tolerance=Decimal(str(tolerance)),
    )

    transaction_records = []
    for index, row in enumerate(rows):
        amount = row.get("amount")
        provenance = {
            name: _cell_provenance(cell, file_name)
            for name, cell in row.get("cells", {}).items()
        }
        transaction_records.append(
            {
                "index": index,
                "page": int(row.get("page") or 0),
                "date": row.get("date", ""),
                "post_date": row.get("post_date"),
                "description": row.get("description", ""),
                "check_number": row.get("check_number"),
                "amount": decimal_to_float(amount),
                "type": (
                    "credit" if amount is not None and amount > 0
                    else "debit" if amount is not None and amount < 0
                    else "checkpoint" if row.get("is_balance_forward")
                    else "unknown"
                ),
                "balance": decimal_to_float(row.get("balance")),
                "is_balance_forward": bool(row.get("is_balance_forward")),
                "provenance": provenance,
            }
        )

    payload = StatementPayload(
        document={
            "source_file": str(path),
            "page_count": plan["page_count"],
            "engine_version": ENGINE_VERSION,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        },
        routing={
            "modes_per_page": plan["modes"],
            "digital_pages": plan["digital_pages"],
            "raster_pages": plan["raster_pages"],
            "raster_diagnostics": {str(k): v for k, v in raster_diagnostics.items()},
        },
        classification={
            "statement_type": classification["statement_type"],
            "confidence": classification["confidence"],
            "document_scores": classification["document_scores"],
        },
        account={
            "account_number": summary.get("account_number"),
            "routing_number": summary.get("routing_number"),
            "statement_period": summary.get("statement_period"),
        },
        summary_balances={
            "opening_balance": decimal_to_float(opening),
            "opening_balance_source": opening_source,
            "closing_balance": decimal_to_float(closing),
            "closing_balance_source": closing_source,
            "reported_totals": {
                k: decimal_to_float(v) for k, v in summary.get("reported_totals", {}).items()
            },
            "sources": summary.get("summary_sources", {}),
        },
        transactions=transaction_records,
        reconciliation={
            "status": recon.status.value,
            "formula": recon.formula,
            "opening_balance": decimal_to_float(recon.opening_balance),
            "closing_balance": decimal_to_float(recon.closing_balance),
            "expected_closing_balance": decimal_to_float(recon.expected_closing_balance),
            "difference": decimal_to_float(recon.difference),
            "components": {k: decimal_to_float(v) for k, v in recon.components.items()},
            "running_balance_breaks": recon.running_balance_breaks,
            "checkpoints": checkpoints,
            "segments": segments,
            "flags": recon.flags,
        },
        diagnostics=parsed["diagnostics"],
    )
    return payload.model_dump(mode="json")
