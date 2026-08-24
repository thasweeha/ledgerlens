"""
FastAPI application for LedgerLens.
Provides REST endpoints for PDF upload, 300 DPI page image serving,
drag-to-select re-OCR, real-time reconciliation calculation, and JSON/XLSX export.

Extraction is delegated to the unified hybrid pipeline (pipeline.engine),
which routes each page through the digital-vector or TrOCR raster path.
"""
from typing import Dict, Any, Optional, List
import io
import os
import uuid
import re
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
import pymupdf

from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query
from fastapi.responses import Response, FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pipeline.engine import process_statement
from pipeline.money import parse_amount_decimal
from backend.schemas import (
    StatementPayload,
    TransactionRow,
    PageInfo,
    BBox,
    ReconciliationSummary,
    ReOCRRequest,
    ReOCRResponse,
    ValidateRequest,
    ExportRequest
)
from backend.services.export_service import generate_json_bytes, generate_xlsx_bytes


# In-memory storage for active statement processing sessions
SESSIONS: Dict[str, Dict[str, Any]] = {}

# Lazily-initialized TrOCR recognizer (model weights load on first re-OCR use)
_OCR_RECOGNIZER = None

app = FastAPI(
    title="LedgerLens API",
    description="Bank Statement Analyzer & Intelligent Document Processing Backend",
    version="1.0.0"
)

# CORS middleware for local frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base UI directory
BASE_DIR = Path(__file__).resolve().parent.parent
UI_DIR = BASE_DIR / "ui"

# 300 DPI render resolution for the interactive viewer canvas
VIEWER_DPI = 300


def _get_recognizer():
    """Lazily loads the shared TrOCR recognizer used by /api/re-ocr."""
    global _OCR_RECOGNIZER
    if _OCR_RECOGNIZER is None:
        from pipeline.ocr_trocr import TrOCRRecognizer

        _OCR_RECOGNIZER = TrOCRRecognizer()
    return _OCR_RECOGNIZER


def _render_page_images(pdf_path: str | Path, dpi: int = VIEWER_DPI) -> List[Dict[str, Any]]:
    """Renders every PDF page to a PIL image (for the viewer) plus text previews."""
    pages: List[Dict[str, Any]] = []
    doc = pymupdf.open(str(pdf_path))
    try:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append({
                "image": img,
                "width": pix.width,
                "height": pix.height,
                "text_preview": (page.get_text() or "")[:200],
            })
    finally:
        doc.close()
    return pages


def _mode_to_page_type(mode: str) -> str:
    """Maps pipeline PageMode strings to the UI's 'vector'/'scanned' labels."""
    return "vector" if mode == "DIGITAL_VECTOR" else "scanned"


_DATE_PATTERNS = [
    r"\b\d{4}[-/.]\d{2}[-/.]\d{2}\b",
    r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{4}\b",
    r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2}\b",
    r"\b\d{1,2}\s*[-/.]?\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*[-/.]?\s*\d{2,4}\b",
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b",
]


def _parse_date_text(text: Optional[str]) -> Optional[str]:
    """Extracts a recognized date string from arbitrary OCR text."""
    if not text:
        return None
    clean = text.strip()
    for pat in _DATE_PATTERNS:
        m = re.search(pat, clean, re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return None


def _parse_amount_text(text: Optional[str]) -> Optional[float]:
    """Extracts a currency amount from OCR text using the pipeline's money parser."""
    if not text:
        return None
    direct = parse_amount_decimal(text.strip())
    if direct is not None:
        return float(direct)
    # Amounts usually sit rightmost on a row; scan tokens right-to-left.
    for token in reversed(re.split(r"\s+", text.strip())):
        val = parse_amount_decimal(token)
        if val is not None:
            return float(val)
    return None


def _provenance_bbox_to_schema(bbox: Optional[List[float]], page_number: int) -> BBox:
    """
    Converts a pipeline provenance bbox [x0, y0, x1, y1] -- in PDF point
    space (72 DPI), per pipeline/tokens.py -- into the UI's pixel-space
    {x, y, width, height, page} shape, matching the 300 DPI page images
    served by _render_page_images (VIEWER_DPI).
    """
    if not bbox or len(bbox) != 4:
        return BBox(x=0.0, y=0.0, width=0.0, height=0.0, page=page_number)
    scale = VIEWER_DPI / 72.0
    x0, y0, x1, y1 = (float(v) * scale for v in bbox)
    return BBox(
        x=x0,
        y=y0,
        width=max(0.0, x1 - x0),
        height=max(0.0, y1 - y0),
        page=page_number
    )


def _map_transaction(record: Dict[str, Any]) -> TransactionRow:
    """Maps one pipeline transaction record into the UI's TransactionRow shape."""
    amount = record.get("amount")
    amount = float(amount) if amount is not None else 0.0

    t_type = record.get("type") or "unknown"
    if t_type not in ("credit", "debit"):
        t_type = "credit" if amount >= 0 else "debit"

    # Pipeline pages are 0-based; the UI grid filters on 1-based page numbers.
    page_number = int(record.get("page") or 0) + 1

    prov: Dict[str, Any] = record.get("provenance") or {}
    date_bbox = _provenance_bbox_to_schema((prov.get("date") or {}).get("bbox"), page_number)
    desc_bbox = _provenance_bbox_to_schema((prov.get("description") or {}).get("bbox"), page_number)
    amount_prov = prov.get("amount") or prov.get("debit") or prov.get("credit")
    amount_bbox = _provenance_bbox_to_schema((amount_prov or {}).get("bbox"), page_number)

    # Row-level box = union of all per-cell provenance boxes (raw coordinates).
    boxes = [c.get("bbox") for c in prov.values() if c.get("bbox") and len(c["bbox"]) == 4]
    if boxes:
        x0 = min(float(b[0]) for b in boxes)
        y0 = min(float(b[1]) for b in boxes)
        x1 = max(float(b[2]) for b in boxes)
        y1 = max(float(b[3]) for b in boxes)
        row_bbox = BBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0, page=page_number)
    else:
        row_bbox = _provenance_bbox_to_schema(None, page_number)

    balance = record.get("balance")

    return TransactionRow(
        id=str(uuid.uuid4())[:8],
        date=record.get("date") or "",
        description=record.get("description") or "",
        amount=amount,
        type=t_type,
        balance=float(balance) if balance is not None else None,
        page=page_number,
        bbox=row_bbox,
        date_bbox=date_bbox,
        desc_bbox=desc_bbox,
        amount_bbox=amount_bbox
    )


def _map_reconciliation(payload: Dict[str, Any], tx_rows: List[TransactionRow]) -> ReconciliationSummary:
    """Maps the pipeline reconciliation block onto the UI's summary schema."""
    rec = payload.get("reconciliation") or {}
    components = rec.get("components") or {}

    credits = components.get("deposits_credits", components.get("payments_and_credits"))
    debits = components.get("withdrawals_debits", components.get("purchases_and_charges"))
    if credits is None:
        credits = round(sum(t.amount for t in tx_rows if t.type == "credit"), 2)
    if debits is None:
        debits = round(sum(abs(t.amount) for t in tx_rows if t.type == "debit"), 2)

    opening = rec.get("opening_balance")
    opening = round(float(opening), 2) if opening is not None else 0.0
    closing = rec.get("closing_balance")
    closing = round(float(closing), 2) if closing is not None else 0.0

    calc = rec.get("expected_closing_balance")
    calc = round(float(calc), 2) if calc is not None else round(opening + credits - debits, 2)
    diff = rec.get("difference")
    diff = round(abs(float(diff)), 2) if diff is not None else round(abs(calc - closing), 2)

    return ReconciliationSummary(
        reconciled=(rec.get("status") == "BALANCED"),
        opening_balance=opening,
        closing_balance=closing,
        total_credits=round(float(credits), 2),
        total_debits=round(float(debits), 2),
        calculated_closing=calc,
        difference=diff,
        transaction_count=len(tx_rows),
        tolerance=0.01
    )


@app.get("/api/health")
async def health_check():
    """Returns engine health status and OCR capability."""
    return {
        "status": "healthy",
        "service": "LedgerLens",
        "version": "1.0.0",
        "trocr_model_loaded": _OCR_RECOGNIZER is not None
    }


@app.post("/api/upload", response_model=StatementPayload)
async def upload_statement(file: UploadFile = File(...)):
    """
    Ingests a bank statement PDF through the unified hybrid pipeline
    (digital vector extraction + TrOCR raster path), renders 300 DPI page
    images for the viewer, maps results onto the UI payload schema, and
    caches the session.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        tmp.write(content)
        tmp.close()

        try:
            result = process_statement(tmp.name)
            rendered = _render_page_images(tmp.name, dpi=VIEWER_DPI)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {str(e)}")
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    session_id = str(uuid.uuid4())
    modes: List[str] = result.get("routing", {}).get("modes_per_page", [])

    tx_rows = [_map_transaction(rec) for rec in result.get("transactions", [])]

    page_infos: List[PageInfo] = []
    for idx, img_data in enumerate(rendered):
        mode = modes[idx] if idx < len(modes) else "DIGITAL_VECTOR"
        page_infos.append(PageInfo(
            page_number=idx + 1,
            page_index=idx,
            type=_mode_to_page_type(mode),
            width=img_data["width"],
            height=img_data["height"],
            image_url=f"/api/page-image/{session_id}/{idx}",
            text_preview=img_data["text_preview"]
        ))

    balances = result.get("summary_balances") or {}
    opening = balances.get("opening_balance")
    closing = balances.get("closing_balance")

    recon_summary = _map_reconciliation(result, tx_rows)

    statement_payload = StatementPayload(
        session_id=session_id,
        filename=file.filename,
        page_count=result.get("document", {}).get("page_count", len(rendered)),
        pages=page_infos,
        opening_balance=round(float(opening), 2) if opening is not None else recon_summary.opening_balance,
        closing_balance=round(float(closing), 2) if closing is not None else recon_summary.closing_balance,
        account_number=(result.get("account") or {}).get("account_number"),
        statement_period=(result.get("account") or {}).get("statement_period"),
        transactions=tx_rows,
        reconciliation=recon_summary
    )

    # Cache session
    SESSIONS[session_id] = {
        "statement": statement_payload,
        "pages": rendered
    }

    return statement_payload


@app.get("/api/session/{session_id}", response_model=StatementPayload)
async def get_session(session_id: str):
    """Retrieves cached statement session payload."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    return SESSIONS[session_id]["statement"]


@app.get("/api/page-image/{session_id}/{page_index}")
async def get_page_image(session_id: str, page_index: int):
    """Serves high-resolution 300 DPI rendered page image."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")

    session_pages = SESSIONS[session_id]["pages"]
    if page_index < 0 or page_index >= len(session_pages):
        raise HTTPException(status_code=404, detail="Page index out of range.")

    img: Image.Image = session_pages[page_index]["image"]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


@app.post("/api/re-ocr", response_model=ReOCRResponse)
async def re_ocr_endpoint(req: ReOCRRequest):
    """
    Performs precision re-OCR on a user-selected canvas bounding box using
    the pipeline's TrOCR recognizer. Extracts text, parses date, and parses amount.
    """
    page_img = None
    if req.session_id and req.session_id in SESSIONS:
        pages = SESSIONS[req.session_id]["pages"]
        if 0 <= req.page_index < len(pages):
            page_img = pages[req.page_index]["image"]

    target_field = req.target_field or "all"
    if req.image_base64:
        crop = _decode_base64_image(req.image_base64)
    elif page_img is not None:
        crop = _crop_region(page_img, req.bbox)
    else:
        return ReOCRResponse(raw_text="", cleaned_text="", target_field=target_field, confidence=0.0)

    try:
        recognizer = _get_recognizer()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"TrOCR model unavailable: {str(e)}")

    raw_text, confidence = recognizer.recognize(np.array(crop.convert("L")))
    cleaned_text = raw_text.strip()

    parsed_date = _parse_date_text(cleaned_text)
    parsed_amount = _parse_amount_text(cleaned_text)

    return ReOCRResponse(
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        parsed_date=parsed_date,
        parsed_amount=parsed_amount,
        target_field=target_field,
        confidence=round(confidence, 4) if cleaned_text else 0.0
    )


def _crop_region(page_image: Image.Image, bbox: BBox) -> Image.Image:
    """
    Safely crops the bounding box region from a 300 DPI page image.
    Clamps coordinates to image boundaries.
    """
    img_w, img_h = page_image.size
    x1 = max(0, int(round(bbox.x)))
    y1 = max(0, int(round(bbox.y)))
    x2 = min(img_w, int(round(bbox.x + bbox.width)))
    y2 = min(img_h, int(round(bbox.y + bbox.height)))

    if x2 <= x1 or y2 <= y1:
        # Return tiny valid fallback patch if degenerate bbox
        return Image.new("RGB", (10, 10), color="white")

    return page_image.crop((x1, y1, x2, y2))


def _decode_base64_image(b64_string: str) -> Image.Image:
    """Decodes a base64 data URI or raw base64 string to a PIL Image."""
    import base64

    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


@app.post("/api/reconcile", response_model=ReconciliationSummary)
async def reconcile_transactions(req: ValidateRequest):
    """
    Recalculates ledger arithmetic reconciliation across active transactions:
    Opening Balance + Sum(Credits) - Sum(Debits) == Closing Balance +/- 0.01
    """
    credits = round(sum(t.amount for t in req.transactions if t.type == "credit"), 2)
    debits = round(sum(abs(t.amount) for t in req.transactions if t.type == "debit"), 2)

    opening = round(float(req.opening_balance), 2)
    closing = round(float(req.closing_balance), 2)
    calc = round(opening + credits - debits, 2)
    diff = round(abs(calc - closing), 2)

    return ReconciliationSummary(
        reconciled=diff <= 0.01,
        opening_balance=opening,
        closing_balance=closing,
        total_credits=credits,
        total_debits=debits,
        calculated_closing=calc,
        difference=diff,
        transaction_count=len(req.transactions),
        tolerance=0.01
    )


@app.post("/api/export")
async def export_statement(req: ExportRequest):
    """
    Generates and returns verified statement download file (.json or multi-tab .xlsx).
    """
    stmt = req.statement
    base_name = Path(stmt.filename).stem or "statement"

    if req.format == "json":
        buf = generate_json_bytes(stmt)
        filename = f"{base_name}_verified.json"
        return StreamingResponse(
            buf,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    elif req.format == "xlsx":
        buf = generate_xlsx_bytes(stmt)
        filename = f"{base_name}_verified.xlsx"
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Use 'json' or 'xlsx'.")


# Static file serving & Single Page App route
if UI_DIR.exists():
    app.mount("/css", StaticFiles(directory=str(UI_DIR / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(UI_DIR / "js")), name="js")

    @app.get("/favicon.svg", include_in_schema=False)
    async def serve_favicon_svg():
        return FileResponse(str(UI_DIR / "favicon.svg"), media_type="image/svg+xml")

    @app.get("/favicon.ico", include_in_schema=False)
    async def serve_favicon_ico():
        return FileResponse(str(UI_DIR / "favicon.ico"), media_type="image/x-icon")

    @app.get("/apple-touch-icon.png", include_in_schema=False)
    async def serve_apple_touch_icon():
        return FileResponse(str(UI_DIR / "apple-touch-icon.png"), media_type="image/png")

    @app.get("/")
    async def serve_index():
        index_file = UI_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return HTMLResponse("<h1>LedgerLens UI Loading...</h1>")
