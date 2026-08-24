# LedgerLens — Bank Statement Analyzer

Local-first Intelligent Document Processing (IDP) engine that extracts transaction
ledgers from bank statement PDFs (digital vector and scanned raster), enforces
ledger reconciliation arithmetic, and provides an interactive dual-pane
verification workspace with drag-to-select re-OCR.

**Architecture:** hybrid vector/raster page routing + unified extraction engine +
FastAPI backend + vanilla-JS dual-pane UI (canvas viewer / spreadsheet editor).

---

## 1. How It Works

1. **Routing (`pipeline/router.py`)** — `route_document()` classifies each PDF page
   as `DIGITAL_VECTOR` (embedded text stream, parsed natively via PyMuPDF in
   `pipeline/digital.py`) or `FLATBED_RASTER` (page image, sent through the OCR path).
2. **Extraction (`pipeline/engine.py`)** — `process_statement()` orchestrates the
   full run: tokenization (`pipeline/tokens.py`, bboxes in PDF point space),
   column segmentation (`pipeline/columns.py`), transaction assembly, statement-type
   classification (`pipeline/classifier.py`), summary balances (`pipeline/summary.py`),
   and reconciliation (`pipeline/reconciler.py`). Raster pages are recognized with a
   TrOCR model (`pipeline/ocr_trocr.py`, default `microsoft/trocr-small-printed`).
3. **Money parsing (`pipeline/money.py`)** — `parse_amount_decimal()` handles
   `$1,250.50`, `-45.00`, `($100.25)`, `45.00 CR/DR`, `+12.00`, etc.
4. **API layer (`backend/app.py`)** — FastAPI app serving the endpoints below,
   rendering 300 DPI page images for the viewer, mapping the engine payload onto the
   UI schema (`backend/schemas.py`; engine-side schemas live in `service/schemas.py`),
   and caching sessions in memory.
5. **UI (`ui/`)** — zero-CDN static frontend: canvas viewer with bbox overlays and
   drag-to-select re-OCR (`ui/js/viewer.js`) plus a keyboard-driven spreadsheet
   editor with live reconciliation status (`ui/js/editor.js`).

The engine payload shape is defined by `service/schemas.py`
(`DocumentMeta`, `RoutingInfo`, `ClassificationInfo`, `TransactionRecord`,
`ReconciliationBlock`, engine `StatementPayload`). The REST/UI payload shape is
defined by `backend/schemas.py` (`BBox`, `TransactionRow`, `PageInfo`,
`ReconciliationSummary`, API `StatementPayload`, request/response models).
`backend/app.py` maps between the two.

---

## 2. Directory Structure

```text
ledgerlens/
├── main.py                      # Unified entry point: parse / test-ocr / train / serve
├── requirements.txt             # Python dependency manifest
├── backend/
│   ├── app.py                   # FastAPI server: upload, page images, re-OCR, reconcile, export
│   ├── schemas.py               # Pydantic v2 models for the REST/UI payload
│   └── services/
│       ├── export_service.py    # JSON bytes + styled multi-tab XLSX for /api/export
│       └── xlsx_exporter.py     # Audit-linked XLSX (per-cell provenance comments), used by CLI
├── service/
│   └── schemas.py               # Shared engine-output schemas (provenance, routing, recon block)
├── pipeline/
│   ├── __init__.py
│   ├── engine.py                # process_statement(): end-to-end orchestration
│   ├── router.py                # route_document(): DIGITAL_VECTOR vs FLATBED_RASTER per page
│   ├── digital.py               # Native vector-text extraction
│   ├── raster.py                # Raster-page OCR path (TrOCR)
│   ├── ocr_trocr.py             # TrOCRRecognizer wrapper (microsoft/trocr-small-printed)
│   ├── classifier.py            # Statement type classification
│   ├── columns.py               # Column segmentation within rows
│   ├── tokens.py                # Tokenization; bboxes in PDF point space (72 DPI)
│   ├── money.py                 # Currency/negative-notation amount parsing
│   ├── summary.py               # Opening/closing balance extraction
│   └── reconciler.py            # Ledger reconciliation math
├── ui/
│   ├── index.html               # Dual-pane workspace shell
│   ├── css/style.css
│   └── js/
│       ├── viewer.js            # Canvas viewer, bbox overlays, drag-to-select re-OCR
│       └── editor.js            # Spreadsheet grid, hotkeys, live PASS/FAIL status
├── tests/
│   └── test_ledgerlens_engine.py  # pytest suite: money, router, engine, exports, API
└── sample/                      # Manual test fixtures (statement PDFs + parse outputs)
```

---

## 3. Installation & Setup

Prerequisites:
- Python 3.10+
- No system OCR binary required — raster pages use TrOCR
  (`torch` + `transformers` are pulled in via `requirements.txt`).

```bash
pip install -r requirements.txt
```

Note: the first time a scanned/raster page is processed (or `/api/re-ocr` is used),
the TrOCR weights (`microsoft/trocr-small-printed`) are downloaded from Hugging Face
and cached locally.

---

## 4. Usage

### A. Interactive Dual-Pane UI

```bash
python main.py serve --host 127.0.0.1 --port 8000 --reload
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) (API docs at `/docs`).

- **Left pane:** 300 DPI canvas viewer with zoom/pan, bounding-box overlays per
  cell, click-to-focus row, and a drag-to-select region re-OCR popover.
- **Right pane:** spreadsheet editor with inline editing, keyboard navigation,
  live opening/closing balance cards, PASS/FAIL reconciliation badge, and
  one-click JSON/XLSX export.

### B. CLI Commands

Parse a statement (prints a summary; optionally writes JSON and/or XLSX):

```bash
python main.py parse <pdf_path> [--output out.json] [--export-xlsx out.xlsx] [--verbose]
```

Example using the bundled fixtures:

```bash
python main.py parse sample/C.pdf --output c.json --export-xlsx c.xlsx --verbose
```

Run a single synthetic OCR inference check:

```bash
python main.py test-ocr
```

Training placeholder (simulated progress only — no model artifacts are produced):

```bash
python main.py train --epochs 25 --samples 5000
```

### C. HTTP Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Service status + whether the TrOCR recognizer is loaded |
| POST | `/api/upload` | Multipart PDF ingest → UI `StatementPayload` (session cached) |
| GET | `/api/session/{session_id}` | Retrieve cached session payload |
| GET | `/api/page-image/{session_id}/{page_index}` | 300 DPI JPEG page render |
| POST | `/api/re-ocr` | Re-OCR a selected bbox (or base64 image); parses date/amount |
| POST | `/api/reconcile` | Recompute credits/debits/closing-balance arithmetic |
| POST | `/api/export` | Download verified JSON or multi-tab XLSX |

### D. Exports

Two Excel writers exist by design:

- **`/api/export` (and the UI button)** → `backend/services/export_service.py` —
  styled Transactions + Reconciliation Summary workbook generated from the UI payload.
- **CLI `parse --export-xlsx`** → `backend/services/xlsx_exporter.py` — audit-linked
  workbook where every extracted cell carries an OpenPyXL comment citing its source
  (`file | page | mode | bbox`) from the engine provenance data.

---

## 5. Running Tests

```bash
pytest tests/test_ledgerlens_engine.py -v
```

Covers money parsing, vector/raster routing, end-to-end digital extraction,
JSON/XLSX export generation, and all FastAPI endpoints. A separate TrOCR
re-OCR endpoint test is gated behind `LEDGERLENS_TEST_TROCR=1` (it downloads model
weights):

```bash
LEDGERLENS_TEST_TROCR=1 pytest tests/test_ledgerlens_engine.py -v
```
