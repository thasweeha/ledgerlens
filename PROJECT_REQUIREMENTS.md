# LedgerLens — Functional & Technical Specifications
**System Name:** LedgerLens (formerly "STAC Engine" — Statement Transaction Auto-Capture)  
**Type:** Local-First Deep Learning Intelligent Document Processing (IDP) Engine  

---

## 1. Executive Overview
LedgerLens is a high-performance IDP engine designed to extract transaction ledgers, validate balances, and export structured JSON/Excel reports from multi-page bank statements. It employs a hybrid architecture: direct PDF parsing for vector text streams and a custom-trained PyTorch CRNN (CNN + BiLSTM + CTC) for scanned image-based statements.

---

## 2. Technical Stack
- **Runtime:** Python 3.10+
- **OCR Engine:** Tesseract OCR (pytesseract wrapper)
- **Computer Vision:** OpenCV (`opencv-python-headless`), Pillow (`PIL`)
- **PDF Extraction Engine:** PyMuPDF (`fitz`), `pdfplumber`
- **Data Validation & Export:** Pydantic v2, Pandas, OpenPyXL
- **CLI Interface:** `argparse`

---

## 3. Architecture & Functional Modules

### 3.1. Document Routing & Ingestion (`pipeline/pdf_loader.py`)
- Automatically detect document type:
  - Vector PDF (embedded selectable text stream) -> Direct native extraction.
  - Scanned/Raster PDF (embedded full-page images or zero text stream) -> Image pipeline.
- Render scanned PDF pages to 300 DPI high-resolution OpenCV images via PyMuPDF.

### 3.2. Visual Preprocessing & Slicing (`pipeline/image_slicer.py`)
- Apply grayscale conversion and Otsu binarization.
- Compute horizontal projection profiles to segment rows into individual transaction lines.
- Segment lines into modular column bounding boxes: `[Date Box]`, `[Description Box]`, `[Amount Box]`.

### 3.3. OCR Engine
- Uses Tesseract via `pytesseract` for scanned image OCR. The earlier prototype CRNN-based trainer is removed in this build; the pipeline focuses on robust preprocessing (deskew, adaptive threshold) and Tesseract text extraction.

### 3.4. Synthetic Data Generation & Training
- Training and model artifacts (PyTorch training scripts and .pth weights) have been removed from this repository to keep the package focused on robust PDF/image parsing and extraction via Tesseract.

### 3.5. Reconciliation & Export (`pipeline/validator.py`)
- Strict Pydantic models for transactions (`date`, `description`, `amount`, `balance`, `type`).
- Ledger reconciliation rule:
  $$\text{Opening Balance} + \sum \text{Credits} - \sum \text{Debits} == \text{Closing Balance} \pm 0.01$$
- Export verified payloads to structured `.json` and multi-tab `.xlsx` files.

---

## 4. CLI Specifications
- `python main.py test-ocr`: Runs single-sample synthetic inference and displays ground truth vs prediction.
- `python main.py train --epochs 25 --samples 5000`: Generates synthetic data and trains the CRNN model.
- `python main.py parse <pdf_path> [--output <file.json>] [--export-xlsx <file.xlsx>]`: Processes any bank statement.