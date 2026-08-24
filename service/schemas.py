"""Shared response schemas for the LedgerLens API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str


class ApiResponse(BaseModel):
    """Standardized envelope: {success, data, error}."""

    success: bool
    data: Optional[dict] = Field(default=None)
    error: Optional[ErrorDetail] = Field(default=None)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_epoch: Optional[int] = None


class DocumentMeta(BaseModel):
    source_file: str
    page_count: int
    engine_version: str = "ledgerlens-unified-2.0"
    processed_at: Optional[str] = None


class RoutingInfo(BaseModel):
    modes_per_page: List[str] = Field(default_factory=list)
    digital_pages: List[int] = Field(default_factory=list)
    raster_pages: List[int] = Field(default_factory=list)
    raster_diagnostics: Dict[str, dict] = Field(default_factory=dict)


class ClassificationInfo(BaseModel):
    statement_type: str
    confidence: float
    document_scores: Dict[str, float] = Field(default_factory=dict)


class CellProvenance(BaseModel):
    text: str = ""
    page: Optional[int] = None
    mode: Optional[str] = None
    bbox: Optional[List[float]] = None
    confidence: Optional[float] = None

    def audit_string(self, file_name: str) -> str:
        page = (self.page + 1) if self.page is not None else 0
        mode = self.mode or "UNKNOWN"
        bbox = self.bbox or [0, 0, 0, 0]
        coords = ", ".join(f"{v:.0f}" for v in bbox)
        return f"Source: {file_name} | Page: {page} | Mode: {mode} | BBox: [{coords}]"


class TransactionRecord(BaseModel):
    index: int
    page: int
    date: str = ""
    post_date: Optional[str] = None
    description: str = ""
    check_number: Optional[str] = None
    amount: Optional[float] = None
    type: str = "unknown"
    balance: Optional[float] = None
    is_balance_forward: bool = False
    provenance: Dict[str, CellProvenance] = Field(default_factory=dict)


class ReconciliationBlock(BaseModel):
    status: str = Field(description="BALANCED | UNBALANCED | INSUFFICIENT_DATA")
    formula: str
    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None
    expected_closing_balance: Optional[float] = None
    difference: Optional[float] = None
    components: Dict[str, float] = Field(default_factory=dict)
    running_balance_breaks: int = 0
    checkpoints: List[Dict[str, Any]] = Field(default_factory=list)
    segments: List[Dict[str, Any]] = Field(default_factory=list)
    flags: Dict[str, Any] = Field(default_factory=dict)


class StatementPayload(BaseModel):
    """Unified JSON payload produced by the LedgerLens engine (spec §6)."""

    document: DocumentMeta
    routing: RoutingInfo
    classification: ClassificationInfo
    account: Dict[str, Optional[str]] = Field(default_factory=dict)
    summary_balances: Dict[str, Any] = Field(default_factory=dict)
    transactions: List[TransactionRecord] = Field(default_factory=list)
    reconciliation: ReconciliationBlock
    diagnostics: Dict[str, Any] = Field(default_factory=dict)