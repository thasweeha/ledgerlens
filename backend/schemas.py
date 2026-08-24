"""
Pydantic v2 schemas for the LedgerLens FastAPI backend.
"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, model_validator
import uuid


class BBox(BaseModel):
    x: float = Field(0.0, description="Top-left X coordinate in 300 DPI image pixels")
    y: float = Field(0.0, description="Top-left Y coordinate in 300 DPI image pixels")
    width: float = Field(0.0, description="Width in 300 DPI image pixels")
    height: float = Field(0.0, description="Height in 300 DPI image pixels")
    page: Optional[int] = Field(1, description="1-based page number")


class TransactionRow(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    date: str = ""
    description: str = ""
    amount: float = 0.0
    type: Literal["credit", "debit"] = "credit"
    balance: Optional[float] = None
    page: int = 1
    bbox: Optional[BBox] = None
    date_bbox: Optional[BBox] = None
    desc_bbox: Optional[BBox] = None
    amount_bbox: Optional[BBox] = None

    @model_validator(mode="after")
    def validate_type(self) -> "TransactionRow":
        t = str(self.type).lower().strip()
        if t in ("credit", "cr", "+"):
            self.type = "credit"
        elif t in ("debit", "dr", "-"):
            self.type = "debit"
        else:
            self.type = "credit" if self.amount >= 0 else "debit"
        return self


class PageInfo(BaseModel):
    page_number: int
    page_index: int
    type: Literal["vector", "scanned"]
    width: int
    height: int
    image_url: str
    text_preview: Optional[str] = ""


class ReconciliationSummary(BaseModel):
    reconciled: bool
    opening_balance: float
    closing_balance: float
    total_credits: float
    total_debits: float
    calculated_closing: float
    difference: float
    transaction_count: int
    tolerance: float = 0.01


class StatementPayload(BaseModel):
    session_id: str
    filename: str
    page_count: int
    pages: List[PageInfo]
    opening_balance: float = 0.0
    closing_balance: float = 0.0
    currency: Optional[str] = "USD"
    account_number: Optional[str] = None
    statement_period: Optional[str] = None
    transactions: List[TransactionRow] = Field(default_factory=list)
    reconciliation: ReconciliationSummary


class ReOCRRequest(BaseModel):
    session_id: Optional[str] = None
    page_index: int = 0
    bbox: BBox
    image_base64: Optional[str] = None
    target_field: Optional[str] = "all"  # 'date', 'description', 'amount', 'all'


class ReOCRResponse(BaseModel):
    raw_text: str
    cleaned_text: str
    parsed_date: Optional[str] = None
    parsed_amount: Optional[float] = None
    target_field: str = "all"
    confidence: float = 1.0


class ValidateRequest(BaseModel):
    opening_balance: float = 0.0
    closing_balance: float = 0.0
    transactions: List[TransactionRow] = Field(default_factory=list)


class ExportRequest(BaseModel):
    statement: StatementPayload
    format: Literal["json", "xlsx"] = "xlsx"
