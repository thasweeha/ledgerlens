"""Mathematical reconciliation and continuity engine (spec §5).

All arithmetic uses decimal.Decimal. Supports:

  * Bank formula:      Opening + Σ(Deposits) - Σ(Withdrawals) - Σ(Checks) == Ending
  * Credit card:       Previous - Σ(Payments/Credits) + Σ(Purchases) + Σ(Fees)
                       + Σ(Interest) == New Balance
  * Running-balance verification per transaction.
  * Multi-page continuity via Balance Forward / Carried Forward checkpoints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

from pipeline.money import parse_amount_decimal, quantize_cents

ZERO = Decimal("0")
CENT = Decimal("0.01")

FEE_RE = re.compile(r"\bfee\b|\bfees?\b|s/c|service\s+charge", re.I)
INTEREST_RE = re.compile(r"\binterest\b", re.I)
PAYMENT_RE = re.compile(r"\bpayment\b|\bcredit\s+memo\b|\brefund\b", re.I)


class ReconciliationStatus(str, Enum):
    BALANCED = "BALANCED"
    UNBALANCED = "UNBALANCED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class LedgerEntry:
    """A normalized transaction ready for reconciliation."""

    date: str = ""
    description: str = ""
    amount: Optional[Decimal] = None
    balance: Optional[Decimal] = None
    check_number: Optional[str] = None
    page: int = 0
    is_balance_forward: bool = False


@dataclass
class ReconciliationResult:
    status: ReconciliationStatus
    formula: str
    opening_balance: Optional[Decimal] = None
    closing_balance: Optional[Decimal] = None
    expected_closing_balance: Optional[Decimal] = None
    difference: Optional[Decimal] = None
    components: Dict[str, Decimal] = field(default_factory=dict)
    running_balance_breaks: int = 0
    continuity_checkpoints: List[dict] = field(default_factory=list)
    flags: Dict[str, object] = field(default_factory=dict)

    def to_payload(self) -> dict:
        return {
            "status": self.status.value,
            "formula": self.formula,
            "opening_balance": self.opening_balance,
            "closing_balance": self.closing_balance,
            "expected_closing_balance": self.expected_closing_balance,
            "difference": self.difference,
            "components": {k: str(v) for k, v in self.components.items()},
            "running_balance_breaks": self.running_balance_breaks,
            "continuity_checkpoints": self.continuity_checkpoints,
            "flags": self.flags,
        }


def _sum(entries: List[LedgerEntry], predicate) -> Decimal:
    total = ZERO
    for entry in entries:
        if entry.amount is not None and predicate(entry):
            total += entry.amount
    return quantize_cents(total)


def _categorize(entry: LedgerEntry) -> str:
    text = entry.description or ""
    if INTEREST_RE.search(text):
        return "interest"
    if FEE_RE.search(text):
        return "fees"
    if entry.check_number:
        return "checks"
    if PAYMENT_RE.search(text):
        return "payments"
    return "purchases"


def reconcile(
    entries: List[LedgerEntry],
    statement_type: str,
    opening_balance: Optional[Decimal],
    closing_balance: Optional[Decimal],
    tolerance: Decimal = CENT,
) -> ReconciliationResult:
    """Cross-foot the ledger against the reported balances."""
    entries = [e for e in entries if not e.is_balance_forward]
    negative = [e for e in entries if e.amount is not None and e.amount < 0]
    positive = [e for e in entries if e.amount is not None and e.amount > 0]

    if statement_type == "CREDIT_CARD":
        credits_total = -_sum(negative, lambda e: True)
        charges_total = _sum(positive, lambda e: _categorize(e) == "purchases")
        fees_total = _sum(positive, lambda e: _categorize(e) == "fees")
        interest_total = _sum(positive, lambda e: _categorize(e) == "interest")
        components = {
            "payments_and_credits": credits_total,
            "purchases_and_charges": charges_total,
            "fees_charged": fees_total,
            "interest_charged": interest_total,
        }
        formula = (
            "Previous Balance - Σ(Payments/Credits) + Σ(Purchases) "
            "+ Σ(Fees) + Σ(Interest) == New Balance"
        )
        expected = (
            (opening_balance or ZERO)
            - credits_total
            + charges_total
            + fees_total
            + interest_total
        )
    else:
        deposits_total = _sum(positive, lambda e: True)
        withdrawals_total = -_sum(negative, lambda e: not e.check_number)
        checks_total = -_sum(negative, lambda e: bool(e.check_number))
        components = {
            "deposits_credits": deposits_total,
            "withdrawals_debits": withdrawals_total,
            "checks_paid": checks_total,
        }
        formula = (
            "Opening Balance + Σ(Deposits) - Σ(Withdrawals) - Σ(Checks) == Ending Balance"
        )
        expected = (
            (opening_balance or ZERO)
            + deposits_total
            - withdrawals_total
            - checks_total
        )

    expected = quantize_cents(expected)

    if closing_balance is None or opening_balance is None:
        result = ReconciliationResult(
            status=ReconciliationStatus.INSUFFICIENT_DATA,
            formula=formula,
            components=components,
            expected_closing_balance=expected,
        )
        result.flags["reason"] = "opening or closing balance not found on statement"
        return result

    difference = quantize_cents(expected - closing_balance)
    status = (
        ReconciliationStatus.BALANCED
        if abs(difference) <= tolerance
        else ReconciliationStatus.UNBALANCED
    )

    result = ReconciliationResult(
        status=status,
        formula=formula,
        opening_balance=quantize_cents(opening_balance),
        closing_balance=quantize_cents(closing_balance),
        expected_closing_balance=expected,
        difference=difference,
        components=components,
    )

    breaks = 0
    running = opening_balance
    for i, entry in enumerate(entries):
        if entry.amount is None:
            continue
        running += entry.amount
        if entry.balance is not None and abs(running - entry.balance) > tolerance:
            breaks += 1
            result.flags.setdefault("running_balance_mismatches", []).append(
                {"index": i, "date": entry.date, "expected": str(quantize_cents(running)),
                 "reported": str(entry.balance)}
            )
    result.running_balance_breaks = breaks
    return result


def extract_continuity_checkpoints(rows: List[dict]) -> List[dict]:
    """Pull Balance Forward / Carried Forward rows out of the parsed ledger."""
    checkpoints = []
    for row in rows:
        if row.get("is_balance_forward"):
            checkpoints.append(
                {
                    "page": row.get("page"),
                    "date": row.get("date"),
                    "description": row.get("description"),
                    "balance": row.get("balance"),
                }
            )
    return checkpoints


def split_ledger_segments(
    entries: List[LedgerEntry], tolerance: Decimal = CENT
) -> List[dict]:
    """Split a ledger into contiguous segments at continuity checkpoints.

    A Balance Forward row whose balance matches the running total continues
    the same ledger; one that does not match marks a new sub-ledger (common on
    multi-product statements). Checkpoints themselves are never transactions.
    """
    segments: List[dict] = []
    current: List[LedgerEntry] = []
    opening: Optional[Decimal] = None
    running: Optional[Decimal] = None

    def close():
        nonlocal current
        if current or opening is not None:
            segments.append({"opening": opening, "entries": current})
            current = []

    for entry in entries:
        if entry.is_balance_forward:
            claimed = entry.balance
            if current and running is not None and claimed is not None:
                if abs(quantize_cents(running - claimed)) > tolerance:
                    close()
                    opening = claimed
                    running = claimed
                    continue
                running = claimed
                continue
            if opening is None:
                opening = claimed
            running = claimed if claimed is not None else running
            continue
        current.append(entry)
        if entry.amount is not None:
            running = entry.amount if running is None else running + entry.amount
    close()
    return segments


def _last_balance(entries: List[LedgerEntry]) -> Optional[Decimal]:
    trailing = [e.balance for e in entries if e.balance is not None]
    return trailing[-1] if trailing else None


def reconcile_document(
    entries: List[LedgerEntry],
    statement_type: str,
    opening_balance: Optional[Decimal],
    closing_balance: Optional[Decimal],
    tolerance: Decimal = CENT,
) -> tuple:
    """Reconcile every ledger segment; BALANCED requires all to cross-foot.

    Returns (aggregate ReconciliationResult, per-segment payloads).
    """
    segments = split_ledger_segments(entries, tolerance)
    if not segments:
        result = reconcile(entries, statement_type, opening_balance, closing_balance, tolerance)
        return result, []

    segment_payloads = []
    aggregate_components: Dict[str, Decimal] = {}
    all_balanced = True
    total_difference = ZERO
    breaks = 0
    flags: Dict[str, object] = {}
    resolved_openings: List[Optional[Decimal]] = []

    for index, segment in enumerate(segments):
        seg_entries = segment["entries"]
        seg_opening = segment["opening"]
        if seg_opening is None:
            seg_opening = opening_balance if index == 0 else None
        resolved_openings.append(seg_opening)
        if len(segments) == 1:
            seg_closing = closing_balance
        else:
            seg_closing = _last_balance(seg_entries)

        result = reconcile(seg_entries, statement_type, seg_opening, seg_closing, tolerance)
        if result.status is ReconciliationStatus.UNBALANCED:
            all_balanced = False
        if result.difference is not None:
            total_difference += result.difference
        breaks += result.running_balance_breaks
        for key, value in result.components.items():
            aggregate_components[key] = aggregate_components.get(key, ZERO) + value
        if result.flags.get("running_balance_mismatches"):
            flags.setdefault("running_balance_mismatches", []).extend(
                result.flags["running_balance_mismatches"]
            )

        segment_payloads.append(
            {
                "segment": index,
                "opening_page": seg_entries[0].page if seg_entries else None,
                "closing_page": seg_entries[-1].page if seg_entries else None,
                "transaction_count": len(seg_entries),
                "status": result.status.value,
                "difference": decimal_or_none(result.difference),
            }
        )

    multi = len(segments) > 1
    insufficient = any(o is None for o in resolved_openings) or (
        not multi and closing_balance is None
    )

    status = (
        ReconciliationStatus.INSUFFICIENT_DATA
        if insufficient
        else ReconciliationStatus.BALANCED if all_balanced else ReconciliationStatus.UNBALANCED
    )

    formula = (
        "Previous Balance - Σ(Payments/Credits) + Σ(Purchases) + Σ(Fees) "
        "+ Σ(Interest) == New Balance"
        if statement_type == "CREDIT_CARD"
        else "Opening Balance + Σ(Deposits) - Σ(Withdrawals) - Σ(Checks) == Ending Balance"
    )

    aggregate = ReconciliationResult(
        status=status,
        formula=formula,
        opening_balance=quantize_cents(resolved_openings[0])
        if resolved_openings[0] is not None
        else None,
        closing_balance=quantize_cents(closing_balance)
        if not multi and closing_balance is not None
        else _last_balance(segments[-1]["entries"]),
        expected_closing_balance=None,
        difference=quantize_cents(total_difference),
        components={k: quantize_cents(v) for k, v in aggregate_components.items()},
        running_balance_breaks=breaks,
        flags=flags,
    )
    aggregate.flags["segments"] = segment_payloads
    return aggregate, segment_payloads


def decimal_or_none(value: Optional[Decimal]) -> Optional[float]:
    return float(value) if value is not None else None


def verify_continuity(checkpoints: List[dict], entries: List[LedgerEntry]) -> Dict[str, object]:
    """Check that carried-forward balances match the running ledger across pages."""
    report: Dict[str, object] = {"checkpoints": len(checkpoints), "breaks": []}
    if not checkpoints:
        return report

    ordered = sorted(
        [c for c in checkpoints if c.get("page") is not None], key=lambda c: c["page"]
    )
    running: Optional[Decimal] = None
    checkpoint_index = 0
    for entry in entries:
        while (
            checkpoint_index < len(ordered)
            and ordered[checkpoint_index].get("page") is not None
            and entry.page >= ordered[checkpoint_index]["page"]
        ):
            claimed = parse_amount_decimal(str(ordered[checkpoint_index].get("balance")))
            if running is not None and claimed is not None:
                diff = quantize_cents(running - claimed)
                if abs(diff) > CENT:
                    report["breaks"].append(
                        {
                            "page": ordered[checkpoint_index]["page"],
                            "expected": str(running),
                            "carried_forward": str(claimed),
                        }
                    )
            running = claimed if claimed is not None else running
            checkpoint_index += 1
        if entry.amount is not None:
            running = entry.amount if running is None else running + entry.amount

    while checkpoint_index < len(ordered):
        claimed = parse_amount_decimal(str(ordered[checkpoint_index].get("balance")))
        if running is not None and claimed is not None:
            diff = quantize_cents(running - claimed)
            if abs(diff) > CENT:
                report["breaks"].append(
                    {
                        "page": ordered[checkpoint_index]["page"],
                        "expected": str(running),
                        "carried_forward": str(claimed),
                    }
                )
        checkpoint_index += 1
    return report
