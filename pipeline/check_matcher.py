"""Match ICR-extracted checks to parsed ledger rows.

Algorithm:
1. For each check detected by pipeline/check_ocr.py, look at transactions on the
   same page that are debits (negative signed amounts) and have not already been
   paired with a check.
2. Pick the debit whose absolute amount is closest to the check amount within a
   configurable tolerance (default +/- $0.50).
3. Update the winning transaction with check_number, payee_name, and source
   "check_ocr".
4. If no suitable debit exists, append a new transaction row flagged for manual
   review so the check is not lost.

All monetary comparisons use Decimal to avoid floating-point rounding issues.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional

DEFAULT_AMOUNT_TOLERANCE = Decimal("0.50")


def _to_decimal(value: Any) -> Optional[Decimal]:
    """Convert an amount value to Decimal, returning None on failure."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _is_unmatched_debit(txn: Dict[str, Any]) -> bool:
    """True when a transaction row is a debit and not already check-paired."""
    amount = _to_decimal(txn.get("amount"))
    if amount is None or amount >= 0:
        return False
    if txn.get("check_number"):
        return False
    txn_type = (txn.get("type") or "").lower()
    if txn_type and txn_type not in ("debit", "unknown", ""):
        return False
    return True


def match_checks_to_transactions(
    checks: List[Dict[str, Any]],
    transactions: List[Dict[str, Any]],
    tolerance: Decimal = DEFAULT_AMOUNT_TOLERANCE,
) -> List[Dict[str, Any]]:
    """Pair extracted checks with debit transactions and append orphans.

    Args:
        checks: List of check dicts from check_ocr.detect_checks().
        transactions: List of transaction dicts from columns.parse_rows().
        tolerance: Maximum allowed amount difference for a match.

    Returns:
        Updated transaction list.  Matched rows gain check_number, payee_name,
        and source="check_ocr".  Unmatched checks become new rows with
        review_flag="manual_review".
    """
    working = [dict(t) for t in transactions]

    for check in checks or []:
        check_amount = _to_decimal(check.get("amount"))
        if check_amount is None:
            continue

        best_index: Optional[int] = None
        best_diff: Optional[Decimal] = None

        for idx, txn in enumerate(working):
            if txn.get("page") != check.get("page_number"):
                continue
            if not _is_unmatched_debit(txn):
                continue
            txn_amount = abs(_to_decimal(txn.get("amount")))
            diff = abs(txn_amount - abs(check_amount))
            if diff > tolerance:
                continue
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_index = idx

        if best_index is not None:
            txn = working[best_index]
            txn["check_number"] = check.get("check_number") or txn.get("check_number")
            txn["payee_name"] = check.get("payee_name")
            txn["source"] = "check_ocr"
        else:
            working.append(
                {
                    "page": check.get("page_number"),
                    "date": "",
                    "post_date": None,
                    "description": f"CHECK {check.get('check_number') or 'UNKNOWN'}",
                    "check_number": check.get("check_number"),
                    "payee_name": check.get("payee_name"),
                    "amount": -abs(check_amount),
                    "type": "debit",
                    "balance": None,
                    "is_balance_forward": False,
                    "source": "check_ocr",
                    "review_flag": "manual_review",
                    "cells": {},
                }
            )

    return working
