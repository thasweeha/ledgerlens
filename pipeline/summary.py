"""Statement summary extraction (spec §3/§5).

Geometry-aware reader for summary bands where labels sit on one baseline and
amounts on the next (e.g. STARTING BALANCE / TOTAL DEPOSITS / TOTAL
WITHDRAWALS / ENDING BALANCE followed by four money values). Also pulls
account number, routing number, and statement period from free text.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from pipeline.money import looks_like_amount, parse_amount_decimal
from pipeline.tokens import Token, cluster_baselines, median, page_heights

LABEL_PHRASES: List[Tuple[str, re.Pattern]] = [
    ("opening_balance", re.compile(r"^(?:starting|opening|beginning)\s+balance$", re.I)),
    ("closing_balance", re.compile(r"^(?:ending|closing|new|total)\s+balance$", re.I)),
    ("previous_balance", re.compile(r"^previous\s+balance$", re.I)),
    ("new_balance", re.compile(r"^new\s+balance$", re.I)),
    ("total_deposits", re.compile(r"^total\s+(?:deposits|credits)$", re.I)),
    ("total_withdrawals", re.compile(r"^total\s+(?:withdrawals|debits)$", re.I)),
    ("purchases_and_charges", re.compile(r"^(?:total\s+charges?|purchases?)$", re.I)),
    ("payments_and_credits", re.compile(r"^payments?\s*(?:&|and)\s*credits?$|^payments?$", re.I)),
    ("fees_charged", re.compile(r"^fees?\s+charged?$|^fees?$", re.I)),
    ("interest_charged", re.compile(r"^interest\s+charged?$|^interest$", re.I)),
]

BARE_BALANCE_LABELS: List[Tuple[str, re.Pattern]] = [
    ("summary_balance", re.compile(r"^balance$", re.I)),
]

LABEL_START_RE = re.compile(
    r"^(?:starting|opening|beginning|ending|closing|previous|new|total|payments?|fees?|interest)$",
    re.I,
)
LABEL_CONTINUATION_RE = re.compile(
    r"^(?:balance|deposits?|credits?|withdrawals?|debits?|charged?|&|and)$", re.I
)

ACCOUNT_RE = re.compile(
    r"account\s*(?:number|#|no\.?)\s*[:\-]?\s*([0-9A-Z*#][0-9A-Z\-*# ]{2,})", re.I
)
ROUTING_RE = re.compile(
    r"(?:routing\s*(?:number|no\.?|#)?|transit\s*(?:number|no\.?|#)?)\s*[:\-]?\s*(\d{5,9})",
    re.I,
)
_DATE_PART = (
    r"(?:[A-Za-z]{3,9}\.?\s+\d{1,2},?\s*\d{4}"
    r"|\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}"
    r"|\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2})"
)
PERIOD_RE = re.compile(
    rf"(?:statement\s*period|period\s*ending)\s*[:\-]?\s*"
    rf"({_DATE_PART}(?:\s*[–—-]\s*(?:[A-Za-z]{{3,9}}\.?\s*)?{_DATE_PART})?)",
    re.I | re.VERBOSE,
)


def _match_label_phrase(tokens: List[Token]) -> Optional[Tuple[str, int]]:
    """Match a known label phrase starting at tokens[i]; returns (name, length)."""
    joined_2 = " ".join(t.text for t in tokens[:2])
    joined_3 = " ".join(t.text for t in tokens[:3])
    for name, pattern in LABEL_PHRASES:
        if pattern.match(joined_3):
            return name, min(3, len(tokens))
        if pattern.match(joined_2):
            return name, min(2, len(tokens))
    return None


def _summary_pairs_from_baselines(rows: List[List[Token]], found: Dict[str, Token]) -> None:
    """Pair label phrases with amounts on the following baseline by x-order.

    Mutates `found` with setdefault semantics so summary blocks that appear
    above transaction tables keep priority.
    """
    for idx, row in enumerate(rows):
        labels: List[Tuple[str, int, float]] = []
        i = 0
        while i < len(row):
            token = row[i]
            if LABEL_START_RE.match(token.text):
                match = _match_label_phrase(row[i:])
                if match:
                    name, length = match
                    phrase_tokens = row[i : i + length]
                    center = sum(t.x_center for t in phrase_tokens) / len(phrase_tokens)
                    labels.append((name, i + length, center))
                    i += length
                    continue
            i += 1

        # A bare BALANCE label only counts on sparse baselines (summary
        # tables), never on full transaction-table header rows.
        sparse = len(labels) == 0 and len(row) <= 3
        if 0 < len(labels) <= 2 or sparse:
            for j, token in enumerate(row):
                for name, pattern in BARE_BALANCE_LABELS:
                    if pattern.match(token.text) and not any(
                        lbl[0] == name or lbl[0].endswith("balance") for lbl in labels
                    ):
                        labels.append((name, j + 1, token.x_center))
                        break

        if not labels:
            continue

        # Inline amounts first: "<label> ... $123.45" on the same baseline.
        # Summary boxes park values in a right-hand column, so accept any
        # amount to the label's right, not just within a tight radius.
        remaining_labels = []
        for name, consumed, center in labels:
            right_amounts = [
                t
                for t in row[consumed:]
                if looks_like_amount(t.text) and t.x_center >= center
            ]
            if right_amounts and name not in found:
                found[name] = min(
                    right_amounts, key=lambda t: abs(t.x_center - center)
                )
            else:
                remaining_labels.append((name, consumed, center))

        if not remaining_labels:
            continue

        # Otherwise take amounts from the next baseline(s).
        value_row: List[Token] = []
        for offset in (1, 2):
            if idx + offset < len(rows):
                value_row.extend(rows[idx + offset])
        amounts = [t for t in value_row if looks_like_amount(t.text)]
        amounts.sort(key=lambda t: t.x_center)
        used = set()
        for name, _, center in remaining_labels:
            if name in found:
                continue
            candidates = [
                (abs(t.x_center - center), j)
                for j, t in enumerate(amounts)
                if j not in used
            ]
            if candidates:
                _, best = min(candidates)
                found[name] = amounts[best]
                used.add(best)


def extract_statement_summary(pages_tokens: Dict[int, List[Token]]) -> dict:
    """Extract reported balances and account metadata from all pages."""
    all_tokens = [t for page in sorted(pages_tokens) for t in pages_tokens[page]]
    text_lines = []
    for page in sorted(pages_tokens):
        info = page_heights(pages_tokens[page])
        tolerance = (info.get(page, (0, 10.0))[1]) * 0.6 or 5.0
        for row in cluster_baselines(pages_tokens[page], tolerance):
            text_lines.append(" ".join(t.text for t in row))
    full_text = "\n".join(text_lines)

    rows_by_page = {}
    for page in sorted(pages_tokens):
        info = page_heights(pages_tokens[page])
        tolerance = (info.get(page, (0, 10.0))[1]) * 0.6 or 5.0
        rows_by_page[page] = cluster_baselines(pages_tokens[page], tolerance)

    found: Dict[str, Token] = {}
    for page in sorted(rows_by_page):
        _summary_pairs_from_baselines(rows_by_page[page], found)

    def value(name: str):
        token = found.get(name)
        return parse_amount_decimal(token.text) if token else None

    account = ACCOUNT_RE.search(full_text)
    routing = ROUTING_RE.search(full_text)
    period = PERIOD_RE.search(full_text)

    return {
        "opening_balance": value("opening_balance") or value("previous_balance"),
        "closing_balance": value("closing_balance") or value("new_balance") or value("summary_balance"),
        "reported_totals": {
            name: value(name)
            for name in (
                "total_deposits",
                "total_withdrawals",
                "purchases_and_charges",
                "payments_and_credits",
                "fees_charged",
                "interest_charged",
            )
            if value(name) is not None
        },
        "account_number": account.group(1).strip() if account else None,
        "routing_number": routing.group(1) if routing else None,
        "statement_period": period.group(1).strip() if period else None,
        "summary_sources": {
            name: {
                "page": token.page,
                "mode": token.mode.short,
                "bbox": token.bbox,
            }
            for name, token in found.items()
        },
    }
