"""Financial token normalization (spec §4/§5).

Handles every negative-notation variant found on statements:
-$45.00, $45.00-, ($45.00), 45.00 CR / 45.00 DR, +$12.00, 1,234.56-
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, localcontext
from typing import Optional

MAX_AMOUNT_DIGITS = 15

AMOUNT_CORE_RE = re.compile(
    r"""^
    (?P<lead_sign>[+\-])?
    \(?
    \$?\s*
    (?P<digits>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)
    \s*
    (?P<crdr>CR|DR)?
    \)?
    (?P<trail_sign>[+\-])?
    $""",
    re.IGNORECASE | re.VERBOSE,
)

CR_RE = re.compile(r"(?:^|\s)(CR)\s*$", re.IGNORECASE)
DR_RE = re.compile(r"(?:^|\s)(DR)\s*$", re.IGNORECASE)


THOUSAND_DOT_RE = re.compile(r"^\$?\s*(\d{1,3}(?:\.\d{3})+)(?:[.,](\d{2}))?$")
EU_DECIMAL_RE = re.compile(r"^\$?\s*(\d{1,3}(?:\.\d{3})+),(\d{1,2})$")


def _normalize_separators(cleaned: str) -> str:
    """Repair OCR separator damage: '$9.099.73' -> '9,099.73',
    '30.000,00' -> '30,000.00'."""
    match = EU_DECIMAL_RE.match(cleaned)
    if match:
        return match.group(1).replace(".", ",") + "." + match.group(2)
    match = THOUSAND_DOT_RE.match(cleaned)
    if match:
        cents = match.group(2) or "00"
        return match.group(1).replace(".", ",") + "." + cents
    return cleaned


def parse_amount_decimal(text: Optional[str]) -> Optional[Decimal]:
    """Normalize an amount string to a signed Decimal (credit positive).

    Conventions:
      * parentheses or minus -> negative
      * trailing/leading CR suffix/prefix -> positive credit marker stripped
      * trailing/leading DR -> negative
      * '+' -> positive
    """
    if text is None:
        return None
    cleaned = text.strip()
    if not cleaned or cleaned in {"-", "--", "—", "–"}:
        return None

    cr_dr_positive = bool(CR_RE.search(cleaned))
    dr_negative = bool(DR_RE.search(cleaned))
    cleaned = CR_RE.sub("", cleaned)
    cleaned = DR_RE.sub("", cleaned)
    cleaned = cleaned.replace("$", "").replace(" ", "")
    cleaned = _normalize_separators(cleaned)

    match = AMOUNT_CORE_RE.match(cleaned)
    if not match:
        try:
            return Decimal(cleaned)
        except (InvalidOperation, ValueError):
            return None

    digits = match.group("digits")
    if len(digits.split(".")[0].replace(",", "").lstrip("0")) > MAX_AMOUNT_DIGITS:
        return None
    try:
        value = Decimal(digits.replace(",", ""))
    except InvalidOperation:
        return None

    negative = (
        dr_negative
        or match.group("trail_sign") == "-"
        or match.group("lead_sign") == "-"
        or (cleaned.startswith("(") and cleaned.endswith(")"))
    )
    if negative:
        value = -abs(value)
    elif cr_dr_positive or match.group("lead_sign") == "+":
        value = abs(value)
    return value


def looks_like_amount(text: Optional[str]) -> bool:
    """True when a token parses as a monetary value."""
    return parse_amount_decimal(text) is not None


def decimal_to_float(value: Optional[Decimal]) -> Optional[float]:
    """JSON-safe float conversion with cent rounding."""
    if value is None:
        return None
    return float(round(value, 2))


def quantize_cents(value: Decimal) -> Decimal:
    """Round to cents, surviving garbage OCR values that exceed the default
    decimal context precision."""
    try:
        return value.quantize(Decimal("0.01"))
    except InvalidOperation:
        with localcontext() as ctx:
            ctx.prec = max(28, len(value.as_tuple().digits) + 4)
            return value.quantize(Decimal("0.01"))
