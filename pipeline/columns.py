"""Dynamic column anchoring and multi-line parsing engine (spec §4).

Unified extraction applied identically to vector tokens and neural OCR tokens:

  1. Header anchor localization across synonym variations.
  2. X-lane construction from header centroids (coordinate projection).
  3. Y-clustering of tokens into visual baseline rows.
  4. Row state machine: transaction rows, multi-line description continuation,
     balance-forward checkpoints, footer noise rejection.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from pipeline.money import looks_like_amount, parse_amount_decimal
from pipeline.tokens import Token, cluster_baselines, median, overlap_fraction, page_heights

DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(r"\b[A-Z][a-z]{2,8}\.?\s+\d{1,2},?\s*\d{4}\b"),
    re.compile(r"\b\d{4}[.-]\d{2}[.-]\d{2}\b"),
    # Year-less table dates: 'JUN 24', 'June 7'
    re.compile(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b",
        re.I,
    ),
]

HEADER_SYNONYMS: Dict[str, re.Pattern] = {
    "date": re.compile(r"^(?:trans(?:action)?\s*date|posting\s*date|post\s*date|date)$", re.I),
    "description": re.compile(r"^(?:description|details|activity|transaction|memo|narrative)$", re.I),
    "categories": re.compile(r"^(?:spend\s+)?categor(?:y|ies)$", re.I),
    "check_number": re.compile(r"^(?:check#|cheque#|check\s*#|cheque\s*#|check\s*num(?:ber)?|cheque\s*num(?:ber)?|ref(?:erence)?#?|reference)$", re.I),
    "debit": re.compile(r"^(?:withdrawals?|debits?|purchases?|charges?|money\s*out)$", re.I),
    "credit": re.compile(r"^(?:deposits?|credits?|payments?|money\s*in)$", re.I),
    "amount": re.compile(
        r"^(?:amount|charge)(?:\s*(?:\([^)]{0,24}\)|\[\w{1,3}\]|\$))?$", re.I
    ),
    "balance": re.compile(r"^(?:balance|running\s*balance|new\s*balance)$", re.I),
}

BALANCE_FORWARD_RE = re.compile(
    r"balance\s+(?:forward|carried\s+forward|brought\s+forward)|(?:^|\b)b/?f\b", re.I
)

NOISE_ROW_RE = re.compile(
    r"(?:\(continued\)|continued\.|page\s+\d+(\s+of\s+\d+)?|\d+\s+of\s+\d+"
    r"|statement\s+period|routing\s+number|account\s+(?:number|#|no)"
    r"|starting\s+balance|ending\s+balance|beginning\s+balance|opening\s+balance|closing\s+balance"
    r"|total\s+(?:deposits|withdrawals|debits|credits)|previous\s+balance|new\s+balance"
    r"|payments?\s*&\s*credits?|fees?\s+charged|interest\s+charged|payment\s+due"
    r"|tel\s*:|fax\s*:|www\.|https?://|visit\s+us\s+online|moyafinancial|fdic"
    r"|please\s+review|this\s+statement|^\W+$)",
    re.I,
)

DESCRIPTION_NOISE_CUT_RE = re.compile(
    r"(?:total\s+(?:deposits|withdrawals|debits|credits)\b.*|statement\s+period\b.*"
    r"|account\s*(?:#|number|no\.?)\b.*|page\s+\d+\b.*|\(continued\).*)",
    re.I,
)

ATTACH_WINDOW_FACTOR = 2.2
MIN_ATTACH_WINDOW_PT = 12.0

HEADER_LANE_TOLERANCE_PT = 18.0
ROW_Y_TOLERANCE_FACTOR = 0.6

# TrOCR frequently splits a leading capital into its own word ('P AYPAL',
# 'C ONFIRMATION'); rejoin it when a real word follows.
SPLIT_LETTER_RE = re.compile(r"\b([A-Za-z])\s+(?=[A-Z]{2,})")

# Date cells may absorb debris from neighbouring lanes ('E', ':', '=',
# stray digits). Only month/day/year-shaped fragments survive.
_DATE_PART_RE = re.compile(
    r"[A-Za-z]{3,9}\.?|\d{1,2},?|\d{4},?|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?,?"
)

SECTION_CREDIT_RE = re.compile(
    r"(?:your\s+)?payments?(?:\s*(?:&|and)\s*credits?)?|credits?", re.I
)
SECTION_DEBIT_RE = re.compile(
    r"(?:your\s+)?(?:new\s+)?(?:money\s+out\s+)?(?:purchases?|charges?|debits?|transactions?|fees?)"
    r"(?:\s*(?:&|and|,)\s*(?:credits?|debits?|fees?))?",
    re.I,
)


def _row_text(row: List[Token]) -> str:
    return " ".join(t.text for t in row)


def _clean_date_text(raw: str) -> str:
    """Drop OCR debris from a date cell while preserving good values.

    Cells made only of date-safe characters pass through untouched when
    they parse as dates; anything with debris ('=', ':') keeps only
    month/day/year-shaped fragments and is dropped entirely when nothing
    date-like remains."""
    text = re.sub(r"\s+", " ", raw).strip()
    if not text:
        return ""
    if re.fullmatch(r"[A-Za-z0-9 ,/-]+", text) and _matches_date(text):
        return text
    parts = [
        m.group(0).rstrip(".,")
        for m in _DATE_PART_RE.finditer(text)
    ]
    cleaned = " ".join(p for p in parts if p)
    return cleaned if _matches_date(cleaned) else ""


def _matches_date(text: str) -> bool:
    return any(p.search(text) for p in DATE_PATTERNS)


def _canonical_matches(token: Token) -> List[str]:
    return [
        canonical
        for canonical, pattern in HEADER_SYNONYMS.items()
        if pattern.match(token.text)
    ]


def _split_fused_header(token: Token) -> List[Tuple[str, Token]]:
    """Split an over-fused OCR token ('DATE DESCRIPTION BALANCE') into
    per-synonym sub-anchors with proportional x geometry."""
    text = token.text
    hits: List[Tuple[int, int, str]] = []
    for canonical, pattern in HEADER_SYNONYMS.items():
        for match in pattern.finditer(text):
            hits.append((match.start(), match.end(), canonical))
    if len(hits) <= 1:
        return []
    hits.sort()
    width = token.width or 1.0
    out: List[Tuple[str, Token]] = []
    for start, end, canonical in hits:
        fx0 = token.x0 + width * (start / len(text))
        fx1 = token.x0 + width * (end / len(text))
        out.append(
            (
                canonical,
                Token(
                    text=text[start:end],
                    page=token.page,
                    bbox=(fx0, token.y0, fx1, token.y1),
                    mode=token.mode,
                    confidence=token.confidence,
                    font_size=token.font_size,
                ),
            )
        )
    return out


def find_header_anchors(pages_tokens: Dict[int, List[Token]]) -> Dict[int, Dict[str, List[Token]]]:
    """Per page: canonical column -> header tokens anchoring that column.

    Candidate header baselines are scored by how many distinct canonical
    columns they anchor (with a bonus for an explicit date label) so summary
    bands like STARTING BALANCE / TOTAL DEPOSITS never win against the real
    transaction-table header. Over-fused OCR tokens are split into per-label
    anchors so raster header rows score equally with vector ones.
    """
    anchors: Dict[int, Dict[str, List[Token]]] = {}
    for page, tokens in pages_tokens.items():
        heights = page_heights(tokens)
        tolerance = (heights.get(page, (0, 10.0))[1]) * ROW_Y_TOLERANCE_FACTOR or 5.0
        best_row: Optional[Tuple[Tuple[int, int], Dict[str, List[Token]]]] = None
        for row in cluster_baselines(tokens, tolerance):
            row_anchors: Dict[str, List[Token]] = {}
            for token in row:
                exact = _canonical_matches(token)
                if exact:
                    for canonical in exact:
                        row_anchors.setdefault(canonical, []).append(token)
                    continue
                for canonical, sub_token in _split_fused_header(token):
                    row_anchors.setdefault(canonical, []).append(sub_token)
            if len(row_anchors) < 2:
                continue
            money_bonus = 1 if any(
                key in row_anchors for key in ("amount", "balance", "debit", "credit")
            ) else 0
            score = (
                len(row_anchors),
                money_bonus,
                1 if "date" in row_anchors else 0,
            )
            if best_row is None or score > best_row[0]:
                best_row = (score, row_anchors)
        if best_row is not None:
            anchors[page] = best_row[1]
    return anchors


def build_lanes(
    anchors: Dict[int, Dict[str, List[Token]]], all_tokens: List[Token]
) -> Tuple[List[Tuple[float, float]], Dict[str, int]]:
    """Project header centroids into lane intervals [x_min, x_max].

    Returns lanes sorted left-to-right plus canonical-column -> lane index.
    """
    centers_by_column: Dict[str, List[float]] = {}
    for page_anchors in anchors.values():
        for canonical, tokens in page_anchors.items():
            centers_by_column.setdefault(canonical, []).extend(t.x_center for t in tokens)

    representatives: List[Tuple[float, str]] = []
    for canonical, centers in centers_by_column.items():
        clusters: List[List[float]] = []
        for center in sorted(centers):
            if clusters and abs(center - median(clusters[-1])) <= HEADER_LANE_TOLERANCE_PT:
                clusters[-1].append(center)
            else:
                clusters.append([center])
        best = max(clusters, key=len)
        representatives.append((median(best), canonical))

    # Collapse distinct columns sharing the same physical position (e.g. a
    # single AMOUNT column detected as both 'debit' and 'amount').
    representatives.sort()
    collapsed: List[Tuple[float, str]] = []
    for center, canonical in representatives:
        if collapsed and abs(center - collapsed[-1][0]) <= HEADER_LANE_TOLERANCE_PT:
            kept_center, kept = collapsed[-1]
            preference = {"amount": 2, "debit": 1, "credit": 1}
            if preference.get(canonical, 0) > preference.get(kept, 0):
                collapsed[-1] = (kept_center, canonical)
            continue
        collapsed.append((center, canonical))

    if not collapsed:
        return [], {}

    xs = [t.x0 for t in all_tokens] + [t.x1 for t in all_tokens]
    left_edge = min(xs) if xs else 0.0
    right_edge = max(xs) if xs else 0.0

    lanes: List[Tuple[float, float]] = []
    column_index: Dict[str, int] = {}
    boundaries = [left_edge]
    for i in range(len(collapsed) - 1):
        boundaries.append((collapsed[i][0] + collapsed[i + 1][0]) / 2.0)
    boundaries.append(right_edge + 1.0)

    for i, (_, canonical) in enumerate(collapsed):
        lanes.append((boundaries[i], boundaries[i + 1]))
        column_index[canonical] = i
    return lanes, column_index


def _is_table_layout(colidx: Dict[str, int]) -> bool:
    """A credible transaction table pairs a description column with at
    least one date or money column; cover letters and analytics grids
    anchor stray label words without ever forming this shape."""
    return "description" in colidx and bool(
        {"date", "debit", "credit", "amount", "balance"} & set(colidx)
    )


def build_page_lanes(
    anchors: Dict[int, Dict[str, List[Token]]],
    pages_tokens: Dict[int, List[Token]],
) -> Dict[int, Tuple[List[Tuple[float, float]], Dict[str, int]]]:
    """Per-page lane geometry.

    Scanned statements drift horizontally between pages, so lane intervals
    must come from each page's own header anchors; a page without anchors
    inherits the previous page's layout."""
    result: Dict[int, Tuple[List[Tuple[float, float]], Dict[str, int]]] = {}
    previous: Optional[Tuple[List[Tuple[float, float]], Dict[str, int]]] = None
    blocked = False
    for page in sorted(pages_tokens):
        page_anchors = anchors.get(page)
        if page_anchors:
            candidate = build_lanes({page: page_anchors}, pages_tokens[page])
            if candidate[0] and _is_table_layout(candidate[1]):
                result[page] = candidate
                previous = candidate
                continue
            # Label words that never form a credible table mark the end of
            # the transaction section; later pages must not inherit lanes.
            blocked = True
            continue
        if previous is not None and not blocked:
            result[page] = previous
    return result


def assign_lanes(
    tokens: List[Token], lanes: List[Tuple[float, float]]
) -> Dict[Token, int]:
    """Assign each token to the lane with maximal horizontal overlap."""
    assignment: Dict[Token, int] = {}
    for token in tokens:
        fractions = [
            overlap_fraction(token, start, end) for start, end in lanes
        ]
        best = max(range(len(lanes)), key=lambda i: fractions[i]) if lanes else 0
        if fractions[best] <= 0:
            distances = [
                min(abs(token.x_center - start), abs(token.x_center - end))
                for start, end in lanes
            ]
            best = min(range(len(lanes)), key=lambda i: distances[i])
        assignment[token] = best
    return assignment


class Cell:
    """A parsed field value with provenance back to its source tokens."""

    def __init__(self) -> None:
        self.tokens: List[Token] = []

    def add(self, token: Token) -> None:
        self.tokens.append(token)

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.tokens)

    @property
    def bbox(self) -> Optional[Tuple[float, float, float, float]]:
        if not self.tokens:
            return None
        return (
            min(t.x0 for t in self.tokens),
            min(t.y0 for t in self.tokens),
            max(t.x1 for t in self.tokens),
            max(t.y1 for t in self.tokens),
        )

    @property
    def page(self) -> Optional[int]:
        return self.tokens[0].page if self.tokens else None

    @property
    def mode(self) -> str:
        return self.tokens[0].mode.short if self.tokens else ""

    def amount(self) -> Optional[Decimal]:
        parsed = parse_amount_decimal(self.text)
        if parsed is not None:
            return parsed
        # OCR lane spill (e.g. 'RECREATION 47.49'): the true amount sits at
        # the right edge, so fall back to the rightmost parseable token.
        for token in reversed(self.tokens):
            parsed = parse_amount_decimal(token.text)
            if parsed is not None:
                return parsed
        return None


class TransactionRow:
    """One parsed ledger line with per-cell provenance."""

    def __init__(self) -> None:
        self.cells: Dict[str, Cell] = {name: Cell() for name in HEADER_SYNONYMS}
        self.cells["post_date"] = Cell()
        self.page = 0
        self.is_balance_forward = False
        self.polarity = "debit"

    def cell_for(self, canonical: Optional[str]) -> Cell:
        """Map a lane canonical to its cell; a second date lane is post_date."""
        name = canonical if canonical else "description"
        if name == "date" and self.cells["date"].tokens:
            return self.cells["post_date"]
        return self.cells.setdefault(name, Cell())

    def finalize(self) -> dict:
        date_cell = self.cells["date"]
        post_date_cell = self.cells["post_date"]
        # A post date whose month token lost its day to the description lane
        # (common when the POST DATE column straddles a lane boundary).
        if post_date_cell.tokens and not _matches_date(post_date_cell.text):
            desc_tokens = self.cells["description"].tokens
            if desc_tokens and re.fullmatch(r"\d{1,2}", desc_tokens[0].text):
                post_date_cell.add(desc_tokens.pop(0))
        desc_cell = self.cells["description"]
        debit = self.cells["debit"].amount() if self.cells["debit"].tokens else None
        credit = self.cells["credit"].amount() if self.cells["credit"].tokens else None
        amount_cell = self.cells["amount"]
        single_amount = amount_cell.amount() if amount_cell.tokens else None
        balance = self.cells["balance"].amount() if self.cells["balance"].tokens else None

        description = re.sub(r"\s+", " ", desc_cell.text).strip()
        description = DESCRIPTION_NOISE_CUT_RE.sub("", description).strip(" -,")
        description = SPLIT_LETTER_RE.sub(r"\1", description)
        self.is_balance_forward = bool(BALANCE_FORWARD_RE.search(description))

        signed_amount: Optional[Decimal] = None
        if debit is not None:
            signed_amount = -abs(debit)
        elif credit is not None:
            signed_amount = abs(credit)
        elif single_amount is not None:
            signed_amount = single_amount
        if self.polarity == "credit" and (
            debit is None and credit is None and single_amount is not None
        ):
            # Card statements park payments in an unsigned amount column;
            # the enclosing section header carries the direction. Debit and
            # credit lanes are pre-signed and must never be flipped.
            signed_amount = -abs(signed_amount)

        return {
            "page": date_cell.page if date_cell.page is not None else self.page,
            "date": _clean_date_text(date_cell.text),
            "post_date": _clean_date_text(post_date_cell.text) or None,
            "description": description,
            "check_number": self.cells["check_number"].text.strip() or None,
            "debit": debit,
            "credit": credit,
            "amount": signed_amount,
            "balance": balance,
            "is_balance_forward": self.is_balance_forward,
            "cells": {
                name: {
                    "text": cell.text.strip(),
                    "bbox": cell.bbox,
                    "page": cell.page,
                    "mode": cell.mode,
                    "tokens": list(cell.tokens),
                }
                for name, cell in self.cells.items()
                if cell.tokens
            },
        }


def _is_header_row(row: List[Token]) -> bool:
    """True when every token on the baseline is a column label, including
    over-fused OCR tokens that pack several labels into one box."""
    if not row:
        return False
    for token in row:
        if _canonical_matches(token):
            continue
        if len(_split_fused_header(token)) >= 2:
            continue
        return False
    return True


def _group_date_tokens(bucket: List[Token]) -> List[List[Token]]:
    """Split a date lane's tokens into consecutive date values ('JUN' + '24'
    -> one group) so dual-date tables can fill date and post_date cells."""
    ordered = sorted(bucket, key=lambda t: t.x0)
    groups: List[List[Token]] = []
    current: List[Token] = []
    for token in ordered:
        if current and _matches_date(" ".join(t.text for t in current)):
            groups.append(current)
            current = [token]
        else:
            current.append(token)
    if current:
        groups.append(current)
    return groups


def parse_rows(pages_tokens: Dict[int, List[Token]]) -> dict:
    """Run the full anchoring + clustering + state machine over all tokens."""
    anchors = find_header_anchors(pages_tokens)
    page_lanes = build_page_lanes(anchors, pages_tokens)

    rows_out: List[dict] = []
    columns_by_page = {
        page: {canonical: idx for idx, canonical in sorted(colidx.items(), key=lambda kv: kv[1])}
        for page, (_, colidx) in page_lanes.items()
    }
    diagnostics = {
        "header_pages": sorted(anchors.keys()),
        "lanes": [
            {"page": page, "x_min": round(s, 1), "x_max": round(e, 1)}
            for page, (lanes, _) in sorted(page_lanes.items())
            for s, e in lanes
        ],
        "columns": columns_by_page,
        "rows_seen": 0,
        "continuations": 0,
        "balance_forward_checkpoints": 0,
    }

    if not page_lanes:
        diagnostics["warning"] = "no header anchors detected; table extraction skipped"
        return {"transactions": rows_out, "diagnostics": diagnostics}

    current: Optional[TransactionRow] = None
    completed: List[TransactionRow] = []
    current_section = "debit"

    for page in sorted(pages_tokens):
        if page not in page_lanes:
            continue
        lanes, column_index = page_lanes[page]
        tokens = pages_tokens[page]
        info = page_heights(tokens)
        _, typical_height = info.get(page, (0, 10.0))
        tolerance = typical_height * ROW_Y_TOLERANCE_FACTOR or 5.0
        window = max(ATTACH_WINDOW_FACTOR * typical_height, MIN_ATTACH_WINDOW_PT)
        assignment = assign_lanes(tokens, lanes)
        date_lane = column_index.get("date")
        desc_lane = column_index.get("description", 1)
        amount_lanes = {
            column_index[c]
            for c in ("debit", "credit", "amount", "balance")
            if c in column_index
        }

        classified: List[dict] = []
        for row in cluster_baselines(tokens, tolerance):
            diagnostics["rows_seen"] += 1
            joined = _row_text(row)
            if NOISE_ROW_RE.search(joined):
                continue
            stripped = joined.strip()
            if re.match(r"^\s*total\b", stripped, re.I):
                continue
            if SECTION_CREDIT_RE.fullmatch(stripped):
                current_section = "credit"
                continue
            if SECTION_DEBIT_RE.fullmatch(stripped) or (
                re.search(
                    r"\b(?:charges?|purchases?|transactions?|withdrawals?|deposits?|fees?)\b",
                    stripped,
                    re.I,
                )
                and not looks_like_amount(stripped)
                and not _matches_date(stripped)
            ):
                current_section = "debit"
                continue
            if _is_header_row(row):
                # Card layouts repeat column headers inside every section;
                # they never change the running polarity.
                continue
            lane_buckets: Dict[int, List[Token]] = {}
            for token in row:
                lane_buckets.setdefault(assignment[token], []).append(token)
            has_date = date_lane is not None and _matches_date(
                " ".join(t.text for t in lane_buckets.get(date_lane, []))
            )
            desc_tokens = [
                t for t in row if assignment[t] == desc_lane
            ]
            amount_tokens = [
                t for t in row if looks_like_amount(t.text) and assignment[t] in amount_lanes
            ]
            # Fallback admission only trusts debit/credit/amount lanes; a bare
            # number in a balance lane is a checkpoint or account header, not
            # a transaction amount.
            money_lanes = {
                column_index[c] for c in ("debit", "credit", "amount") if c in column_index
            }
            strict_amounts = [
                t for t in row if looks_like_amount(t.text) and assignment[t] in money_lanes
            ]
            if has_date or strict_amounts:
                # Rows whose date cell is OCR-damaged still count as
                # transactions when they carry a monetary amount in a money
                # lane; running 'total' lines never do, and neither do
                # letterless numeric clusters (analytics grids misread as
                # tables), rows with nothing in the description lane, nor
                # rows whose date lane holds no trace of a date at all
                # (account/segment headers).
                date_bucket = (
                    " ".join(t.text for t in lane_buckets.get(date_lane, []))
                    if date_lane is not None
                    else ""
                )
                date_hint = bool(
                    re.search(
                        r"\b\d{1,2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
                        date_bucket,
                        re.I,
                    )
                )
                if not has_date and (
                    not date_hint
                    or re.match(r"^\s*total\b", joined, re.I)
                    or not re.search(r"[A-Za-z]", joined)
                    or not desc_tokens
                ):
                    continue
                kind = "txn"
            elif desc_tokens:
                kind = "desc"
            else:
                continue
            classified.append(
                {
                    "kind": kind,
                    "y": sum(t.y_center for t in row) / len(row),
                    "lane_buckets": lane_buckets,
                    "desc_tokens": desc_tokens,
                    "amount_tokens": amount_tokens,
                    "section": current_section,
                }
            )

        txn_positions = [i for i, item in enumerate(classified) if item["kind"] == "txn"]
        pending_for_next: List[Token] = []

        def nearest_txn(index: int, forward: bool) -> Optional[dict]:
            positions = [p for p in txn_positions if (p > index if forward else p < index)]
            if not positions:
                return None
            return classified[min(positions) if forward else max(positions)]

        for index, item in enumerate(classified):
            if item["kind"] == "txn":
                if current is not None:
                    completed.append(current)
                current = TransactionRow()
                current.page = page
                current.polarity = item.get("section", "debit")
                for lane, bucket in item["lane_buckets"].items():
                    canonical = next(
                        (c for c, i in column_index.items() if i == lane), None
                    )
                    if canonical == "date":
                        for group in _group_date_tokens(bucket):
                            target = (
                                current.cells["date"]
                                if not current.cells["date"].tokens
                                else current.cells["post_date"]
                            )
                            for token in sorted(group, key=lambda t: t.x0):
                                target.add(token)
                        continue
                    cell = current.cell_for(canonical)
                    for token in sorted(bucket, key=lambda t: t.x0):
                        cell.add(token)
                if pending_for_next:
                    merged = Cell()
                    merged.tokens = pending_for_next + current.cells["description"].tokens
                    current.cells["description"] = merged
                    pending_for_next = []
                continue

            if item["kind"] == "amounts":
                if current is not None:
                    for token in item["amount_tokens"]:
                        lane = assignment[token]
                        canonical = next(
                            (c for c, i in column_index.items() if i == lane), None
                        )
                        current.cell_for(canonical).add(token)
                continue

            # Description-only baseline: attach to the vertically nearest
            # transaction row inside the attachment window (spec §4 multi-line
            # continuation), preferring the preceding parent row on ties.
            prev_item = nearest_txn(index, forward=False)
            next_item = nearest_txn(index, forward=True)
            dist_up = item["y"] - prev_item["y"] if prev_item else None
            dist_down = next_item["y"] - item["y"] if next_item else None

            attach_prev = (
                dist_up is not None
                and dist_up <= window
                and (dist_down is None or dist_down > window or dist_up <= dist_down)
            )
            attach_next = (
                dist_down is not None
                and dist_down <= window
                and not attach_prev
            )

            if attach_prev and current is not None:
                for token in item["desc_tokens"]:
                    current.cells["description"].add(token)
                diagnostics["continuations"] += 1
            elif attach_next:
                pending_for_next.extend(item["desc_tokens"])
                diagnostics["continuations"] += 1

    if current is not None:
        completed.append(current)

    for row in completed:
        rows_out.append(row.finalize())

    return {"transactions": rows_out, "diagnostics": diagnostics}
