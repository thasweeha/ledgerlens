"""Dynamic statement classification (spec §3).

Scores bank-account vs credit-card feature keywords over the extracted token
text to pick the target extraction schema, at document level with per-page
breakdown.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List

from pipeline.tokens import Token


class StatementType(str, Enum):
    BANK_ACCOUNT = "BANK_ACCOUNT"
    CREDIT_CARD = "CREDIT_CARD"


BANK_FEATURES: Dict[str, str] = {
    "opening_balance": r"(?:beginning|opening|starting)\s+balance",
    "deposits_credits": r"\bdeposits?\b|\bcredits?\b",
    "withdrawals_debits": r"\bwithdrawals?\b|\bdebits?\b",
    "checks_paid": r"\bchecks?\s+paid\b|\bcheques?\s+paid\b|\bcheque#?\b",
    "ending_balance": r"(?:ending|closing)\s+balance",
    "routing_number": r"\brouting\s+(?:number|no\.?|#)\b|\btransit\s+(?:number|no\.?)\b",
}

CREDIT_CARD_FEATURES: Dict[str, str] = {
    "previous_balance": r"previous\s+balance",
    "payments_and_credits": r"payments?\s*(?:&|and)\s*credits?",
    "purchases_charges": r"\bpurchases?\b|\bcharges?\b",
    "fees_charged": r"\bfees?\s+charged?\b",
    "interest_charged": r"\binterest\s+charged?\b",
    "new_balance": r"new\s+balance",
    "payment_due_date": r"payment\s+due(?:\s+date)?\b|\bdue\s+date\b",
}

STRONG_SIGNALS: Dict[StatementType, List[str]] = {
    StatementType.BANK_ACCOUNT: [
        r"\bchecking\b|\bchequing\b|\bsavings?\b",
        r"\brouting\s+number\b",
        r"\bbank\s+statement\b",
    ],
    StatementType.CREDIT_CARD: [
        r"\bcredit\s+card\b",
        r"\bcibc\b\s+dividend|\bvisa\b.*statement|\bmastercard\b",
        r"\bpurchases?\s*&\s*cash\s+advances\b",
        r"\bminimum\s+payment\b",
    ],
}

FEATURE_WEIGHTS: Dict[str, float] = {
    "opening_balance": 1.0,
    "deposits_credits": 0.5,
    "withdrawals_debits": 0.5,
    "checks_paid": 1.5,
    "ending_balance": 1.0,
    "routing_number": 2.0,
    "previous_balance": 1.5,
    "payments_and_credits": 1.0,
    "purchases_charges": 0.75,
    "fees_charged": 1.25,
    "interest_charged": 1.5,
    "new_balance": 1.5,
    "payment_due_date": 1.5,
}


def _page_text(tokens: List[Token]) -> str:
    return "\n".join(t.text for t in tokens)


def _score_features(text: str, features: Dict[str, str]) -> tuple:
    score = 0.0
    evidence = []
    for name, pattern in features.items():
        if re.search(pattern, text, re.IGNORECASE):
            score += FEATURE_WEIGHTS.get(name, 1.0)
            evidence.append(name)
    return score, evidence


def _score_strong(text: str, patterns: List[str]) -> float:
    return sum(2.5 for p in patterns if re.search(p, text, re.IGNORECASE))


def classify_statement(pages_tokens: Dict[int, List[Token]]) -> dict:
    """Classify the document; returns type, scores, and per-page evidence."""
    per_page = {}
    total_bank = 0.0
    total_cc = 0.0
    for page, tokens in sorted(pages_tokens.items()):
        text = _page_text(tokens)
        bank_score, bank_ev = _score_features(text, BANK_FEATURES)
        cc_score, cc_ev = _score_features(text, CREDIT_CARD_FEATURES)
        bank_score += _score_strong(text, STRONG_SIGNALS[StatementType.BANK_ACCOUNT])
        cc_score += _score_strong(text, STRONG_SIGNALS[StatementType.CREDIT_CARD])
        total_bank += bank_score
        total_cc += cc_score
        per_page[page] = {
            "bank_score": round(bank_score, 2),
            "credit_card_score": round(cc_score, 2),
            "bank_evidence": bank_ev,
            "credit_card_evidence": cc_ev,
        }

    if total_cc > total_bank:
        statement_type = StatementType.CREDIT_CARD
    else:
        statement_type = StatementType.BANK_ACCOUNT

    return {
        "statement_type": statement_type.value,
        "confidence": round(
            abs(total_bank - total_cc) / max(total_bank + total_cc, 1e-6), 4
        ),
        "document_scores": {
            "bank_account": round(total_bank, 2),
            "credit_card": round(total_cc, 2),
        },
        "per_page": per_page,
    }
