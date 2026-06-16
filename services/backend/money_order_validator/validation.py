from __future__ import annotations

import re
from typing import Any, Dict, Optional

from money_order_validator.parsers import ISSUERS, Issuer


IDENTITY_RULES_VERSION = "2026-06-09.1"


def digits(value: Any) -> str:
    return re.sub(r"\D", "", str(value or ""))


def is_aba_routing_number(value: Any) -> bool:
    value_digits = digits(value)
    if len(value_digits) != 9:
        return False
    weights = (3, 7, 1)
    return sum(int(char) * weights[index % 3] for index, char in enumerate(value_digits)) % 10 == 0


def identify_issuer(*values: Any) -> Optional[Issuer]:
    """First registry issuer whose alias appears in the joined text (alias sets
    are disjoint, so match order is immaterial)."""
    text = " ".join(str(value or "") for value in values).upper()
    return next((i for i in ISSUERS if i.aliases and any(a in text for a in i.aliases)), None)


def parse_micr_evidence(value: Any) -> Dict[str, Any]:
    """Parse MICR conservatively without assuming one bank-specific layout."""
    groups = [digits(group) for group in re.findall(r"\d+", str(value or ""))]
    groups = [group for group in groups if group]
    routing = next((group for group in groups if is_aba_routing_number(group)), None)
    non_routing = [group for group in groups if group != routing]
    short = [group for group in non_routing if 2 <= len(group.lstrip("0")) <= 6]
    check_number = (short[-1] if short else None)
    return {
        "raw": str(value or "") or None,
        "groups": groups,
        "routing_number": routing,
        "check_number": check_number.lstrip("0") if check_number else None,
        "account_candidates": [group for group in non_routing if group not in short],
        "rules_version": IDENTITY_RULES_VERSION,
    }


def serial_provenance(row: Dict[str, Any]) -> list[str]:
    sources: list[str] = []
    evidence = str(row.get("serial_evidence") or "").lower()
    if evidence:
        sources.append(evidence)
    for correction in row.get("corrections") or []:
        if correction.get("field") == "serial_number" and correction.get("source"):
            sources.append(str(correction["source"]))
    if row.get("serial_number") and not sources:
        sources.append("extracted_serial")
    return list(dict.fromkeys(sources))


def amount_provenance(row: Dict[str, Any]) -> list[str]:
    sources: list[str] = []
    if row.get("amount_numeric") is not None:
        sources.append(str(row.get("amount_evidence") or "numeric_amount"))
    if row.get("amount_words"):
        sources.append("written_amount")
    for correction in row.get("corrections") or []:
        if correction.get("field") == "amount_numeric" and correction.get("source"):
            sources.append(str(correction["source"]))
    return list(dict.fromkeys(sources))


def identity_match_status(row: Dict[str, Any]) -> str:
    flags = set(row.get("review_flags") or [])
    if row.get("match_status") == "conflicting" or "serial_micr_conflict" in flags:
        return "conflicting"
    if "missing_document_identity" in flags or not row.get("serial_number"):
        return "unclear"
    if any("serial" in flag and "conflict" in flag for flag in flags):
        return "conflicting"
    provenance = serial_provenance(row)
    return "confirmed" if any(source in {"both", "labeled_number", "check_micr_structure"} for source in provenance) else "unmatched"


from collections import Counter, defaultdict
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from money_order_validator.schemas import BatchContext, Instrument
from money_order_validator.parsers import build_property_aliases, similarity

# Derived from the issuer registry: every standard issuer's display name, plus
# None (unset issuer is not flagged as non-standard).
STANDARD_ISSUERS = {issuer.name for issuer in ISSUERS if issuer.standard} | {None}

# Flags that require human review regardless of score. Pure image-quality
# indicators (unclear_instrument_image, low_confidence_extraction) are NOT
# listed here — they are informational only. An instrument whose amount was
# independently verified by two LLM views should not be forced to REVIEW
# just because the underlying scan was hard to read.
_REVIEW_TRIGGER_PREFIXES = (
    "missing_serial",
    "duplicate_serial",
    "date_outside",
    "missing_issue_date",
    "payee_mismatch",
    "missing_payee",
    "manual_review_required",
    "non_standard_issuer",
    "amount_evidence_conflict",
    "unverified_amount",
    "missing_amount",
    "missing_physical",
    "high_value",
    "split_payment",
    "mobile_deposit_prohibited",
)


_PAYEE_TOKEN_RE = re.compile(r"[A-Za-z0-9]{5,}")


def _has_hard_flag(flags: List[str]) -> bool:
    return any(f.startswith(p) for f in flags for p in _REVIEW_TRIGGER_PREFIXES)


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def compute_ocr_confidence(raw: Dict[str, Any]) -> float:
    fields = [
        raw.get("serial_number"),
        raw.get("amount_numeric"),
        raw.get("issue_date"),
        raw.get("payee_raw"),
        raw.get("issuer") or raw.get("micr_line"),
    ]
    return round(sum(1 for x in fields if x not in (None, "", [])) / len(fields), 2)


def _compact_name(value: Optional[str]) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())




def apply_batch_reconciliation(batch: BatchContext, instruments: List[Instrument]) -> None:
    """Add batch-level amount/count reconciliation and adjust the final decision.

    A matched dollar total is necessary for ACCEPT, but not sufficient. If any
    item still has review/manual flags, the batch remains REVIEW. If totals or
    counts do not match, the batch becomes REJECT.
    """
    instrument_sum = round(sum(float(i.amount_numeric or 0.0) for i in instruments), 2)
    deposit_total = round(float(batch.batch_amount), 2) if batch.batch_amount is not None else None
    difference = None if deposit_total is None else round(instrument_sum - deposit_total, 2)
    amount_tolerance = 0.01
    amounts_match = bool(difference is not None and abs(difference) <= amount_tolerance)

    instrument_count = len(instruments)
    expected_count = int(batch.total_items) if batch.total_items is not None else None
    item_count_match = bool(expected_count is None or expected_count == instrument_count)

    item_statuses = {(inst.validation or {}).get("overall_status") for inst in instruments}
    has_invalid_items = "INVALID" in item_statuses
    has_review_items = "REVIEW" in item_statuses

    if not amounts_match or not item_count_match:
        decision = "FAIL"
        overall = "REJECT"
    elif has_invalid_items or has_review_items:
        decision = "PASS_WITH_REVIEW"
        overall = "REVIEW"
    else:
        decision = "PASS"
        overall = "ACCEPT"

    flags = []
    if amounts_match:
        flags.append("amounts_reconciled")
    else:
        flags.append("amount_mismatch")
    if item_count_match:
        flags.append("item_count_reconciled")
    else:
        flags.append("item_count_mismatch")
    if decision == "PASS_WITH_REVIEW":
        flags.append("manual_review_required_for_item_flags")

    batch.reconciliation = {
        "instrument_sum": instrument_sum,
        "deposit_total": deposit_total,
        "batch_amount": deposit_total,
        "difference": difference,
        "amounts_match": amounts_match,
        "instrument_count": instrument_count,
        "expected_item_count": expected_count,
        "item_count_match": item_count_match,
        "decision": decision,
        "flags": flags,
    }

    # Keep risk_summary aligned with reconciliation.
    batch.overall_decision = overall
    batch.risk_summary["overall_decision"] = overall
    batch.risk_summary["reconciliation_decision"] = decision
    batch.risk_summary["reconciliation_flags"] = flags


def _payee_match_score(payee: Optional[str], property_name: Optional[str], extra_aliases: Optional[List[str]] = None) -> float:
    if not payee or not property_name:
        return 0.0
    aliases = build_property_aliases(property_name)
    for alias in (extra_aliases or []):
        if alias and alias not in aliases:
            aliases.append(alias)
    scores = [similarity(payee, alias) for alias in aliases]
    payee_c = _compact_name(payee)
    for alias in aliases:
        alias_c = _compact_name(alias)
        if payee_c and alias_c and min(len(payee_c), len(alias_c)) >= 5:
            if payee_c in alias_c or alias_c in payee_c:
                scores.append(0.95)
        for token in _PAYEE_TOKEN_RE.findall(alias):
            token_score = similarity(payee, token)
            if token_score >= 0.80:
                scores.append(max(0.88, token_score))
    return round(max(scores or [0.0]), 3)


def validate_instruments(batch: BatchContext, instruments: List[Instrument]) -> None:
    serials = [i.serial_number for i in instruments if i.serial_number]
    serial_counts = Counter(serials)

    split_groups: Dict[tuple, List[Instrument]] = defaultdict(list)
    for inst in instruments:
        if inst.unit and inst.issue_date:
            split_groups[(inst.unit, inst.issue_date)].append(inst)

    items_valid = items_review = items_invalid = items_flagged = 0
    today = date.today()

    for inst in instruments:
        flags: List[str] = []
        score = 0.0

        if inst.missing_from_scan:
            flags.append("missing_physical_instrument - present in batch/register but no matching scan was extracted")
            score += 0.35

        if inst.image_quality == "unclear":
            flags.append("unclear_instrument_image")
            score += 0.18
            # For verified instruments two LLM views already agreed on the amount.
            # Flag the image quality informatively but don't require manual review.
            if inst.amount_status != "verified":
                flags.append("low_confidence_extraction")
                flags.append("manual_review_required")
        if "amount_evidence_conflict" in (inst.validation.get("flags", []) if inst.validation else []):
            flags.append("amount_evidence_conflict")

        if not inst.serial_number:
            flags.append("missing_serial_number")
            score += 0.20

        if inst.amount_numeric is None:
            flags.append("missing_amount")
            if inst.amount_candidate is not None:
                flags.append("unverified_amount_candidate")
            score += 0.30
        elif inst.amount_numeric >= 1000:
            flags.append(f"high_value - ${inst.amount_numeric:,.2f} at or above review threshold")
            score += 0.10

        if inst.serial_number and serial_counts[inst.serial_number] > 1 and not inst.missing_from_scan:
            flags.append("duplicate_serial_number")
            score += 0.35

        dt = _parse_iso_date(inst.issue_date)
        date_ok = True
        if dt:
            age = abs((today - dt).days)
            date_ok = age <= 120
            if not date_ok:
                flags.append("date_outside_120_days")
                score += 0.15
        else:
            flags.append("missing_issue_date")
            score += 0.12
            date_ok = False

        payee_match_score = _payee_match_score(inst.payee_raw, batch.property_name, list(batch.property_aliases or [])) if batch.property_name else 0.0
        if batch.property_name and inst.payee_raw and payee_match_score < 0.62:
            flags.append("payee_mismatch")
            score += 0.20
        elif batch.property_name and not inst.payee_raw and not inst.missing_from_scan:
            flags.append("missing_payee")
            score += 0.10

        if inst.issuer not in STANDARD_ISSUERS:
            flags.append("non_standard_issuer")
            score += 0.10

        if inst.mobile_deposit_prohibited:
            flags.append("mobile_deposit_prohibited - physical deposit required")

        grouped = split_groups.get((inst.unit, inst.issue_date), []) if inst.unit and inst.issue_date else []
        if len(grouped) >= 3:
            flags.append("split_payment_group - three or more instruments same unit/date")
            score += 0.05

        status = "VALID"
        if score >= 0.60 or inst.amount_numeric is None:
            status = "INVALID"
            items_invalid += 1
        elif score >= 0.25 or _has_hard_flag(flags):
            status = "REVIEW"
            items_review += 1
        else:
            items_valid += 1
        if _has_hard_flag(flags):
            items_flagged += 1

        inst.validation = {
            "overall_status": status,
            "risk_score": round(min(score, 1.0), 3),
            "payee_match_score": payee_match_score,
            "date_within_120_days": date_ok,
            "serial_duplicate": bool(inst.serial_number and serial_counts[inst.serial_number] > 1),
            "fraud_check": {"status": "PASS" if status != "INVALID" else "FAIL", "findings": flags},
            "flags": flags,
        }

    avg = round(sum(i.validation.get("risk_score", 0) for i in instruments) / max(1, len(instruments)), 3)
    overall = "ACCEPT"
    if items_invalid > 0:
        overall = "REJECT"
    elif items_review > 0 or items_flagged > 0:
        overall = "REVIEW"

    split_summary = []
    for (unit, issue_date), group in split_groups.items():
        if len(group) >= 3:
            split_summary.append(
                {
                    "unit": unit,
                    "date": issue_date,
                    "item_nos": [g.item_no for g in group],
                    "total": round(sum(g.amount_numeric or 0 for g in group), 2),
                    "note": f"{len(group)} instruments same unit/date - possible split payment",
                }
            )

    duplicate_serials = sorted([s for s, c in serial_counts.items() if c > 1])
    batch.risk_summary = {
        "average_risk_score": avg,
        "overall_decision": overall,
        "items_valid": items_valid,
        "items_review": items_review,
        "items_invalid": items_invalid,
        "items_flagged": items_flagged,
        "split_payment_groups": split_summary,
        "duplicate_serials": duplicate_serials,
    }
    batch.overall_decision = overall
