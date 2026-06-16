from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, List, Optional

from PIL import Image

from money_order_validator.schemas import TokenUsage
from money_order_validator.page_classifier import PageKind


@dataclass
class RegionEvidence:
    region_id: str
    page_number: int
    image: Image.Image
    bbox: tuple[float, float, float, float]
    source: str
    ocr_text: str = ""
    orientation: int = 0
    confidence: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        page_number: int,
        image: Image.Image,
        bbox: tuple[float, float, float, float],
        source: str,
        ocr_text: str = "",
        orientation: int = 0,
        confidence: float = 0.0,
    ) -> "RegionEvidence":
        signature = f"{page_number}:{source}:" + ":".join(f"{value:.4f}" for value in bbox)
        region_id = f"REG-{page_number:04d}-{hashlib.sha1(signature.encode('ascii')).hexdigest()[:12]}"
        return cls(region_id, page_number, image, bbox, source, ocr_text, orientation, confidence)


@dataclass
class PageEvidence:
    page_number: int
    image: Image.Image
    ocr_text: str
    angle: Optional[float]
    kind: PageKind
    scores: Dict[str, int]
    ocr_words: List[Any] = field(default_factory=list)
    spatial: Dict[str, Any] = field(default_factory=dict)
    regions: List[RegionEvidence] = field(default_factory=list)
    batch: Dict[str, Any] = field(default_factory=dict)
    deposit: Dict[str, Any] = field(default_factory=dict)
    register_items: List[Dict[str, Any]] = field(default_factory=list)
    instruments: List[Dict[str, Any]] = field(default_factory=list)
    log: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageStats:
    total: TokenUsage = field(default_factory=TokenUsage)
    calls: int = 0
    by_phase: Dict[str, int] = field(default_factory=dict)

    def add(self, phase: str, usage: TokenUsage, used: bool) -> None:
        if not used:
            return
        self.calls += 1
        self.total.merge(usage)
        self.by_phase[phase] = self.by_phase.get(phase, 0) + usage.total_tokens


@dataclass
class DocumentEvidence:
    file_name: str
    pages: List[PageEvidence]
    pdf_content: bytes = b""
    batch_data: Dict[str, Any] = field(default_factory=dict)
    deposit_data: Dict[str, Any] = field(default_factory=dict)
    deposit_candidates: List[Dict[str, Any]] = field(default_factory=list)
    register_items: List[Dict[str, Any]] = field(default_factory=list)
    instrument_candidates: List[Dict[str, Any]] = field(default_factory=list)
    usage: UsageStats = field(default_factory=UsageStats)


from collections import Counter
from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from money_order_validator.parsers import (
    normalize_payee,
    normalize_serial,
    parse_money,
)


REGISTER_SOURCES = {
    "regions_deposit_report",
    "transaction_detail_report",
    "vision_register_items",
    "yottareal_batch_detail",
}

AUTHORITATIVE_PLACEHOLDER_SOURCES = {
    "regions_deposit_report",
    "transaction_detail_report",
    "yottareal_batch_detail",
}

DEPOSIT_SOURCE_PRIORITY = {
    "yottareal_report": 100,
    "deposit_detail_report": 90,
    "bank_receipt": 80,
    "regions_report": 75,
    "deposit_ticket": 50,
    "handwritten_ocr": 30,
    "summary_page": 20,
    "metadata": 10,
}


@dataclass
class EvidenceResult:
    deposit_data: Dict[str, Any]
    deposits: List[Dict[str, Any]]
    instruments: List[Dict[str, Any]]
    expected_item_count: Optional[int]
    deposit_total: Optional[float]


def as_money(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    return parse_money(value)


def positive_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def deposit_amount(row: Dict[str, Any]) -> Optional[float]:
    return as_money(
        row.get("deposit_amount")
        or row.get("deposit_total")
        or row.get("check_total")
        or row.get("batch_amount")
    )


_DEP_YOTTA_RE = re.compile(r"\bBATCH\s+DETAILS\b|\bDEPOSIT\s+BATCH\s+DETAIL\s+REPORT\b")
_DEP_RECEIPT_AMOUNT_RE = re.compile(
    r"\b(?:COMMERCIAL\s+DEPOSIT(?:\s+(?:CHECKING|SAVINGS))?|"
    r"(?:CHECKING|SAVINGS)\s+DEPOSIT)\s*:?\s*\$?\s*[\d,]+\.\d{2}\b"
)
_DEP_RECEIPT_RE = re.compile(
    r"MY\s+TRANSACTION\s+SUMMARY|QUICKDEPOSIT\s+RECEIPT|TRANSACTION\s+RECEIPT|"
    r"TOTAL\s+CHECKS\s+AMOUNT"
)
_DEP_TICKET_RE = re.compile(r"DEPOSIT\s+TICKET|TOTAL\s+ITEMS|ADDITIONAL\s+CHECK\s+LISTING")


def classify_deposit_source(page_kind: str, text: str, row: Dict[str, Any]) -> str:
    explicit = str(row.get("source_type") or "").strip().lower()
    if explicit in DEPOSIT_SOURCE_PRIORITY:
        return explicit

    source = str(row.get("source_system") or row.get("bank_name") or "").upper()
    upper = (text or "").upper()
    if "YOTTA" in source or _DEP_YOTTA_RE.search(upper):
        return "yottareal_report"
    if "DEPOSIT DETAIL REPORT" in source or "DEPOSIT DETAIL REPORT" in upper:
        return "deposit_detail_report"
    receipt_amount = _DEP_RECEIPT_AMOUNT_RE.search(upper)
    if page_kind == "receipt" or receipt_amount or _DEP_RECEIPT_RE.search(upper):
        return "bank_receipt"
    if "REGIONS" in source or "DETAILS OF DEPOSITS BY ACCOUNT" in upper:
        return "regions_report"
    if page_kind in {"batch_header", "deposit_report"}:
        return "summary_page"
    if _DEP_TICKET_RE.search(upper):
        return "handwritten_ocr"
    if row.get("deposit_transaction"):
        return "bank_receipt"
    if deposit_amount(row) is not None or positive_int(row.get("item_count")):
        return "deposit_ticket"
    return "metadata"


def deposit_quality_score(row: Dict[str, Any]) -> float:
    source_type = str(row.get("source_type") or "metadata")
    score = float(DEPOSIT_SOURCE_PRIORITY.get(source_type, 0))
    if row.get("deposit_transaction"):
        score += 20
    if positive_int(row.get("item_count")):
        score += 15
    if deposit_amount(row) is not None:
        score += 15
    if row.get("deposit_account") or row.get("account_last4"):
        score += 8
    if row.get("deposit_date"):
        score += 5
    try:
        score += max(0.0, min(float(row.get("confidence") or 0.0), 1.0)) * 10
    except (TypeError, ValueError):
        pass
    return score


def _compact(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _account(row: Dict[str, Any]) -> str:
    values = re.findall(r"\d{4,}", str(row.get("deposit_account") or row.get("account_last4") or ""))
    return values[-1] if values else ""


def _page(row: Dict[str, Any]) -> Optional[int]:
    try:
        page = int(row.get("page_number"))
    except (TypeError, ValueError):
        return None
    return page if page > 0 else None


def _amounts_are_ocr_variants(left: Optional[float], right: Optional[float]) -> bool:
    if left is None or right is None:
        return True
    if abs(left - right) <= 0.01:
        return True
    high, low = max(left, right), min(left, right)
    if low <= 0:
        return False
    ratio = high / low
    return abs(ratio - 10.0) <= 0.25 or abs(high - low) <= max(1000.0, high * 0.10)


def _same_logical_deposit(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    left_tx = _compact(left.get("deposit_transaction"))
    right_tx = _compact(right.get("deposit_transaction"))
    # Regions Bank reports can produce multiple pages that describe the same
    # deposit with different internal reference codes. Don't hard-reject on
    # differing transaction IDs when both records share the same report format
    # and the same account — let account + date + amount settle identity.
    both_same_report = (
        left.get("source_type") == right.get("source_type")
        and left.get("source_type") == "regions_report"
    )
    if left_tx and right_tx and left_tx != right_tx:
        if not both_same_report:
            return False

    left_account, right_account = _account(left), _account(right)
    if left_account and right_account and left_account != right_account:
        return False
    if left_tx and right_tx and left_tx == right_tx:
        return True

    left_date, right_date = left.get("deposit_date"), right.get("deposit_date")
    if left_date and right_date and left_date != right_date:
        return False

    left_count = positive_int(left.get("item_count"))
    right_count = positive_int(right.get("item_count"))
    # For regions_report records, item_count reflects database row count in the
    # report view rather than physical instrument count, and varies between
    # report formats for the same deposit.
    if left_count and right_count and abs(left_count - right_count) > 1:
        if not both_same_report:
            return False

    left_amount, right_amount = deposit_amount(left), deposit_amount(right)
    if not _amounts_are_ocr_variants(left_amount, right_amount):
        return False

    exact_amount_count = (
        left_amount is not None
        and right_amount is not None
        and abs(left_amount - right_amount) <= 0.01
        and left_count is not None
        and left_count == right_count
    )
    stable_identity = bool(
        (left_tx and left_tx == right_tx)
        or (left_account and left_account == right_account and left_date and left_date == right_date)
        or (left_account and left_account == right_account and left_count and left_count == right_count)
        or (left_date and left_date == right_date and left_count and left_count == right_count)
    )
    left_page, right_page = _page(left), _page(right)
    nearby = bool(left_page and right_page and abs(left_page - right_page) <= 4)
    same_source_family = bool(
        _compact(left.get("source_system") or left.get("bank_name"))
        and _compact(left.get("source_system") or left.get("bank_name"))
        == _compact(right.get("source_system") or right.get("bank_name"))
    )
    exact_amount = bool(
        left_amount is not None
        and right_amount is not None
        and abs(left_amount - right_amount) <= 0.01
    )
    same_date = bool(left_date and left_date == right_date)
    duplicate_representation = bool(
        exact_amount
        and same_source_family
        and nearby
        and {
            str(left.get("source_type") or ""),
            str(right.get("source_type") or ""),
        }
        <= {"bank_receipt", "deposit_ticket", "handwritten_ocr", "summary_page"}
    )
    complementary_exact_amount = bool(
        exact_amount
        and {
            str(left.get("source_type") or ""),
            str(right.get("source_type") or ""),
        }
        in (
            {"bank_receipt", "deposit_ticket"},
            {"bank_receipt", "handwritten_ocr"},
        )
    )
    authority_variant = bool(
        same_source_family
        and left_count
        and left_count == right_count
        and _amounts_are_ocr_variants(left_amount, right_amount)
        and (
            "bank_receipt" in {left.get("source_type"), right.get("source_type")}
            or "summary_page" in {left.get("source_type"), right.get("source_type")}
        )
    )
    cross_control_pair = bool(
        "yottareal_report" in {left.get("source_type"), right.get("source_type")}
        and left_count
        and left_count == right_count
        and _amounts_are_ocr_variants(left_amount, right_amount)
    )
    return stable_identity or cross_control_pair or authority_variant or duplicate_representation or complementary_exact_amount or (exact_amount_count and nearby) or (exact_amount and same_source_family and same_date and nearby)


def _merge_missing(best: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(best)
    for key, value in other.items():
        if merged.get(key) in (None, "", [], {}) and value not in (None, "", [], {}):
            merged[key] = value
    pages = {
        page
        for page in (
            _page(best),
            _page(other),
            *(best.get("source_pages") or []),
            *(other.get("source_pages") or []),
        )
        if isinstance(page, int) and page > 0
    }
    if pages:
        merged["page_number"] = min(pages)
        merged["source_pages"] = sorted(pages)
    return merged


def normalize_deposit_controls(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    meaningful = [
        dict(row)
        for row in candidates
        if row and (deposit_amount(row) is not None or positive_int(row.get("item_count")))
    ]
    groups: List[List[Dict[str, Any]]] = []
    for row in meaningful:
        row_amount = deposit_amount(row)
        group = next(
            (
                candidate
                for candidate in groups
                if any(_same_logical_deposit(row, item) for item in candidate)
                and not any(
                    row.get("source_type") == item.get("source_type") == "bank_receipt"
                    and row_amount is not None
                    and (item_amount := deposit_amount(item)) is not None
                    and abs(row_amount - item_amount) > 0.01
                    for item in candidate
                )
                and (
                    row_amount is None
                    or all(
                        _amounts_are_ocr_variants(row_amount, amount)
                        for amount in (deposit_amount(item) for item in candidate)
                        if amount is not None
                    )
                )
            ),
            None,
        )
        if group is None:
            groups.append([row])
        else:
            group.append(row)

    logical: List[Dict[str, Any]] = []
    for group in groups:
        ordered = sorted(group, key=deposit_quality_score, reverse=True)
        best = dict(ordered[0])
        for duplicate in ordered[1:]:
            best = _merge_missing(best, duplicate)
        best["source_type"] = best.get("source_type") or "metadata"
        if len(group) > 1:
            best.setdefault("flags", []).append("duplicate_deposit_candidates_collapsed")
        logical.append(best)

    # A physical bank receipt and a weaker handwritten/ticket OCR candidate with
    # the same exact amount are alternate representations of one deposit, even
    # when OCR failed to recover matching account/date metadata.
    receipts_by_amount = {
        deposit_amount(row): row
        for row in logical
        if row.get("source_type") == "bank_receipt" and deposit_amount(row) is not None
    }
    deduped: List[Dict[str, Any]] = []
    for row in logical:
        amount = deposit_amount(row)
        receipt = receipts_by_amount.get(amount)
        if receipt is not None and row is not receipt and row.get("source_type") in {
            "deposit_ticket", "handwritten_ocr", "summary_page", "metadata"
        }:
            merged = _merge_missing(receipt, row)
            merged.setdefault("flags", []).append("same_amount_weaker_deposit_candidate_collapsed")
            receipts_by_amount[amount] = merged
            continue
        deduped.append(row)
    logical = [
        receipts_by_amount.get(deposit_amount(row), row)
        if row.get("source_type") == "bank_receipt"
        else row
        for row in deduped
    ]

    # Distinct bank receipt transaction IDs prove that the file contains
    # multiple physical deposits. Prefer those receipts because a YottaReal
    # report can describe only one of the deposits in the uploaded PDF.
    controls = [row for row in logical if row.get("source_type") == "yottareal_report"]
    if controls:
        control = max(controls, key=deposit_quality_score)
        components = [row for row in logical if row is not control and row.get("source_type") != "yottareal_report"]
        amount_components = [row for row in components if deposit_amount(row) is not None]
        component_amounts = [deposit_amount(row) for row in amount_components]
        control_amount = deposit_amount(control)
        if (
            len(component_amounts) >= 2
            and all(row.get("source_type") == "bank_receipt" for row in amount_components)
            and control_amount is not None
            and abs(sum(component_amounts) - control_amount) <= 0.01
        ):
            return sorted(amount_components, key=lambda row: _page(row) or 10**9)
        receipts = [
            row
            for row in components
            if row.get("source_type") == "bank_receipt"
            and deposit_amount(row) is not None
        ]
        if len(receipts) > 1:
            return sorted(receipts, key=lambda row: _page(row) or 10**9)

        # Otherwise, the YottaReal batch report is the aggregate accounting
        # control. Keep it once instead of adding component representations.
        if components:
            for component in components:
                control = _merge_missing(control, component)
            control.setdefault("flags", []).append("aggregate_control_superseded_component_deposits")
        return [control]
    return sorted(logical, key=lambda row: _page(row) or 10**9)


def aggregate_deposits(deposits: List[Dict[str, Any]], fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not deposits:
        return dict(fallback or {})

    out: Dict[str, Any] = {}
    from money_order_validator.parsers import _clean_property_name, valid_bank_name

    for row in sorted(deposits, key=deposit_quality_score, reverse=True):
        for key in (
            "bank_name",
            "source_system",
            "account_last4",
            "account_name",
            "deposit_account",
            "deposit_date",
            "deposit_transaction",
            "listed_amounts",
        ):
            if not out.get(key) and row.get(key) not in (None, "", [], {}):
                value = row[key]
                if key == "bank_name":
                    value = valid_bank_name(value)
                elif key == "account_name":
                    value = _clean_property_name(value)
                if value not in (None, "", [], {}):
                    out[key] = value

    amounts = [deposit_amount(row) for row in deposits]
    amounts = [amount for amount in amounts if amount is not None]
    if amounts:
        total = round(sum(amounts), 2)
        out.update(deposit_amount=total, deposit_total=total, check_total=total)

    counts = [positive_int(row.get("item_count")) for row in deposits]
    counts = [count for count in counts if count is not None]
    if counts:
        out["item_count"] = sum(counts)

    public_deposits = [dict(row) for row in deposits]
    for row in public_deposits:
        row["bank_name"] = valid_bank_name(row.get("bank_name"))
        row["account_name"] = _clean_property_name(row.get("account_name"))
        row["property_name"] = _clean_property_name(row.get("property_name"))
        row["property_aliases"] = [
            alias for alias in (row.get("property_aliases") or []) if _clean_property_name(alias)
        ]
        for key in ("bank_name", "account_name", "property_name"):
            if row.get(key) is None:
                row.pop(key, None)
    out["deposit_slip_count"] = len(public_deposits)
    out["deposit_slips"] = public_deposits
    return out


def mark_unlisted_visible_amounts(
    rows: List[Dict[str, Any]],
    listed_amounts: Any,
) -> List[Dict[str, Any]]:
    listed = Counter(as_money(value) for value in (listed_amounts or []))
    listed.pop(None, None)
    if not listed or sum(listed.values()) != len(rows):
        return rows
    output: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        amount = as_money(item.get("amount_numeric"))
        if amount is not None and listed[amount] > 0:
            listed[amount] -= 1
        elif item.get("page_number") is not None:
            item["amount_numeric"] = None
            item["image_quality"] = "unclear"
            item.setdefault("review_flags", []).extend(
                ["amount_not_in_deposit_listing", "manual_review_required"]
            )
        output.append(item)
    return output


def expected_item_count(
    *,
    batch_data: Dict[str, Any],
    deposit_data: Dict[str, Any],
    deposits: List[Dict[str, Any]],
    register_items: List[Dict[str, Any]],
) -> Optional[int]:
    counts = [positive_int(row.get("item_count")) for row in deposits]
    counts = [count for count in counts if count is not None]
    batch_count = positive_int(batch_data.get("item_count") or batch_data.get("total_items"))
    if len(deposits) > 1 and len(counts) < len(deposits):
        # A partial set of physical-receipt counts is not a document-level
        # expected count. Prefer a complete batch control or authoritative
        # register; otherwise leave count unresolved for visible-front census.
        if batch_count:
            return batch_count
        authoritative = [
            row
            for row in register_items
            if str(row.get("source") or "") in AUTHORITATIVE_PLACEHOLDER_SOURCES
        ]
        return len(authoritative) or None
    if counts:
        return sum(counts)
    for row in (deposit_data, batch_data):
        count = positive_int(row.get("item_count") or row.get("total_items"))
        if count:
            return count
    authoritative = [row for row in register_items if str(row.get("source") or "") in AUTHORITATIVE_PLACEHOLDER_SOURCES]
    return len(authoritative) or None


def authoritative_deposit_total(deposits: List[Dict[str, Any]], deposit_data: Dict[str, Any]) -> Optional[float]:
    amounts = [deposit_amount(row) for row in deposits]
    amounts = [amount for amount in amounts if amount is not None]
    return round(sum(amounts), 2) if amounts else deposit_amount(deposit_data)


def _identity_present(row: Dict[str, Any]) -> bool:
    return bool(
        normalize_serial(row.get("serial_number"))
        or row.get("micr_line")
        or normalize_payee(row.get("payee_raw"))
        or row.get("amount_words")
        or row.get("issue_date")
    )


def is_strong_visible_instrument(row: Dict[str, Any], total_pages: int) -> bool:
    if str(row.get("source") or "") in REGISTER_SOURCES or row.get("missing_from_scan"):
        return False
    page = _page(row)
    return bool(page and page <= total_pages and as_money(row.get("amount_numeric")) is not None and _identity_present(row))


def _instrument_quality(row: Dict[str, Any], total_pages: int) -> int:
    score = 0
    if is_strong_visible_instrument(row, total_pages):
        score += 50
    if normalize_serial(row.get("serial_number")):
        score += 12
    if row.get("micr_line"):
        score += 10
    if row.get("amount_words"):
        score += 10
    if normalize_payee(row.get("payee_raw")):
        score += 8
    if row.get("issue_date"):
        score += 5
    if str(row.get("image_quality") or "") == "unclear":
        score -= 5
    return score


def _instrument_match_keys(row: Dict[str, Any]) -> Tuple[set[str], Optional[Tuple[str, float]]]:
    serial_keys: set[str] = set()
    for value in (row.get("serial_number"), row.get("micr_line"), row.get("check_number")):
        serial = normalize_serial(value)
        digits = re.sub(r"\D", "", serial or "").lstrip("0")
        if digits:
            serial_keys.add(digits)
            if len(digits) > 4:
                serial_keys.add(digits[-10:])
    amount = as_money(row.get("amount_numeric"))
    unit_amount = (str(row["unit"]), amount) if row.get("unit") and amount is not None else None
    return serial_keys, unit_amount


def _placeholder(row: Dict[str, Any]) -> Dict[str, Any]:
    source = str(row.get("source") or "")
    report_row = source in {"transaction_detail_report", "regions_deposit_report"}
    out = dict(row)
    page = _page(out)
    if page:
        out.setdefault("report_page_number" if report_row else "slip_page_number", page)
    out["page_number"] = None
    out["matched_register_item"] = True
    out["llm_used"] = False
    out["processing_tier"] = 1
    out["missing_from_scan"] = not report_row
    out["image_quality"] = out.get("image_quality") or ("thumbnail_report_image" if report_row else "not_extracted_from_scan")
    identity, _ = _instrument_match_keys(out)
    if not report_row and not identity:
        # A ticket/register amount without a physical front or document number is
        # a reconciliation candidate, not a verified instrument amount.
        out["amount_status"] = "candidate"
        out["ocr_confidence"] = min(float(out.get("ocr_confidence") or 0.2), 0.2)
        out.setdefault("review_flags", []).extend(
            ["unverified_register_amount", "missing_document_identity", "manual_review_required"]
        )
    out.setdefault("review_flags", []).append(
        "report_row_instrument" if report_row else "register_item_not_matched_to_clear_instrument"
    )
    return out


def _dedupe_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[Tuple[Any, ...]] = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("source") or ""),
            _page(row),
            normalize_serial(row.get("serial_number")) or "",
            as_money(row.get("amount_numeric")),
            str(row.get("unit") or ""),
            row.get("item_no"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


def _subset_matching_total(
    rows: List[Dict[str, Any]],
    expected_count: int,
    target_total: Optional[float],
    total_pages: int,
) -> Optional[List[Dict[str, Any]]]:
    if target_total is None or expected_count <= 0 or len(rows) < expected_count:
        return None
    target = int(round(target_total * 100))
    if target < 0:
        return None
    candidates = [
        row
        for row in rows
        if (amount := as_money(row.get("amount_numeric"))) is not None
        and 0 <= int(round(amount * 100)) <= target
    ]
    if len(candidates) < expected_count:
        return None
    dp: Dict[Tuple[int, int], Tuple[int, Tuple[int, ...]]] = {(0, 0): (0, tuple())}
    for index, row in enumerate(candidates):
        amount = as_money(row.get("amount_numeric"))
        if amount is None:
            return None
        cents = int(round(amount * 100))
        for (count, total), (score, picks) in list(dp.items()):
            if count >= expected_count or total + cents > target:
                continue
            key = (count + 1, total + cents)
            candidate = (score + _instrument_quality(row, total_pages), picks + (index,))
            if key not in dp or candidate[0] > dp[key][0]:
                dp[key] = candidate
        if len(dp) > 250_000:
            return None
    match = dp.get((expected_count, target))
    return [candidates[index] for index in match[1]] if match else None


def _coherent_register(
    rows: List[Dict[str, Any]],
    expected_count: Optional[int],
    deposit_total: Optional[float],
) -> List[Dict[str, Any]]:
    if not expected_count or deposit_total is None:
        return rows
    groups: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
    for row in rows:
        source = str(row.get("source") or "")
        page = row.get("report_page_number") or row.get("slip_page_number")
        groups.setdefault((source, page), []).append(row)
    matches = [
        group
        for group in groups.values()
        if len(group) == expected_count
        and abs(sum(as_money(row.get("amount_numeric")) or 0.0 for row in group) - deposit_total) <= 0.01
    ]
    return max(matches, key=lambda group: sum(bool(_instrument_match_keys(row)[0]) for row in group)) if matches else rows


def select_public_instruments(
    rows: Iterable[Dict[str, Any]],
    register_items: Iterable[Dict[str, Any]],
    *,
    total_pages: int,
    expected_count: Optional[int],
    deposit_total: Optional[float],
    batch_number: Optional[str] = None,
) -> List[Dict[str, Any]]:
    candidates = _dedupe_rows([*rows, *register_items])
    visible: List[Dict[str, Any]] = []
    register: List[Dict[str, Any]] = []
    for row in candidates:
        page = _page(row)
        if page and page > total_pages:
            continue
        if is_strong_visible_instrument(row, total_pages):
            visible.append(row)
        elif str(row.get("source") or "") in AUTHORITATIVE_PLACEHOLDER_SOURCES and as_money(row.get("amount_numeric")) is not None:
            register.append(row)

    coherent_register = _coherent_register(register, expected_count, deposit_total)
    coherent_report = bool(
        coherent_register
        and all(
            str(row.get("source") or "") in {"transaction_detail_report", "regions_deposit_report", "yottareal_batch_detail"}
            for row in coherent_register
        )
        and expected_count
        and deposit_total is not None
        and len(coherent_register) == expected_count
        and abs(sum(as_money(row.get("amount_numeric")) or 0.0 for row in coherent_register) - deposit_total) <= 0.01
    )
    register = coherent_register
    visible_serials: set[str] = set()
    visible_unit_amounts: set[Tuple[str, float]] = set()
    for row in visible:
        serials, unit_amount = _instrument_match_keys(row)
        visible_serials.update(serials)
        if unit_amount:
            visible_unit_amounts.add(unit_amount)
    unmatched_register = []
    for row in register:
        serials, unit_amount = _instrument_match_keys(row)
        if serials & visible_serials or (unit_amount and unit_amount in visible_unit_amounts):
            continue
        unmatched_register.append(row)

    visible_amounts = Counter(as_money(row.get("amount_numeric")) for row in visible)
    register = []
    for row in unmatched_register:
        amount = as_money(row.get("amount_numeric"))
        if row.get("source") == "deposit_ticket_sequence" and amount is not None and visible_amounts[amount] > 0:
            visible_amounts[amount] -= 1
            continue
        register.append(row)
    visible.sort(key=lambda row: (_page(row) or 10**9, -_instrument_quality(row, total_pages)))

    selected = list(visible)
    if coherent_report and abs(
        sum(as_money(row.get("amount_numeric")) or 0.0 for row in visible) - float(deposit_total)
    ) > 0.01:
        selected = [_placeholder(row) for row in coherent_register]
    elif expected_count and len(visible) == expected_count:
        selected = _subset_matching_total(visible, expected_count, deposit_total, total_pages) or sorted(
            visible, key=lambda row: _instrument_quality(row, total_pages), reverse=True
        )[:expected_count]
    elif expected_count and len(visible) > expected_count:
        # Extra crop observations may be safely removed only when an exact-count
        # subset also exactly reconciles to the authoritative deposit total.
        selected = _subset_matching_total(visible, expected_count, deposit_total, total_pages) or visible
        if len(selected) > expected_count:
            for row in selected:
                row.setdefault("review_flags", []).extend(
                    ["visible_instruments_exceed_control_count", "manual_review_required"]
                )
    elif expected_count and len(visible) < expected_count:
        needed = expected_count - len(visible)
        register.sort(key=lambda row: (str(row.get("source") or ""), row.get("report_page_number") or row.get("slip_page_number") or 10**9, row.get("item_no") or 10**9))
        visible_total = round(sum(as_money(row.get("amount_numeric")) or 0.0 for row in visible), 2)
        remaining_total = round(deposit_total - visible_total, 2) if deposit_total is not None else None
        missing = _subset_matching_total(register, needed, remaining_total, total_pages) or register[:needed]
        selected.extend(_placeholder(row) for row in missing)
    elif not expected_count and deposit_total is not None and len(visible) > 1:
        visible_total = round(sum(as_money(row.get("amount_numeric")) or 0.0 for row in visible), 2)
        if abs(visible_total - deposit_total) > 0.01:
            for count in range(len(visible) - 1, 0, -1):
                reconciled = _subset_matching_total(visible, count, deposit_total, total_pages)
                if reconciled:
                    selected = reconciled
                    break
    elif not visible:
        selected = [_placeholder(row) for row in register]

    selected.sort(key=lambda row: (_page(row) or 10**9, row.get("item_no") or 10**9))
    for item_no, row in enumerate(selected, start=1):
        row["item_no"] = item_no
        row["instrument_id"] = f"INS-{batch_number or 'UNKNOWN'}-{item_no:03d}"
        if batch_number:
            row["batch_number"] = batch_number
    return selected


def finalize_evidence(
    *,
    batch_data: Dict[str, Any],
    deposit_data: Dict[str, Any],
    deposit_candidates: List[Dict[str, Any]],
    instrument_candidates: List[Dict[str, Any]],
    register_items: List[Dict[str, Any]],
    total_pages: int,
) -> EvidenceResult:
    batch_control = dict(batch_data)
    if batch_control.get("batch_amount") is not None:
        batch_control["deposit_amount"] = batch_control["batch_amount"]
    if batch_control.get("total_items") is not None:
        batch_control["item_count"] = batch_control["total_items"]
    source = str(batch_control.get("source_system") or "").upper()
    is_control_report = bool(
        batch_control.get("batch_number")
        or any(name in source for name in ("YOTTA", "REGIONS", "DEPOSIT DETAIL"))
    )
    if (batch_control.get("deposit_amount") is not None or batch_control.get("item_count") is not None) and (
        is_control_report or not deposit_candidates
    ):
        if "YOTTA" in source or (batch_control.get("batch_number") and not source):
            batch_control["source_type"] = "yottareal_report"
        elif "REGIONS" in source:
            batch_control["source_type"] = "regions_report"
        elif "DEPOSIT DETAIL" in source:
            batch_control["source_type"] = "deposit_detail_report"
        else:
            batch_control["source_type"] = "summary_page"
        deposit_candidates = [batch_control, *deposit_candidates]

    deposits = normalize_deposit_controls(deposit_candidates)
    aggregate = aggregate_deposits(deposits, deposit_data)
    count = expected_item_count(
        batch_data=batch_data,
        deposit_data=aggregate,
        deposits=deposits,
        register_items=register_items,
    )
    total = authoritative_deposit_total(deposits, aggregate)
    instruments = select_public_instruments(
        instrument_candidates,
        register_items,
        total_pages=total_pages,
        expected_count=count,
        deposit_total=total,
        batch_number=batch_data.get("batch_number"),
    )
    visible_amounts = [
        as_money(row.get("amount_numeric"))
        for row in instruments
        if row.get("page_number") is not None and as_money(row.get("amount_numeric")) is not None
    ]
    visible_total = round(sum(visible_amounts), 2)
    reconciling_receipts = [
        row
        for row in deposits
        if row.get("source_type") == "bank_receipt"
        and (amount := deposit_amount(row)) is not None
        and abs(amount - visible_total) <= 0.01
    ]
    if visible_amounts and len(reconciling_receipts) == 1 and (total is None or abs(total - visible_total) > 0.01):
        receipt = dict(reconciling_receipts[0])
        receipt.setdefault("flags", []).append("receipt_reconciled_visible_instruments")
        deposits = [receipt]
        aggregate = aggregate_deposits(deposits, receipt)
        total = visible_total
        count = len(instruments)
    instruments = mark_unlisted_visible_amounts(instruments, aggregate.get("listed_amounts"))
    return EvidenceResult(
        deposit_data=aggregate,
        deposits=deposits,
        instruments=instruments,
        expected_item_count=count,
        deposit_total=total,
    )


def finalize_document(document: DocumentEvidence) -> EvidenceResult:
    """finalize_evidence for a DocumentEvidence's current in-place state."""
    return finalize_evidence(
        batch_data=document.batch_data,
        deposit_data=document.deposit_data,
        deposit_candidates=document.deposit_candidates,
        instrument_candidates=document.instrument_candidates,
        register_items=document.register_items,
        total_pages=len(document.pages),
    )

