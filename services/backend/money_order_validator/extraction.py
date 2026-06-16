from __future__ import annotations

from typing import Any, Dict


NULLABLE_STRING = {"anyOf": [{"type": "string"}, {"type": "null"}]}
NULLABLE_NUMBER = {"anyOf": [{"type": "number"}, {"type": "null"}]}

INSTRUMENT_VISION_SCHEMA: Dict[str, Any] = {
    "name": "instrument_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "instruments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "instrument_type": {
                            "anyOf": [
                                {"type": "string", "enum": ["MoneyOrder", "Check", "CashiersCheck", "Escrow"]},
                                {"type": "null"},
                            ]
                        },
                        "payment_description": NULLABLE_STRING,
                        "issuer": NULLABLE_STRING,
                        "issuer_agent": NULLABLE_STRING,
                        "serial_number": NULLABLE_STRING,
                        "issue_date": NULLABLE_STRING,
                        "amount_numeric": NULLABLE_NUMBER,
                        "amount_words": NULLABLE_STRING,
                        "payee_raw": NULLABLE_STRING,
                        "unit": NULLABLE_STRING,
                        "payer_name": NULLABLE_STRING,
                        "payer_address": NULLABLE_STRING,
                        "payer_signature": {"type": "boolean"},
                        "payment_for_acct": NULLABLE_STRING,
                        "micr_line": NULLABLE_STRING,
                        "amount_evidence": {
                            "type": "string",
                            "enum": ["numeric_box", "amount_words", "both", "unclear"],
                        },
                        "serial_evidence": {
                            "type": "string",
                            "enum": ["labeled_number", "check_number_micr", "both", "unclear"],
                        },
                        "mobile_deposit_prohibited": {"type": "boolean"},
                        "watermark_present": {"type": "boolean"},
                    },
                    "required": [
                        "instrument_type", "payment_description", "issuer", "issuer_agent",
                        "serial_number", "issue_date", "amount_numeric", "amount_words",
                        "payee_raw", "unit", "payer_name", "payer_address", "payer_signature",
                        "payment_for_acct", "micr_line", "amount_evidence", "serial_evidence",
                        "mobile_deposit_prohibited", "watermark_present",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["instruments"],
        "additionalProperties": False,
    },
}


import asyncio
import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from money_order_validator.clients import llm_client
from money_order_validator.prompts import (
    BATCH_HEADER_PROMPT,
    DEPOSIT_DETAIL_REPORT_ITEMS_PROMPT,
    DEPOSIT_DETAIL_REPORT_ROW_CROP_PROMPT,
    DEPOSIT_SLIP_PROMPT,
    DEPOSIT_TICKET_ITEMS_PROMPT,
    INSTRUMENT_EXTRACTION_PROMPT,
    INSTRUMENT_RECOVERY_PROMPT,
    AMOUNT_VERIFICATION_PROMPT,
    IDENTITY_VERIFICATION_PROMPT,
    LLM_SYSTEM_MSG,
    REGISTER_ITEMS_PROMPT,
)
from money_order_validator.schemas import TokenUsage
from money_order_validator.imaging import (
    assess_image_quality,
    crop_to_content,
    enhance_for_verification,
    maybe_rotate_for_reading,
    overlapping_reading_regions,
    recovery_document_regions,
)
from money_order_validator.evidence import RegionEvidence
from money_order_validator.evidence import finalize_document
from money_order_validator.imaging import compact_ocr_context
from money_order_validator.page_classifier import PageKind
from money_order_validator.parsers import (
    parse_batch_header,
    parse_basic_instrument_from_ocr,
    parse_batch_line_items,
    parse_amount_from_words,
    parse_deposit_info,
    parse_money,
    parse_transaction_detail_items,
    sanitize_instrument,
    serial_from_deposit_detail_account,
)
from money_order_validator.settings import settings

logger = logging.getLogger(__name__)


def _deposit_detail_row_crops(image: Image.Image) -> List[Image.Image]:
    """Find row-level crops on Deposit Detail Report pages.

    The report uses thick black header bars above each transaction row. A full-page
    LLM call often misses one row because the table text is small. Cropping from
    one black header bar to the next gives the model a single row plus its
    thumbnail, which is much more reliable.
    """
    gray = image.convert("L")
    width, height = gray.size
    pix = gray.load()
    dark_rows: List[int] = []
    threshold = max(int(width * 0.42), 180)
    for y in range(height):
        dark = 0
        for x in range(0, width, 3):
            if pix[x, y] < 55:
                dark += 3
        if dark >= threshold:
            dark_rows.append(y)

    groups: List[tuple] = []
    if not dark_rows:
        return []
    start = prev = dark_rows[0]
    for y in dark_rows[1:]:
        if y - prev <= 4:
            prev = y
            continue
        if prev - start >= 4:
            groups.append((start, prev))
        start = prev = y
    if prev - start >= 4:
        groups.append((start, prev))

    bars = [(a, b) for a, b in groups if 0.10 * height <= a <= 0.92 * height and (b - a) >= 6]
    if not bars:
        return []

    crops: List[Image.Image] = []
    for i, (top, bottom) in enumerate(bars):
        next_top = bars[i + 1][0] if i + 1 < len(bars) else min(height, bottom + int(height * 0.26))
        y0 = max(0, top - 18)
        y1 = min(height, max(bottom + 90, next_top - 8))
        if y1 - y0 < 90:
            continue
        crop = image.crop((0, y0, width, y1))
        crops.append(crop)
    return crops[:6]


def _normalize_deposit_detail_row(row: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    item_type = str(row.get("item_type") or "")
    if bool(row.get("is_deposit_total")) or item_type.lower() == "credit":
        return None
    amount = row.get("amount_numeric")
    try:
        amount_f = round(float(str(amount).replace("$", "").replace(",", "")), 2)
    except (TypeError, ValueError):
        return None
    if amount_f <= 0 or amount_f > 5000:
        return None
    account = "".join(ch for ch in str(row.get("account_number") or "") if ch.isdigit()) or None
    check = "".join(ch for ch in str(row.get("check_number") or "") if ch.isdigit()) or None
    aux_raw = row.get("aux_serial") or row.get("serial_number") or check
    serial = str(aux_raw).strip() if aux_raw not in (None, "") else None
    if serial:
        serial = "".join(ch for ch in serial if ch.isalnum()) or None
    if serial and (serial == item_type or serial in {"0003", "0004", "003", "004"}):
        serial = None
    if not serial and account:
        serial = serial_from_deposit_detail_account(account)
    routing = "".join(ch for ch in str(row.get("routing_number") or "") if ch.isdigit()) or None
    return {
        "item_no": row.get("item_no") or idx,
        "routing_number": routing,
        "account_number": account,
        "check_number": check,
        "serial_number": serial,
        "amount_numeric": amount_f,
        "instrument_type": "MoneyOrder",
        "payment_description": "Payment-MoneyOrder",
        "source": "transaction_detail_report",
        "source_system": "Deposit Detail Report",
        "image_quality": "thumbnail_report_image",
        "review_flags": ["report_thumbnail_item", "manual_review_required"],
    }


async def _vision(user_prompt: str, image, *, max_width: int, max_completion_tokens: int):
    """Single vision-read contract: fixed system prompt + high detail; only the
    prompt, image, width, and token budget vary across callers."""
    return await llm_client.json_vision(
        system_prompt=LLM_SYSTEM_MSG,
        user_prompt=user_prompt,
        image=image,
        max_width=max_width,
        detail="high",
        max_completion_tokens=max_completion_tokens,
    )


class VisionExtractor:
    async def recover_instruments(
        self,
        image: Image.Image,
        page_angle: Optional[float] = None,
        max_regions: Optional[int] = None,
        evidence_regions: Optional[List[RegionEvidence]] = None,
    ) -> Tuple[List[Dict[str, Any]], TokenUsage, bool]:
        """Second-pass recovery for a page proven to be missing instruments."""
        if not llm_client.available:
            return [], TokenUsage(), False
        img = crop_to_content(maybe_rotate_for_reading(image, page_angle))
        region_limit = max_regions or settings.max_recovery_regions_per_page
        # OCR polygons are useful as a retry signal, but were unreliable as the
        # primary route: incomplete anchor groups can cut through instruments.
        # During reconciliation recovery, inspect their padded high-resolution
        # crops first while preserving the original computer-vision result.
        regions = [
            enhance_for_verification(region.image, target_width=settings.instrument_region_width)
            for region in (evidence_regions or [])[:region_limit]
        ]
        if len(regions) < region_limit:
            regions.extend(recovery_document_regions(img, max_regions=region_limit - len(regions)))
        if not regions:
            regions = overlapping_reading_regions(
                img,
                target_long_side_ratio=0.48,
                overlap_ratio=0.20,
                max_regions=region_limit,
            )
        usage_total = TokenUsage()
        observations: List[Dict[str, Any]] = []
        for region in regions:
            raw, usage = await _vision(INSTRUMENT_RECOVERY_PROMPT, region, max_width=settings.instrument_region_width, max_completion_tokens=2400)
            usage_total.merge(usage)
            rows = raw.get("instruments") if isinstance(raw, dict) else []
            if isinstance(rows, dict):
                rows = [rows]
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                item = sanitize_instrument(row, ocr_text="")
                identity = sum(
                    bool(item.get(key))
                    for key in ("serial_number", "micr_line", "amount_words", "payee_raw", "issue_date")
                )
                if item.get("amount_numeric") is not None and identity >= 1:
                    observations.append(item)
        return self._merge_instrument_observations(observations), usage_total, True

    async def extract_instruments(
        self,
        image: Image.Image,
        ocr_text: str,
        page_kind: PageKind,
        page_angle: Optional[float] = None,
        regions: Optional[List[RegionEvidence]] = None,
    ) -> Tuple[List[Dict[str, Any]], TokenUsage, bool]:
        """Return instruments, token usage, and whether LLM was used."""
        ocr_context = compact_ocr_context(ocr_text)
        fallback = parse_basic_instrument_from_ocr(ocr_text)

        # OCR-only fallback when LLM is unavailable or disabled.
        if not llm_client.available:
            return ([fallback] if fallback else []), TokenUsage(), False

        upper_ocr = ocr_text.upper()
        # Count only phrases that appear exactly once per instrument face.
        # "MONEY ORDER" can appear many times in a single instrument's fine print / header
        # and is NOT a reliable discriminator for multi-instrument pages.
        payment_face_count = max(
            upper_ocr.count("PAY EXACTLY"),
            upper_ocr.count("PAY TO THE ORDER"),
            upper_ocr.count("CASHIER'S CHECK"),
        )
        ocr_amount = fallback.get("amount_numeric") if fallback else None
        ocr_words_amount = parse_amount_from_words(fallback.get("amount_words")) if fallback else None
        amount_consistent = bool(
            ocr_amount is not None
            and (ocr_words_amount is None or abs(float(ocr_words_amount) - float(ocr_amount)) <= 0.01)
        )
        front_evidence = bool(
            re.search(
                r"\bPAY\s+EXACTLY\b|\bPAY\s+TO\s+THE\s+ORDER\b|\bCASHIER'?S\s+CHECK\b|"
                r"\bMONEY\s*ORDER\b|\bREMITTER\b|\bPURCHASER\b",
                upper_ocr,
            )
        )
        back_evidence = bool(
            re.search(
                r"\bENDORSE\s+ABOVE\b|\bFOR\s+DEPOSIT\s+ONLY\b|\bDEPOSITORY\s+BANK\b|"
                r"\bLOAD\s+THIS\s+DIRECTION\b|\bPURCHASER'?S\s+AGREEMENT\b",
                upper_ocr,
            )
        )
        trusted_single_ocr = bool(
            page_kind == PageKind.INSTRUMENT
            and payment_face_count <= 1
            and fallback
            and front_evidence
            and not back_evidence
            and amount_consistent
        )
        clear_single = bool(
            trusted_single_ocr
            and fallback.get("serial_number")
            and fallback.get("image_quality") != "unclear"
        )
        if clear_single:
            fallback["extraction_route"] = "ocr_clear_single"
            return [fallback], TokenUsage(), False

        if not settings.force_vision_for_instruments:
            if fallback and fallback.get("serial_number") and fallback.get("amount_numeric"):
                return [fallback], TokenUsage(), False

        prompt = INSTRUMENT_EXTRACTION_PROMPT.replace("{ocr_context}", ocr_context or "(no OCR text available)")
        region_prompt = INSTRUMENT_EXTRACTION_PROMPT.replace(
            "{ocr_context}",
            "(This is an isolated high-resolution page region. Use only the visible region; do not infer fields from other instruments.)",
        )
        width = settings.report_image_width if page_kind == PageKind.REPORT_WITH_INSTRUMENTS else settings.max_image_width
        # Normalize obvious upside-down/sideways pages before vision extraction. This is
        # critical for amount fidelity on inverted money orders: the model may otherwise
        # read a plausible-but-wrong value from the amount box/words.
        img = crop_to_content(maybe_rotate_for_reading(image, page_angle))
        total_usage = TokenUsage()
        observations: List[Dict[str, Any]] = []
        # Track whether the caller supplied proven multi-front region crops so
        # _consolidate_single_front is not applied to independently-isolated instruments.
        multi_front_regions = bool(regions)

        async def _read(region: Image.Image, max_width: int, user_prompt: str, max_tokens: int = 4000) -> None:
            raw, usage = await llm_client.json_vision(
                system_prompt=LLM_SYSTEM_MSG, user_prompt=user_prompt, image=region,
                max_width=max_width, detail="high", max_completion_tokens=max_tokens,
                schema=INSTRUMENT_VISION_SCHEMA,
            )
            total_usage.merge(usage)
            rows = raw.get("instruments") if isinstance(raw, dict) else None
            if isinstance(rows, dict):
                rows = [rows]
            if rows is None and isinstance(raw, dict) and any(raw.get(k) for k in ("serial_number", "amount_numeric", "payee_raw")):
                rows = [raw]
            for item in rows if isinstance(rows, list) else []:
                if not isinstance(item, dict):
                    continue
                sanitized = sanitize_instrument(item, ocr_text="")
                if (
                    sanitized.get("amount_numeric") is not None
                    and not sanitized.get("amount_words")
                    and str(sanitized.get("amount_evidence") or "").lower() not in {"numeric_box", "both"}
                ):
                    sanitized["amount_candidate"] = sanitized["amount_numeric"]
                    sanitized["amount_numeric"] = None
                    sanitized["amount_status"] = "candidate"
                    sanitized["image_quality"] = "unclear"
                    sanitized.setdefault("review_flags", []).extend(["weak_amount_evidence", "manual_review_required"])
                elif sanitized.get("amount_numeric") is not None:
                    sanitized["amount_status"] = "verified"
                if any(sanitized.get(k) for k in ("serial_number", "amount_numeric", "payee_raw", "issuer", "micr_line")):
                    observations.append(sanitized)

        # A page classified as one instrument must never be split merely because
        # the scan is portrait-shaped. Use one focused call when OCR is incomplete.
        if regions:
            for region in regions:
                before = len(observations)
                await _read(region.image, settings.primary_region_width, region_prompt,
                            max_tokens=settings.primary_extraction_max_tokens)
                for item in observations[before:]:
                    item["_region_id"] = region.region_id
                    item["_region_source"] = region.source
        elif page_kind == PageKind.INSTRUMENT and payment_face_count <= 1:
            await _read(img, width, prompt, max_tokens=settings.primary_extraction_max_tokens)
            if not observations:
                # A classified single front that produced no evidence gets one
                # bounded enhanced reread. This recovers clear but faint scans
                # without activating multi-instrument segmentation.
                await _read(
                    enhance_for_verification(img, target_width=settings.instrument_region_width),
                    settings.instrument_region_width,
                    prompt,
                )
        else:
            regions = overlapping_reading_regions(img, max_regions=settings.max_instrument_regions)
            # Dense/mixed pages are read as isolated overlapping regions so
            # neighboring instruments do not leak fields into each other.
            for region in regions:
                await _read(region, settings.primary_region_width, region_prompt,
                            max_tokens=settings.primary_extraction_max_tokens)

        instruments = self._merge_instrument_observations(observations)

        if not multi_front_regions and page_kind == PageKind.INSTRUMENT and len(instruments) > 1:
            # One-front pages occasionally produce multiple interpretations of the same
            # document. Consolidate when all confirmed amounts are identical (field-leakage
            # signal). Two or more distinct amount values mean genuinely different instruments.
            # Skip when the caller passed proven multi-front crops — those are distinct.
            _confirmed = []
            for _i in instruments:
                try:
                    _v = _i.get("amount_numeric")
                    if _v is not None:
                        _f = float(_v)
                        if _f > 0:
                            _confirmed.append(round(_f, 2))
                except (TypeError, ValueError):
                    pass
            if len(set(_confirmed)) <= 1:
                instruments = [self._consolidate_single_front(instruments)]

        # For a clear single front, deterministic OCR evidence is immutable.
        # Vision may fill missing fields, but must not replace an OCR-confirmed
        # amount or identity with a different plausible reading.
        ocr_patch = parse_basic_instrument_from_ocr(ocr_text)
        if ocr_patch and len(instruments) == 1:
            if trusted_single_ocr and ocr_patch.get("amount_numeric") is not None:
                instruments[0]["amount_numeric"] = ocr_patch["amount_numeric"]
                instruments[0]["amount_words"] = ocr_patch.get("amount_words") or instruments[0].get("amount_words")
                instruments[0]["amount_status"] = "verified"
                instruments[0]["amount_evidence"] = ocr_patch.get("amount_evidence") or "ocr_labeled_amount"
            if trusted_single_ocr and ocr_patch.get("serial_number"):
                instruments[0]["serial_number"] = ocr_patch["serial_number"]
                instruments[0]["serial_evidence"] = ocr_patch.get("serial_evidence") or "ocr_document_identity"
            for key in ("issuer", "issuer_agent", "serial_number", "amount_numeric", "issue_date", "micr_line"):
                if not instruments[0].get(key) and ocr_patch.get(key):
                    instruments[0][key] = ocr_patch[key]
        elif ocr_patch and not instruments:
            instruments.append(ocr_patch)

        return instruments, total_usage, True

    @staticmethod
    def _consolidate_single_front(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        def score(row: Dict[str, Any]) -> int:
            return (
                5 * bool(row.get("amount_words"))
                + 4 * bool(row.get("micr_line"))
                + 3 * bool(row.get("serial_number"))
                + 2 * bool(row.get("payee_raw"))
                + 2 * bool(row.get("issue_date"))
                + bool(row.get("amount_numeric") is not None)
            )

        best = dict(max(rows, key=score))
        amounts = {
            round(float(row["amount_numeric"]), 2)
            for row in rows
            if row.get("amount_numeric") is not None
        }
        if len(amounts) > 1:
            best["amount_candidate"] = best.get("amount_numeric")
            best["amount_numeric"] = None
            best["amount_status"] = "conflict"
            best.setdefault("review_flags", []).append("conflicting_single_front_observations")
        best["image_quality"] = "unclear"
        best.setdefault("review_flags", []).append("multiple_interpretations_single_front")
        return best

    @staticmethod
    def _merge_instrument_observations(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def _similar(left: str, right: str) -> bool:
            return bool(left and right and SequenceMatcher(None, left, right).ratio() >= 0.82)

        groups: List[List[Dict[str, Any]]] = []
        for row in rows:
            serial = "".join(ch for ch in str(row.get("serial_number") or "") if ch.isalnum()).upper()
            micr = "".join(ch for ch in str(row.get("micr_line") or "") if ch.isdigit())
            amount = row.get("amount_numeric")
            payee = "".join(ch for ch in str(row.get("payee_raw") or "").upper() if ch.isalnum())
            inst_type = str(row.get("instrument_type") or "")
            match = next(
                (
                    group
                    for group in groups
                    if (
                        micr
                        and any(
                            _similar(micr, "".join(ch for ch in str(item.get("micr_line") or "") if ch.isdigit()))
                            for item in group
                        )
                    )
                    or (
                        serial
                        and any(
                            _similar(serial, "".join(ch for ch in str(item.get("serial_number") or "") if ch.isalnum()).upper())
                            or serial in "".join(ch for ch in str(item.get("micr_line") or "") if ch.isalnum()).upper()
                            or "".join(ch for ch in str(item.get("serial_number") or "") if ch.isalnum()).upper() in micr
                            for item in group
                        )
                    )
                    or (
                        amount is not None
                        and payee
                        and any(
                            item.get("amount_numeric") == amount
                            and "".join(ch for ch in str(item.get("payee_raw") or "").upper() if ch.isalnum()) == payee
                            and str(item.get("instrument_type") or "") == inst_type
                            for item in group
                        )
                    )
                ),
                None,
            )
            if match is None:
                groups.append([row])
            else:
                match.append(row)

        merged: List[Dict[str, Any]] = []
        for group in groups:
            best = max(group, key=lambda row: sum(value not in (None, "", [], {}) for value in row.values()))
            item = dict(best)
            for row in group:
                for key, value in row.items():
                    if item.get(key) in (None, "", [], {}) and value not in (None, "", [], {}):
                        item[key] = value
            amounts = {round(float(row["amount_numeric"]), 2) for row in group if row.get("amount_numeric") is not None}
            if len(amounts) > 1:
                item["amount_numeric"] = None
                item["image_quality"] = "unclear"
                item.setdefault("review_flags", []).extend(["conflicting_region_amounts", "manual_review_required"])
            serials = {
                "".join(ch for ch in str(row.get("serial_number") or "") if ch.isalnum()).upper()
                for row in group
                if row.get("serial_number")
            }
            if len(serials) > 1:
                item["serial_number"] = None
                item["image_quality"] = "unclear"
                item.setdefault("review_flags", []).extend(["conflicting_region_document_numbers", "manual_review_required"])
            merged.append(item)
        return merged

    async def verify_instrument_amount(
        self,
        image: Image.Image,
        ocr_text: str,
        candidate: Dict[str, Any],
        page_angle: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], TokenUsage, bool]:
        """Focused second-pass amount verification for a single instrument page.

        This is intentionally amount-only.  It is used only when a batch already
        has the expected number of visible instruments but the visible sum does
        not match the deposit control total.  It gives the model a much simpler
        task than full extraction and prevents register math from silently
        overwriting a visible instrument amount.
        """
        if not llm_client.available:
            return {}, TokenUsage(), False

        candidate_context = {
            "page_number": candidate.get("page_number"),
            "serial_number": candidate.get("serial_number"),
            "issuer": candidate.get("issuer"),
            "current_amount_numeric": candidate.get("amount_numeric"),
            "current_amount_words": candidate.get("amount_words"),
            "payee_raw": candidate.get("payee_raw"),
        }
        prompt = AMOUNT_VERIFICATION_PROMPT.replace(
            "{candidate_context}", str(candidate_context)
        ).replace(
            "{ocr_context}", compact_ocr_context(ocr_text, max_chars=3000) or "(no OCR text available)"
        )
        rotated = maybe_rotate_for_reading(image, page_angle)
        img = crop_to_content(rotated)
        views = [img, enhance_for_verification(img, target_width=settings.instrument_region_width)]
        total = TokenUsage()
        observations: List[Dict[str, Any]] = []

        async def _verify_read(view: Image.Image) -> None:
            raw, usage = await _vision(prompt, view, max_width=settings.instrument_region_width, max_completion_tokens=500)
            total.merge(usage)
            if isinstance(raw, dict):
                observations.append(raw)

        for view in views:
            await _verify_read(view)

        def _majority_amount(obs: List[Dict[str, Any]], min_confidence: float = 0.72) -> Optional[float]:
            """Return the amount agreed on by a strict majority of readings."""
            vals = [
                parse_money(r.get("amount_numeric"))
                for r in obs
                if float(r.get("confidence") or 0.0) >= min_confidence
            ]
            vals = [v for v in vals if v is not None]
            if not vals:
                return None
            for candidate in vals:
                agreeing = [v for v in vals if abs(v - candidate) <= 0.01]
                if len(agreeing) * 2 > len(obs):
                    return candidate
            return None

        majority = _majority_amount(observations)
        # When two views disagree, take a 3rd tiebreaker using the uncropped image
        # with independent framing. Majority of 3 independent reads is trusted.
        if majority is None and len(observations) >= 2:
            await _verify_read(enhance_for_verification(rotated, target_width=settings.instrument_region_width))
            majority = _majority_amount(observations)

        best = max(observations, key=lambda row: float(row.get("confidence") or 0.0), default={})
        if majority is None:
            return {
                "amount_numeric": None,
                "confidence": 0.0,
                "evidence_source": "unclear",
                "verification_views_agree": False,
                "image_quality": assess_image_quality(img),
            }, total, True
        best = dict(best)
        best["amount_numeric"] = majority
        best["verification_views_agree"] = True
        best["image_quality"] = assess_image_quality(img)
        return best, total, True

    async def verify_instrument_identity(
        self,
        image: Image.Image,
        ocr_text: str,
        candidate: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], TokenUsage, bool]:
        """Resolve one conflicting identity from an already-isolated crop."""
        if not llm_client.available:
            return {}, TokenUsage(), False
        context = {
            "instrument_type": candidate.get("instrument_type"),
            "serial_candidate": candidate.get("serial_candidate") or candidate.get("serial_number"),
            "micr_check_number": candidate.get("micr_check_number"),
        }
        prompt = IDENTITY_VERIFICATION_PROMPT.replace(
            "{candidate_context}", str(context)
        ).replace(
            "{ocr_context}", compact_ocr_context(ocr_text, max_chars=2000) or "(no crop-local OCR)"
        )
        img = crop_to_content(image)
        # Check/document numbers normally live in the upper-right territory.
        # Excluding the lower MICR band prevents a neighboring or unrelated MICR
        # group from dominating this field-only verification.
        upper_right = img.crop((int(img.width * 0.48), 0, img.width, int(img.height * 0.48)))
        views = [
            upper_right,
            enhance_for_verification(upper_right, target_width=settings.instrument_region_width),
        ]
        total = TokenUsage()
        numbers: List[str] = []
        for view in views:
            raw, usage = await _vision(prompt, view, max_width=settings.instrument_region_width, max_completion_tokens=400)
            total.merge(usage)
            if isinstance(raw, dict) and float(raw.get("confidence") or 0.0) >= 0.78:
                number = "".join(ch for ch in str(raw.get("document_number") or "") if ch.isalnum())
                if number:
                    numbers.append(number)
        if len(numbers) == 2 and numbers[0] == numbers[1]:
            return {
                "serial_number": numbers[0],
                "serial_evidence": "labeled_number",
                "identity_verification_views_agree": True,
            }, total, True
        return {"identity_verification_views_agree": False}, total, True

    async def extract_batch_header(self, image: Image.Image, ocr_text: str) -> Tuple[Dict[str, Any], TokenUsage, bool]:
        parsed = parse_batch_header(ocr_text)
        enough = bool(parsed.get("batch_number") or parsed.get("batch_amount") or parsed.get("total_items"))
        if enough or not llm_client.available:
            return parsed, TokenUsage(), False
        prompt = BATCH_HEADER_PROMPT.replace("{ocr_context}", compact_ocr_context(ocr_text, max_chars=4000) or "(no OCR text available)")
        raw, usage = await _vision(prompt, crop_to_content(image), max_width=settings.report_image_width, max_completion_tokens=1200)
        merged = {**parsed}
        if isinstance(raw, dict):
            for k, v in raw.items():
                if v not in (None, "", []):
                    merged[k] = v
        return parse_batch_header("\n".join([ocr_text, str(merged)])) | merged, usage, True

    async def extract_deposit(self, image: Image.Image, ocr_text: str) -> Tuple[Dict[str, Any], TokenUsage, bool]:
        parsed = parse_deposit_info(ocr_text)
        enough = bool(parsed.get("deposit_amount") or parsed.get("deposit_total") or parsed.get("check_total"))
        if enough or not llm_client.available:
            return parsed, TokenUsage(), False
        prompt = DEPOSIT_SLIP_PROMPT.replace("{ocr_context}", compact_ocr_context(ocr_text, max_chars=4000) or "(no OCR text available)")

        async def _llm_read(img: Image.Image) -> Tuple[Dict[str, Any], TokenUsage]:
            raw, usage = await _vision(prompt, img, max_width=settings.report_image_width, max_completion_tokens=1200)
            return (raw if isinstance(raw, dict) else {}), usage

        # Two independent views: original crop and an enhanced/sharpened variant.
        # Running both concurrently keeps wall-clock flat. If they agree on the
        # deposit total we have consensus; if not, flag for manual review.
        (raw1, u1), (raw2, u2) = await asyncio.gather(
            _llm_read(crop_to_content(image)),
            _llm_read(enhance_for_verification(image, target_width=settings.report_image_width)),
        )
        total_usage = TokenUsage()
        total_usage.merge(u1)
        total_usage.merge(u2)

        def _primary_total(d: Dict[str, Any]) -> Optional[float]:
            for field in ("deposit_amount", "check_total", "credit_total"):
                v = d.get(field)
                if v is not None:
                    try:
                        return round(float(v), 2)
                    except (TypeError, ValueError):
                        pass
            return None

        t1, t2 = _primary_total(raw1), _primary_total(raw2)
        if t1 is not None and t2 is not None and abs(t1 - t2) <= 0.01:
            raw = raw1  # consensus
        elif t1 is not None and t2 is not None:
            # Disagreement — take a 3rd tiebreaker read (sequential, only on disagreement).
            raw3, u3 = await _llm_read(crop_to_content(image))
            total_usage.merge(u3)
            t3 = _primary_total(raw3)
            if t3 is not None and abs(t3 - t1) <= 0.01:
                raw = raw1  # t1 & t3 agree
            elif t3 is not None and abs(t3 - t2) <= 0.01:
                raw = raw2  # t2 & t3 agree
            else:
                raw = raw1  # all three differ — flag for review
                raw1.setdefault("review_flags", []).append("deposit_amount_uncertain")
        else:
            raw = raw1 if raw1 else raw2

        merged = {**parsed}
        for k, v in raw.items():
            if v not in (None, "", []):
                merged[k] = v
        return merged, total_usage, True

    async def extract_deposit_ticket_items(
        self,
        image: Image.Image,
        ocr_text: str,
        expected_count: Optional[int] = None,
        expected_total: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], TokenUsage, bool]:
        """Extract handwritten deposit-ticket row amounts.

        Chase deposit tickets are frequently scanned sideways or upside down. A
        single vision pass can read the table total but misread several line
        amounts. This method tries the native orientation first, then rotated
        copies when the row count/total does not look trustworthy, and selects
        the most internally consistent candidate.
        """
        if not llm_client.available:
            return [], TokenUsage(), False

        prompt = DEPOSIT_TICKET_ITEMS_PROMPT.replace(
            "{ocr_context}", compact_ocr_context(ocr_text, max_chars=3000) or "(no OCR text available)"
        )

        def _parse_items(raw: Dict[str, Any], orientation: int) -> List[Dict[str, Any]]:
            items_raw = raw.get("items") if isinstance(raw, dict) else None
            if not isinstance(items_raw, list):
                return []
            items: List[Dict[str, Any]] = []
            for idx, item in enumerate(items_raw, start=1):
                if not isinstance(item, dict):
                    continue
                amount = item.get("amount_numeric")
                try:
                    amount_f = round(float(str(amount).replace(",", "").replace("$", "")), 2)
                except (TypeError, ValueError):
                    continue
                if amount_f <= 0:
                    continue
                # Deposit-ticket row amounts are individual payment amounts. Drop obvious
                # copied totals or OCR garbage outside the normal MO/check range.
                if amount_f > 5000:
                    continue
                row = {
                    "item_no": item.get("item_no") or idx,
                    "unit": item.get("unit"),
                    "amount_numeric": amount_f,
                    "source": "deposit_ticket_sequence",
                    "payment_description": "Payment-MoneyOrder",
                    "instrument_type": "MoneyOrder",
                    "orientation_degrees": orientation,
                }
                items.append(row)
            return items

        def _candidate_score(items: List[Dict[str, Any]]) -> float:
            if not items:
                return -1_000_000.0
            row_sum = round(sum(float(i.get("amount_numeric") or 0.0) for i in items), 2)
            score = float(len(items)) * 100.0
            if expected_count:
                score -= abs(len(items) - int(expected_count)) * 500.0
                if len(items) == int(expected_count):
                    score += 1000.0
            # Use the slip total only as a weak signal. Handwritten totals are often OCR'd
            # with a missing leading digit, so a wrong expected_total must not force a bad
            # line-item candidate to win.
            if expected_total:
                diff = abs(row_sum - float(expected_total))
                if diff <= 1.0:
                    score += 400.0
                elif diff <= 250.0:
                    score += 75.0
                else:
                    score -= min(diff, 1000.0) / 25.0
            # If candidates have the same row count, prefer the larger positive sum. In these
            # tickets the common OCR failure is dropping a leading digit or reading 442 as 400,
            # not inventing extra dollars.
            score += row_sum / 1000.0
            return score

        total_usage = TokenUsage()
        candidates: List[Tuple[float, List[Dict[str, Any]]]] = []

        async def _run_for_orientation(degrees: int) -> None:
            img = image.rotate(degrees, expand=True) if degrees else image
            raw, usage = await _vision(prompt, crop_to_content(img), max_width=settings.report_image_width, max_completion_tokens=1800)
            total_usage.merge(usage)
            items = _parse_items(raw, degrees)
            candidates.append((_candidate_score(items), items))

        await _run_for_orientation(0)
        best_score, best_items = max(candidates, key=lambda c: c[0])
        best_sum = round(sum(float(i.get("amount_numeric") or 0.0) for i in best_items), 2)
        count_ok = bool(expected_count and len(best_items) == int(expected_count))
        total_ok = bool(expected_total and abs(best_sum - float(expected_total)) <= 1.0)

        # If native orientation does not match the ticket count/total, try rotated copies.
        # 180 degrees fixes upside-down deposit tickets; 90/270 cover sideways camera scans.
        if not (count_ok and (total_ok or not expected_total)):
            for degrees in (180, 90, 270):
                await _run_for_orientation(degrees)
            best_score, best_items = max(candidates, key=lambda c: c[0])

        return best_items, total_usage, True

    async def extract_deposit_detail_report_items(self, image: Image.Image, ocr_text: str) -> Tuple[List[Dict[str, Any]], TokenUsage, bool]:
        """Extract authoritative rows from Deposit Detail Report pages.

        Reads the printed report row table and ignores embedded thumbnail instrument
        images, which may be too small or duplicated. Also tries per-row crops for
        higher accuracy on pages with small or blurred table text.
        """
        parsed = parse_transaction_detail_items(ocr_text)
        header = parse_deposit_info(ocr_text)
        try:
            expected = int(header.get("debit_items") or 0)
            if not expected and header.get("report_item_count") is not None:
                expected = max(int(header["report_item_count"]) - int(header.get("credit_items") or 1), 0)
        except (TypeError, ValueError):
            expected = 0
        if parsed and expected and len(parsed) >= expected:
            return parsed, TokenUsage(), False
        if not llm_client.available:
            return parsed, TokenUsage(), False

        total_usage = TokenUsage()
        prompt = DEPOSIT_DETAIL_REPORT_ITEMS_PROMPT.replace(
            "{ocr_context}", compact_ocr_context(ocr_text, max_chars=4000) or "(no OCR text available)"
        )
        raw, usage = await _vision(prompt, crop_to_content(image), max_width=settings.report_image_width, max_completion_tokens=3000)
        total_usage.merge(usage)

        rows = raw.get("items") if isinstance(raw, dict) else None
        items: List[Dict[str, Any]] = []
        if isinstance(rows, list):
            for idx, row in enumerate(rows, start=1):
                if not isinstance(row, dict):
                    continue
                normalized = _normalize_deposit_detail_row(row, idx)
                if normalized:
                    items.append(normalized)

        # Row-crop fallback: read each black-header-delimited row crop separately.
        # More reliable than a single full-page call for small/blurred table text.
        crop_rows: List[Dict[str, Any]] = []
        for crop_idx, crop in enumerate(_deposit_detail_row_crops(image), start=1):
            crop_prompt = DEPOSIT_DETAIL_REPORT_ROW_CROP_PROMPT.replace(
                "{ocr_context}", compact_ocr_context(ocr_text, max_chars=1200) or "(no OCR text available)"
            )
            crop_raw, crop_usage = await _vision(crop_prompt, crop_to_content(crop), max_width=settings.report_image_width, max_completion_tokens=900)
            total_usage.merge(crop_usage)
            row_obj = crop_raw.get("item") if isinstance(crop_raw, dict) else None
            if isinstance(row_obj, dict):
                normalized = _normalize_deposit_detail_row(row_obj, crop_idx)
                if normalized:
                    crop_rows.append(normalized)

        # Merge OCR regex rows with full-page and row-crop vision rows. Prefer the
        # largest unique set; downstream reconciliation will use the control total.
        combined = parsed + items + crop_rows
        seen: set = set()
        unique: List[Dict[str, Any]] = []
        for item in combined:
            key = (
                str(item.get("serial_number") or item.get("account_number") or ""),
                round(float(item.get("amount_numeric") or 0.0), 2),
            )
            if key in seen or key == ("", 0.0):
                continue
            seen.add(key)
            unique.append(item)
        return unique, total_usage, True

    async def extract_register_items(self, image: Image.Image, ocr_text: str) -> Tuple[List[Dict[str, Any]], TokenUsage, bool]:
        """Extract bank/property register rows from a deposit report page.

        Focused fallback for when Azure OCR does not preserve table rows well enough for
        regex parsing. Intentionally separate from instrument vision.
        """
        parsed = parse_batch_line_items(ocr_text)
        if parsed or not llm_client.available:
            return parsed, TokenUsage(), False
        prompt = REGISTER_ITEMS_PROMPT.replace("{ocr_context}", compact_ocr_context(ocr_text, max_chars=4000) or "(no OCR text available)")
        raw, usage = await _vision(prompt, crop_to_content(image), max_width=settings.report_image_width, max_completion_tokens=2500)
        items_raw = raw.get("items") if isinstance(raw, dict) else None
        if not isinstance(items_raw, list):
            return [], usage, True
        items: List[Dict[str, Any]] = []
        for idx, item in enumerate(items_raw, start=1):
            if not isinstance(item, dict):
                continue
            amount = item.get("amount_numeric")
            serial = item.get("serial_number") or item.get("check_number")
            if amount in (None, "") or serial in (None, ""):
                continue
            inst_type = item.get("instrument_type") or ("MoneyOrder" if len(str(serial).lstrip("0")) >= 7 else "Check")
            if inst_type not in {"Check", "MoneyOrder", "CashiersCheck", "Escrow"}:
                inst_type = "MoneyOrder" if len(str(serial).lstrip("0")) >= 7 else "Check"
            row = {
                "item_no": item.get("item_no") or idx,
                "routing_number": item.get("routing_number"),
                "account_number": item.get("account_number"),
                "check_number": item.get("check_number") or serial,
                "serial_number": serial,
                "amount_numeric": amount,
                "instrument_type": inst_type,
                "payment_description": item.get("payment_description") or ("Payment-MoneyOrder" if inst_type == "MoneyOrder" else "Payment-Check"),
                "source": "vision_register_items",
            }
            items.append(row)
        return items, usage, True


vision_extractor = VisionExtractor()


from typing import Any, Dict, Iterable, List, Optional, Tuple

from money_order_validator.clients import OcrPage
from money_order_validator.clients import custom_vision_detector
from money_order_validator.evidence import DocumentEvidence, PageEvidence, RegionEvidence
from money_order_validator.evidence import (
    as_money,
    classify_deposit_source,
    expected_item_count,
    is_strong_visible_instrument,
)
from money_order_validator.imaging import (
    document_region_proposals,
    merge_region_proposals,
    ocr_anchor_instrument_regions,
    split_region_proposals,
)
from money_order_validator.page_classifier import PageKind, classify_page, spatial_page_signals
from money_order_validator.parsers import (
    is_deposit_detail_report,
    normalize_payee,
    normalize_serial,
    parse_amount_from_words,
    parse_batch_header,
    parse_batch_line_items,
    parse_deposit_info,
    sanitize_instrument,
    validate_graph_local_instrument,
)
from money_order_validator.imaging import pdf_renderer
from money_order_validator.imaging import (
    forced_territory_partition,
    rotation_sweep_census,
    segment_page,
    page_front_census,
    topology_check_regions,
    verify_recovered_instrument,
)


REPORT_KINDS = {
    PageKind.BATCH_HEADER,
    PageKind.DEPOSIT_REPORT,
    PageKind.DEPOSIT_SLIP,
    PageKind.RECEIPT,
    PageKind.REPORT_WITH_INSTRUMENTS,
}

SKIP_INSTRUMENT_KINDS = {
    PageKind.BACK,
    PageKind.BLANK,
    PageKind.BATCH_HEADER,
    PageKind.DEPOSIT_REPORT,
    PageKind.DEPOSIT_SLIP,
    PageKind.RECEIPT,
}


def merge_non_empty(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in (source or {}).items():
        if value not in (None, "", [], {}):
            target[key] = value


def _ocr_by_page(pages: List[OcrPage], total_pages: int) -> Tuple[List[str], List[Optional[float]], List[List[Any]]]:
    texts = [""] * total_pages
    angles: List[Optional[float]] = [None] * total_pages
    words: List[List[Any]] = [[] for _ in range(total_pages)]
    for page in pages:
        index = page.page_number - 1
        if 0 <= index < total_pages:
            texts[index] = page.text or ""
            angles[index] = page.angle
            words[index] = list(page.words or [])
    return texts, angles, words


def _unknown_may_be_instrument(text: str, front_score: int = 0) -> bool:
    if not settings.vision_on_unknown_pages:
        return False
    stripped = text.strip()
    if not stripped:
        return llm_client.available
    upper = stripped.upper()
    if re.search(
        r"DETAILS\s+OF\s+DEPOSITS\s+BY\s+ACCOUNT|DEPOSIT\s+DETAIL\s+REPORT|"
        r"CAPTURE\s+SEQ|TOTAL\s+OF\s+DEPOSITS\s+SUBMITTED",
        upper,
    ):
        return False
    # Sparse OCR with front-classifier evidence suggests a poorly-scanned instrument.
    # Require front_score >= 1 so truly blank/back pages don't waste LLM calls.
    if len(stripped) < 500 and front_score >= 1:
        return llm_client.available
    # A non-trivial front score is sufficient even when OCR captured enough text
    # to pass the sparse threshold but no explicit instrument keywords appear.
    if front_score >= 2:
        return llm_client.available
    return bool(
        re.search(
            r"\b(PAY\s+EXACTLY|PAY\s+TO\s+THE\s+ORDER|MONEY\s*ORDER|CASHIER|"
            r"WESTERN\s+UNION|MONEYGRAM|INTERMEX|BARRI|FIDELITY|PLS)\b|\$\s*\d",
            upper,
        )
    )


def _has_ticket_table(text: str, deposit: Dict[str, Any]) -> bool:
    upper = (text or "").upper()
    if re.search(r"MY\s+TRANSACTION\s+SUMMARY|QUICKDEPOSIT\s+RECEIPT", upper):
        return False
    return bool(
        re.search(
            r"DEPOSIT\s+TICKET|TOTAL\s+ITEMS|ADDITIONAL\s+CHECK\s+LISTING|"
            r"CURRENCY\s+AND\s+COIN|PLEASE\s+BE\s+SURE\s+ALL\s+ITEMS",
            upper,
        )
        and (deposit.get("item_count") or as_money(deposit.get("deposit_amount")))
    )


def _needs_register_vision(page: PageEvidence) -> bool:
    if page.register_items or page.kind not in REPORT_KINDS:
        return False
    return bool(
        re.search(
            r"CAPTURE\s+SEQ|CHECK\s+NUMBER|POST\s+AMOUNT|PAYMENT[-\s]?MONEYORDER|"
            r"PAYMENT[-\s]?CHECK|TRANSACTION\s+DETAIL\s+FOR\s+TRANSACTION",
            page.ocr_text,
            re.IGNORECASE,
        )
    )


def _page_may_contain_instruments(page: PageEvidence) -> bool:
    if page.kind in {PageKind.BLANK, PageKind.BACK, PageKind.BATCH_HEADER}:
        return False
    if page.kind in {PageKind.INSTRUMENT, PageKind.REPORT_WITH_INSTRUMENTS, PageKind.UNKNOWN}:
        return True
    # Mixed pages are common: a deposit ticket or receipt can share a scan page
    # with one or more instrument fronts. Classification describes the dominant
    # page content, not everything visible on it.
    return int(page.scores.get("front") or 0) >= 3


def _identity_count(row: Dict[str, Any]) -> int:
    fields = (
        normalize_serial(row.get("serial_number")),
        row.get("micr_line"),
        normalize_payee(row.get("payee_raw")),
        row.get("amount_words"),
    )
    return sum(value not in (None, "", []) for value in fields)


def _is_page_control_row(row: Dict[str, Any], page: PageEvidence) -> bool:
    amount = as_money(row.get("amount_numeric"))
    controls = {
        as_money(value)
        for value in (
            page.batch.get("batch_amount"),
            page.deposit.get("deposit_amount"),
            page.deposit.get("deposit_total"),
            page.deposit.get("check_total"),
        )
        if as_money(value) is not None
    }
    if amount not in controls:
        return False
    control_page = bool(
        page.kind in REPORT_KINDS
        or page.deposit
        or page.batch
        or int(page.scores.get("deposit") or 0) >= 5
        or int(page.scores.get("batch") or 0) >= 5
    )
    if not control_page:
        return False
    # A control page may contain nearby instrument images. Keep an amount equal
    # to the page aggregate only when the extracted object has independent,
    # instrument-local evidence; a plausible serial alone is insufficient.
    independent = sum(
        bool(value)
        for value in (
            row.get("micr_line"),
            normalize_payee(row.get("payee_raw")),
            row.get("amount_words"),
        )
    )
    return independent < 2


def _needs_deposit_vision(page: PageEvidence) -> bool:
    if any(page.deposit.get(key) for key in ("deposit_amount", "deposit_total", "check_total")):
        return False
    return bool(
        page.kind in {PageKind.DEPOSIT_REPORT, PageKind.DEPOSIT_SLIP, PageKind.RECEIPT}
        or page.deposit.get("source_type") not in (None, "", "metadata")
        or int(page.scores.get("deposit") or 0) >= 5
        or int(page.scores.get("receipt") or 0) >= 4
    )


def _prepare_instrument(
    row: Dict[str, Any], page: PageEvidence, file_name: str, used_llm: bool, index: int
) -> "tuple[Optional[Dict[str, Any]], str]":
    item = sanitize_instrument(row, ocr_text=str(row.get("_region_ocr_text") or page.ocr_text))
    item = validate_graph_local_instrument(item)
    if _is_page_control_row(item, page):
        return None, "control_row"
    if as_money(item.get("amount_numeric")) is None and _identity_count(item) == 0:
        return None, "no_amount_no_identity"
    if page.kind == PageKind.UNKNOWN:
        independent = sum(
            bool(value)
            for value in (
                item.get("micr_line"),
                normalize_serial(item.get("serial_number")),
                normalize_payee(item.get("payee_raw")),
                item.get("amount_words"),
            )
        )
        front = int(page.scores.get("front") or 0)
        # Accept when: ≥2 independent fields (strong), or ≥1 field with front_score≥2
        # (mirrors the _unknown_may_be_instrument attempt threshold).
        # Reject when identity is too sparse to distinguish a real instrument from noise.
        if independent < 2 and (front < 2 or independent < 1):
            return None, f"unknown_weak_identity(indep={independent},front={front})"
    if page.kind == PageKind.BACK and page.log.get("automatic_multi_front_trigger"):
        # Back pages forced into extraction via census override must have strong multi-field
        # identity. Endorsement-text false positives typically only have serial+payee (count=2)
        # from routing codes and deposit stamps. A genuinely misclassified front will have
        # amount_words + micr_line + serial/payee (count≥3). Require at least 3.
        if _identity_count(item) < 3:
            return None, f"back_page_forced_weak_identity(indep={_identity_count(item)})"
    item.update(
        page_number=page.page_number,
        source_file=file_name,
        llm_used=used_llm,
        processing_tier=3 if used_llm else 1,
        _page_item_index=index,
    )
    if _identity_count(item) < 2 or as_money(item.get("amount_numeric")) is None:
        item["image_quality"] = "unclear"
    return item, ""


def _dedupe(rows: Iterable[Dict[str, Any]], *, register: bool = False) -> List[Dict[str, Any]]:
    seen: set[Tuple[Any, ...]] = set()
    output: List[Dict[str, Any]] = []
    # Track best instrument per (serial, page) so same-page same-serial collisions keep
    # the reading with the highest identity score rather than both.
    serial_page_best: dict[Tuple[str, Any], int] = {}
    row_list = list(rows)
    for idx, row in enumerate(row_list):
        key = (
            str(row.get("source") or ""),
            str(row.get("_region_id") or ""),
            normalize_serial(row.get("serial_number") or row.get("check_number")) or "",
            as_money(row.get("amount_numeric")),
            row.get("item_no") if register else row.get("page_number"),
            row.get("slip_page_number"),
        )
        if key in seen:
            continue
        seen.add(key)
        # Collapse same serial + same page to single best reading.
        serial = normalize_serial(row.get("serial_number") or row.get("check_number")) or ""
        pg = row.get("item_no") if register else row.get("page_number")
        sp_key = (serial, pg)
        if serial and (prev_idx := serial_page_best.get(sp_key)) is not None:
            prev = output[prev_idx]
            if _identity_count(row) > _identity_count(prev):
                output[prev_idx] = row
                serial_page_best[sp_key] = prev_idx
            # Either way skip adding this row as a new entry
            continue
        serial_page_best[sp_key] = len(output)
        output.append(row)
    return output


class PageEvidenceCollector:
    async def collect(
        self,
        file_name: str,
        images: List[Any],
        ocr_pages: List[OcrPage],
    ) -> DocumentEvidence:
        texts, angles, words_by_page = _ocr_by_page(ocr_pages, len(images))
        pages = []
        for index, image in enumerate(images):
            text = texts[index]
            spatial = spatial_page_signals(words_by_page[index])
            kind, scores = classify_page(text, spatial)
            if not text.strip() and llm_client.available:
                kind = PageKind.UNKNOWN
                scores = {**scores, "no_ocr_vision_fallback": 1}
            pages.append(
                PageEvidence(
                    page_number=index + 1,
                    image=image,
                    ocr_text=text,
                    angle=angles[index],
                    kind=kind,
                    scores=scores,
                    ocr_words=words_by_page[index],
                    spatial=spatial,
                    log={
                        "page_number": index + 1,
                        "kind": kind.value,
                        "ocr_angle": angles[index],
                        "scores": scores,
                        "ocr_chars": len(text),
                        "ocr_words": len(words_by_page[index]),
                        "spatial": spatial,
                        "llm_used": False,
                        "instruments": 0,
                    },
                )
            )
            page = pages[-1]
            page.regions = merge_region_proposals(
                [
                    *document_region_proposals(
                        image,
                        page_number=page.page_number,
                        max_regions=settings.max_instrument_regions,
                    ),
                    *ocr_anchor_instrument_regions(
                        image,
                        page.ocr_words,
                        page_number=page.page_number,
                        max_regions=settings.max_instrument_regions,
                    ),
                ]
            )[: settings.max_instrument_regions]

        document = DocumentEvidence(file_name=file_name, pages=pages)
        # Stage 1 — Segmentation: determine instrument regions for every page
        # before any LLM extraction begins. This ensures extraction always
        # operates on pre-determined segments rather than discovering them
        # reactively after a full-page miss.
        self._segment_pages(document, images)
        await self._detect_custom_vision_regions(document)
        self._parse_text_evidence(document)
        await self._extract_primary_vision(document)
        self._roll_up(document)
        return document

    # ------------------------------------------------------------------
    # Segmentation stage
    # ------------------------------------------------------------------

    def _segment_pages(self, document: DocumentEvidence, images: List[Any]) -> None:
        """Stage 1 — run segmentation for every page before LLM extraction starts."""
        for page, image in zip(document.pages, images):
            self._segment_page(image, page)

    def _segment_page(self, image: Any, page: "PageEvidence") -> None:
        """Determine instrument regions for a single page using visual census and OCR phrase counts.

        Sets page.regions and page.log["automatic_multi_front_trigger"] when ≥2
        instruments are detected with sufficient confidence. This runs before any
        LLM extraction so the extractor always receives pre-determined segments.
        """
        census = page_front_census(
            image,
            page.ocr_words,
            page.page_number,
            page_kind=page.kind.value,
            front_score=int(page.scores.get("front") or 0),
            max_regions=settings.max_instrument_regions,
        )
        page.log["front_census"] = {
            "classification": census.classification,
            "visible_front_count": census.visible_front_count,
            "confidence": census.confidence,
            "signals": census.signals,
        }
        upper_text = page.ocr_text.upper()
        # Broad count: confirms a visual census multi-front call.
        # WU/HEB money orders print "PAY EXACTLY" in both the dollar-header and the
        # written-words line, so count = 2 for a single instrument — acceptable here
        # because we just need payment-phrase evidence to approve an existing census signal.
        ocr_face_count = max(
            upper_text.count("PAY EXACTLY"),
            upper_text.count("PAY TO THE ORDER"),
            upper_text.count("CASHIER'S CHECK"),
        )
        # Narrow count: drives the OCR-only multi-front trigger when visual census misses.
        # WU/HEB money orders print "PAY EXACTLY" TWICE per instrument (header + words line).
        # On a WU/HEB page, count = 2 means 1 instrument and odd counts like 3 mean 1.5
        # (partial read of 2 stacked MOs). The ceiling formula handles both: (3+1)//2 = 2.
        # For all other issuers "PAY EXACTLY" appears exactly once per instrument, so we
        # use the raw count directly.
        pay_exactly_raw = upper_text.count("PAY EXACTLY")
        is_wu_heb = "WESTERN UNION" in upper_text
        pay_exactly_instrument_count = (pay_exactly_raw + 1) // 2 if is_wu_heb else pay_exactly_raw
        ocr_instrument_count = max(
            pay_exactly_instrument_count,
            upper_text.count("PAY TO THE ORDER"),
            upper_text.count("CASHIER'S CHECK"),
        )
        page.log["ocr_face_count"] = ocr_face_count
        page.log["ocr_instrument_count"] = ocr_instrument_count

        if census.classification in {"multi_front", "report_with_fronts"} and census.visible_front_count >= 2:
            # Require OCR phrase count to confirm census for instrument/unknown pages.
            # Back pages may be misclassified fronts; OCR evidence is unreliable there
            # (payment phrases don't appear on backs), so honor visual census directly.
            if page.kind == PageKind.BACK or ocr_face_count >= 2:
                page.regions = census.regions
                page.log["automatic_multi_front_trigger"] = True
        elif ocr_instrument_count >= 2 and page.kind in {PageKind.INSTRUMENT, PageKind.UNKNOWN}:
            # Visual census found no strong regions but OCR confirms ≥2 instruments.
            # Partition into horizontal bands for isolated per-instrument extraction.
            bands = forced_territory_partition(
                image,
                page.page_number,
                ocr_instrument_count,
                max_regions=settings.max_instrument_regions,
            )
            if len(bands) >= 2:
                page.regions = bands
                page.log["automatic_multi_front_trigger"] = True
                page.log["ocr_only_multi_front_trigger"] = True

    async def _detect_custom_vision_regions(self, document: DocumentEvidence) -> None:
        if not custom_vision_detector.available:
            for page in document.pages:
                page.log["custom_vision"] = "not_configured"
            return
        detected = await asyncio.gather(
            *(custom_vision_detector.detect(page.image, page.page_number) for page in document.pages)
        )
        for page, regions in zip(document.pages, detected):
            fronts = [
                region for region in regions
                if region.source.endswith("instrument_front")
            ]
            page.log["custom_vision"] = "detected" if regions else "no_confident_boxes"
            page.log["custom_vision_regions"] = len(regions)
            page.log["custom_vision_fronts"] = len(fronts)
            if fronts:
                page.regions = merge_region_proposals(fronts)[: settings.max_instrument_regions]

    def _parse_text_evidence(self, document: DocumentEvidence) -> None:
        for page in document.pages:
            page.batch = parse_batch_header(page.ocr_text) if page.kind not in {PageKind.BACK, PageKind.BLANK, PageKind.INSTRUMENT} else {}
            if page.batch:
                page.batch["page_number"] = page.page_number
                if re.search(r"\bYOTTAREAL\b", page.ocr_text, re.IGNORECASE):
                    page.batch.setdefault("source_system", "YottaReal")
            page.deposit = parse_deposit_info(page.ocr_text)
            if page.deposit:
                page.deposit["page_number"] = page.page_number
                page.deposit["source_type"] = classify_deposit_source(page.kind.value, page.ocr_text, page.deposit)
                page.log["deposit_detected"] = True
            page.register_items = parse_batch_line_items(page.ocr_text)
            for row in page.register_items:
                row.setdefault("report_page_number", page.page_number)
            if is_deposit_detail_report(page.ocr_text):
                page.log["deposit_detail_report"] = True

    async def _extract_primary_vision(self, document: DocumentEvidence) -> None:
        await asyncio.gather(*(self._extract_page(document, page) for page in document.pages))

    async def _extract_page(self, document: DocumentEvidence, page: PageEvidence) -> None:
        detail_report = bool(page.log.get("deposit_detail_report"))

        if detail_report:
            rows, usage, used = await vision_extractor.extract_deposit_detail_report_items(page.image, page.ocr_text)
            document.usage.add("deposit_detail_rows", usage, used)
            if rows:
                page.register_items = rows
                for row in page.register_items:
                    row["report_page_number"] = page.page_number
            page.log["deposit_detail_report_items"] = len(page.register_items)
            page.log["llm_used"] = used
            return

        if page.kind == PageKind.BATCH_HEADER and not any(
            page.batch.get(key) for key in ("batch_number", "batch_amount", "total_items")
        ):
            patch, usage, used = await vision_extractor.extract_batch_header(page.image, page.ocr_text)
            document.usage.add("batch_header", usage, used)
            merge_non_empty(page.batch, patch)
            page.log["llm_used"] = page.log["llm_used"] or used

        if _needs_deposit_vision(page):
            patch, usage, used = await vision_extractor.extract_deposit(page.image, page.ocr_text)
            document.usage.add("deposit", usage, used)
            if patch:
                patch["page_number"] = page.page_number
                patch["source_type"] = classify_deposit_source(page.kind.value, page.ocr_text, patch)
                merge_non_empty(page.deposit, patch)
            page.log["llm_used"] = page.log["llm_used"] or used

        if _needs_register_vision(page):
            rows, usage, used = await vision_extractor.extract_register_items(page.image, page.ocr_text)
            document.usage.add("register", usage, used)
            if rows:
                page.register_items = rows
                for row in page.register_items:
                    row["report_page_number"] = page.page_number
            page.log["llm_used"] = page.log["llm_used"] or used

        should_extract = _page_may_contain_instruments(page)
        if page.kind == PageKind.UNKNOWN:
            should_extract = _unknown_may_be_instrument(page.ocr_text, int(page.scores.get("front") or 0))
        # Census-proven multi-front pages must be extracted regardless of classifier label.
        # The visual census is more reliable than OCR-derived page type when both disagree.
        if not should_extract and page.log.get("automatic_multi_front_trigger") and page.kind != PageKind.BLANK:
            should_extract = True
        page.log["should_extract"] = should_extract
        if not should_extract:
            page.log["skip_reason"] = (
                "unknown_keyword_miss" if page.kind == PageKind.UNKNOWN else "page_kind_excluded"
            )
            return

        rows, usage, used = await vision_extractor.extract_instruments(
            page.image,
            page.ocr_text,
            page.kind,
            page_angle=page.angle,
            regions=page.regions if page.log.get("automatic_multi_front_trigger") else None,
        )
        document.usage.add("instrument", usage, used)
        page.log["llm_used"] = bool(page.log.get("llm_used")) or used
        page.log["llm_rows_returned"] = len(rows)
        instruments_before = len(page.instruments)
        reject_reasons: List[str] = []
        for index, row in enumerate(rows, start=1):
            item, reject_reason = _prepare_instrument(row, page, document.file_name, used, index)
            if reject_reason:
                reject_reasons.append(reject_reason)
            if item:
                if (
                    item.get("amount_numeric") is None
                    and item.get("amount_candidate") is not None
                    and page.kind in {PageKind.INSTRUMENT, PageKind.UNKNOWN}
                ):
                    verified, verify_usage, verify_used = await vision_extractor.verify_instrument_amount(
                        page.image,
                        page.ocr_text,
                        item,
                        page_angle=page.angle,
                    )
                    document.usage.add("amount_verify", verify_usage, verify_used)
                    if verified.get("verification_views_agree") and as_money(verified.get("amount_numeric")) is not None:
                        item["amount_numeric"] = as_money(verified["amount_numeric"])
                        item["amount_words"] = verified.get("amount_words") or item.get("amount_words")
                        item["amount_status"] = "verified"
                        item["amount_evidence"] = verified.get("evidence_source") or "focused_amount_verification"
                        item["amount_candidate"] = None
                        item["review_flags"] = [
                            flag
                            for flag in (item.get("review_flags") or [])
                            if flag not in {"weak_amount_evidence", "manual_review_required"}
                        ]
                        if not item["review_flags"]:
                            item["image_quality"] = None
                    else:
                        # Verification inconclusive — promote candidate so the instrument
                        # is counted in reconciliation but flagged for manual review.
                        candidate_val = as_money(item.get("amount_candidate"))
                        if candidate_val is not None:
                            item["amount_numeric"] = candidate_val
                            item["amount_status"] = "candidate_promoted"
                            item["image_quality"] = "unclear"
                            flags = item.setdefault("review_flags", [])
                            if "unverified_amount" not in flags:
                                flags.append("unverified_amount")
                            if "manual_review_required" not in flags:
                                flags.append("manual_review_required")
                page.instruments.append(item)
        instruments_accepted = len(page.instruments) - instruments_before
        page.log["prepare_accepted"] = instruments_accepted
        page.log["prepare_rejected"] = len(rows) - instruments_accepted
        if reject_reasons:
            page.log["prepare_reject_reasons"] = reject_reasons
        if page.kind == PageKind.INSTRUMENT and not page.regions and len(page.instruments) > 1:
            # Clamp to one instrument when all confirmed amounts are identical — that is the
            # field-leakage fingerprint (one instrument read multiple times).
            # Two or more distinct non-zero amounts mean genuinely different instruments.
            _best_amounts = [
                as_money(i.get("amount_numeric")) if as_money(i.get("amount_numeric")) is not None
                else as_money(i.get("amount_candidate"))
                for i in page.instruments
            ]
            _confirmed_amounts = [a for a in _best_amounts if a is not None and a > 0]
            _clearly_distinct = len(set(_confirmed_amounts)) > 1
            if not _clearly_distinct:
                page.log["single_front_clamped"] = True
                page.instruments = [
                    max(
                        page.instruments,
                        key=lambda item: (
                            bool(item.get("amount_words")),
                            bool(item.get("micr_line")),
                            bool(normalize_serial(item.get("serial_number"))),
                            bool(normalize_payee(item.get("payee_raw"))),
                            _identity_count(item),
                        ),
                    )
                ]
                page.instruments[0].setdefault("review_flags", []).append(
                    "multiple_interpretations_single_front"
                )
                page.instruments[0]["image_quality"] = "unclear"
        page.log["instruments"] = len(page.instruments)
        page.log["instrument_candidates"] = [
            {
                "serial_number": row.get("serial_number"),
                "amount_numeric": row.get("amount_numeric"),
                "amount_candidate": row.get("amount_candidate"),
                "image_quality": row.get("image_quality"),
                "review_flags": row.get("review_flags"),
            }
            for row in page.instruments
        ]
        page.log["instrument_regions"] = len(page.regions)

    def _roll_up(self, document: DocumentEvidence) -> None:
        batch_candidates = [page.batch for page in document.pages if page.batch]
        document.batch_data = dict(
            max(
                batch_candidates,
                key=lambda row: (
                    bool(row.get("batch_number")),
                    int(row.get("total_items") or 0),
                    as_money(row.get("batch_amount")) or 0.0,
                    len(row),
                ),
            )
        ) if batch_candidates else {}
        document.deposit_data = {}
        document.deposit_candidates = []
        for page in document.pages:
            for key, value in page.batch.items():
                if document.batch_data.get(key) in (None, "", [], {}) and value not in (None, "", [], {}):
                    document.batch_data[key] = value
            merge_non_empty(document.deposit_data, page.deposit)
            if page.deposit:
                document.deposit_candidates.append(dict(page.deposit))
        document.register_items = _dedupe(
            (row for page in document.pages for row in page.register_items),
            register=True,
        )
        document.instrument_candidates = _dedupe(
            row for page in document.pages for row in page.instruments
        )

    async def recover_missing_instruments(
        self,
        document: DocumentEvidence,
        expected_count: Optional[int],
        *,
        reconciliation_gap: bool = False,
        automatic_multi_front: bool = False,
    ) -> bool:
        """Run finer vision crops when trusted controls prove evidence is incomplete."""
        rollout = str(settings.segmentation_rollout_mode or "assist").lower()
        if rollout == "off" or not expected_count or not llm_client.available:
            return False
        visible = sum(
            is_strong_visible_instrument(row, len(document.pages))
            for row in document.instrument_candidates
        )
        if visible >= expected_count and not reconciliation_gap:
            return False
        # Shadow mode: allow recovery when either:
        #   (a) a page already has a confirmed multi-front segmentation, or
        #   (b) the deposit count proves ≥2 instruments are missing — the
        #       segmentation stage ran for every page, so segment proposals
        #       are available to guide recovery even without an explicit trigger.
        count_proven_gap = (expected_count - visible) >= 2
        if rollout == "shadow" and not automatic_multi_front and not count_proven_gap and not reconciliation_gap:
            return False
        known_ids = {
            normalize_serial(value)
            for row in document.instrument_candidates
            for value in (row.get("serial_number"), row.get("micr_line"))
            if normalize_serial(value)
        }

        pages = [
            page
            for page in document.pages
            if page.kind == PageKind.INSTRUMENT
            or page.kind == PageKind.BACK
            or (
                page.kind == PageKind.UNKNOWN
                and int(page.scores.get("front") or 0) >= 3
            )
        ]
        if automatic_multi_front and rollout == "shadow":
            pages = [
                page
                for page in pages
                if page.log.get("automatic_multi_front_trigger")
            ]
        pages.sort(
            key=lambda page: (
                bool(page.instruments),
                -int(page.scores.get("front") or 0),
                page.page_number,
            )
        )
        changed = False
        recovery_calls = 0
        for page in pages[: settings.max_recovery_pages]:
            if (
                visible >= expected_count
                and not reconciliation_gap
            ) or recovery_calls >= settings.max_recovery_calls_per_document:
                break
            remaining_calls = settings.max_recovery_calls_per_document - recovery_calls
            region_limit = min(settings.max_recovery_regions_per_page, remaining_calls)
            existing = list(page.instruments)
            dpis = [
                int(value)
                for value in str(settings.recovery_render_dpis).split(",")
                if value.strip().isdigit()
            ] or [400]
            proposals = page.regions or document_region_proposals(
                page.image,
                page_number=page.page_number,
                max_regions=region_limit,
            )
            consensus = segment_page(
                page.image,
                page.ocr_words,
                page.page_number,
                max_regions=region_limit,
            )
            if consensus:
                page.log["segmentation_consensus_regions"] = len(consensus)
                if rollout == "shadow":
                    page.log["segmentation_shadow_only"] = True
                else:
                    proposals = merge_region_proposals([*consensus, *proposals])[:region_limit]
            topology = topology_check_regions(
                page.image,
                page.page_number,
                words=page.ocr_words,
                page_kind=page.kind.value,
                front_score=int(page.scores.get("front") or 0),
                max_regions=region_limit,
            )
            page.log["topology_check_regions"] = len(topology)
            strong_existing = sum(
                is_strong_visible_instrument(row, len(document.pages))
                for row in existing
            )
            # Segmentation recovery discovers missing fronts. A reconciliation
            # gap alone must not replace already-complete visible regions.
            proposed_fronts = max(len(topology), len(page.regions))
            if strong_existing and proposed_fronts <= strong_existing:
                page.log["recovery_status"] = "complete_visible_regions_preserved"
                continue
            # Census-confirmed multi-front pages have already proven multiple
            # instruments exist on this page. Enable advanced recovery features
            # (topology, rotation, forced partition) for those pages even in
            # shadow mode — the census is the authorization signal.
            page_confirmed_multi = bool(page.log.get("automatic_multi_front_trigger"))
            use_advanced = rollout != "shadow" or page_confirmed_multi

            topology_selected = bool(use_advanced and topology)
            if use_advanced and topology:
                # Complete, page-role-filtered topology crops take precedence
                # over broad component proposals during failure recovery.
                proposals = topology[:region_limit]
            missing_on_page = max(0, expected_count - visible)
            rotation_regions = rotation_sweep_census(
                page.image,
                page.page_number,
                max_regions=region_limit,
            )
            page.log["rotation_census_regions"] = len(rotation_regions)
            if use_advanced and rotation_regions and not topology_selected:
                proposals = merge_region_proposals([*rotation_regions, *proposals])[:region_limit]
            if use_advanced and not topology_selected and missing_on_page > 1 and len(proposals) < missing_on_page:
                forced = forced_territory_partition(
                    page.image,
                    page.page_number,
                    min(missing_on_page, region_limit),
                    max_regions=region_limit,
                )
                page.log["forced_partition_regions"] = len(forced)
                proposals = merge_region_proposals([*proposals, *forced], iou_threshold=0.68)[:region_limit]
            if not topology_selected:
                proposals = split_region_proposals(page.image, proposals, max_depth=1)[:region_limit]
            if not proposals:
                proposals = [
                    RegionEvidence.create(
                        page_number=page.page_number,
                        image=page.image,
                        bbox=(0.0, 0.0, 1.0, 1.0),
                        source="full_page_recovery",
                        confidence=0.2,
                    )
                ]
            for dpi in dpis:
                if recovery_calls >= settings.max_recovery_calls_per_document:
                    break
                rows: List[Dict[str, Any]] = []
                recovery_images: Dict[str, Any] = {}
                used = False
                for proposal in proposals[:region_limit]:
                    recovery_image = proposal.image
                    if document.pdf_content:
                        recovery_image = await pdf_renderer.render_region(
                            document.pdf_content,
                            page.page_number,
                            proposal.bbox,
                            dpi,
                            proposal.orientation,
                        )
                    recovery_images[proposal.region_id] = recovery_image
                    if proposal.source == "topology_check_band":
                        recovered, usage, region_used = await vision_extractor.extract_instruments(
                            recovery_image,
                            proposal.ocr_text,
                            PageKind.INSTRUMENT,
                            page_angle=None,
                        )
                    else:
                        recovered, usage, region_used = await vision_extractor.recover_instruments(
                            recovery_image,
                            page_angle=None,
                            max_regions=1,
                            evidence_regions=None,
                        )
                    recovery_calls += 1
                    document.usage.add(f"instrument_recovery_{dpi}dpi", usage, region_used)
                    used = used or region_used
                    for row in recovered:
                        row["_region_id"] = proposal.region_id
                        row["_region_bbox"] = proposal.bbox
                        row["_region_orientation"] = proposal.orientation
                        row["_region_confidence"] = proposal.confidence
                        row["_region_ocr_text"] = proposal.ocr_text
                        accepted, gate_flags = verify_recovered_instrument(row, proposal)
                        row["_recovery_gate"] = "accepted" if accepted else "rejected"
                        row.setdefault("review_flags", []).extend(gate_flags)
                        if settings.recovery_require_identity_and_amount and not accepted:
                            row["_reject_recovery"] = True
                    rows.extend(recovered)
                before_dpi = len(page.instruments)
                accepted_this_dpi: List[Dict[str, Any]] = []
                for index, row in enumerate(rows, start=len(page.instruments) + 1):
                    if row.get("_reject_recovery"):
                        page.log["recovery_gate_rejections"] = int(page.log.get("recovery_gate_rejections") or 0) + 1
                        continue
                    recovered_ids = {
                        normalize_serial(value)
                        for value in (row.get("serial_number"), row.get("micr_line"))
                        if normalize_serial(value)
                    }
                    if recovered_ids & known_ids:
                        continue
                    item, _ = _prepare_instrument(row, page, document.file_name, used, index)
                    if item:
                        if (
                            item.get("match_status") == "conflicting"
                            and row.get("_region_id") in recovery_images
                        ):
                            verified_identity, identity_usage, identity_used = (
                                await vision_extractor.verify_instrument_identity(
                                    recovery_images[row["_region_id"]],
                                    str(row.get("_region_ocr_text") or ""),
                                    item,
                                )
                            )
                            document.usage.add("identity_verify", identity_usage, identity_used)
                            if verified_identity.get("identity_verification_views_agree"):
                                item["serial_number"] = verified_identity["serial_number"]
                                item["serial_evidence"] = verified_identity["serial_evidence"]
                                item["match_status"] = "confirmed"
                                item["image_quality"] = None
                                item["review_flags"] = [
                                    flag
                                    for flag in (item.get("review_flags") or [])
                                    if flag not in {"serial_micr_conflict", "manual_review_required"}
                                ]
                        item["_recovery_dpi"] = dpi
                        page.instruments.append(item)
                        accepted_this_dpi.append(item)
                        known_ids.update(recovered_ids)
                        visible += 1
                if topology_selected and not existing and len(accepted_this_dpi) == len(proposals):
                    page.instruments = accepted_this_dpi
                page.instruments = _dedupe(page.instruments)
                # Stop at the lowest DPI that adds independently identified evidence.
                if len(page.instruments) > before_dpi:
                    page.log.setdefault("recovery_dpis", []).append(dpi)
                    break
            page.instruments = _dedupe(page.instruments)
            recovered = len(page.instruments) - len(existing)
            if recovered > 0:
                changed = True
                page.log["recovered_instruments"] = recovered
        if changed:
            self._roll_up(document)
        if visible < expected_count:
            for page in document.pages:
                page.log.setdefault("recovery_status", "budget_exhausted_or_unresolved")
        return changed

    def multi_front_census(self, document: DocumentEvidence) -> tuple[int, bool]:
        """Estimate visible fronts and prove whether any page is multi-front."""
        total = 0
        automatic_multi = False
        for page in document.pages:
            existing = sum(
                is_strong_visible_instrument(row, len(document.pages))
                for row in page.instruments
            )
            census = dict(page.log.get("front_census") or {})
            census_count = int(census.get("visible_front_count") or 0)
            fronts = max(existing, census_count)
            total += fronts
            if (
                census.get("classification") in {"multi_front", "report_with_fronts"}
                and census_count >= 2
                and census_count > existing
            ):
                automatic_multi = True
                page.log["automatic_multi_front_trigger"] = True
        return total, automatic_multi

    async def _extract_ticket_rows_if_needed(self, document: DocumentEvidence) -> None:
        preliminary = finalize_document(document)
        expected = preliminary.expected_item_count
        visible = sum(
            is_strong_visible_instrument(row, len(document.pages))
            for row in document.instrument_candidates
        )
        if not expected or visible >= expected:
            return

        seen_controls: set[Tuple[Optional[float], Optional[int], str]] = set()
        for page in document.pages:
            if not _has_ticket_table(page.ocr_text, page.deposit):
                continue
            count = expected_item_count(
                batch_data={},
                deposit_data=page.deposit,
                deposits=[page.deposit],
                register_items=[],
            )
            total = as_money(
                page.deposit.get("deposit_amount")
                or page.deposit.get("deposit_total")
                or page.deposit.get("check_total")
            )
            key = (total, count, str(page.deposit.get("deposit_transaction") or ""))
            if key in seen_controls:
                continue
            seen_controls.add(key)
            rows, usage, used = await vision_extractor.extract_deposit_ticket_items(
                page.image,
                page.ocr_text,
                expected_count=count,
                expected_total=total,
            )
            document.usage.add("deposit_items", usage, used)
            page.log["llm_used"] = page.log["llm_used"] or used
            valid: List[Dict[str, Any]] = []
            for index, row in enumerate(rows, start=1):
                item_no = row.get("item_no") or index
                try:
                    item_no = int(item_no)
                except (TypeError, ValueError):
                    continue
                if count and not 1 <= item_no <= count:
                    continue
                if as_money(row.get("amount_numeric")) is None:
                    continue
                row["item_no"] = item_no
                row["slip_page_number"] = page.page_number
                row["source"] = "deposit_ticket_sequence"
                valid.append(row)
            page.register_items.extend(valid)
            page.log["deposit_ticket_items"] = len(valid)


async def verify_amounts_if_needed(
    document: DocumentEvidence,
    rows: List[Dict[str, Any]],
    expected_count: Optional[int],
    deposit_total: Optional[float],
) -> List[Dict[str, Any]]:
    if deposit_total is None or not llm_client.available:
        return rows
    # When an authoritative item count is available and we're still short,
    # amount re-reads won't close the gap — instruments are missing, not misread.
    if expected_count and len(rows) < expected_count:
        return rows

    def total(items: List[Dict[str, Any]]) -> float:
        return round(sum(as_money(row.get("amount_numeric")) or 0.0 for row in items), 2)

    gap = round(deposit_total - total(rows), 2)
    if abs(gap) <= 0.01:
        return rows
    # For financial-gap batches (no authoritative expected_count), only attempt
    # re-verification when the gap is small enough that a misread amount is the
    # plausible explanation. Large gaps indicate missing instruments, not wrong reads.
    if not expected_count and abs(gap) > 150.0:
        return rows

    candidates = []
    for index, row in enumerate(rows):
        try:
            page_number = int(row.get("page_number") or 0)
        except (TypeError, ValueError):
            continue
        if not 1 <= page_number <= len(document.pages):
            continue
        words = parse_amount_from_words(row.get("amount_words"))
        amount = as_money(row.get("amount_numeric"))
        if (
            not row.get("amount_words")
            or (words is not None and amount is not None and abs(words - amount) > 0.01)
            or row.get("image_quality") == "unclear"
        ):
            candidates.append((index, page_number))

    output = [dict(row) for row in rows]
    for index, page_number in candidates[:4]:
        page = document.pages[page_number - 1]
        raw, usage, used = await vision_extractor.verify_instrument_amount(
            page.image,
            page.ocr_text,
            output[index],
            page_angle=page.angle,
        )
        document.usage.add("amount_verify", usage, used)
        try:
            confidence = float(raw.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        amount = as_money(raw.get("amount_numeric"))
        if amount is None or confidence < 0.72:
            continue
        if not raw.get("verification_views_agree"):
            continue
        if "amount_evidence_conflict" in (output[index].get("review_flags") or []) and raw.get("evidence_source") != "both":
            continue
        candidate = [dict(row) for row in output]
        candidate[index]["amount_numeric"] = amount
        if raw.get("amount_words"):
            candidate[index]["amount_words"] = raw["amount_words"]
        if abs(deposit_total - total(candidate)) < abs(deposit_total - total(output)):
            output = candidate
            if abs(deposit_total - total(output)) <= 0.01:
                break
    return output


page_evidence_collector = PageEvidenceCollector()
