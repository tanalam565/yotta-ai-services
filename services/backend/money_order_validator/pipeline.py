from __future__ import annotations

from collections import Counter
from datetime import date
import re
import uuid
from typing import Any, Dict, Iterable, List, Optional

from money_order_validator.schemas import BatchContext, Instrument
from money_order_validator.evidence import DocumentEvidence
from money_order_validator.evidence import EvidenceResult, as_money
from money_order_validator.page_classifier import PageKind
from money_order_validator.imaging import pdf_renderer
from money_order_validator.parsers import (
    build_property_aliases,
    normalize_payee,
    normalize_serial,
    sanitize_instrument,
)
from money_order_validator.validation import compute_ocr_confidence
from money_order_validator.settings import settings


DEBUG_FIELDS = {
    "corrections",
    "_ocr_text",
    "_page_item_index",
    "orientation_degrees",
    "review_flags",
    "flags",
}


def clean_public(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: clean_public(item)
            for key, item in value.items()
            if key not in DEBUG_FIELDS
        }
    if isinstance(value, list):
        return [clean_public(item) for item in value]
    return value


def _account(value: Any) -> Optional[str]:
    matches = re.findall(r"\d{4,}", str(value or ""))
    return matches[-1] if matches else None


def _valid_bank(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip()
    if not text or not re.search(r"[A-Za-z]", text):
        return None
    if text[0] in ",.;:-" or re.fullmatch(r"(?:N\.?A\.?|BANK)\s*\d*", text, re.IGNORECASE):
        return None
    from money_order_validator.parsers import valid_bank_name

    return valid_bank_name(text)


def _bad_property(value: Optional[str]) -> bool:
    compact = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
    upper = str(value or "").upper()
    return (
        not compact
        or len(compact) > 80
        or bool(re.search(r"\b(SECURITY\s+FEATURES?|WATERMARK|HEAT\s+SENSITIVE|HOLOGRAM|SERVICE\s+CHARGE|ENDORSE)\b", upper))
        or compact in {
        "CENTS",
        "DOLLARS",
        "TOTAL",
        "CHASE",
        "JPMORGANCHASEBANK",
        "REGIONS",
        "DEPOSIT",
        "DEPOSITTICKET",
        }
        or sum(len(word) <= 3 for word in re.findall(r"[A-Za-z]+", str(value or "")))
        / max(1, len(re.findall(r"[A-Za-z]+", str(value or ""))))
        > 0.65
        or bool(re.search(r"[a-z][A-Z]|[A-Z][a-z][A-Z]", str(value or "")))
    )


def _infer_property(rows: Iterable[Dict[str, Any]]) -> Optional[str]:
    names: Counter[str] = Counter()
    for row in rows:
        payee = normalize_payee(row.get("payee_raw"))
        if not payee or _bad_property(payee):
            continue
        if re.search(r"\b(BANK|CHASE|REGIONS|WESTERN\s+UNION|MONEYGRAM|INTERMEX|FIDELITY)\b", payee, re.IGNORECASE):
            continue
        names[payee.title()] += 1
    return names.most_common(1)[0][0] if names else None


def build_batch(document: DocumentEvidence, evidence: EvidenceResult) -> BatchContext:
    batch_data = document.batch_data
    deposit = evidence.deposit_data
    batch_number = re.sub(r"\D", "", str(batch_data.get("batch_number") or "")) or None
    property_name = (
        batch_data.get("property_name")
        or deposit.get("account_name")
        or deposit.get("property_name")
    )
    bank_name = _valid_bank(batch_data.get("bank_name")) or _valid_bank(deposit.get("bank_name"))
    source_identity = str(batch_data.get("source_system") or deposit.get("source_system") or "")
    property_compact = re.sub(r"[^A-Z0-9]", "", str(property_name or "").upper())
    bank_compact = re.sub(r"[^A-Z0-9]", "", str(bank_name or "").upper()).removesuffix("BANK")
    source_compact = re.sub(r"[^A-Z0-9]", "", source_identity.upper()).removesuffix("BANK")
    if _bad_property(property_name) or (
        property_compact
        and property_compact in {bank_compact, source_compact}
    ):
        property_name = _infer_property(evidence.instruments)

    source_system = (
        batch_data.get("source_system")
        or deposit.get("source_system")
        or ("Deposit Detail Report" if any(page.log.get("deposit_detail_report") for page in document.pages) else None)
        or "Unknown"
    )
    deposited_date = (
        batch_data.get("deposited_date")
        or deposit.get("deposit_date")
        or batch_data.get("printed_on")
        or date.today().isoformat()
    )
    return BatchContext(
        batch_id=str(uuid.uuid4()),
        batch_number=batch_number,
        document_number=(
            batch_data.get("document_number")
            or deposit.get("document_number")
            or deposit.get("deposit_id")
        ),
        batch_type=batch_data.get("batch_type") or "Check/MO",
        batch_status=batch_data.get("batch_status"),
        pay_period=batch_data.get("pay_period"),
        bank_name=bank_name,
        account_number=_account(batch_data.get("account_number") or deposit.get("deposit_account")),
        property_name=property_name,
        property_aliases=[
            alias
            for alias in (batch_data.get("property_aliases") or build_property_aliases(property_name))
            if not _bad_property(alias)
        ],
        property_address=batch_data.get("property_address"),
        deposited_date=deposited_date,
        deposit_transaction=batch_data.get("deposit_transaction") or deposit.get("deposit_transaction"),
        total_items=evidence.expected_item_count,
        batch_amount=evidence.deposit_total,
        printed_on=batch_data.get("printed_on"),
        source_system=source_system,
    )


def build_instruments(batch: BatchContext, rows: List[Dict[str, Any]], file_name: str) -> List[Instrument]:
    from money_order_validator.validation import (
        IDENTITY_RULES_VERSION,
        amount_provenance,
        identify_issuer,
        identity_match_status,
        serial_provenance,
    )

    instruments: List[Instrument] = []
    for index, raw in enumerate(rows, start=1):
        row = sanitize_instrument(raw, ocr_text="")
        item_no = index
        instrument_type = row.get("instrument_type") or "MoneyOrder"
        description = row.get("payment_description") or (
            "Payment-Check" if instrument_type in {"Check", "CashiersCheck"} else "Payment-MoneyOrder"
        )
        payee = normalize_payee(row.get("payee_raw"))
        image_quality = row.get("image_quality")
        if row.get("review_flags") and not image_quality:
            image_quality = "unclear"
        issuer_rule = identify_issuer(row.get("issuer"), row.get("issuer_agent"))
        instruments.append(
            Instrument(
                item_no=item_no,
                instrument_id=f"INS-{batch.batch_number or 'UNKNOWN'}-{item_no:03d}",
                batch_number=batch.batch_number,
                unit=row.get("unit"),
                resident_name=row.get("resident_name") or row.get("payer_name"),
                instrument_type=instrument_type,
                payment_description=description,
                issuer=row.get("issuer"),
                issuer_id=issuer_rule.issuer_id if issuer_rule else None,
                issuer_agent=row.get("issuer_agent"),
                serial_number=normalize_serial(row.get("serial_number")),
                micr_line=row.get("micr_line"),
                issue_date=row.get("issue_date"),
                amount_numeric=as_money(row.get("amount_numeric")),
                amount_candidate=as_money(row.get("amount_candidate")),
                amount_status=str(row.get("amount_status") or (
                    "verified" if as_money(row.get("amount_numeric")) is not None
                    else "candidate" if as_money(row.get("amount_candidate")) is not None
                    else "conflict" if "amount_evidence_conflict" in (row.get("review_flags") or [])
                    else "missing"
                )),
                amount_words=row.get("amount_words"),
                payee_raw=payee,
                payee_normalized=payee.title() if payee else None,
                payer_name=row.get("payer_name"),
                payer_address=row.get("payer_address"),
                payer_signature=bool(row.get("payer_signature")),
                payment_for_acct=row.get("payment_for_acct"),
                mobile_deposit_prohibited=bool(row.get("mobile_deposit_prohibited")),
                watermark_present=bool(row.get("watermark_present")),
                posted_by=row.get("posted_by"),
                posted_date=row.get("posted_date"),
                ocr_confidence=compute_ocr_confidence(row),
                processing_tier=int(row.get("processing_tier") or 1),
                llm_used=bool(row.get("llm_used")),
                missing_from_scan=bool(row.get("missing_from_scan")),
                page_number=row.get("page_number"),
                source_file=file_name,
                image_quality=image_quality,
                serial_provenance=serial_provenance(row),
                amount_provenance=amount_provenance(row),
                match_status=identity_match_status(row),
                transaction_group_id=row.get("transaction_group_id"),
                evidence_rules_version=IDENTITY_RULES_VERSION,
            )
        )
    return instruments


def build_processing_stats(document: DocumentEvidence, instrument_count: int, deposit_count: int) -> Dict[str, Any]:
    kinds = Counter(page.kind.value for page in document.pages)
    return {
        "total_pages": len(document.pages),
        "ocr_pages": sum(bool(page.ocr_text.strip()) for page in document.pages),
        "page_kinds": dict(kinds),
        "vision_pages": sum(bool(page.log.get("llm_used")) for page in document.pages),
        "skipped_pages": sum(
            kinds.get(kind.value, 0)
            for kind in (
                PageKind.BACK,
                PageKind.BLANK,
                PageKind.RECEIPT,
                PageKind.DEPOSIT_REPORT,
                PageKind.DEPOSIT_SLIP,
                PageKind.BATCH_HEADER,
            )
        ),
        "register_items_extracted": len(document.register_items),
        "instruments_extracted": instrument_count,
        "detected_region_count": sum(len(page.regions) for page in document.pages),
        "confirmed_instrument_count": sum(
            1
            for page in document.pages
            for row in page.instruments
            if row.get("serial_number") and row.get("amount_numeric") is not None
        ),
        "deposit_slips_extracted": deposit_count,
        "llm_calls_total": document.usage.calls,
        "prompt_tokens": document.usage.total.prompt_tokens,
        "completion_tokens": document.usage.total.completion_tokens,
        "total_tokens": document.usage.total.total_tokens,
        "tokens_by_phase": dict(document.usage.by_phase),
        "token_budget": settings.max_total_tokens_per_document,
        "token_budget_exceeded": document.usage.total.total_tokens > settings.max_total_tokens_per_document,
        "recovery_limits": {
            "pages": settings.max_recovery_pages,
            "regions_per_page": settings.max_recovery_regions_per_page,
            "calls_per_document": settings.max_recovery_calls_per_document,
        },
        "segmentation_rollout_mode": settings.segmentation_rollout_mode,
        "recovery_gate_rejections": sum(int(page.log.get("recovery_gate_rejections") or 0) for page in document.pages),
        "rotation_census_regions": sum(int(page.log.get("rotation_census_regions") or 0) for page in document.pages),
        "forced_partition_regions": sum(int(page.log.get("forced_partition_regions") or 0) for page in document.pages),
        "region_render_cache": {
            "hits": pdf_renderer.region_cache_hits,
            "misses": pdf_renderer.region_cache_misses,
        },
        "token_saving_strategy": "two-pass page routing; ticket rows and amount rereads only on reconciliation gaps",
        "custom_vision": (
            "configured"
            if any(page.log.get("custom_vision") != "not_configured" for page in document.pages)
            else "not_configured"
        ),
        "custom_vision_regions": sum(int(page.log.get("custom_vision_regions") or 0) for page in document.pages),
        "automatic_multi_front_pages": [
            page.page_number
            for page in document.pages
            if page.log.get("automatic_multi_front_trigger")
        ],
        "topology_front_regions": sum(int(page.log.get("topology_check_regions") or 0) for page in document.pages),
        "per_page_trace": [
            {
                "pg": page.page_number,
                "kind": page.kind.value,
                "front_score": int((page.log.get("scores") or {}).get("front") or 0),
                "ocr_chars": page.log.get("ocr_chars", 0),
                "should_extract": page.log.get("should_extract"),
                "skip_reason": page.log.get("skip_reason"),
                "census": (page.log.get("front_census") or {}).get("classification"),
                "census_fronts": (page.log.get("front_census") or {}).get("visible_front_count"),
                "llm_rows_returned": page.log.get("llm_rows_returned"),
                "prepare_accepted": page.log.get("prepare_accepted"),
                "prepare_rejected": page.log.get("prepare_rejected"),
                "prepare_reject_reasons": page.log.get("prepare_reject_reasons"),
                "single_front_clamped": page.log.get("single_front_clamped"),
                "instruments": page.log.get("instruments", 0),
                "regions": len(page.regions),
            }
            for page in document.pages
        ],
    }


def public_pages(document: DocumentEvidence) -> Optional[List[Dict[str, Any]]]:
    if not settings.return_debug_pages:
        return None
    return clean_public([page.log for page in document.pages])


import asyncio
import logging
import math
from typing import List, Tuple

from money_order_validator.clients import adi_reader
from money_order_validator.schemas import ValidationResult
from money_order_validator.evidence import finalize_document, finalize_evidence
from money_order_validator.extraction import (
    page_evidence_collector,
    verify_amounts_if_needed,
)
from money_order_validator.validation import (
    apply_batch_reconciliation,
    validate_instruments,
)

logger = logging.getLogger(__name__)


def _instrument_total(rows: List[dict]) -> float:
    """Rounded sum of instrument amounts, treating missing/None as 0."""
    return round(sum(float(row.get("amount_numeric") or 0.0) for row in rows), 2)


class DocumentProcessor:
    """Coordinates the staged evidence pipeline.

    Stages:
      1. Render and OCR once.
      2. Collect page-scoped evidence.
      3. Normalize controls and select public instruments.
      4. Re-read only suspicious amounts when reconciliation requires it.
      5. Build and validate the public response.
    """

    async def process_batch(self, file_payloads: List[Tuple[str, bytes]]) -> List[ValidationResult]:
        if len(file_payloads) > settings.max_files_per_batch:
            raise ValueError(f"Too many files. Maximum is {settings.max_files_per_batch}.")
        return await asyncio.gather(
            *(self.process_file(file_name, content) for file_name, content in file_payloads)
        )

    async def process_file(self, file_name: str, content: bytes) -> ValidationResult:
        self._validate_file(file_name, content)
        logger.info("Processing %s (%d bytes)", file_name, len(content))

        images, ocr_pages = await asyncio.gather(
            pdf_renderer.render(content),
            adi_reader.analyze_pdf(content),
        )
        if not images:
            raise ValueError(f"PDF rendered zero pages: {file_name}")

        document = await page_evidence_collector.collect(file_name, images, ocr_pages)
        document.pdf_content = content
        evidence = finalize_document(document)
        visible_front_census, automatic_multi_front = page_evidence_collector.multi_front_census(document)
        # When the register/deposit slip provides an authoritative expected count, use it
        # rather than the inflated census count. False-positive multi_front census results
        # (back pages, single instruments with repeated text) inflate visible_front_census
        # and cause unnecessary multi-DPI recovery passes. Only fall back to census when
        # no authoritative count is available.
        recovery_expected_count = (
            int(evidence.expected_item_count)
            if evidence.expected_item_count
            else (int(visible_front_census) if visible_front_census else None)
        )
        visible_total = _instrument_total(evidence.instruments)
        reconciliation_gap = bool(
            evidence.expected_item_count
            and evidence.deposit_total is not None
            and (
                len(evidence.instruments) > evidence.expected_item_count
                or abs(visible_total - evidence.deposit_total) > 0.01
            )
        )
        # When no authoritative item count exists but the deposit total reveals a significant
        # financial gap, derive a floor count so recovery can still run.
        financial_gap = (
            round(evidence.deposit_total - visible_total, 2)
            if evidence.deposit_total is not None
            else 0.0
        )
        if not recovery_expected_count and financial_gap > 50.0:
            recovery_expected_count = len(evidence.instruments) + max(
                1, int(visible_front_census) if visible_front_census else 1
            )

        def _refine_expected(instruments, deposit_total, gap):
            """Estimate expected count from deposit gap and average instrument value."""
            if not instruments or gap <= 50.0:
                return len(instruments)
            avg_val = round(
                sum(float(r.get("amount_numeric") or 0) for r in instruments) / len(instruments), 2
            )
            avg_val = max(50.0, min(avg_val, 1500.0))
            return len(instruments) + min(math.ceil(gap / avg_val), 15)

        if (
            recovery_expected_count
            and len(evidence.instruments) < recovery_expected_count
            and await page_evidence_collector.recover_missing_instruments(
                document,
                recovery_expected_count,
                reconciliation_gap=reconciliation_gap or financial_gap > 50.0,
                automatic_multi_front=automatic_multi_front,
            )
        ):
            evidence = finalize_document(document)

        # For financial-gap batches (no authoritative item count), iterate recovery
        # until the gap closes, the token budget is strained, or no new instruments are found.
        if not evidence.expected_item_count and evidence.deposit_total is not None:
            for _iter in range(3):
                cur_total = _instrument_total(evidence.instruments)
                cur_gap = round(evidence.deposit_total - cur_total, 2)
                if cur_gap <= 50.0:
                    break
                # Stop iterating if we've already spent 40% of the per-document token budget
                # on recovery — further passes are unlikely to improve results.
                recovery_tokens = sum(
                    v for k, v in document.usage.by_phase.items() if "recovery" in k
                )
                if recovery_tokens > settings.max_total_tokens_per_document * 0.40:
                    logger.debug(
                        "Stopping iterative recovery: spent %d recovery tokens (>40%% budget)",
                        recovery_tokens,
                    )
                    break
                refined = _refine_expected(evidence.instruments, evidence.deposit_total, cur_gap)
                prev_count = len(evidence.instruments)
                if refined <= prev_count:
                    break
                ran = await page_evidence_collector.recover_missing_instruments(
                    document,
                    refined,
                    reconciliation_gap=True,
                    automatic_multi_front=automatic_multi_front,
                )
                if ran:
                    evidence = finalize_document(document)
                if len(evidence.instruments) <= prev_count:
                    break

        verified = await verify_amounts_if_needed(
            document,
            evidence.instruments,
            evidence.expected_item_count,
            evidence.deposit_total,
        )
        if verified != evidence.instruments:
            evidence = finalize_evidence(
                batch_data=document.batch_data,
                deposit_data=evidence.deposit_data,
                deposit_candidates=evidence.deposits,
                instrument_candidates=verified,
                register_items=document.register_items,
                total_pages=len(document.pages),
            )

        # When the final instrument sum is very close to — but not equal to — the
        # deposit total (small gap ≤ $20, count matches, instruments are all verified),
        # the deposit total itself may be the misread. Flag so reviewers inspect the slip.
        final_sum = _instrument_total(evidence.instruments)
        final_gap = abs(final_sum - evidence.deposit_total) if evidence.deposit_total is not None else 0.0
        # "Trusted" means no instrument has an unverified-candidate or None amount.
        amounts_trusted = all(
            r.get("amount_numeric") is not None and r.get("amount_status") != "candidate_promoted"
            for r in evidence.instruments
        )
        deposit_possibly_misread = bool(
            evidence.deposit_total is not None
            and 0 < final_gap <= 20.0
            and evidence.expected_item_count is not None
            and len(evidence.instruments) == evidence.expected_item_count
            and amounts_trusted
        )
        if deposit_possibly_misread and evidence.deposit_data:
            evidence.deposit_data.setdefault("review_flags", []).append("possible_deposit_misread")

        batch = build_batch(document, evidence)
        instruments = build_instruments(batch, evidence.instruments, file_name)
        validate_instruments(batch, instruments)
        apply_batch_reconciliation(batch, instruments)
        batch.processing_stats = build_processing_stats(
            document,
            instrument_count=len(instruments),
            deposit_count=len(evidence.deposits),
        )

        deposit_data = clean_public(evidence.deposit_data) if evidence.deposit_data else None
        deposits = clean_public(evidence.deposits) if evidence.deposits else None
        return ValidationResult(
            file_name=file_name,
            batch=batch,
            instruments=instruments,
            deposit_slip=deposit_data,
            deposit_slips=deposits,
            pages=public_pages(document),
        )

    @staticmethod
    def _validate_file(file_name: str, content: bytes) -> None:
        if not file_name.lower().endswith(".pdf"):
            raise ValueError(f"Only PDF files are supported: {file_name}")
        if len(content) > settings.max_file_size_mb * 1024 * 1024:
            raise ValueError(f"File exceeds {settings.max_file_size_mb} MB: {file_name}")


document_processor = DocumentProcessor()
