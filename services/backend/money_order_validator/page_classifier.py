from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, Iterable, Tuple


class PageKind(str, Enum):
    BLANK = "blank"
    BACK = "back_page"
    BATCH_HEADER = "batch_header"
    DEPOSIT_REPORT = "deposit_report"
    DEPOSIT_SLIP = "deposit_slip"
    RECEIPT = "receipt"
    INSTRUMENT = "instrument_front"
    REPORT_WITH_INSTRUMENTS = "report_with_instruments"
    UNKNOWN = "unknown"


def normalize_text(text: str) -> str:
    t = (text or "").upper()
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    t = re.sub(r"[ \t]+", " ", t)
    return t


def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE))


def spatial_page_signals(words: Iterable[Any]) -> Dict[str, Any]:
    anchors = {
        "front": re.compile(r"PAY|EXACTLY|ORDER|MONEY|CASHIER|REMITTER|PURCHASER", re.I),
        "back": re.compile(r"ENDORSE|DEPOSITORY|AGREEMENT|SERVICE|DIRECTION|TERMS", re.I),
        "control": re.compile(r"DEPOSIT|TOTAL|BATCH|TRANSACTION|ACCOUNT|ITEMS", re.I),
        "amount": re.compile(r"^\$?[\d,]+\.\d{2}$"),
    }
    counts = {key: 0 for key in anchors}
    high_confidence = 0
    low_confidence = 0
    vertical_bins: set[int] = set()
    for word in words or []:
        content = str(getattr(word, "content", "") or "")
        confidence = float(getattr(word, "confidence", 0.0) or 0.0)
        if confidence >= 0.80:
            high_confidence += 1
        elif confidence < 0.50:
            low_confidence += 1
        for key, pattern in anchors.items():
            if pattern.search(content):
                counts[key] += 1
        polygon = tuple(getattr(word, "polygon", ()) or ())
        if len(polygon) >= 4:
            ys = polygon[1::2]
            y = sum(ys) / len(ys)
            if 0 <= y <= 1:
                vertical_bins.add(min(int(y * 8), 7))
    total = high_confidence + low_confidence
    return {
        "front_anchors": counts["front"],
        "back_anchors": counts["back"],
        "control_anchors": counts["control"],
        "amount_anchors": counts["amount"],
        "high_confidence_words": high_confidence,
        "low_confidence_words": low_confidence,
        "confidence_ratio": round(high_confidence / total, 3) if total else 0.0,
        "occupied_vertical_bins": len(vertical_bins),
    }


def classify_page(text: str, spatial: Dict[str, Any] | None = None) -> Tuple[PageKind, Dict[str, int]]:
    t = normalize_text(text)
    words = re.findall(r"[A-Z0-9$#:/.-]{2,}", t)
    if len(words) < 6:
        return PageKind.BLANK, {"words": len(words)}

    # Hard skip rules for back-side pages. Azure OCR often sees faint bleed-through from the
    # front of a money order/check; these deterministic back signals should win unless there is
    # a clearly visible front-face payment area.
    hard_back_signals = sum(
        1
        for pat in (
            r"\bLOAD\s+THIS\s+DIRECTION\b",
            r"\bPURCHASER'?S\s+AGREEMENT\b",
            r"\bSERVICE\s+CHARGE\b",
            r"\bTERMS\s+AND\s+CONDITIONS\b",
            r"\bPAYEE\s+ENDORSEMENT\b",
            r"\bENDORSE\s+ABOVE\s+THIS\s+LINE\b",
            r"\bDEPOSITORY\s+BANK\s+ENDORSEMENT\b",
            r"\bDO\s+NOT\s+WRITE\s*/?\s*SIGN\s*/?\s*STAMP\s+BELOW\b",
            r"\bFOR\s+DEPOSIT\s+ONLY\b",
        )
        if re.search(pat, t, flags=re.IGNORECASE)
    )
    strong_front_face = bool(
        re.search(r"\bPAY\s+EXACTLY\b|\bPAY\s+ONLY\b", t)
        and re.search(r"\bPAY\s+TO\s+THE\s+ORDER\b|\bPAY\s+TO\b", t)
        and re.search(r"\$\s*\d{2,5}[,.]\d{2}\b", t)
    )
    check_back_only = bool(
        re.search(r"\bFOR\s+DEPOSIT\s+ONLY\b", t)
        and re.search(r"\bENDORSE|DEPOSITORY\s+BANK|DO\s+NOT\s+WRITE", t)
        and not strong_front_face
    )
    if (hard_back_signals >= 2 and not strong_front_face) or check_back_only:
        return PageKind.BACK, {"words": len(words), "hard_back_signals": hard_back_signals}

    # Scores are deliberately simple and explainable. Front-side instrument signals must beat
    # legal/back-side disclosure signals to avoid extracting bleed-through from back pages.
    front = 0
    front += 5 * _count(r"\bPAY\s+EXACTLY\b", t)
    front += 4 * _count(r"\bPAY\s+TO\s+THE\s+ORDER\b", t)
    front += 3 * _count(r"\bCASHIER'?S\s+CHECK\b|\bOFFICIAL\s+CHECK\b", t)
    front += 3 * _count(r"\bREMITTER\b", t)
    front += 3 * _count(r"\bWESTERN\s+UNION\b|\bMONEYGRAM\b|\bINTERMEX\b|\bFIDELITY\s+EXPRESS\b|\bBARRI\b|\bDOLEX\b|\bPLS\b", t)
    # Count money order text only when not just a YottaReal description column.
    if re.search(r"\bMONEY\s*ORDER\b|\bMONEYORDER\b", t) and "PAYMENT-MONEYORDER" not in t[:600]:
        front += 2
    front += 1 * _count(r"\$\s*\d{2,5}[,.]\d{2}\b", t)
    front += 1 * _count(r"\b\d{2}[-\s]?\d{7,12}\b", t)
    front += 1 * _count(r"\bMICR\b|\bMEMO\b|\bPURCHASER\b", t)

    back = 0
    back += 5 * _count(r"\bSERVICE\s+CHARGE\b", t)
    back += 4 * _count(r"\bLOAD\s+THIS\s+DIRECTION\b", t)
    back += 4 * _count(r"\bENDORSE\s+ABOVE\s+THIS\s+LINE\b|\bPAYEE\s+ENDORSEMENT\b", t)
    back += 3 * _count(r"\bPURCHASER'?S\s+AGREEMENT\b|\bLIMITED\s+RECOURSE\b", t)
    back += 2 * _count(r"\bFOR\s+DEPOSIT\s+ONLY\b|\bDEPOSITORY\s+BANK\s+ENDORSEMENT\b", t)
    back += 1 * _count(r"\bSECURITY\s+FEATURES\b|\bVOID\s+IF\b|\bTERMS\s+AND\s+CONDITIONS\b", t)

    # Dense endorsement/legal evidence must beat faint front-side bleed-through.
    # Mixed pages remain eligible only when front evidence is comparable.
    if back >= 12 and back >= front * 2:
        return PageKind.BACK, {"words": len(words), "front": front, "back": back}

    batch = 0
    batch += 5 * _count(r"\bBATCH\s+DETAILS\b|\bDEPOSIT\s+BATCH\s+DETAIL\s+REPORT\b", t)
    batch += 4 * _count(r"\bBATCH\s*#\b|\bBATCH\s+AMOUNT\b|\bACTUAL\s+ITEMS\b|\bLINE\s+ITEMS\b", t)
    batch += 2 * _count(r"\bG/L\s+ACCOUNT\b|\bGENERAL\s+LEDGER\b|\bPOSTED\s+BY\b", t)
    batch += 1 * _count(r"PAYMENT[-\s]?MONEYORDER|PAYMENT[-\s]?CHECK", t)

    deposit = 0
    deposit += 5 * _count(r"\bTRANSACTION\s+DETAIL\s+FOR\s+TRANSACTION\b|\bDEPOSIT\s+CONTROL\s+INFORMATION\b", t)
    deposit += 5 * _count(r"\bDETAILS\s+OF\s+DEPOSITS\s+BY\s+ACCOUNT\b|\bTOTAL\s+OF\s+DEPOSITS\s+SUBMITTED\b", t)
    deposit += 4 * _count(r"\bDEPOSIT\s+ACCOUNT\b|\bDEPOSIT\s+TOTAL\b|\bCHECKS?\s+TOTAL\b", t)
    deposit += 4 * _count(r"\bTOTAL\s+NUMBER\s+OF\s+ITEMS\b|\bACCOUNT\s+NAME/NUMBER\b", t)
    deposit += 3 * _count(r"\bCAPTURE\s+SEQ\b|\bPOST\s+AMOUNT\b|\bCREDIT\s+AMOUNT\b|\bDEPOSIT\s+NUMBER\b", t)
    deposit += 2 * _count(r"\bCREDIT\s+TOTAL\b|\bDEBIT\s+TOTAL\b|\bITEM\s+COUNT\b", t)
    deposit += 2 * _count(r"\bDEPOSIT\s+SLIP\b|\bDEPOSIT\s+TICKET\b", t)

    receipt = 0
    receipt += 4 * _count(r"\bCHASE\b.*\bTRANSACTION\s+SUMMARY\b|\bMY\s+TRANSACTION\s+SUMMARY\b", t)
    receipt += 5 * _count(r"\bTRANSACTION\s+RECEIPT\b", t)
    receipt += 4 * _count(r"\bTOTAL\s+CHECKS\s+AMOUNT\b|\bTOTAL\s+DEPOSIT\b", t)
    receipt += 3 * _count(r"\bNUMBER\s+OF\s+CHECKS\b|\bCHECK\s+LISTING\b", t)
    receipt += 3 * _count(r"\bDEPOSIT\s+CASH\s+OR\s+CHECKS\b|\bCHECKING\s+DEPOSIT\b", t)
    receipt += 1 * _count(r"\bCASHBOX\b|\bBUSINESS\s+DATE\b", t)
    receipt += 5 * _count(
        r"\b(?:COMMERCIAL\s+DEPOSIT(?:\s+(?:CHECKING|SAVINGS))?|"
        r"(?:CHECKING|SAVINGS)\s+DEPOSIT)\s+\$?\s*[\d,]+\.\d{2}\b",
        t,
    )

    scores = {
        "words": len(words),
        "front": front,
        "back": back,
        "batch": batch,
        "deposit": deposit,
        "receipt": receipt,
    }
    spatial = spatial or {}
    spatial_front = int(spatial.get("front_anchors") or 0)
    spatial_back = int(spatial.get("back_anchors") or 0)
    spatial_control = int(spatial.get("control_anchors") or 0)
    scores["spatial_front"] = spatial_front
    scores["spatial_back"] = spatial_back
    scores["spatial_control"] = spatial_control

    if receipt >= 8 and not strong_front_face:
        return PageKind.RECEIPT, scores

    if front >= 6:
        if deposit >= 5 or batch >= 5:
            return PageKind.REPORT_WITH_INSTRUMENTS, scores
        return PageKind.INSTRUMENT, scores

    if receipt >= 4 and front < 4:
        return PageKind.RECEIPT, scores

    if deposit >= 8 and front < 9:
        return PageKind.DEPOSIT_SLIP, scores

    if back >= 5 and front < 8:
        return PageKind.BACK, scores

    if deposit >= 5 and front < 6:
        return PageKind.DEPOSIT_REPORT, scores

    if batch >= 5 and front < 6:
        return PageKind.BATCH_HEADER, scores

    # Some cropped deposit tickets are handwritten and contain totals but no strong report labels.
    if re.search(r"\bTOTAL\s+ITEMS\b|\bITEMS\s+TOTAL\b|\bTOTAL\s+DEPOSIT\b", t) and re.search(r"\$\s*\d", t):
        return PageKind.DEPOSIT_SLIP, scores

    return PageKind.UNKNOWN, scores
