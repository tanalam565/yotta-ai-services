from __future__ import annotations

"""Deterministic OCR parsing and normalization implementation."""

import re
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Issuer:
    """Single source of truth for a money-order issuer.

    One record carries every facet that used to live in separate tables:
    OCR detection (`detect`/`agent`), serial validation (`issuer_id`/`aliases`/
    `serial_lengths`), and the standard-issuer allowlist (`standard`). The
    derived views below reproduce the original tables exactly.
    """

    name: str                                       # display name; detect result + allowlist key
    detect: Optional[str] = None                    # OCR-detection regex (None -> not auto-detected)
    agent: Optional[str] = None                     # issuing agent
    issuer_id: Optional[str] = None                 # stable id surfaced on the instrument
    aliases: Tuple[str, ...] = ()                   # substring aliases for identify_issuer
    serial_lengths: Tuple[int, ...] = ()            # valid serial lengths for this issuer
    standard: bool = True                           # member of the standard-issuer allowlist


# Order matters: detect_issuer returns the first matching `detect` regex.
ISSUERS: List[Issuer] = [
    Issuer("Western Union", r"WESTERN\s+UNION|\bWALMART\b|\bKROGER\b|\bH[-\s]?E[-\s]?B\b|\bCVS\b",
           issuer_id="western_union", aliases=("WESTERN UNION",), serial_lengths=(10, 11)),
    Issuer("MoneyGram", r"MONEY\s*GRAM|MONEYGRAM|CITIZENS\s+ALLIANCE",
           issuer_id="moneygram", aliases=("MONEYGRAM",), serial_lengths=(8, 9, 10, 11)),
    Issuer("Intermex", r"INTERMEX", agent="Intermex Wire Transfer LLC"),
    Issuer("Fidelity Express", r"FIDELITY\s+EXPRESS", agent="Fidelity Express",
           issuer_id="fidelity_express", aliases=("FIDELITY EXPRESS",), serial_lengths=(8, 9, 10, 11)),
    Issuer("DolEx", r"BARRI|DOLEX", agent="Bank of Texas",
           issuer_id="dolex", aliases=("DOLEX",), serial_lengths=(10, 11)),
    Issuer("PLS", r"\bPLS\b|BANCFIRST", agent="BancFirst"),
    Issuer("JPMorgan Chase", r"JPMORGAN|JP\s*MORGAN|CHASE"),
    Issuer("Wells Fargo", r"WELLS\s+FARGO"),
    Issuer("Prosperity Bank", r"PROSPERITY\s+BANK"),
    Issuer("Comerica Bank", r"COMERICA"),
    # Identify-only (serial validation); never auto-detected, not on the allowlist.
    Issuer("USPS", issuer_id="usps", standard=False, serial_lengths=(10, 11),
           aliases=("UNITED STATES POSTAL SERVICE", "US POSTAL SERVICE", "USPS")),
]

BANK_PATTERNS: List[Tuple[str, str]] = [
    (r"J\s*P\s*MORGAN|JPMORGAN|JP\s*MORGAN|CHASE", "JPMorgan Chase Bank"),
    (r"WELLS\s+FARGO", "Wells Fargo Bank"),
    (r"REGIONS", "Regions Bank"),
    (r"BANK\s+OF\s+TEXAS", "Bank of Texas"),
    (r"MORGAN\s+CHASE", "JPMorgan Chase Bank"),
    (r"IMPERIAL\s+CHASE", "Imperial Chase"),
]

# Compiled once at import; detect_issuer/normalize_bank_name run per instrument.
_ISSUER_COMPILED: List[Tuple[Any, str, Optional[str]]] = [
    (re.compile(i.detect, re.IGNORECASE), i.name, i.agent) for i in ISSUERS if i.detect
]
_BANK_COMPILED: List[Tuple[Any, str]] = [
    (re.compile(pat, re.IGNORECASE), name) for pat, name in BANK_PATTERNS
]

ONES = {
    "ZERO": 0,
    "ONE": 1,
    "TWO": 2,
    "THREE": 3,
    "FOUR": 4,
    "FIVE": 5,
    "SIX": 6,
    "SEVEN": 7,
    "EIGHT": 8,
    "NINE": 9,
    "TEN": 10,
    "ELEVEN": 11,
    "TWELVE": 12,
    "THIRTEEN": 13,
    "FOURTEEN": 14,
    "FIFTEEN": 15,
    "SIXTEEN": 16,
    "SEVENTEEN": 17,
    "EIGHTEEN": 18,
    "NINETEEN": 19,
}
TENS = {
    "TWENTY": 20,
    "THIRTY": 30,
    "FORTY": 40,
    "FOURTY": 40,
    "FIFTY": 50,
    "SIXTY": 60,
    "SEVENTY": 70,
    "EIGHTY": 80,
    "NINETY": 90,
}

# Compiled once at import; parse_amount_from_words runs per instrument amount.
_FIRST_CARDINAL_WORD = re.compile(
    r"\b(?:ZERO|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|"
    r"ELEVEN|TWELVE|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|"
    r"EIGHTEEN|NINETEEN|TWENTY|THIRTY|FORTY|FIFTY|SIXTY|SEVENTY|"
    r"EIGHTY|NINETY|HUNDRED|THOUSAND)\b"
)


def norm_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"[ \t]+", " ", text)


def normalize_bank_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    for pat, name in _BANK_COMPILED:
        if pat.search(value):
            return name
    return value.strip()


def valid_bank_name(value: Optional[str]) -> Optional[str]:
    name = normalize_bank_name(value)
    if not name:
        return None
    words = re.findall(r"[A-Za-z]+", name)
    generic = {"NAME", "ACCOUNT", "DEPOSIT", "FINANCIAL", "INSTITUTION", "BANK"}
    meaningful = [word for word in words if word.upper() not in generic]
    if len(words) < 2 or not meaningful or len("".join(meaningful)) < 3:
        return None
    return name


def infer_bank_name(text: str) -> Optional[str]:
    """Extract a visible bank heading without relying on a bank allowlist."""
    stopwords = {
        "AVAILABLE",
        "COPY",
        "DEPOSIT",
        "EXPECT",
        "FROM",
        "OF",
        "PART",
        "RECORD",
        "SERVICES",
        "THE",
        "TICKET",
        "YOU",
    }
    candidates: List[Tuple[int, int, str]] = []
    for line in (text or "").splitlines()[:16]:
        clean = re.sub(r"\s+", " ", line).strip(" :-")
        words = re.findall(r"[A-Z0-9&.'-]+", clean.upper())
        for index, word in enumerate(words):
            if word != "BANK":
                continue
            for width in range(1, min(index, 3) + 1):
                parts = words[index - width : index + 1]
                penalty = sum(part in stopwords for part in parts[:-1])
                candidates.append((penalty, -width, " ".join(parts)))
        for match in re.finditer(r"\bBANK\s+OF\s+[A-Z][A-Z0-9&.'-]+\b", clean.upper()):
            candidates.append((0, -3, match.group(0)))
    if not candidates:
        return None
    name = min(candidates)[2]
    normalized = normalize_bank_name(name)
    candidate = normalized if normalized != name else name.title().replace("N. A.", "N.A.")
    return valid_bank_name(candidate)


def detect_issuer(text: str) -> Tuple[Optional[str], Optional[str]]:
    for pat, issuer, agent in _ISSUER_COMPILED:
        if pat.search(text or ""):
            return issuer, agent
    return None, None


def parse_money(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Remove max-cap language such as NOT VALID OVER $1000.00.
    s = re.sub(r"NOT\s+VALID\s+OVER\s*\$?\s*[\d,.]+", "", s, flags=re.IGNORECASE)

    # OCR often drops the decimal point in the amount box: "$356 56" or "$356,56".
    # Treat these as dollars+cents only when the text is clearly amount-like.
    amount_like = bool(re.search(r"\$|DOLLARS?|CENTS?|PAY\s+EXACTLY|PAY\s+ONLY|AMOUNT", s, re.IGNORECASE))
    m_sep = re.search(r"(?:\$\s*)?\b([0-9]{1,5})\s+([0-9]{2})\b", s)
    if m_sep and (amount_like or re.fullmatch(r"\s*[0-9]{1,5}\s+[0-9]{2}\s*", s)):
        dollars = int(m_sep.group(1).replace(",", ""))
        cents_i = int(m_sep.group(2))
        if 0 <= cents_i <= 99 and 0 <= dollars <= 100000:
            return round(dollars + cents_i / 100.0, 2)

    # Decimal comma in OCR: "$356,56" should be 356.56, but "1,234" is thousands.
    m_comma_decimal = re.search(r"(?:\$\s*)?\b([0-9]{1,5}),([0-9]{2})\b", s)
    if m_comma_decimal and (amount_like or re.fullmatch(r"\s*[0-9]{1,5},[0-9]{2}\s*", s)):
        dollars = int(m_comma_decimal.group(1))
        cents_i = int(m_comma_decimal.group(2))
        if 0 <= cents_i <= 99 and 0 <= dollars <= 100000:
            return round(dollars + cents_i / 100.0, 2)

    m = re.search(r"[-+]?\$?\s*((?:[0-9]{1,3}(?:,[0-9]{3})+)|[0-9]+)(?:[.](\d{1,2}))?", s)
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    cents = m.group(2) or "00"
    if len(cents) == 1:
        cents += "0"
    try:
        val = float(f"{raw}.{cents}")
    except ValueError:
        return None
    if 0.0 <= val <= 1000000:
        return round(val, 2)
    return None


def parse_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")
    m = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", s)
    if m:
        return _date_from_parts(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"\b(\d{1,2})[./|\-](\d{1,2})[./|\-](\d{2,4})\b", s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
        year = int("20" + y) if len(y) == 2 else int(y)
        return _date_from_parts(year, a, b) or _date_from_parts(year, b, a)
    # Western Union internal date code: D 050926.
    m = re.search(r"\bD\s*(\d{2})(\d{2})(\d{2})\b", s, flags=re.IGNORECASE)
    if m:
        return _date_from_parts(2000 + int(m.group(3)), int(m.group(1)), int(m.group(2)))
    for fmt in ("%B %d %Y", "%b %d %Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s.title(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    m = re.search(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(20\d{2})\b", s)
    if m:
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                return datetime.strptime(f"{m.group(1).title()} {m.group(2)} {m.group(3)}", fmt).strftime("%Y-%m-%d")
            except ValueError:
                pass
    return None


def _date_from_parts(year: int, month: int, day: int) -> Optional[str]:
    try:
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _parse_cardinal_words(value: str) -> Optional[int]:
    """Parse simple English cardinal words up to hundreds of thousands."""
    if not value:
        return None
    value = re.sub(r"[^A-Z0-9 -]", " ", value.upper())
    tokens = [tok for tok in re.split(r"[\s-]+", value) if tok]
    total = 0
    current = 0
    seen = False
    for tok in tokens:
        if tok in {"AND", "DOLLAR", "DOLLARS", "CENT", "CENTS", "ONLY", "NO"}:
            continue
        if tok == "ZERO":
            seen = True
            continue
        if tok.isdigit():
            current += int(tok)
            seen = True
        elif tok in ONES:
            current += ONES[tok]
            seen = True
        elif tok in TENS:
            current += TENS[tok]
            seen = True
        elif tok == "HUNDRED":
            current = max(current, 1) * 100
            seen = True
        elif tok == "THOUSAND":
            total += max(current, 1) * 1000
            current = 0
            seen = True
    if not seen:
        return None
    return total + current


def parse_amount_from_words(text: Optional[str]) -> Optional[float]:
    """Parse written check/MO amounts.

    Separates the dollar segment from the cents segment so cents are not counted
    as extra dollars. Handles common handwritten/OCR variants:
      - "FOURTEEN HUNDRED NINETY NINE 74" -> 1499.74
      - "TWO HUNDRED SIX AND 17 DOLLARS" -> 206.17
      - "THIRTEEN HUNDRED TWENTY THREE DOLLARS 74/XX" -> 1323.74
    """
    if not text:
        return None
    t = str(text).upper()
    t = t.replace("NO/100", "00/100")
    # Normalize legal-fraction variants before stripping punctuation.
    # Woodforest/Chase checks often render cents as "63/100ths dollars";
    # without this, the trailing 63 was counted as extra dollars
    # ("One Hundred Thirty Two and 63/100ths" -> 195.00).
    t = re.sub(r"/(?:\s*)100(?:THS?|TH)?\b", "/100", t, flags=re.IGNORECASE)
    # OCR sometimes reads /100 as /XX on handwritten checks.
    t = re.sub(r"/\s*(?:XX|X{2})\b", "/100", t, flags=re.IGNORECASE)
    t = re.sub(r"[^A-Z0-9/ .'-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None
    # Legal amount lines sometimes include form/date/reference codes before the
    # written phrase. Once cardinal words begin, numeric prefixes must not be
    # interpreted as dollars.
    first_word = _FIRST_CARDINAL_WORD.search(t)
    if first_word and re.search(r"\d", t[: first_word.start()]):
        t = t[first_word.start():]

    cents = 0
    dollar_part = t

    # Numeric cents: 04/100, 74/XX->74/100, 09 CENTS, 9 CENTS.
    m = re.search(r"\b(\d{1,2})\s*/\s*100\b", t)
    if m:
        cents = int(m.group(1))
        dollar_part = t[:m.start()]
    else:
        # Handwriting/OCR: "Two Hundred Six and 17 Dollars" — cents before DOLLARS label.
        m = re.search(r"\bAND\s+(\d{1,2})\s+DOLLARS?\b", t)
        if m:
            cents = int(m.group(1))
            dollar_part = t[:m.start()]
        else:
            m = re.search(r"\b(\d{1,2})\s+CENTS?\b", t)
            if m:
                cents = int(m.group(1))
                dollar_part = t[:m.start()]
            elif re.search(r"\bCENTS?\b", t):
                cents_head = t[: re.search(r"\bCENTS?\b", t).start()]
                m_words = re.search(r"\bDOLLARS?\b\s*(?:AND\s+)?([A-Z -]+)$", cents_head)
                if m_words:
                    cents_words = m_words.group(1).strip()
                    dollar_part = cents_head[: m_words.start()]
                else:
                    m_words = re.search(r"\bAND\s+([A-Z -]+)$", cents_head)
                    if m_words:
                        cents_words = m_words.group(1).strip()
                        dollar_part = cents_head[: m_words.start()]
                    else:
                        cents_words = cents_head.strip()
                        dollar_part = ""
                if re.fullmatch(r"(?:NO|ZERO|00)", cents_words.strip()):
                    cents = 0
                else:
                    parsed_cents = _parse_cardinal_words(cents_words)
                    cents = parsed_cents if parsed_cents is not None else 0
            else:
                # Bare trailing two-digit cents with no DOLLARS marker, e.g.
                # "FOURTEEN HUNDRED NINETY NINE 74".
                m = re.search(r"\b(?:AND\s+)?(\d{1,2})\s*$", t)
                if m and re.search(r"[A-Z]", t[:m.start()]):
                    cents = int(m.group(1))
                    dollar_part = t[:m.start()]

    m_dollars = re.search(r"\bDOLLARS?\b", dollar_part)
    if m_dollars:
        dollar_part = dollar_part[:m_dollars.start()]
    dollar_part = re.sub(r"\b(PAY|EXACTLY|ONLY|THE|SUM|OF|AMOUNT|PAYABLE)\b", " ", dollar_part)
    dollar_part = re.sub(r"\bAND\s*$", " ", dollar_part).strip()

    dollars = _parse_cardinal_words(dollar_part)
    if dollars is None:
        return None
    if not (0 <= cents <= 99):
        return None
    total = round(float(dollars) + cents / 100.0, 2)
    if 0 < total <= 100000:
        return total
    return None


def extract_legal_amount_words(text: str) -> Optional[str]:
    """Extract a multiline legal amount following a PAY EXACTLY label."""
    cleaned = re.sub(r"\s+", " ", re.sub(r"[*_=]+", " ", norm_text(text)))
    match = re.search(
        r"\bPAY\s+EXACTLY\b\s*([A-Z0-9 /'-]{3,180}?\bDOLLARS?\b(?:\s+(?:AND\s+)?[A-Z0-9 /'-]{1,50}\bCENTS?\b|\s+\d{1,2}/100)?)",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    value = re.sub(r"\s+", " ", match.group(1)).strip()
    return value if parse_amount_from_words(value) is not None else None


def extract_labeled_amount(text: str) -> Optional[float]:
    t = norm_text(text)
    t = re.sub(r"NOT\s+VALID\s+OVER[^\n]*", "", t, flags=re.IGNORECASE)
    patterns = [
        # OCR/vision sometimes reads the amount box as "$356 56" or "$356,56".
        # Put these before whole-dollar patterns so cents are not dropped.
        r"PAY\s+EXACTLY[^$\n]{0,80}\$\s*([0-9]{1,5}\s+[0-9]{2})",
        r"PAY\s+ONLY[^$\n]{0,50}\$\s*([0-9]{1,5}\s+[0-9]{2})",
        r"\$\s*([0-9]{1,5}\s+[0-9]{2})(?!\d)",
        r"\$\s*([0-9]{1,5},[0-9]{2})(?!\d)",
        # Normal decimal amount.
        r"PAY\s+EXACTLY[^$\n]{0,80}\$\s*([\d,]+(?:\.\d{1,2})?)",
        r"PAY\s+ONLY[^$\n]{0,50}\$\s*([\d,]+(?:\.\d{1,2})?)",
        r"\*+\s*\$?\s*([\d,]+\.\d{1,2})\s*\*+",
        r"\$\s*([\d,]+\.\d{1,2})",
    ]
    for pat in patterns:
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            val = parse_money(m.group(1))
            if val is not None and 0.01 <= val <= 100000:
                return val
    return None


# Compiled once at import; extract_serial runs per instrument.
_SERIAL_LABELED_PATTERNS = [
    re.compile(r"(?:MONEY\s*ORDER|DOCUMENT|SERIAL|CHECK)\s*(?:NO|NUMBER|#)?\.?\s*[:#-]?\s*(\d{7,14})"),
    re.compile(r"\b([12]\d[-\s]?\d{7,12})\b"),
]
_SERIAL_EXPLICIT_PATTERNS = [
    re.compile(r"CHECK\s*NO\.?\s*[:#]?\s*(\d{6,12})"),
    re.compile(r"SERIAL\s*[:#]?\s*(\d{5,12})"),
]
_SERIAL_BARE_PATTERN = re.compile(r"\b\d{7,11}\b")


def extract_serial(text: str) -> Optional[str]:
    t = norm_text(text).upper()
    candidates: List[str] = []

    # Long serials printed near an explicit identifier. Prefixes vary by issuer
    # and production series; they are not limited to historically common values.
    for pat in _SERIAL_LABELED_PATTERNS:
        candidates += [re.sub(r"\D", "", m.group(1)) for m in pat.finditer(t)]
    # MoneyGram explicit check/no.
    for pat in _SERIAL_EXPLICIT_PATTERNS:
        for m in pat.finditer(t):
            candidates.append(m.group(1))
    # BARRI/Intermex/Fidelity vertical serials often 7-10 digits near issuer text.
    for m in _SERIAL_BARE_PATTERN.finditer(t):
        raw = m.group(0)
        if is_aba_routing_number(raw) or set(raw) == {"0"}:
            continue
        candidates.append(raw)
    for c in candidates:
        c = re.sub(r"\D", "", c)
        if 5 <= len(c) <= 12:
            return c
    return None


def extract_micr(text: str) -> Optional[str]:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    candidates = []
    for ln in lines:
        digits = re.sub(r"[^0-9 ]", " ", ln)
        groups = re.findall(r"\d{4,}", digits)
        if len(groups) >= 2 and sum(len(g) for g in groups) >= 14:
            candidates.append(" ".join(groups))
    if candidates:
        return max(candidates, key=len)
    return None


def extract_unit_from_text(*values: Optional[str]) -> Optional[str]:
    joined = "\n".join(v for v in values if v)
    if not joined:
        return None
    patterns = [
        r"(?:APT|APARTMENT|UNIT|SUITE|STE|#)\s*\.?\s*#?\s*([A-Z]?\d{3,5}[A-Z]?)\b",
        r"\b(?:RENT|MEMO|FOR|ACCOUNT|ACCT)\D{0,20}([A-Z]?\d{3,5}[A-Z]?)\b",
        r"\b([A-Z]?\d{3,5}[A-Z]?)\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, joined, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            unit = m.group(1).upper()
            if re.match(r"^\d+$", unit) and not (3 <= len(unit) <= 5):
                continue
            return unit
    return None


def serial_from_deposit_detail_account(account: Any) -> Optional[str]:
    """Return the public serial for a Deposit Detail Report MICR account.

    Western Union rows print as Account=40<serial><check-digit>.
    The trailing MICR check digit must not be included in serial_number.
    Example: 40197995806964 -> 19799580696.
    """
    acct = re.sub(r"\D", "", str(account or ""))
    if not acct:
        return None
    if acct.startswith("40"):
        tail = acct[2:]
        if len(tail) >= 10:
            return tail[:-1] if len(tail) >= 12 else tail
    return acct[-11:] if len(acct) >= 9 else None


def normalize_serial(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    s = str(value).strip().upper()
    if s in {"NULL", "NONE", "N/A", "NA"}:
        return None
    s = s.replace("O", "0").replace("I", "1").replace("L", "1")
    s = re.sub(r"[^0-9A-Z-]", "", s)
    if re.match(r"^\d{2}-?X+$", s):
        return None
    # Long numeric serials normalize to digits so matching against register/MICR works.
    if re.match(r"^\d{2}-?\d{7,12}$", s):
        return re.sub(r"\D", "", s)
    return s or None


def is_aba_routing_number(value: Any) -> bool:
    from money_order_validator.validation import is_aba_routing_number as validate_routing

    return validate_routing(value)


def validate_graph_local_instrument(row: Dict[str, Any]) -> Dict[str, Any]:
    """Reject fields not supported by the same isolated region."""
    out = dict(row)
    text = str(out.get("_region_ocr_text") or "")
    if not out.get("_region_id") and out.get("_region_source") != "evidence_graph":
        return out
    if not text.strip():
        return out

    serial = re.sub(r"\D", "", str(out.get("serial_number") or ""))
    local_digits = {re.sub(r"\D", "", value) for value in re.findall(r"\d[\d,.$/-]*", text)}
    local_digits.discard("")
    if serial and not any(serial == value or serial in value or value in serial for value in local_digits):
        out["serial_number"] = None
        out["image_quality"] = "unclear"
        out.setdefault("review_flags", []).extend(["serial_outside_local_graph", "manual_review_required"])

    micr = str(out.get("micr_line") or "")
    micr_groups = [re.sub(r"\D", "", value) for value in re.findall(r"\d+", micr)]
    if micr_groups and not any(
        group and any(group == value or group in value or value in group for value in local_digits)
        for group in micr_groups
    ):
        out["micr_line"] = None
        out["micr_evidence"] = None
        out["image_quality"] = "unclear"
        out.setdefault("review_flags", []).extend(["micr_outside_local_graph", "manual_review_required"])

    amount = parse_money(out.get("amount_numeric"))
    words_amount = parse_amount_from_words(out.get("amount_words"))
    if amount is not None and words_amount is not None and abs(amount - words_amount) > 0.01:
        out["amount_candidate"] = amount
        out["amount_numeric"] = None
        out["amount_status"] = "conflict"
        out["image_quality"] = "unclear"
        out.setdefault("review_flags", []).extend(["graph_local_amount_conflict", "manual_review_required"])
    return out


def normalize_payee(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    s = re.sub(r"\s+", " ", str(value)).strip(" .,-")
    if not s or s.upper() in {"PAY TO THE ORDER OF", "FOR DEPOSIT ONLY", "ADDRESS", "PURCHASER"}:
        return None
    # Strip unit suffix/prefix from payee value.
    s = re.sub(r"\s+(?:APT|APARTMENT|UNIT|#)\s*\.?\s*#?\s*[A-Z]?\d{3,5}[A-Z]?\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^#\s*[A-Z]?\d{3,5}[A-Z]?\s+", "", s, flags=re.IGNORECASE)
    return s.strip(" .,-") or None


def similarity(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b:
        return 0.0
    aa = re.sub(r"[^A-Z0-9]", "", a.upper())
    bb = re.sub(r"[^A-Z0-9]", "", b.upper())
    if not aa or not bb:
        return 0.0
    return round(SequenceMatcher(None, aa, bb).ratio(), 3)


def sanitize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "visible"}


def sanitize_instrument(raw: Dict[str, Any], ocr_text: str = "") -> Dict[str, Any]:
    out = dict(raw or {})
    issuer, agent = detect_issuer("\n".join([str(out.get("issuer") or ""), ocr_text]))
    if issuer:
        out["issuer"] = issuer
        out["issuer_agent"] = out.get("issuer_agent") or agent

    inst_type = out.get("instrument_type") or ""
    joined = f"{ocr_text}\n{out.get('issuer') or ''}\n{out.get('payment_description') or ''}".upper()
    if not inst_type:
        if "CASHIER" in joined or "OFFICIAL CHECK" in joined:
            inst_type = "CashiersCheck"
        elif "CHECK" in joined and "MONEY" not in joined:
            inst_type = "Check"
        else:
            inst_type = "MoneyOrder"
    if inst_type not in {"MoneyOrder", "Check", "CashiersCheck", "Escrow"}:
        inst_type = "MoneyOrder"
    out["instrument_type"] = inst_type

    if not out.get("payment_description"):
        if inst_type == "Escrow":
            out["payment_description"] = "Escrow Deposit Paid In"
        elif inst_type in {"Check", "CashiersCheck"}:
            out["payment_description"] = "Payment-Check"
        else:
            out["payment_description"] = "Payment-MoneyOrder"

    out["serial_number"] = normalize_serial(out.get("serial_number")) or extract_serial(ocr_text)
    serial_digits = re.sub(r"\D", "", str(out.get("serial_number") or ""))
    weak_nine_digit_check_serial = bool(
        inst_type in {"Check", "CashiersCheck"}
        and len(serial_digits) == 9
        and str(out.get("serial_evidence") or "").lower() not in {"labeled_number", "both"}
    )
    if is_aba_routing_number(out.get("serial_number")) or weak_nine_digit_check_serial:
        out.setdefault("corrections", []).append(
            {
                "field": "serial_number",
                "old": out["serial_number"],
                "new": None,
                "source": "reject_aba_routing_as_serial",
            }
        )
        out["serial_number"] = None
        out.setdefault("review_flags", []).extend(["missing_document_identity", "manual_review_required"])
    out["issue_date"] = parse_date(out.get("issue_date")) or parse_date(ocr_text)

    amt = parse_money(out.get("amount_numeric"))
    labeled_amt = extract_labeled_amount(ocr_text)
    if labeled_amt is not None:
        if amt is None:
            amt = labeled_amt
        else:
            # If the model/OCR captured only whole dollars but the labeled amount box
            # has cents, restore the cents. Example: model read 356, box reads 356.56.
            same_dollar_part = int(float(amt)) == int(float(labeled_amt))
            amt_has_no_cents = abs(float(amt) - int(float(amt))) < 0.005
            labeled_has_cents = abs(float(labeled_amt) - int(float(labeled_amt))) >= 0.005
            if same_dollar_part and amt_has_no_cents and labeled_has_cents:
                out.setdefault("corrections", []).append(
                    {
                        "field": "amount_numeric",
                        "old": amt,
                        "new": labeled_amt,
                        "source": "labeled_amount_cents",
                    }
                )
                amt = labeled_amt
    if not out.get("amount_words"):
        out["amount_words"] = extract_legal_amount_words(ocr_text)
    words_amt = parse_amount_from_words(out.get("amount_words"))
    # Written amount is strong evidence, but OCR on faint/rotated instruments can
    # produce plausible contradictory words. Never silently turn disagreement
    # into a confident amount.
    words_text = str(out.get("amount_words") or "").upper()
    words_specify_cents = bool(
        re.search(r"\b\d{1,2}\s*/\s*100\b|\b\d{1,2}\s+CENTS?\b|\bNO\s+CENTS?\b", words_text)
    )
    amount_disagrees = bool(
        words_amt is not None
        and amt is not None
        and (
            int(float(words_amt)) != int(float(amt))
            or (words_specify_cents and abs(float(amt) - float(words_amt)) >= 0.01)
        )
    )
    if amount_disagrees:
        out["amount_candidate"] = amt
        out["amount_status"] = "conflict"
        out.setdefault("review_flags", []).extend(
            ["amount_evidence_conflict", "manual_review_required"]
        )
        out["image_quality"] = "unclear"
        out.setdefault("corrections", []).append(
            {
                "field": "amount_numeric",
                "old": amt,
                "new": None,
                "source": "conflicting_numeric_and_written_amounts",
            }
        )
        amt = None
    elif words_amt is not None and amt is None:
        out.setdefault("corrections", []).append(
            {
                "field": "amount_numeric",
                "old": amt,
                "new": words_amt,
                "source": "amount_words",
            }
        )
        amt = words_amt
        out["amount_status"] = "verified"
    out["amount_numeric"] = amt
    if amt is not None and not out.get("amount_status"):
        out["amount_status"] = "verified"

    out["payee_raw"] = normalize_payee(out.get("payee_raw"))
    out["unit"] = extract_unit_from_text(out.get("unit"), out.get("payment_for_acct"), out.get("payer_address"), out.get("payee_raw"), ocr_text)

    for key in ("payer_name", "payer_address", "amount_words", "payment_for_acct", "micr_line", "issuer_agent"):
        val = out.get(key)
        if isinstance(val, str):
            val = re.sub(r"\s+", " ", val).strip()
            if val.upper() in {"NULL", "NONE", "N/A", "NA", "PURCHASER", "ADDRESS", "REMITTER"}:
                val = None
        out[key] = val or None

    if not out.get("micr_line"):
        out["micr_line"] = extract_micr(ocr_text)
    if out.get("micr_line"):
        from money_order_validator.validation import parse_micr_evidence

        micr_evidence = parse_micr_evidence(out["micr_line"])
        out["micr_evidence"] = micr_evidence
        current = re.sub(r"\D", "", str(out.get("serial_number") or "")).lstrip("0")
        check_number = micr_evidence.get("check_number")
        non_identity_groups = {
            str(value).lstrip("0")
            for value in [
                micr_evidence.get("routing_number"),
                *(micr_evidence.get("account_candidates") or []),
            ]
            if value
        }
        if current and check_number and current in non_identity_groups:
            out.setdefault("corrections", []).append(
                {
                    "field": "serial_number",
                    "old": current,
                    "new": None,
                    "source": "reject_micr_account_or_routing_as_serial",
                }
            )
            out["serial_number"] = None
            out["image_quality"] = "unclear"
            out.setdefault("review_flags", []).extend(
                ["missing_document_identity", "manual_review_required"]
            )
    if out.get("instrument_type") in {"Check", "CashiersCheck"} and out.get("micr_line"):
        from money_order_validator.validation import parse_micr_evidence

        micr_evidence = out.get("micr_evidence") or parse_micr_evidence(out["micr_line"])
        check_number = micr_evidence.get("check_number")
        current = re.sub(r"\D", "", str(out.get("serial_number") or "")).lstrip("0")
        serial_evidence = str(out.get("serial_evidence") or "").lower()
        labeled_serial = serial_evidence in {"labeled_number", "both"}
        existing_conflict = bool(
            out.get("match_status") == "conflicting"
            or "serial_micr_conflict" in (out.get("review_flags") or [])
        )
        if existing_conflict:
            out["serial_number"] = None
            out["match_status"] = "conflicting"
            out["image_quality"] = "unclear"
        elif check_number and current and current != check_number and labeled_serial:
            out["serial_candidate"] = current
            out["micr_check_number"] = check_number
            out["serial_number"] = None
            out["match_status"] = "conflicting"
            out["image_quality"] = "unclear"
            out.setdefault("review_flags", []).extend(
                ["serial_micr_conflict", "manual_review_required"]
            )
        elif check_number and current != check_number:
            out.setdefault("corrections", []).append(
                {"field": "serial_number", "old": current or None, "new": check_number, "source": "check_micr_structure"}
            )
            out["serial_number"] = check_number

    out["payer_signature"] = sanitize_bool(out.get("payer_signature"))
    # Trust only the explicit printed phrase for this flag. Vision models sometimes infer this
    # on ordinary money orders/checks, which creates noisy risk flags.
    mdp_text = "\n".join([ocr_text or "", str(out.get("mobile_deposit_prohibited") or "")])
    out["mobile_deposit_prohibited"] = bool(
        re.search(r"MOBILE\s+DEPOSIT\s+PROHIBITED|NOT\s+FOR\s+MOBILE\s+DEPOSIT", mdp_text, re.IGNORECASE)
    )
    out["watermark_present"] = sanitize_bool(out.get("watermark_present"))

    return out


def parse_basic_instrument_from_ocr(text: str) -> Optional[Dict[str, Any]]:
    issuer, agent = detect_issuer(text)
    serial = extract_serial(text)
    amount = extract_labeled_amount(text)
    issue_date = parse_date(text)
    if not any([issuer, serial, amount, issue_date]):
        return None
    inst_type = "MoneyOrder" if issuer or re.search(r"MONEY\s*ORDER", text or "", re.IGNORECASE) else "Check"
    return sanitize_instrument(
        {
            "instrument_type": inst_type,
            "issuer": issuer,
            "issuer_agent": agent,
            "serial_number": serial,
            "amount_numeric": amount,
            "issue_date": issue_date,
            "micr_line": extract_micr(text),
        },
        text,
    )


def _one_line(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" :-")


def is_deposit_detail_report(text: str) -> bool:
    """Detect bank Deposit Detail Report pages with transaction item rows.

    These reports already contain authoritative transaction id, account, row amounts,
    and row serial/account values. Thumbnail images on the report must not be treated
    as separate physical instruments.
    """
    t = norm_text(text).upper()
    return bool(
        re.search(r"\bDEPOSIT\s+DETAIL\s+REPORT\b", t)
        or (
            re.search(r"\bDEPOSIT\s+DETAIL\s+FOR\s+DEPOSIT\s+ID\b", t)
            and re.search(r"\bTRANSACTION\s+DETAIL\s+FOR\s+TRANSACTION\s+ID\b", t)
        )
        or (
            re.search(r"\bDEPOSIT\s+CONTROL\s+INFORMATION\b", t)
            and re.search(r"\bTRANSACTION\s+CONTROL\s+INFORMATION\b", t)
        )
    )


def parse_deposit_detail_report_header(text: str) -> Dict[str, Any]:
    """Parse Deposit Detail Report header/summary fields.

    This report type has one logical deposit spread across multiple pages.
    Per-page item rows must not be aggregated as separate deposit slips.
    """
    if not is_deposit_detail_report(text):
        return {}
    t = norm_text(text)
    out: Dict[str, Any] = {
        "source_system": "Deposit Detail Report",
    }

    if m := re.search(r"Deposit\s+Detail\s+for\s+Deposit\s+ID\s*:?\s*(\d{4,})", t, flags=re.IGNORECASE):
        out["deposit_id"] = m.group(1)
        out["document_number"] = m.group(1)
    if m := re.search(r"Transaction\s+Detail\s+for\s+Transaction\s+ID\s*:?\s*(\d{4,})", t, flags=re.IGNORECASE):
        out["deposit_transaction"] = m.group(1)
    if m := re.search(r"Batch\s+ID\s*:?\s*(\d{4,})", t, flags=re.IGNORECASE):
        out["batch_number"] = m.group(1)
    if m := re.search(r"Processing\s+Date\s*:?\s*(20\d{2}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4})", t, flags=re.IGNORECASE):
        out["deposit_date"] = parse_date(m.group(1))
        out["deposited_date"] = out["deposit_date"]
        out["printed_on"] = out["deposit_date"]
    if m := re.search(r"Account\s+Name\s*:?\s*([^\n]+?)(?=\s+Location\s+ID\b|\s+Transaction\s+Detail\b|$)", t, flags=re.IGNORECASE):
        name = _clean_property_name(m.group(1))
        if name:
            out["account_name"] = name
            out["property_name"] = name
            out["property_aliases"] = build_property_aliases(name)
    if m := re.search(r"Deposit\s+Account\s*:?\s*(\d{4,})(?:\s*-\s*([^\n]+?))?(?=\s+Partnership\b|\s+AUX|$)", t, flags=re.IGNORECASE):
        out["deposit_account"] = m.group(1)
        out["account_number"] = m.group(1)
        if m.group(2):
            name = _clean_property_name(m.group(2))
            if name:
                out.setdefault("account_name", name)
                out.setdefault("property_name", name)
    for key, pat in (
        ("deposit_total", r"Deposit\s+Total\s*:?\s*\$?\s*([\d,]+\.\d{2})"),
        ("check_total", r"Checks?\s+Total\s*:?\s*\$?\s*([\d,]+\.\d{2})"),
        ("credit_total", r"Credit\s+Total\s*:?\s*\$?\s*([\d,]+\.\d{2})"),
        ("debit_total", r"Debit\s+Total\s*:?\s*\$?\s*([\d,]+\.\d{2})"),
        ("deposit_amount", r"Deposit\s+Amount\s*:?\s*\$?\s*([\d,]+\.\d{2})"),
    ):
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            val = parse_money(m.group(1))
            if val is not None:
                out[key] = val
    if out.get("deposit_total") is not None:
        out.setdefault("deposit_amount", out["deposit_total"])
        out.setdefault("check_total", out["deposit_total"])
    for key, pat in (
        ("credit_items", r"Credit\s+Items\s*:?\s*(\d{1,4})"),
        ("debit_items", r"Debit\s+Items\s*:?\s*(\d{1,4})"),
        ("report_item_count", r"Item\s+Count\s*:?\s*(\d{1,4})"),
    ):
        if m := re.search(pat, t, flags=re.IGNORECASE):
            try:
                out[key] = int(m.group(1))
            except ValueError:
                pass
    if out.get("debit_items") is not None:
        out["item_count"] = out["debit_items"]
    return {k: v for k, v in out.items() if v not in (None, "", [])}


def _clean_property_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    s = _one_line(value)
    s = re.sub(r"\bAccount\s+Currency\b.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bNumber\s+of\s+Deposits\b.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTotal\s+of\s+Deposits\b.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"/\s*\d{4,}\b.*$", "", s)
    s = s.strip(" -:/")
    words = re.findall(r"[A-Za-z]+", s)
    short_word_ratio = sum(len(word) <= 3 for word in words) / max(1, len(words))
    mixed_case_noise = sum(bool(re.search(r"[a-z][A-Z]|[A-Z][a-z][A-Z]", word)) for word in words)
    if (
        not s
        or re.fullmatch(r"REGIONS|CHASE|JPMORGAN|BANK", s, flags=re.IGNORECASE)
        or len(words) < 2
        or short_word_ratio > 0.65
        or mixed_case_noise > 0
        or bool(re.fullmatch(r"[A-Z]{1,4}\d{1,5}\s*\(COPY\)", s, flags=re.IGNORECASE))
    ):
        return None
    return s


def build_property_aliases(property_name: Optional[str]) -> List[str]:
    """Return match aliases for property/payee validation."""
    if not property_name:
        return []
    candidates: List[str] = []

    def add(value: Optional[str]) -> None:
        value = _clean_property_name(value)
        if not value:
            return
        compact = re.sub(r"[^A-Z0-9]", "", value.upper())
        seen = {re.sub(r"[^A-Z0-9]", "", x.upper()) for x in candidates}
        if compact and compact not in seen:
            candidates.append(value)

    add(property_name)
    m = re.search(r"\bd\s*/?\s*b\s*/?\s*a\b\s+(.+)$|\bdba\b\s+(.+)$", property_name, flags=re.IGNORECASE)
    alias_base = None
    if m:
        alias_base = m.group(1) or m.group(2)
        add(alias_base)
    else:
        alias_base = property_name

    if alias_base:
        alias_base = _clean_property_name(alias_base) or alias_base
        no_suffix = re.sub(r"\b(APARTMENTS?|APTS?|VILLAGE|LP|LLC|LTD|LIMITED\s+PARTNERSHIP)\b", "", alias_base, flags=re.IGNORECASE)
        add(no_suffix)
        for sep in (" in ", " at ", " dba "):
            if sep in alias_base.lower():
                add(re.split(sep, alias_base, flags=re.IGNORECASE)[0])
        words = re.findall(r"[A-Za-z0-9]+", alias_base)
        if len(words) >= 2:
            add(" ".join(words[:2]))
        if words and len(words[0]) >= 5:
            add(words[0])
    return candidates


def is_regions_deposit_report(text: str) -> bool:
    """Detect Regions "Details of Deposits by Account" reports.

    These pages look like a payment register but are not payment instruments. If
    they are misrouted to vision extraction, routing/account/check values become
    fake instruments.  Keep this detector deterministic and conservative.
    """
    t = norm_text(text).upper()
    signals = [
        r"\bDETAILS\s+OF\s+DEPOSITS\s+BY\s+ACCOUNT\b",
        r"\bTOTAL\s+OF\s+DEPOSITS\s+SUBMITTED\b",
        r"\bTOTAL\s+NUMBER\s+OF\s+ITEMS\b",
        r"\bACCOUNT\s+NAME/NUMBER\b",
        r"\bCAPTURE\s+SEQ\b",
        r"\bPOST\s+AMOUNT\b",
        r"\bCREDIT\s+AMOUNT\b",
        r"\bDEPOSIT\s+NUMBER\b",
    ]
    return sum(1 for pat in signals if re.search(pat, t, flags=re.IGNORECASE)) >= 2


def _clean_regions_name(value: str) -> Optional[str]:
    if not value:
        return None
    value = re.sub(r"\s+", " ", value).strip(" -:/")
    value = re.split(
        r"\b(?:NUMBER\s+OF\s+DEPOSITS|TOTAL\s+OF\s+DEPOSITS|TOTAL\s+NUMBER|ACCOUNT\s+CURRENCY|USD|DEPOSIT\s+NUMBER)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = value.strip(" -:/")
    return value or None


def parse_regions_deposit_header(text: str) -> Dict[str, Any]:
    """Parse Regions bank deposit-register header fields.

    Example page text contains:
      Account Name/Number: Example Holdings LP dba Example Apartments/0293323581
      Total of Deposits Submitted: 25,213.12
      Total Number of Items: 17
    """
    if not is_regions_deposit_report(text):
        return {}

    t = norm_text(text)
    out: Dict[str, Any] = {
        "bank_name": "Regions Bank",
        "source_system": "Regions",
    }

    # The account name can wrap before the slash/account number.
    m = re.search(
        r"Account\s+Name/Number\s*:\s*(?P<name>.*?)/\s*(?P<acct>\d{4,})",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        name = _clean_regions_name(m.group("name"))
        if name:
            out["property_name"] = name
        out["account_number"] = m.group("acct")

    if not out.get("property_name"):
        m = re.search(
            r"Details\s+of\s+Deposits\s+by\s+Account\s*-\s*(?P<name>.+?)\s*-",
            t,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m:
            name = _clean_regions_name(m.group("name"))
            if name:
                out["property_name"] = name

    for key, pat in [
        ("batch_amount", r"Total\s+of\s+Deposits\s+Submitted\s*:?[ \t]*\$?\s*([\d,]+\.\d{2})"),
        ("deposit_amount", r"Total\s+of\s+Deposits\s+Submitted\s*:?[ \t]*\$?\s*([\d,]+\.\d{2})"),
        ("deposit_total", r"Deposit\s+Total\s*:?[ \t]*\$?\s*([\d,]+\.\d{2})"),
    ]:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            out[key] = parse_money(m.group(1))

    m = re.search(r"Total\s+Number\s+of\s+Items\s*:?[ \t]*(\d+)", t, flags=re.IGNORECASE)
    if m:
        out["total_items"] = int(m.group(1))
        out["item_count"] = int(m.group(1))

    # Regions deposit number appears in the summary row below the header.
    m = re.search(
        r"Deposit\s+Number\s+Item\s+Count.*?\n\s*(\d{4,})\s+\d+\s+[\d,]+\.\d{2}",
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        out["deposit_transaction"] = m.group(1)

    # Prefer the actual deposit date row if present; otherwise any report date is acceptable.
    m = re.search(r"Deposit\s+Date.*?(\d{1,2}/\d{1,2}/20\d{2})", t, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        m = re.search(r"\b(\d{1,2}/\d{1,2}/20\d{2})\s+\d{1,2}:\d{2}\s*(?:AM|PM)?\b", t, flags=re.IGNORECASE)
    if m:
        out["deposited_date"] = parse_date(m.group(1))
        out.setdefault("printed_on", parse_date(m.group(1)))

    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def parse_regions_deposit_items(text: str) -> List[Dict[str, Any]]:
    """Parse Regions table rows into authoritative register items."""
    if not is_regions_deposit_report(text):
        return []

    items: List[Dict[str, Any]] = []
    row_re = re.compile(
        r"^\s*(?P<capture_seq>\d{6})\s+"
        r"(?P<routing_number>\d{9})\s+"
        r"(?P<account_number>\d{4,20})\s+"
        r"(?P<check_number>\d{1,12})\s+"
        r"(?P<post_amount>[\d,]+\.\d{2})\s+"
        r"(?P<credit_amount>[\d,]+\.\d{2})"
        r"(?:\s+(?P<adjustment>[\d,]+\.\d{2}))?\s*$",
        flags=re.IGNORECASE,
    )
    for raw_line in (text or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line.strip())
        m = row_re.match(line)
        if not m:
            continue
        raw_check = m.group("check_number")
        serial = raw_check if len(raw_check) <= 5 else (raw_check.lstrip("0") or raw_check)
        amount = parse_money(m.group("post_amount"))
        payment_description = "Payment-MoneyOrder" if len(serial) >= 6 else "Payment-Check"
        items.append(
            {
                "item_no": int(m.group("capture_seq")),
                "serial_number": serial,
                "check_number": raw_check,
                "routing_number": m.group("routing_number"),
                "drawee_account_number": m.group("account_number"),
                "amount_numeric": amount,
                "credit_amount": parse_money(m.group("credit_amount")),
                "adjustment": parse_money(m.group("adjustment")) if m.group("adjustment") else 0.0,
                "payment_description": payment_description,
                "source": "regions_deposit_report",
            }
        )
    return items


def parse_transaction_detail_items(text: str) -> List[Dict[str, Any]]:
    t = norm_text(text)
    if not re.search(r"Transaction\s+Detail\s+for\s+Transaction|Deposit\s+Control\s+Information", t, flags=re.IGNORECASE):
        return []
    items: List[Dict[str, Any]] = []
    row_re = re.compile(
        r"^\s*(?:(?P<aux>[A-Z0-9-]{4,14})\s+)?"
        r"(?P<routing>\d{9})\s+"
        r"(?P<account>\d{4,20})"
        r"(?:\s+(?P<check>\d{1,12}))?\s+"
        r"\$?\s*(?P<amount>[\d,]+\.\d{2})\s+"
        r"(?P<item_type>\d{3,4}|Credit|Debit)?\b",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    item_no = 0
    for m in row_re.finditer(t):
        line = m.group(0)
        if re.search(r"\bCredit\b", line, flags=re.IGNORECASE):
            continue
        amount = parse_money(m.group("amount"))
        if amount is None or amount <= 0 or amount > 5000:
            continue
        serial = normalize_serial(m.group("aux") or m.group("check"))
        acct = m.group("account")
        item_type = normalize_serial(m.group("item_type"))
        if serial and (serial == item_type or serial in {"0003", "0004", "003", "004"}):
            serial = None
        if not serial:
            serial = normalize_serial(serial_from_deposit_detail_account(acct)) if acct else None
        item_no += 1
        items.append(
            {
                "item_no": item_no,
                "routing_number": m.group("routing"),
                "account_number": m.group("account"),
                "check_number": m.group("check"),
                "serial_number": serial,
                "amount_numeric": amount,
                "payment_description": "Payment-MoneyOrder",
                "instrument_type": "MoneyOrder",
                "source": "transaction_detail_report",
                "source_system": "Deposit Detail Report",
            }
        )
    return items


def parse_batch_header(text: str) -> Dict[str, Any]:
    t = norm_text(text)
    result: Dict[str, Any] = {}
    _regions = parse_regions_deposit_header(t)
    if _regions:
        result.update(_regions)

    deposit_detail = parse_deposit_detail_report_header(t)
    if deposit_detail:
        # Keep deposit-detail fields but do not let generic header OCR overwrite
        # key values with labels like Worktype as the property.
        result.update({k: v for k, v in deposit_detail.items() if k not in {"deposit_amount", "deposit_total", "check_total", "credit_total", "debit_total", "report_item_count"}})
        if deposit_detail.get("deposit_total") is not None:
            result["batch_amount"] = deposit_detail["deposit_total"]

    def after_label(label_pat: str, stop: str = r"\n|Batch\s+|Bank\s+|Account\s+|Actual\s+|Line\s+|Period\s+|Printed\s+|Deposit\s+") -> Optional[str]:
        m = re.search(r"(?:" + label_pat + r")\s*:?[ \t]*([^\n]+)", t, flags=re.IGNORECASE)
        if not m:
            return None
        value = re.split(stop, m.group(1), maxsplit=1, flags=re.IGNORECASE)[0]
        return value.strip(" :-") or None

    m = re.search(r"\bBatch\s*(?:#|Number)\s*:?\s*([0-9]{6,12})\b", t, flags=re.IGNORECASE)
    if m:
        result["batch_number"] = m.group(1)

    for key, pat in [
        ("batch_type", r"Batch\s+Type"),
        ("batch_status", r"Batch\s+Status"),
        ("pay_period", r"Period"),
    ]:
        val = after_label(pat)
        if val:
            result[key] = val

    bank = after_label(r"Bank\s+Name") or after_label(r"Bank")
    if bank and not result.get("bank_name"):
        result["bank_name"] = normalize_bank_name(bank)

    acct = after_label(r"Account\s*#|Account\s+Number|Deposit\s+Account")
    if acct and not result.get("account_number"):
        nums = re.findall(r"\d{4,}", acct)
        if nums:
            result["account_number"] = nums[-1]

    total = None
    for pat in [
        r"Actual\s+Items\s*:?\s*(\d+)",
        r"Line\s+Items\s*:?\s*(\d+)",
        r"Total\s+Items\s*:?\s*(\d+)",
        r"Total\s+Number\s+of\s+Items\s*:?\s*(\d+)",
        r"Totals?\s*\((\d+)\)",
    ]:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            total = int(m.group(1))
            break
    if total is not None:
        result["total_items"] = total

    if not result.get("batch_amount"):
        for pat in [
            r"Batch\s+Amount\s*:?\s*\$?\s*([\d,]+\.\d{2})",
            r"Total\s+of\s+Deposits\s+Submitted\s*:?\s*\$?\s*([\d,]+\.\d{2})",
            r"Deposit\s+Amount\s*:?[ \t]*\$?\s*([\d,]+\.\d{2})",
            r"Deposit\s+Total\s*:?[ \t]*\$?\s*([\d,]+\.\d{2})",
            r"Totals?\s*\(\d+\).*?\$?\s*([\d,]+\.\d{2})",
        ]:
            m = re.search(pat, t, flags=re.IGNORECASE | re.DOTALL)
            if m:
                result["batch_amount"] = parse_money(m.group(1))
                break

    m = re.search(r"Printed\s+On\s*:?[ \t]*([^\n]+)", t, flags=re.IGNORECASE)
    if m:
        result["printed_on"] = parse_date(m.group(1))
    if not result.get("printed_on"):
        result["printed_on"] = parse_date(t)

    m = re.search(r"Transaction\s+(?:ID|#)\s*:?[ \t]*([0-9]{5,})", t, flags=re.IGNORECASE)
    if m:
        result["deposit_transaction"] = m.group(1)

    if not result.get("property_name"):
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        for i, line in enumerate(lines[:10]):
            if re.search(r"BATCH\s+DETAIL|DEPOSIT\s+BATCH|DETAILS\s+OF\s+DEPOSITS|YOTTAREAL|PRINTED\s+ON|REGIONS|CHASE|JPMORGAN", line, re.IGNORECASE):
                continue
            if re.search(r"\b[A-Z][A-Z .'-]{3,}\b", line) and not re.search(r"BANK|ACCOUNT|BATCH|PERIOD|REPORT|TOTAL", line, re.IGNORECASE):
                name = _clean_property_name(line)
                if name:
                    result.setdefault("property_name", name.title())
                    result.setdefault("property_aliases", build_property_aliases(name))
                if i + 1 < len(lines) and re.search(r"\b[A-Z]{2}\s+\d{5}\b", lines[i + 1], re.IGNORECASE):
                    result.setdefault("property_address", lines[i + 1].title())
                break

    if result.get("property_name") and not result.get("property_aliases"):
        result["property_aliases"] = build_property_aliases(result.get("property_name"))

    return {k: v for k, v in result.items() if v not in (None, "", [])}


def parse_deposit_info(text: str) -> Dict[str, Any]:
    t = norm_text(text)
    out: Dict[str, Any] = {}

    deposit_detail = parse_deposit_detail_report_header(t)
    if deposit_detail:
        out.update(deposit_detail)

    deposit_context = bool(
        re.search(
            r"MY\s+TRANSACTION\s+SUMMARY|CHECKING\s+DEPOSIT|COMMERCIAL\s+DEPOSIT|"
            r"DEPOSIT\s+ACCOUNT|DEPOSIT\s+TOTAL|CHECKS?\s+TOTAL|ACCOUNT\s+NUMBER\s+ENDING\s+IN|"
            r"TRANSACTION\s+DETAIL\s+FOR\s+TRANSACTION|DEPOSIT\s+CONTROL\s+INFORMATION|"
            r"DEPOSIT\s+TICKET|TOTAL\s+ITEMS|FOR\s+CASH\s+DEPOSIT|"
            r"CHECKS\s+AND\s+OTHER\s+ITEMS\s+ARE\s+RECEIVED\s+FOR\s+DEPOSIT|"
            r"DETAILS\s+OF\s+DEPOSITS\s+BY\s+ACCOUNT|ACCOUNT\s+NAME/NUMBER|"
            r"TRANSACTION\s+RECEIPT|TOTAL\s+CHECKS\s+AMOUNT|TOTAL\s+DEPOSIT",
            t,
            flags=re.IGNORECASE,
        )
    )

    if deposit_context and re.search(r"CHASE|JPMORGAN", t, flags=re.IGNORECASE):
        out.setdefault("bank_name", "JPMorgan Chase Bank")
        out.setdefault("source_system", "Chase")
    if deposit_context:
        bank_name = infer_bank_name(t)
        if bank_name:
            out.setdefault("bank_name", valid_bank_name(bank_name))
            if out.get("bank_name"):
                out.setdefault("source_system", out["bank_name"])

    # Bank receipt parsing.
    if re.search(r"MY\s+TRANSACTION\s+SUMMARY|CHECKING\s+DEPOSIT|COMMERCIAL\s+DEPOSIT", t, flags=re.IGNORECASE):
        for pat in (
            r"(?:COMMERCIAL\s+DEPOSIT(?:\s+(?:CHECKING|SAVINGS))?|"
            r"(?:CHECKING|SAVINGS)\s+DEPOSIT)\s*:?\s*\$?\s*([\d,]+\.\d{2})",
            r"DEPOSIT\s+AMOUNT\s*:?\s*\$?\s*([\d,]+\.\d{2})",
        ):
            m = re.search(pat, t, flags=re.IGNORECASE)
            if m:
                amount = parse_money(m.group(1))
                if amount is not None:
                    out.setdefault("deposit_amount", amount)
                    out.setdefault("deposit_total", amount)
                    out.setdefault("check_total", amount)
                break
        if m := re.search(r"ACCOUNT\s+NUMBER\s+ENDING\s+IN\s*:?\s*(\d{2,6})", t, flags=re.IGNORECASE):
            out.setdefault("account_last4", m.group(1)[-4:])
            out.setdefault("deposit_account", m.group(1)[-4:])
        if m := re.search(r"TRANSACTION\s*#\s*:?\s*(\d{1,12})", t, flags=re.IGNORECASE):
            out.setdefault("deposit_transaction", m.group(1))
        if m := re.search(r"BUSINESS\s+DATE\s*:?\s*(\d{1,2}/\d{1,2}/\d{2,4}|20\d{2}[-/]\d{1,2}[-/]\d{1,2})", t, flags=re.IGNORECASE):
            out.setdefault("deposit_date", parse_date(m.group(1)))
        if m := re.search(r"\b(?:PD|CD)\s+(\d{1,2}/\d{1,2}/\d{2,4})", t, flags=re.IGNORECASE):
            out.setdefault("deposit_date", parse_date(m.group(1)))
        if m := re.search(r"\bX{3,}\s*(\d{3,6})\b", t, flags=re.IGNORECASE):
            out.setdefault("account_last4", m.group(1)[-4:])
            out.setdefault("deposit_account", m.group(1)[-4:])

    # Deposit tickets often carry the property/legal entity near the top.
    if m := re.search(r"\bDBA\s+([A-Z0-9 .&'/-]+?)\s+OPERATING\s+ACCOUNT\b", t, flags=re.IGNORECASE):
        prop = _clean_property_name(m.group(1))
        if prop:
            out.setdefault("account_name", prop.title())

    _regions = parse_regions_deposit_header(t)
    if _regions:
        if _regions.get("batch_amount") is not None:
            out["deposit_total"] = _regions["batch_amount"]
            out["check_total"] = _regions["batch_amount"]
        if _regions.get("total_items") is not None:
            out["item_count"] = _regions["total_items"]
        for src, dst in [
            ("account_number", "deposit_account"),
            ("property_name", "account_name"),
            ("deposited_date", "deposit_date"),
            ("deposit_transaction", "deposit_transaction"),
            ("bank_name", "bank_name"),
            ("source_system", "source_system"),
        ]:
            if _regions.get(src) is not None:
                out[dst] = _regions[src]

    if m := re.search(r"Transaction\s+(?:ID|#)\s*:?[ \t]*([0-9]{2,})", t, flags=re.IGNORECASE):
        out["deposit_transaction"] = m.group(1)
    if m := re.search(r"Deposit\s+Account\s*:?[ \t]*([0-9]{4,})(?:\s*-\s*([^\n]+))?", t, flags=re.IGNORECASE):
        out["deposit_account"] = m.group(1)
        if m.group(2):
            account_name = _clean_property_name(m.group(2))
            if account_name:
                out["account_name"] = account_name
    if m := re.search(r"Account\s+Number\s+Ending\s+In\s*:?[ \t]*([0-9]{3,6})", t, flags=re.IGNORECASE):
        out.setdefault("deposit_account", m.group(1))
        out["account_last4"] = m.group(1)[-4:]
    if m := re.search(r"\bX{3,}\s*(\d{3,6})\b", t, flags=re.IGNORECASE):
        out.setdefault("deposit_account", m.group(1)[-4:])
        out.setdefault("account_last4", m.group(1)[-4:])
    if deposit_context and re.search(r"\bCHASE\b|JPMORGAN\s+CHASE", t, flags=re.IGNORECASE):
        out.setdefault("bank_name", "JPMorgan Chase Bank")
        out.setdefault("source_system", "Chase")
    if deposit_context and re.search(r"\bREGIONS\b", t, flags=re.IGNORECASE):
        out.setdefault("bank_name", "Regions Bank")

    for key, pat in [
        ("deposit_amount", r"Deposit\s+Amount\s*:?[ \t]*\$?([\d,]+\.\d{2})"),
        ("deposit_total", r"Deposit\s+Total\s*:?[ \t]*\$?([\d,]+\.\d{2})"),
        ("deposit_total", r"Checking\s+Deposit\s*:?[ \t]*\$?([\d,]+\.\d{2})"),
        ("deposit_total", r"\$?\s*([\d,]+\.\d{2})\s+Checking\s+Deposit\b"),
        (
            "deposit_total",
            r"(?:Commercial\s+Deposit(?:\s+(?:Checking|Savings))?|"
            r"(?:Checking|Savings)\s+Deposit)\s*:?[ \t]*\$?([\d,]+\.\d{2})",
        ),
        ("check_total", r"Checks?\s+Total\s*:?[ \t]*\$?([\d,]+\.\d{2})"),
        ("credit_total", r"Credit\s+Total\s*:?[ \t]*\$?([\d,]+\.\d{2})"),
        ("debit_total", r"Debit\s+Total\s*:?[ \t]*\$?([\d,]+\.\d{2})"),
        ("difference", r"Difference\s*:?[ \t]*\$?([\d,]+\.\d{2})"),
        ("cash_back", r"Cash\s+Back\s*:?[ \t]*\$?([\d,]+\.\d{2})"),
        ("deposit_total", r"Total\s+of\s+Deposits\s+Submitted\s*:?\s*\$?([\d,]+\.\d{2})"),
        ("deposit_total", r"Total\s+Deposit\s*:?\s*\$?\s*([\d,]+\.\d{2})"),
        ("deposit_total", r"Total\s+Enter\s+Please\s+\$?\s*([\d,]+\.\d{2})"),
        ("check_total", r"Total\s+Checks\s+Amount\s*:?\s*\$?\s*([\d,]+\.\d{2})"),
    ]:
        if m := re.search(pat, t, flags=re.IGNORECASE):
            val = parse_money(m.group(1))
            if val is not None:
                out[key] = val
                if key == "deposit_total":
                    out.setdefault("check_total", val)

    if m := re.search(r"Item\s+Count\s*:?[ \t]*(\d+)", t, flags=re.IGNORECASE):
        out["item_count"] = int(m.group(1))
    if m := re.search(r"Total\s+Number\s+of\s+Items\s*:?\s*(\d+)", t, flags=re.IGNORECASE):
        out["item_count"] = int(m.group(1))
    if m := re.search(r"Total\s+Items\s*:?[ \t]*(\d+)", t, flags=re.IGNORECASE):
        out["item_count"] = int(m.group(1))
    if m := re.search(r"\b(\d{1,4})\s+Items\s+Total\b", t, flags=re.IGNORECASE):
        out["item_count"] = int(m.group(1))
    if m := re.search(r"Number\s+of\s+Checks\s*:?\s*(?:Check\s+Listing\s*)?(\d+)", t, flags=re.IGNORECASE):
        out["item_count"] = int(m.group(1))
    if m := re.search(r"Check\s+Listing\s+(.*?)\s+Total\s+Checks\s+Amount", t, flags=re.IGNORECASE | re.DOTALL):
        listed = [
            amount
            for amount in (parse_money(value) for value in re.findall(r"\$?\s*([\d,]+\.\d{2})", m.group(1)))
            if amount is not None
        ]
        if listed:
            out["listed_amounts"] = listed

    for date_pat in [
        r"Business\s+Date\s*:?[ \t]*(\d{1,2}/\d{1,2}/\d{2,4}|20\d{2}[-/]\d{1,2}[-/]\d{1,2})",
        r"Report\s+Time\s*:?[ \t]*(20\d{2}[-/]\d{1,2}[-/]\d{1,2})",
        r"\bDate\s*:?[ \t]*(\d{1,2}/\d{1,2}/\d{2,4}|20\d{2}[-/]\d{1,2}[-/]\d{1,2})",
    ]:
        if m := re.search(date_pat, t, flags=re.IGNORECASE):
            parsed = parse_date(m.group(1))
            if parsed:
                out["deposit_date"] = parsed
                break

    clean = {k: v for k, v in out.items() if v not in (None, "", [])}
    if not deposit_context and not _regions:
        return {}
    return clean


def parse_batch_line_items(text: str) -> List[Dict[str, Any]]:
    lines = [re.sub(r"\s+", " ", ln.strip()) for ln in (text or "").splitlines() if ln.strip()]
    items: List[Dict[str, Any]] = []

    items.extend(parse_regions_deposit_items(text))

    existing_keys = {(i.get("source"), i.get("serial_number"), i.get("amount_numeric")) for i in items}
    for item in parse_transaction_detail_items(text):
        key = (item.get("source"), item.get("serial_number"), item.get("amount_numeric"))
        if key not in existing_keys:
            items.append(item)
            existing_keys.add(key)

    row_pat = re.compile(
        r"^(?P<unit>[A-Z]?\d{1,5}[A-Z]?)\s+"
        r"(?P<resident>.+?)\s+"
        r"(?P<desc>Payment[- ](?:MoneyOrder|Check)|Escrow[^\s]*)\s+"
        r"(?P<serial>[A-Z0-9-]{4,18})\s+"
        r"(?P<date>\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+"
        r"\$?(?P<amount>[\d,]+\.\d{2})"
        r"(?:\s+(?P<posted_by>.+))?$",
        flags=re.IGNORECASE,
    )
    for line in lines:
        m = row_pat.match(line)
        if not m:
            continue
        serial = normalize_serial(m.group("serial"))
        desc = m.group("desc").replace(" ", "-")
        item = {
            "item_no": len(items) + 1,
            "unit": m.group("unit"),
            "resident_name": m.group("resident").strip(),
            "payment_description": desc,
            "instrument_type": "Check" if "Check" in desc else "MoneyOrder",
            "serial_number": serial,
            "posted_date": parse_date(m.group("date")),
            "amount_numeric": parse_money(m.group("amount")),
            "posted_by": (m.group("posted_by") or "").strip() or None,
            "source": "yottareal_batch_detail",
        }
        key = (item.get("source"), item.get("serial_number"), item.get("amount_numeric"))
        if key not in existing_keys:
            items.append(item)
            existing_keys.add(key)
    return items
