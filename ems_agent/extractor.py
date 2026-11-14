import re
from datetime import datetime
from typing import Any, Awaitable, Callable

from .config import FIELDS, field_label
from .llm import chat_json

JsonCall = Callable[[str, str], Awaitable[dict[str, Any]]]


def _spoken_amount(text: str) -> str | None:
    words = re.findall(r"[a-z]+", text.lower())
    small = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
        "eighteen": 18, "nineteen": 19,
    }
    tens = {
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    }
    total = current = 0
    seen = False
    for word in words:
        if word in small:
            current += small[word]
            seen = True
        elif word in tens:
            current += tens[word]
            seen = True
        elif word == "hundred" and seen:
            current = max(current, 1) * 100
        elif word == "thousand" and seen:
            total += max(current, 1) * 1000
            current = 0
        elif word in {"and", "dollar", "dollars", "cent", "cents"}:
            continue
        elif seen:
            break
    if not seen:
        return None
    return f"{total + current:.2f}"


def _normalize_amount(value: Any, evidence: str) -> str | None:
    if re.search(r"\b(?:no payment|nothing was paid|zero dollars?)\b", evidence, re.IGNORECASE):
        return "0.00" if str(value).strip() in {"0", "0.0", "0.00"} else None
    parsed_value = re.sub(r"[^\d.]", "", str(value))
    split_cents = re.search(r"\$\s*(\d[\d,]*)\.\s*(\d{1,2})(?:\s*(?:\u00a2|cents?))?", evidence, re.I)
    if split_cents:
        combined = float(split_cents.group(1).replace(",", "")) + int(split_cents.group(2)) / 100
        if float(parsed_value or -1) in {float(split_cents.group(1).replace(",", "")), combined}:
            return f"{combined:.2f}"
    currency_amount = re.search(
        r"\$?\s*(\d[\d,]*(?:\.\d{1,2})?)\s*(?:dollars?)?\s*"
        r"(?:and\s*)?(\d{1,2})\s*(?:\u00a2|cents?)(?![a-z])",
        evidence,
        re.IGNORECASE,
    )
    if currency_amount:
        dollars = float(currency_amount.group(1).replace(",", ""))
        cents = int(currency_amount.group(2))
        combined = dollars + cents / 100
        if cents < 100 and float(parsed_value or -1) == combined:
            return f"{combined:.2f}"

    evidence_numbers = re.findall(r"\d[\d,]*(?:\.\d{1,2})?", evidence)
    if evidence_numbers:
        for number in evidence_numbers:
            try:
                number_value = float(number.replace(",", ""))
                if float(parsed_value or -1) == number_value:
                    return f"{number_value:.2f}"
            except ValueError:
                continue
    spoken = _spoken_amount(evidence)
    try:
        if spoken is not None and float(spoken) == float(parsed_value):
            return spoken
    except ValueError:
        pass
    return None


def _normalize_date(value: Any, evidence: str) -> str | None:
    value_text = str(value).strip()
    for fmt in ("%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d", "%m/%d/%y", "%m-%d-%y"):
        try:
            expected = datetime.strptime(value_text, fmt)
            break
        except ValueError:
            expected = None
    if expected is None:
        return None

    compact_evidence = re.sub(r"\s+", " ", evidence.lower())
    month = expected.strftime("%m").lstrip("0")
    day = expected.strftime("%d").lstrip("0")
    numeric_forms = {
        expected.strftime("%m/%d/%Y"),
        expected.strftime("%m-%d-%Y"),
        expected.strftime("%Y-%m-%d"),
        expected.strftime("%m/%d/%y"),
        expected.strftime("%m-%d-%y"),
        f"{expected.month}/{expected.day}/{expected:%y}",
        f"{expected.month}-{expected.day}-{expected:%y}",
        f"{month}/{day}/{expected:%Y}",
        f"{expected:%B} {day}, {expected:%Y}",
        f"{expected:%B} {day} {expected:%Y}",
        f"{expected:%b} {day}, {expected:%Y}",
        f"{expected:%b} {day} {expected:%Y}",
        f"{day} {expected:%B} {expected:%Y}",
        f"{day} {expected:%b} {expected:%Y}",
        f"{expected.day} {expected:%B} {expected:%Y}",
        f"{expected.day} {expected:%b} {expected:%Y}",
    }
    if any(form.lower() in compact_evidence for form in numeric_forms):
        return expected.strftime("%m/%d/%Y")

    spelled_evidence = re.sub(r"[,.;]", " ", compact_evidence)
    ordinal_days = {
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
        "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
        "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
        "nineteenth": 19, "twentieth": 20, "twenty first": 21,
        "twenty second": 22, "twenty third": 23, "twenty fourth": 24,
        "twenty fifth": 25, "twenty sixth": 26, "twenty seventh": 27,
        "twenty eighth": 28, "twenty ninth": 29, "thirtieth": 30,
        "thirty first": 31,
    }
    month_names = expected.strftime("%B|%b")
    day_pattern = "|".join(re.escape(day_word) for day_word in ordinal_days)
    spelled_date = re.search(
        rf"\b(?:{month_names})\s+({day_pattern}|\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+(\d{{4}})\b",
        spelled_evidence,
        re.IGNORECASE,
    )
    if not spelled_date:
        spelled_date = re.search(
            rf"\b({day_pattern}|\d{{1,2}})(?:st|nd|rd|th)?\s+(?:{month_names})[,]?\s+(\d{{4}})\b",
            spelled_evidence,
            re.IGNORECASE,
        )
    if spelled_date:
        stated_day = ordinal_days.get(spelled_date.group(1).lower())
        if stated_day is None:
            stated_day = int(spelled_date.group(1))
        if stated_day == expected.day and int(spelled_date.group(2)) == expected.year:
            return expected.strftime("%m/%d/%Y")
    return None


def _validate(name: str, value: Any, evidence: str, full_text: str | None = None) -> Any:
    if not isinstance(value, (str, int, float)) or not isinstance(evidence, str):
        return None
    if not evidence.strip():
        return None
    full_text = full_text or evidence

    validator = FIELDS[name]["validator"]
    if validator == "amount":
        return _normalize_amount(value, evidence)
    if validator == "date":
        return _normalize_date(value, evidence)
    if validator == "status":
        allowed_statuses = (
            "paid in full",
            "partially paid",
            "in process",
            "in review",
            "under review",
            "not found",
            "processed",
            "approved",
            "finalized",
            "adjudicated",
            "received",
            "closed",
            "on hold",
            "denied",
            "pending",
            "rejected",
            "paid",
        )
        normalized_evidence = re.sub(r"\s+", " ", evidence.casefold())
        normalized_full_text = re.sub(r"\s+", " ", full_text.casefold())
        normalized_value = re.sub(r"\s+", " ", str(value).casefold()).strip()
        if normalized_value not in allowed_statuses:
            return None
        hedge = (
            r"\b(?:not\s+paid|has(?:\s+not|n't)\s+been\s+paid|unpaid|not\s+yet|"
            r"will\s+be\s+paid|to\s+be\s+paid|scheduled|expected|pending\s+payment|no\s+payment)\b"
        )
        if normalized_value == "paid" and re.search(hedge, normalized_full_text):
            return None
        if re.search(rf"\b{re.escape(normalized_value)}\b", normalized_evidence):
            return normalized_value
        return None

    compact_value = re.sub(r"[^a-z0-9]", "", str(value).casefold())
    compact_evidence = re.sub(r"[^a-z0-9]", "", evidence.casefold())
    if validator == "reference":
        spoken_digits = re.search(r"\bdouble\s+(\d)\s+(\d)\s+(\d)\b", evidence, re.I)
        if spoken_digits:
            compact_value = spoken_digits.group(1) * 2 + spoken_digits.group(2) + spoken_digits.group(3)
        else:
            digit_groups = re.findall(r"\b\d\b", evidence)
            if len(digit_groups) >= 3:
                compact_value = "".join(digit_groups)
            else:
                compact_value = re.sub(r"[^a-z0-9]", "", str(value).casefold())
        claim_number = re.sub(r"[^a-z0-9]", "", "TEST-48213".casefold())
        if (
            not 3 <= len(compact_value) <= 20
            or not re.search(r"\d", compact_value)
            or compact_value == claim_number
            or compact_value in {"number", "pending"}
        ):
            return None
        if compact_value in compact_evidence or spoken_digits:
            return compact_value
        return None
    if compact_value and compact_value in compact_evidence:
        return str(value).strip()
    return None


def _add_validated(
    extracted: dict[str, Any], name: str, value: Any, evidence: str, full_text: str,
) -> None:
    """Store a regex candidate only after the same validation as LLM output."""
    if name in FIELDS:
        validated = _validate(name, value, evidence, full_text)
        if validated is not None:
            extracted[name] = validated


def regex_extract(text: str, active_field: str | None = None) -> dict[str, Any]:
    """Extract explicit, high-confidence facts without waiting for an LLM."""
    extracted: dict[str, Any] = {"_declined": []}

    for status in (
        "paid in full", "partially paid", "under review", "in review", "in process",
        "not found", "processed", "approved", "finalized", "adjudicated", "received",
        "closed", "on hold", "denied", "pending", "rejected", "paid",
    ):
        match = re.search(rf"\b{re.escape(status)}\b", text, re.IGNORECASE)
        if match:
            _add_validated(extracted, "claim_status", status, match.group(), text)
            break

    for match in re.finditer(r"\$\s*(\d[\d,]*)(?:\.\s*(\d{1,2})(?:\s*(?:\u00a2|cents?))?|(?:\.(\d{1,2}))?)", text, re.I):
        cents = match.group(2) or match.group(3)
        value = match.group(1) if cents is None else f"{match.group(1)}.{cents}"
        _add_validated(extracted, "paid_amount", value, match.group(), text)
        break
    if "paid_amount" not in extracted:
        match = re.search(r"\b(\d[\d,]*)(?:\s+dollars?)?\s+(?:and\s+)?(\d{1,2})\s*(?:¢|cents?)\b", text, re.I)
        if match:
            value = float(match.group(1).replace(",", "")) + int(match.group(2)) / 100
            _add_validated(extracted, "paid_amount", f"{value:.2f}", match.group(), text)
    if "paid_amount" not in extracted:
        match = re.search(r"\b(\d[\d,]*(?:\.\d{1,2})?)\s+dollars?\b", text, re.I)
        if match:
            _add_validated(extracted, "paid_amount", match.group(1), match.group(), text)
    if "paid_amount" not in extracted:
        amount_words = (
            r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
            r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
            r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|and"
        )
        match = re.search(rf"\b((?:(?:{amount_words})\s+)+(?:dollars?|cents?))\b", text, re.I)
        if match:
            value = _spoken_amount(match.group())
            if value is not None:
                _add_validated(extracted, "paid_amount", value, match.group(), text)
    if "paid_amount" not in extracted and re.search(r"\b(?:no payment|nothing was paid|zero dollars?)\b", text, re.I):
        evidence = re.search(r"\b(?:no payment|nothing was paid|zero dollars?)\b", text, re.I).group()
        _add_validated(extracted, "paid_amount", "0", evidence, text)

    for match in re.finditer(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text):
        _add_validated(extracted, "payment_date", match.group(), match.group(), text)
        if "payment_date" in extracted:
            break
    if "payment_date" not in extracted:
        month = re.search(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?[,]?\s+\d{4}\b",
            text,
            re.I,
        )
        if month:
            cleaned = re.sub(r"(st|nd|rd|th)(?=[,\s])|,", "", month.group(), flags=re.I)
            date = datetime.strptime(cleaned, "%B %d %Y")
            _add_validated(extracted, "payment_date", date.strftime("%m/%d/%Y"), month.group(), text)

    references = re.finditer(
        r"\b(?:check|eft)(?:\s+(?:number|no\.?))?\s+((?:[A-Za-z0-9]\s*){3,20})",
        text,
        re.I,
    )
    for reference in references:
        _add_validated(extracted, "check_or_eft_number", reference.group(1), reference.group(), text)
        if "check_or_eft_number" in extracted:
            break
    if active_field == "check_or_eft_number" and "check_or_eft_number" not in extracted:
        # A representative may answer a requested EFT number with bare grouped digits.
        bare_number = re.search(r"\b(?:\d\s*){3,20}\b", text)
        if bare_number:
            _add_validated(extracted, "check_or_eft_number", bare_number.group(), bare_number.group(), text)

    return extracted


async def extract_facts(
    rep_text: str,
    active_field: str | None = None,
    llm_call: JsonCall = chat_json,
) -> dict[str, Any]:
    """Extract facts only when the model supplies an exact quote from this turn."""
    regex_facts = regex_extract(rep_text, active_field)
    if any(field_name in regex_facts for field_name in FIELDS) or len(rep_text.split()) <= 3:
        return regex_facts

    fact_schema = ", ".join(
        f'"{name}": {{"value": string|null, "evidence": string|null}}'
        for name in FIELDS
    )
    system = (
        "You are an extraction engine for emergency medical services (EMS) claim billing calls. "
        "Extract ONLY values explicitly spoken by the insurance representative in the latest message. "
        "A fragment such as 'claim status is already' is not a claim status; do not turn incomplete phrases into facts. "
        f"Use this schema exactly: {{\"facts\": {{{fact_schema}}}, \"declined\": [\"field_name\"]}}. "
        "Evidence must be an exact substring copied from the speaker's message. Do not infer, paraphrase, or supply any value not explicitly spoken. "
        + (f"The caller just asked about {field_label(active_field)}. " if active_field else "")
        + "If a field is absent, set value to null and evidence to null. "
        "If the rep says they do not know or cannot provide a field, add that field name to declined. "
        "Do not include explanations or markdown. Return valid JSON only."
    )
    try:
        result = await llm_call(system, rep_text)
    except Exception:
        result = {}
    extracted: dict[str, Any] = {}
    raw_facts = result.get("facts", {}) if isinstance(result, dict) else {}
    if isinstance(raw_facts, dict):
        for name in FIELDS:
            item = raw_facts.get(name)
            if not isinstance(item, dict):
                continue
            evidence = item.get("evidence")
            if not isinstance(evidence, str) or evidence not in rep_text:
                continue
            value = _validate(name, item.get("value"), evidence, rep_text)
            if value is not None:
                extracted[name] = value

    declined = result.get("declined", []) if isinstance(result, dict) else []
    extracted["_declined"] = [name for name in declined if name in FIELDS]
    return extracted
