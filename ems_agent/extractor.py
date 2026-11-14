import re
from datetime import datetime
from typing import Any, Awaitable, Callable

from .config import FIELDS
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
    parsed_value = re.sub(r"[^\d.]", "", str(value))
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
    for fmt in ("%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d"):
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
        f"{month}/{day}/{expected:%Y}",
        f"{expected:%B} {day}, {expected:%Y}",
        f"{expected:%b} {day}, {expected:%Y}",
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


def _validate(name: str, value: Any, evidence: str) -> Any:
    if not isinstance(value, (str, int, float)) or not isinstance(evidence, str):
        return None
    if not evidence.strip():
        return None

    validator = FIELDS[name]["validator"]
    if validator == "amount":
        return _normalize_amount(value, evidence)
    if validator == "date":
        return _normalize_date(value, evidence)
    if validator == "status":
        allowed_statuses = (
            "partially paid",
            "in process",
            "not found",
            "processed",
            "approved",
            "denied",
            "pending",
            "rejected",
            "paid",
        )
        normalized_evidence = re.sub(r"\s+", " ", evidence.casefold())
        normalized_value = re.sub(r"\s+", " ", str(value).casefold()).strip()
        if normalized_value not in allowed_statuses:
            return None
        explicit_status_context = re.search(
            r"\b(?:claim\s+(?:status|is|was)|status(?:\s+is)?)\b",
            normalized_evidence,
        )
        if (
            normalized_value == "paid"
            and re.search(r"\bnot\s+paid\b", normalized_evidence)
        ):
            return None
        if (
            re.search(rf"\b{re.escape(normalized_value)}\b", normalized_evidence)
            and (explicit_status_context or normalized_evidence.strip() == normalized_value)
        ):
            return normalized_value
        return None

    compact_value = re.sub(r"[^a-z0-9]", "", str(value).casefold())
    compact_evidence = re.sub(r"[^a-z0-9]", "", evidence.casefold())
    if compact_value and compact_value in compact_evidence:
        return str(value).strip()
    return None


async def extract_facts(rep_text: str, llm_call: JsonCall = chat_json) -> dict[str, Any]:
    """Extract facts only when the model supplies an exact quote from this turn."""
    system = (
        "You are an extraction engine for emergency medical services (EMS) claim billing calls. "
        "Extract ONLY values explicitly spoken by the insurance representative in the latest message. "
        "A fragment such as 'claim status is already' is not a claim status; do not turn incomplete phrases into facts. "
        "Use this schema exactly: {\"facts\": {\"claim_status\": {\"value\": string|null, \"evidence\": string|null}, "
        "\"paid_amount\": {\"value\": string|null, \"evidence\": string|null}, "
        "\"check_or_eft_number\": {\"value\": string|null, \"evidence\": string|null}, "
        "\"payment_date\": {\"value\": string|null, \"evidence\": string|null}}, \"declined\": [\"field_name\"]}. "
        "Evidence must be an exact substring copied from the speaker's message. Do not infer, paraphrase, or supply any value not explicitly spoken. "
        "If a field is absent, set value to null and evidence to null. "
        "If the rep says they do not know or cannot provide a field, add that field name to declined. "
        "Do not include explanations or markdown. Return valid JSON only."
    )
    result = await llm_call(system, rep_text)
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
            value = _validate(name, item.get("value"), evidence)
            if value is not None:
                extracted[name] = value

    declined = result.get("declined", []) if isinstance(result, dict) else []
    extracted["_declined"] = [name for name in declined if name in FIELDS]
    return extracted