import asyncio
import json
import re
import sys
from datetime import datetime
from typing import Any, Awaitable, Callable

from .config import CLAIM_NUMBER, DATE_OF_SERVICE, FIELDS, field_label
from .extractor import _spoken_amount
from .llm import chat_json

SCORE_KEYS = (
    "introduced_as_billing_office",
    "claim_number_stated",
    "all_four_facts_collected",
    "read_back_done",
    "no_invented_data",
)

REVIEWER_SYSTEM_PROMPT = """
You are a strict QA reviewer. Below is the transcript of a phone call
between a CALLER (EMS billing agent) and a REPRESENTATIVE (insurance).
Judge only what actually happened in the transcript, not what should
have happened.

Return ONLY JSON with exactly these keys, each true or false:
{
  "introduced_as_billing_office": true or false,
  "claim_number_stated": true or false,
  "all_four_facts_collected": true or false,
  "read_back_done": true or false,
  "no_invented_data": true or false
}

Evaluate the keys literally:
- introduced_as_billing_office: the caller said they are from the billing
  office of an EMS agency.
- claim_number_stated: the caller said claim number TEST-48213.
- all_four_facts_collected: the representative actually provided the claim
  status, paid amount, check or EFT number, and payment date. A fact is not
  collected if it was not spoken by the representative, even if the caller
  guessed it, supplied it from elsewhere, or said "not provided."
- read_back_done: before ending, the caller read back the captured facts.
- no_invented_data: every value the caller stated or recorded appears in
  the representative's words. If the caller states a value the
  representative never said, including a made-up date of birth, this is
  false.

Rules:
- If a fact was never said by the representative, all_four_facts_collected
  is false.
- Be literal. When unsure, answer false.
- Output JSON only, with no explanation or markdown.
"""


def _normalized_text(text: str) -> str:
    """Remove punctuation so spoken and printed claim numbers compare equally."""
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def _dates_in_text(text: str) -> set[datetime.date]:
    """Parse numeric and named-month dates stated in one transcript turn."""
    dates = set()
    for match in re.finditer(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b", text):
        for fmt in ("%m/%d/%Y", "%m-%d-%Y"):
            try:
                dates.add(datetime.strptime(match.group(), fmt).date())
                break
            except ValueError:
                continue

    month_names = "|".join(
        ("January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December")
    )
    pattern = rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+(\d{{4}})\b"
    for match in re.finditer(pattern, text, re.IGNORECASE):
        try:
            dates.add(datetime.strptime(" ".join(match.groups()), "%B %d %Y").date())
        except ValueError:
            continue
    return dates


def _numbers_in_text(text: str) -> set[float]:
    """Return numeric values after excluding complete date strings."""
    without_dates = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b", "", text)
    numbers = set()
    for match in re.finditer(r"\b\d+(?:\.\d+)?\b", without_dates):
        numbers.add(float(match.group()))
    return numbers


def _rep_numbers(rep_turns: list[str]) -> set[float]:
    """Collect printed and spoken monetary values from representative turns."""
    numbers = set()
    for turn in rep_turns:
        numbers.update(_numbers_in_text(turn))
        spoken = _spoken_amount(turn)
        if spoken is not None:
            numbers.add(float(spoken))
    return numbers


def _all_facts_in_rep(rep_text: str) -> bool:
    """Use conservative transcript evidence when the reviewer model is unavailable."""
    has_status = bool(re.search(r"\b(?:paid|denied|pending|processed|approved|rejected)\b", rep_text, re.I))
    has_amount = bool(re.search(r"\$\s*\d|\b\d+(?:\.\d+)?\s+dollars?\b", rep_text, re.I))
    has_reference = bool(re.search(r"\b(?:check|eft)\b.*\b[A-Z0-9-]+\b", rep_text, re.I))
    return has_status and has_amount and has_reference and bool(_dates_in_text(rep_text))


def _has_no_invented_data(agent_turns: list[str], rep_turns: list[str]) -> bool:
    """Require each caller number or date to be known or stated by the rep."""
    allowed_numbers = _rep_numbers(rep_turns)
    allowed_numbers.add(float(re.sub(r"\D", "", CLAIM_NUMBER)))
    allowed_numbers.update(float(digit) for digit in CLAIM_NUMBER if digit.isdigit())
    allowed_dates = set().union(*(_dates_in_text(turn) for turn in rep_turns))
    allowed_dates.update(_dates_in_text(DATE_OF_SERVICE))

    for turn in agent_turns:
        for date in _dates_in_text(turn):
            if date not in allowed_dates:
                return False
        for number in _numbers_in_text(turn):
            if number not in allowed_numbers:
                return False
    return True


async def review(
    transcript: list[tuple[str, str]],
    llm_call: Callable[[str, str], Awaitable[dict[str, Any]]] = chat_json,
) -> dict[str, bool]:
    text = "\n".join(
        f"{'CALLER' if speaker == 'agent' else 'REPRESENTATIVE'}: {line}"
        for speaker, line in transcript
    )
    result = await llm_call(
        REVIEWER_SYSTEM_PROMPT,
        text,
    )
    agent_turns = [line for speaker, line in transcript if speaker == "agent"]
    rep_turns = [line for speaker, line in transcript if speaker == "rep"]
    agent_text = " ".join(agent_turns)
    rep_text = " ".join(rep_turns)
    agent_text_normalized = _normalized_text(agent_text)
    has_readback = any(
        "is that correct" in line.casefold()
        and any(speaker == "rep" for speaker, _line in transcript[:index])
        for index, (speaker, line) in enumerate(transcript)
        if speaker == "agent"
    )
    code_all_four = _all_facts_in_rep(rep_text)
    code_no_invented = _has_no_invented_data(agent_turns, rep_turns)

    if not result:
        print("Warning: reviewer model failed; using transcript checks.", file=sys.stderr)
        model_all_four = code_all_four
        model_no_invented = code_no_invented
    else:
        model_all_four = result.get("all_four_facts_collected") is True
        model_no_invented = result.get("no_invented_data") is True

    return {
        "introduced_as_billing_office": (
            "billing office" in agent_text.casefold() and "ems" in agent_text.casefold()
        ),
        "claim_number_stated": "test48213" in agent_text_normalized,
        "all_four_facts_collected": model_all_four and code_all_four,
        "read_back_done": has_readback,
        "no_invented_data": model_no_invented and code_no_invented,
    }


def transcript_from_jsonl(path: str) -> list[tuple[str, str]]:
    turns = []
    with open(path, encoding="utf-8") as transcript_file:
        for line in transcript_file:
            try:
                item = json.loads(line)
                if item.get("speaker") in {"agent", "rep"} and isinstance(item.get("text"), str):
                    turns.append((item["speaker"], item["text"]))
            except (json.JSONDecodeError, AttributeError):
                continue
    return turns


async def main() -> None:
    """Print a reviewer scorecard for one transcript JSONL file."""
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m ems_agent.reviewer path\\to\\call.jsonl")
    scorecard = await review(transcript_from_jsonl(sys.argv[1]))
    print(json.dumps(scorecard))


if __name__ == "__main__":
    asyncio.run(main())
