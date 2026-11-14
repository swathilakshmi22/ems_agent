import json
import re
from typing import Any, Awaitable, Callable

from .config import CLAIM_NUMBER, FIELDS
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


async def review(
    transcript: list[tuple[str, str]],
    llm_call: Callable[[str, str], Awaitable[dict[str, Any]]] = chat_json,
) -> dict[str, bool]:
    text = "\n".join(f"{speaker}: {line}" for speaker, line in transcript)
    result = await llm_call(
        REVIEWER_SYSTEM_PROMPT,
        text,
    )
    score = {key: result.get(key) is True for key in SCORE_KEYS}
    agent_turns = [line.casefold() for speaker, line in transcript if speaker == "agent"]
    rep_turns = [line.casefold() for speaker, line in transcript if speaker == "rep"]
    agent_text = " ".join(agent_turns)
    rep_text = " ".join(rep_turns)

    score["introduced_as_billing_office"] &= "billing office" in agent_text
    score["claim_number_stated"] &= CLAIM_NUMBER.casefold() in agent_text
    score["read_back_done"] &= "is that correct" in agent_text
    if "not provided" in agent_text or not all(
        name.replace("_", " ") in agent_text for name in FIELDS
    ):
        score["all_four_facts_collected"] = False

    spoken_numbers = re.findall(r"\b\d+(?:[.,/]\d+)*\b", agent_text)
    allowed_numbers = re.findall(r"\b\d+(?:[.,/]\d+)*\b", rep_text)
    allowed_numbers += re.findall(r"\b\d+(?:[.,/]\d+)*\b", f"{CLAIM_NUMBER} 03/14/2026")
    score["no_invented_data"] &= all(number in allowed_numbers for number in spoken_numbers)
    return score


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