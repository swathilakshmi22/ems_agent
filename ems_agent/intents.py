from enum import Enum
import re
from typing import Any, Awaitable, Callable

from .llm import chat_json


class Intent(str, Enum):
    HOLD = "HOLD"
    SPELL_SLOWLY = "SPELL_SLOWLY"
    PROVIDES_FACTS = "PROVIDES_FACTS"
    CORRECTION = "CORRECTION"
    DONT_KNOW = "DONT_KNOW"
    CONFIRM = "CONFIRM"
    GOODBYE = "GOODBYE"
    OTHER = "OTHER"


async def classify_intent(
    text: str,
    llm_call: Callable[[str, str], Awaitable[dict[str, Any]]] = chat_json,
) -> Intent:
    normalized = text.casefold()
    if any(
        term in normalized
        for term in (
            "please hold",
            "one moment",
            "just a moment",
            "hold on",
            "wait a minute",
            "let me check",
            "i'm checking",
            "i am checking",
        )
    ):
        return Intent.HOLD
    if any(
        term in normalized
        for term in (
            "slowly",
            "spell the claim",
            "say the claim number",
            "one character",
            "repeat that",
            "repeat again",
            "say that again",
        )
    ):
        return Intent.SPELL_SLOWLY
    if any(term in normalized for term in ("don't know", "do not know", "not sure", "unknown")):
        return Intent.DONT_KNOW
    if any(
        term in normalized
        for term in (
            "actually",
            "correction",
            "correct something",
            "want to correct",
            "change the",
            "i meant",
            "that's incorrect",
            "that is incorrect",
            "not correct",
            "that's wrong",
            "that is wrong",
        )
    ):
        return Intent.CORRECTION
    confirmation = re.sub(r"[^\w\s']", "", normalized).strip()
    clear_confirmations = {
        "yes",
        "yeah",
        "yep",
        "right",
        "correct",
        "that's correct",
        "that is correct",
        "yes that's correct",
        "yes that is correct",
        "yes that's right",
        "yes that is right",
        "yeah that's right",
        "yeah that is right",
        "yeah it's correct",
        "yeah it is correct",
        "yes it's correct",
        "yes it is correct",
    }
    if confirmation in clear_confirmations:
        return Intent.CONFIRM
    if any(term in normalized for term in ("goodbye", "that's all", "that is all", "have a good day", "hang up")):
        return Intent.GOODBYE
    if normalized.strip(" .,!") in {"hello", "hi", "hello there"}:
        return Intent.OTHER

    result = await llm_call(
        "You classify one latest turn from an insurance representative in an EMS claim-status call. "
        "Return exactly one intent: PROVIDES_FACTS, CORRECTION, DONT_KNOW, CONFIRM, GOODBYE, or OTHER. "
        "Choose CONFIRM only when the representative clearly confirms the caller's immediately preceding complete read-back. "
        "A casual 'yes', 'correct', or 'right' during fact collection is not confirmation. "
        "If a turn includes a fact, correction, or both confirmation words and a new fact, classify it as PROVIDES_FACTS "
        "or CORRECTION, never CONFIRM. 'Let me check' and hold requests are not facts. "
        "Use only the representative's actual wording. Return JSON only in the shape {\"intent\": \"...\"}.",
        text,
    )
    try:
        intent = Intent(result.get("intent", "OTHER"))
    except (ValueError, AttributeError):
        return Intent.OTHER
    if intent == Intent.CONFIRM and confirmation not in clear_confirmations:
        return Intent.OTHER
    return intent