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


def is_filler(text: str) -> bool:
    """Return true for acknowledgements that are not answers to a field question."""
    normalized = re.sub(r"[^a-z\s]", "", text.casefold()).strip()
    phrases = (
        "okay", "ok", "go ahead", "okay go ahead", "thanks", "thank you", "thanks for waiting",
        "sorry for the wait", "let me pull that up", "hello", "hi",
    )
    return normalized in phrases or normalized.startswith(("thanks", "thank you", "sorry for the wait"))


def is_confirmation(text: str) -> bool:
    """Recognize a short, unambiguous confirmation without fact content."""
    normalized = re.sub(r"[^\w\s']", "", text.casefold()).strip()
    words = normalized.split()
    affirmations = (
        "yes", "yeah", "yep", "yup", "correct", "right", "exactly", "sure",
        "okay", "ok", "that's correct", "that's right", "sounds good", "all good",
        "perfect", "uh huh",
    )
    fact_words = ("paid", "denied", "pending", "amount", "check", "eft", "date", "status")
    exclusions = ("but", "not", "wrong", "incorrect", "change", "actually", "correct something", "no")
    return (
        len(words) <= 10
        and not re.search(r"\d", normalized)
        and normalized.startswith(affirmations)
        and not any(re.search(rf"\b{re.escape(word)}\b", normalized) for word in fact_words)
        and not any(re.search(rf"\b{re.escape(word)}\b", normalized) for word in exclusions)
    )


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
            "slow down",
            "spell",
            "spell the claim",
            "character by character",
            "letter by letter",
            "digit by digit",
            "one character at a time",
            "one letter at a time",
            "one digit at a time",
            "say the claim number",
            "repeat the claim number",
            "say the claim number again",
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
            "correct the",
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
    if is_confirmation(text):
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
    if intent == Intent.CONFIRM and not is_confirmation(text):
        return Intent.OTHER
    return intent
