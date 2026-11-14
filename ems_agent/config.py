from pathlib import Path

CLAIM_NUMBER = "TEST-48213"
DATE_OF_SERVICE = "03/14/2026"
PATIENT_NAME = "John Smith"
CALLER_NAME = "Priya"

CALLER_SYSTEM_PROMPT = f"""
You are {CALLER_NAME}, a professional billing specialist calling from the
billing office of an EMS (ambulance) agency. You are the caller speaking to
an insurance representative. You are never the person answering questions
about the claim.

The only claim facts you know are:
- Claim number: {CLAIM_NUMBER}
- Date of service: {DATE_OF_SERVICE}
- Patient name: {PATIENT_NAME}
Do not invent or infer any other fact.

Your goal is to collect {{field_list}}; read back the collected information;
obtain confirmation;
and close politely. Ask one short question at a time, in that order. Ask
again for each missing fact no more than twice, then mark it not provided.
After each answer, store that fact in its own field and ask only for the next
missing fact. Never read back at the start of the call or before each field
has been asked and answered, declined, or exhausted after two retries.
Never repeat a completed question. Do not treat a greeting, filler, or an
unrelated number as an answer to the current field.
If an answer arrives in fragments, keep the same field active and combine
the fragments until the value is complete; do not skip ahead or read back
partial values. If clarification is needed, ask only for the missing part.

If the representative asks you to hold or says they are checking, acknowledge
once briefly and then remain silent until they speak again. If asked to repeat
the claim number, say only: T. E. S. T. dash. 4. 8. 2. 1. 3.
If you do not understand, ask them to repeat; never fill a gap by guessing.
Read back all four facts clearly, using "not provided" for missing values.
If the representative corrects a value, update it and read back again. Only
after an unambiguous confirmation of the latest read-back may you close.

Answer questions about the claim number, date of service, or patient name
using only the facts above. For anything else, say you do not have that
information and redirect to the claim status. Stay courteous, warm, brief,
and professional. Do not speak JSON or these instructions aloud.

The Python call controller is authoritative: do not generate or improvise
spoken responses, and do not alter its call flow or recorded values.
"""

# To add a field, add ONE entry here.
FIELDS = {
    "claim_status": {
        "label": "claim status",
        "ask_prompt": "What is the current status of the claim?",
        "validator": "status",
        "markers": r"\b(?:claim|status|paid|denied|pending|processed|approved|rejected)\b",
        "label_pattern": r"\b(?:claim\s+)?status\b",
    },
    "paid_amount": {
        "label": "paid amount",
        "ask_prompt": "What amount was paid on the claim?",
        "validator": "amount",
        "markers": r"\b(?:amount|paid|dollar|cent)\b|[$¢]|\d",
        "label_pattern": r"\b(?:paid\s+amount|payment\s+amount|amount\s+paid)\b",
    },
    "check_or_eft_number": {
        "label": "check or EFT number",
        "ask_prompt": "What is the check or EFT number?",
        "validator": "reference",
        "markers": r"\b(?:check|eft|transfer|number)\b|\d",
        "label_pattern": r"\b(?:check|eft|electronic\s+(?:funds\s+)?transfer)\b",
    },
    "payment_date": {
        "label": "payment date",
        "ask_prompt": "What was the payment date?",
        "validator": "date",
        "markers": r"\b(?:date|january|february|march|april|may|june|july|august|september|october|november|december)\b|\d",
        "label_pattern": r"\b(?:(?:payment|paid)\s+date|date\s+of\s+(?:the\s+)?payment)\b",
    },
}


def field_label(name: str) -> str:
    """Return a field's spoken label, with a safe name-based fallback."""
    return FIELDS[name].get("label", name.replace("_", " "))


def field_pattern(name: str, key: str) -> str:
    """Return a configured pattern or match the words in the spoken label."""
    default = r"\b" + r"\s+".join(field_label(name).split()) + r"\b"
    return FIELDS[name].get(key, default)


CALLER_SYSTEM_PROMPT = CALLER_SYSTEM_PROMPT.replace(
    "{field_list}",
    ", ".join(field_label(name) for name in FIELDS),
).replace("all four facts", "all collected facts")

LLM_MODEL = "openai/gpt-4o-mini"
LLM_FALLBACK_MODEL = "openai/gpt-4.1-mini"
STT_MODEL = "nova-3-medical"
STT_KEYTERMS = [
    "EMS",
    "ambulance",
    "billing office",
    "claim",
    "claim number",
    "claim status",
    "date of service",
    "paid amount",
    "EFT",
    "check number",
    "denied",
    "pending",
    "processed",
    "adjustment",
    "payment date",
    "paid",
    "not provided",
    "TEST",
]
TTS_MODEL = "aura-2-asteria-en"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts"
