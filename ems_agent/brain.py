import json
import re
from pathlib import Path
from typing import Any, Awaitable, Callable

from .config import (
    CALLER_NAME,
    CLAIM_NUMBER,
    DATE_OF_SERVICE,
    FIELDS,
    PATIENT_NAME,
)
from .extractor import extract_facts
from .intents import Intent, classify_intent
from .state import CallState

Extractor = Callable[[str], Awaitable[dict[str, Any]]]
IntentClassifier = Callable[[str], Awaitable[Intent]]


class CallingAgent:
    def __init__(
        self,
        transcript_path: Path | None = None,
        extractor: Extractor = extract_facts,
        intent_classifier: IntentClassifier = classify_intent,
    ) -> None:
        self.state = CallState()
        self.extractor = extractor
        self.intent_classifier = intent_classifier
        self.transcript_path = transcript_path

    def _log(self, speaker: str, text: str) -> None:
        self.state.transcript.append((speaker, text))
        if self.transcript_path:
            self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
            with self.transcript_path.open("a", encoding="utf-8") as transcript_file:
                transcript_file.write(json.dumps({"speaker": speaker, "text": text}) + "\n")

    def opening(self) -> str:
        self.state.introduced = True
        message = (
            f"Hello, this is {CALLER_NAME} from the billing office of an EMS agency. "
            f"I'm calling about claim {CLAIM_NUMBER}, date of service {DATE_OF_SERVICE}."
        )
        self._log("agent", message)
        return message

    def _readback(self) -> str:
        labels = {
            "claim_status": "the claim status",
            "paid_amount": "the paid amount",
            "check_or_eft_number": "the check or EFT number",
            "payment_date": "the payment date",
        }
        facts = []
        for name in FIELDS:
            value = self.state.values[name]
            if value is None:
                value = "not provided"
            elif name == "paid_amount":
                value = f"${value}"
            facts.append(f"{labels[name]} is {value}")
        self.state.readback_done = True
        return f"To confirm, {', '.join(facts)}. Is that correct?"

    def _known_data_answer(self, rep_text: str) -> str | None:
        lowered = rep_text.casefold()
        if "?" not in rep_text:
            return None
        if "claim number" in lowered:
            return f"The claim number is {CLAIM_NUMBER}."
        if "date of service" in lowered:
            return f"The date of service is {DATE_OF_SERVICE}."
        if "patient name" in lowered or "who is the patient" in lowered:
            return f"The patient's name is {PATIENT_NAME}."
        unavailable = (
            "date of birth",
            "dob",
            "member id",
            "social security",
            "ssn",
            "npi",
            "tax id",
            "address",
            "billed amount",
        )
        if any(term in lowered for term in unavailable):
            return "I don't have that information in front of me. Could we continue with the claim status?"
        return None

    @staticmethod
    def _field_label(name: str) -> str:
        return {
            "claim_status": "claim status",
            "paid_amount": "paid amount",
            "check_or_eft_number": "check or EFT number",
            "payment_date": "payment date",
        }[name]

    def _ask_next_missing(self) -> str | None:
        if self.state.active_field in self.state.missing_fields():
            return FIELDS[self.state.active_field]["ask_prompt"]
        self.state.active_field = next(iter(self.state.missing_fields()), None)
        if self.state.active_field is None:
            return None
        return FIELDS[self.state.active_field]["ask_prompt"]

    def _retry_active_field(self, count_retry: bool = True) -> str | None:
        name = self.state.active_field
        if name is None:
            return self._ask_next_missing()
        if count_retry and self.state.retries[name] >= 2:
            self.state.declined.add(name)
            self.state.active_field = None
            self.state.field_buffers.pop(name, None)
            return self._ask_next_missing()
        if count_retry:
            self.state.retries[name] += 1
        label = self._field_label(name)
        if self.state.field_buffers.get(name):
            return self._clarification_prompt(name, True)
        if self.state.retries[name] == 1:
            return f"Could you tell me the {label}, please?"
        return f"I may have missed it. Could you repeat the {label}, please?"

    @staticmethod
    def _relevant_fragment(name: str, text: str, has_buffer: bool) -> bool:
        if has_buffer:
            return bool(re.search(r"[a-z0-9]", text, re.IGNORECASE))
        markers = {
            "claim_status": r"\b(?:claim|status|paid|denied|pending|processed|approved|rejected)\b",
            "paid_amount": r"\b(?:amount|paid|dollar|cent)\b|[$¢]|\d",
            "check_or_eft_number": r"\b(?:check|eft|transfer|number)\b|\d",
            "payment_date": r"\b(?:date|january|february|march|april|may|june|july|august|september|october|november|december)\b|\d",
        }
        return re.search(markers[name], text, re.IGNORECASE) is not None

    @staticmethod
    def _clarification_prompt(name: str, has_buffer: bool) -> str:
        if name == "payment_date" and has_buffer:
            return "I heard part of the date. Could you give me the full date, including the year?"
        if name == "paid_amount":
            return "I didn't catch the amount. Could you say the amount in dollars and cents?"
        return f"Please repeat just the {CallingAgent._field_label(name)}."

    @staticmethod
    def _explicitly_labeled(name: str, text: str) -> bool:
        patterns = {
            "claim_status": r"\b(?:claim\s+)?status\b",
            "paid_amount": r"\b(?:paid\s+amount|payment\s+amount|amount\s+paid)\b",
            "check_or_eft_number": r"\b(?:check|eft|electronic\s+(?:funds\s+)?transfer)\b",
            "payment_date": r"\b(?:(?:payment|paid)\s+date|date\s+of\s+(?:the\s+)?payment)\b",
        }
        return re.search(patterns[name], text, re.IGNORECASE) is not None

    def _apply_extracted(self, extracted: dict[str, Any], rep_text: str) -> dict[str, Any]:
        accepted = {}
        expected = self.state.correction_field or self.state.active_field
        for name, value in extracted.items():
            if name not in FIELDS or value is None:
                continue
            if expected and name != expected and not self._explicitly_labeled(name, rep_text):
                continue
            self.state.values[name] = value
            self.state.declined.discard(name)
            accepted[name] = value
            self.state.field_buffers.pop(name, None)
        return accepted

    async def respond(self, rep_text: str) -> str:
        if self.state.ended:
            return ""
        self._log("rep", rep_text)
        known_answer = self._known_data_answer(rep_text)
        if known_answer:
            return self._say(known_answer)

        intent = await self.intent_classifier(rep_text)

        if intent == Intent.HOLD:
            self.state.on_hold = True
            if not self.state.hold_acknowledged:
                self.state.hold_acknowledged = True
                return self._say("Sure, thank you.")
            return ""

        if intent == Intent.SPELL_SLOWLY:
            self.state.on_hold = False
            spelled = ". ".join(
                "dash" if character == "-" else character
                for character in CLAIM_NUMBER.upper()
            ) + "."
            return self._say(f"{spelled} Do you have it?")

        self.state.on_hold = False
        expected_field = self.state.correction_field or self.state.active_field
        previous_buffer = self.state.field_buffers.get(expected_field, "") if expected_field else ""
        extraction_text = rep_text
        if expected_field:
            if previous_buffer and self._relevant_fragment(expected_field, rep_text, True):
                extraction_text = " ".join(part for part in (previous_buffer, rep_text) if part)
        extracted = await self.extractor(extraction_text)
        declined = extracted.pop("_declined", [])
        accepted = self._apply_extracted(extracted, rep_text)

        for name in declined:
            if (
                name in FIELDS
                and (
                    name == self.state.active_field
                    or self._explicitly_labeled(name, rep_text)
                )
                and self.state.values[name] is None
            ):
                self.state.declined.add(name)
                self.state.field_buffers.pop(name, None)
                if self.state.active_field == name:
                    self.state.active_field = None

        if self.state.active_field in accepted:
            self.state.active_field = None

        if self.state.readback_done and accepted:
            self.state.readback_done = False
            self.state.awaiting_correction = False
            self.state.correction_field = None
            return self._say(self._readback())

        if self.state.readback_done and intent == Intent.CONFIRM:
            self.state.ended = True
            return self._say("Thank you so much for your help today. Have a great day. Goodbye.")

        if self.state.readback_done and intent == Intent.CORRECTION:
            field_name = next(
                (name for name in FIELDS if self._explicitly_labeled(name, rep_text)),
                None,
            )
            if field_name:
                self.state.correction_field = field_name
                self.state.awaiting_correction = True
                self.state.correction_attempts = 0
                return self._say(f"What is the corrected {self._field_label(field_name)}?")
            self.state.awaiting_correction = True
            self.state.correction_attempts = 0
            return self._say("Which detail should I change: the status, amount, check or EFT number, or payment date?")

        if self.state.readback_done:
            if self.state.awaiting_correction and self.state.correction_field:
                field_name = self.state.correction_field
                grew = len(extraction_text) > len(previous_buffer)
                if self._relevant_fragment(field_name, rep_text, bool(previous_buffer)):
                    self.state.field_buffers[field_name] = extraction_text
                if not grew:
                    self.state.correction_attempts += 1
                if self.state.correction_attempts >= 2:
                    self.state.awaiting_correction = False
                    self.state.correction_field = None
                    self.state.correction_attempts = 0
                    self.state.field_buffers.pop(field_name, None)
                    return self._say(
                        f"I couldn't hear the correction, so I kept the recorded {self._field_label(field_name)}. "
                        "Is the read-back correct?"
                    )
                return self._say(self._clarification_prompt(field_name, bool(self.state.field_buffers.get(field_name))))
            return self._say("I didn't catch that. Are the read-back details correct, or would you like to change one?")

        if self.state.correction_field and self.state.correction_field in accepted:
            self.state.correction_field = None
            self.state.awaiting_correction = False

        if intent == Intent.DONT_KNOW:
            named = [name for name in FIELDS if self._explicitly_labeled(name, rep_text)]
            declined_fields = named or ([self.state.active_field] if self.state.active_field else [])
            for name in declined_fields:
                if name in FIELDS and self.state.values[name] is None:
                    self.state.declined.add(name)
                    if self.state.active_field == name:
                        self.state.active_field = None

        if not accepted and intent == Intent.CORRECTION:
            field_name = next(
                (name for name in FIELDS if self._explicitly_labeled(name, rep_text)),
                self.state.active_field,
            )
            if field_name:
                self.state.correction_field = field_name
                self.state.awaiting_correction = True
                self.state.correction_attempts = 0
                return self._say(f"What is the corrected {self._field_label(field_name)}?")

        if self.state.awaiting_correction and self.state.correction_field:
            field_name = self.state.correction_field
            grew = len(extraction_text) > len(previous_buffer)
            if self._relevant_fragment(field_name, rep_text, bool(previous_buffer)):
                self.state.field_buffers[field_name] = extraction_text
            if not grew:
                self.state.correction_attempts += 1
            if self.state.correction_attempts >= 2:
                self.state.awaiting_correction = False
                self.state.correction_field = None
                self.state.correction_attempts = 0
                self.state.field_buffers.pop(field_name, None)
                return self._say(
                    "I couldn't hear the correction, so I kept the recorded value. "
                    "Is the read-back correct?"
                )
            return self._say(self._clarification_prompt(field_name, bool(self.state.field_buffers.get(field_name))))

        if self.state.active_field and not accepted:
            name = self.state.active_field
            relevant = self._relevant_fragment(name, rep_text, bool(previous_buffer))
            grew = len(extraction_text) > len(previous_buffer)
            if relevant:
                self.state.field_buffers[name] = extraction_text
            question = self._retry_active_field(count_retry=not (relevant and grew))
        else:
            question = self._ask_next_missing()
            if (
                self.state.active_field
                and expected_field is None
                and not accepted
                and self._relevant_fragment(self.state.active_field, rep_text, False)
            ):
                self.state.field_buffers[self.state.active_field] = rep_text
                question = self._clarification_prompt(self.state.active_field, True)
        if question:
            return self._say(f"Thank you. {question}" if accepted else question)
        if self.state.collection_complete():
            return self._say(self._readback())
        return self._say("Could you repeat that, please?")

    def _say(self, text: str) -> str:
        if text:
            self._log("agent", text)
        return text