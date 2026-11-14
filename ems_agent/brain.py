"""Deterministic call controller.

Phase diagram: OPENING -> COLLECTING -> READBACK -> CLOSED.
Corrections return READBACK to COLLECTING only while a value is being repaired.
"""

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
    field_label,
    field_pattern,
)
from .extractor import extract_facts
from .intents import Intent, classify_intent, is_filler
from .state import CallState, Phase

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
        self.state.phase = Phase.COLLECTING
        message = (
            f"Hello, this is {CALLER_NAME} from the billing office of an EMS agency. "
            f"I'm calling about claim {CLAIM_NUMBER}, date of service {DATE_OF_SERVICE}."
        )
        self._log("agent", message)
        return message

    def _readback(self) -> str:
        facts = []
        for name in FIELDS:
            value = self.state.values[name]
            if value is None:
                value = "not provided"
            elif name == "paid_amount":
                value = f"${value}"
            facts.append(f"the {field_label(name)} is {value}")
        self.state.readback_done = True
        self.state.phase = Phase.READBACK
        self.state.confirm_attempts = 0
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
    def _spelled_claim_number() -> str:
        """Return the fixed claim number one character at a time."""
        characters = ["dash" if character == "-" else character for character in CLAIM_NUMBER]
        return ". ".join(characters) + "."

    @staticmethod
    def _field_label(name: str) -> str:
        return field_label(name)

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
        return re.search(field_pattern(name, "markers"), text, re.IGNORECASE) is not None

    @staticmethod
    def _clarification_prompt(name: str, has_buffer: bool) -> str:
        if name == "payment_date" and has_buffer:
            return "I heard part of the date. Could you give me the full date, including the year?"
        if name == "paid_amount":
            return "I didn't catch the amount. Could you say the amount in dollars and cents?"
        return f"Please repeat just the {CallingAgent._field_label(name)}."

    @staticmethod
    def _explicitly_labeled(name: str, text: str) -> bool:
        return re.search(field_pattern(name, "label_pattern"), text, re.IGNORECASE) is not None

    @staticmethod
    def _has_currency_evidence(text: str) -> bool:
        return bool(re.search(r"[$¢]|\b(?:dollars?|cents?)\b", text, re.IGNORECASE))

    @staticmethod
    def _is_plain_no(text: str) -> bool:
        return re.sub(r"[^a-z]", "", text.casefold()) == "no"

    def _handle_hold(self) -> str:
        """Acknowledge the first hold request and silence later ones."""
        self.state.on_hold = True
        if self.state.hold_acknowledged:
            return ""
        self.state.hold_acknowledged = True
        return self._say("Sure, thank you.")

    def _handle_spelling(self) -> str:
        """Speak the code-controlled claim-number spelling."""
        self.state.on_hold = False
        return self._say(f"{self._spelled_claim_number()} Do you have it?")

    def _handle_closed(self) -> str:
        """Close the state before returning the scripted farewell."""
        self.state.ended = True
        self.state.phase = Phase.CLOSED
        return self._say("Thank you so much for your help today. Have a great day. Goodbye.")

    def _apply_extracted(self, extracted: dict[str, Any], rep_text: str) -> dict[str, Any]:
        accepted = {}
        correction_mode = self.state.correction_field is not None or (
            self.state.readback_done
            and bool(re.search(r"\b(?:actually|correction|change|wrong|incorrect)\b", rep_text, re.IGNORECASE))
        )
        for name, value in extracted.items():
            if name not in FIELDS or value is None:
                continue
            labelled = self._explicitly_labeled(name, rep_text)
            active = name == self.state.active_field
            validator = FIELDS[name]["validator"]
            if self.state.values[name] is not None and not (labelled or correction_mode):
                continue
            # Facts can arrive together. Type-based assignment keeps valid values
            # without treating an unlabeled number as every possible field.
            if validator == "amount" and not (active or self._has_currency_evidence(rep_text)):
                continue
            if validator == "reference":
                compact_value = re.sub(r"[^a-z0-9]", "", str(value).casefold())
                forbidden = {
                    re.sub(r"[^a-z0-9]", "", CLAIM_NUMBER.casefold()),
                    re.sub(r"[^a-z0-9]", "", DATE_OF_SERVICE.casefold()),
                }
                reference_words = r"\b(?:check|eft|trace|reference|number)\b"
                if compact_value in forbidden:
                    continue
                if not (active or re.search(reference_words, rep_text, re.IGNORECASE)):
                    continue
                if active and (self._has_currency_evidence(rep_text) or "/" in str(value)):
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
        intent = await self.intent_classifier(rep_text)

        if intent == Intent.HOLD and len(rep_text.split()) < 12:
            return self._handle_hold()

        if intent == Intent.SPELL_SLOWLY:
            return self._handle_spelling()

        known_answer = self._known_data_answer(rep_text)
        if known_answer:
            return self._say(known_answer)

        if not self.state.readback_done and "?" in rep_text and intent in {Intent.OTHER, Intent.PROVIDES_FACTS}:
            lowered = rep_text.casefold()
            question = self._ask_next_missing() or "What is the current status of the claim?"
            if "who am i speaking with" in lowered:
                return self._say(f"This is {CALLER_NAME} from the billing office. {question}")
            if "what agency" in lowered:
                return self._say(f"An EMS agency. {question}")
            return self._say(f"I only have the claim details in front of me. {question}")

        self.state.on_hold = False
        expected_field = self.state.correction_field or self.state.active_field
        previous_buffer = self.state.field_buffers.get(expected_field, "") if expected_field else ""
        extraction_text = rep_text
        if expected_field:
            if previous_buffer and self._relevant_fragment(expected_field, rep_text, True):
                extraction_text = " ".join(part for part in (previous_buffer, rep_text) if part)
        if self.extractor is extract_facts:
            extracted = await self.extractor(extraction_text, active_field=expected_field)
        else:
            extracted = await self.extractor(extraction_text)
        declined = extracted.pop("_declined", [])
        accepted = self._apply_extracted(extracted, rep_text)

        if intent == Intent.GOODBYE:
            self.state.ended = True
            self.state.phase = Phase.CLOSED
            if any(value is not None for value in self.state.values.values()):
                return self._say(self._readback() + " Thank you. Goodbye.")
            return self._say("Thank you for your time. Goodbye.")

        if self.state.values.get("claim_status") == "denied" and re.search(r"\b(?:no payment|nothing was paid|no payment was issued)\b", rep_text, re.I):
            self.state.values["paid_amount"] = "0.00"
            self.state.declined.update({"check_or_eft_number", "payment_date"})

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
            return self._handle_closed()

        if self.state.readback_done and self._is_plain_no(rep_text):
            self.state.awaiting_correction = True
            self.state.confirm_attempts = 0
            return self._say("Which detail should I change?")

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
            self.state.confirm_attempts += 1
            if self.state.confirm_attempts >= 2:
                return self._say("Shall I go ahead and close the call?")
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
            question = self._retry_active_field(
                count_retry=not (relevant and grew) and not is_filler(rep_text)
            )
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
