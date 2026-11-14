import os
import re
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from .brain import CallingAgent
from . import config
from .config import FIELDS
from .extractor import _normalize_amount, _normalize_date, _validate, extract_facts
from .intents import Intent, classify_intent, is_confirmation
from . import llm
from .llm import chat_json
from .reviewer import REVIEWER_SYSTEM_PROMPT, review
from .voice_agent import _save_results


def fixture_extractor(text: str) -> dict:
    result = {}
    lowered = text.casefold()
    status = re.search(
        r"\b(?:claim\s+(?:status\s+is|is|was)|status(?:\s+is)?)\s+(paid|denied|pending|processed)\b",
        text,
        re.I,
    )
    if status:
        result["claim_status"] = status.group(1).casefold()
    elif lowered.strip(" .,!") in {"paid", "denied", "pending", "processed"}:
        result["claim_status"] = lowered.strip(" .!,")
    amount = re.search(r"\$?\s*(\d+(?:\.\d{2})?)\s*(?:dollars?)?", text, re.I)
    if amount and ("$" in text or "dollar" in lowered):
        result["paid_amount"] = f"{float(amount.group(1)):.2f}"
    date = re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", text)
    if date:
        result["payment_date"] = date.group(0)
    check = re.search(r"(?:check|eft)(?: number| no\.)?\s+(?!number\b)([A-Z0-9-]+)", text, re.I)
    if check:
        result["check_or_eft_number"] = check.group(1)
    if any(term in lowered for term in ("don't know", "do not know", "not sure")):
        result["_declined"] = [
            name for name in FIELDS if name.replace("_", " ") in lowered
        ] or ["check_or_eft_number"]
    return result


async def async_fixture_extractor(text: str) -> dict:
    return fixture_extractor(text)


async def fixture_intent(text: str) -> Intent:
    lowered = text.casefold()
    if "hold" in lowered or "moment" in lowered:
        return Intent.HOLD
    if "slowly" in lowered or "say the claim number" in lowered:
        return Intent.SPELL_SLOWLY
    if any(term in lowered for term in ("don't know", "do not know", "not sure")):
        return Intent.DONT_KNOW
    if any(term in lowered for term in ("actually", "correction", "correct the", "not $", "not correct", "wrong")):
        return Intent.CORRECTION
    if any(term in lowered for term in ("goodbye", "can't help", "cannot help")):
        return Intent.GOODBYE
    if is_confirmation(text):
        return Intent.CONFIRM
    return Intent.PROVIDES_FACTS


async def make_agent() -> CallingAgent:
    return CallingAgent(extractor=async_fixture_extractor, intent_classifier=fixture_intent)


async def replay_extractor(text: str) -> dict:
    """Small fixture-only extractor for obvious status, amount, and date values."""
    result = fixture_extractor(text)
    if re.search(r"\b(?:it|claim|status)\s+(?:was|is)\s+paid\b", text, re.I):
        result["claim_status"] = "paid"
    amount = re.search(r"\b(\d+)\s+dollars?\b", text, re.I)
    if amount:
        result["paid_amount"] = f"{float(amount.group(1)):.2f}"
    date = re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", text)
    if date:
        result["payment_date"] = date.group(0)
    return result


class AgentTests(unittest.IsolatedAsyncioTestCase):
    def test_final_result_is_saved_separately(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            transcript_path = Path(temp_dir) / "call-test.jsonl"
            brain = CallingAgent(transcript_path=transcript_path)
            result = {name: None for name in FIELDS}
            scorecard = {key: False for key in (
                "introduced_as_billing_office", "claim_number_stated",
                "all_four_facts_collected", "read_back_done", "no_invented_data",
            )}
            with patch("ems_agent.voice_agent.RESULTS_DIR", Path(temp_dir)):
                path = _save_results(brain, result, scorecard)
            self.assertEqual(path.name, "call-test-result.json")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["scorecard"], scorecard)

    async def test_spell_slowly_with_question_mark(self):
        agent = CallingAgent(
            extractor=async_fixture_extractor,
            intent_classifier=classify_intent,
        )
        agent.opening()
        for question in (
            "Can you say the claim number slowly?",
            "Could you repeat the claim number one character at a time?",
        ):
            reply = await agent.respond(question)
            self.assertIn("T. E. S. T.", reply)
            self.assertIn("dash", reply)
            self.assertNotIn("The claim number is", reply)

    async def test_plain_claim_number_question(self):
        agent = await make_agent()
        agent.opening()
        self.assertEqual(
            await agent.respond("What is the claim number?"),
            "The claim number is TEST-48213.",
        )

    async def test_new_field_added_to_FIELDS(self):
        added_field = {
            "label": "representative name",
            "ask_prompt": "What is the representative name?",
            "validator": "reference",
        }

        async def extractor(text: str) -> dict:
            result = fixture_extractor(text)
            if "representative name is Ada" in text:
                result["representative_name"] = "Ada"
            return result

        with patch.dict(config.FIELDS, {"representative_name": added_field}):
            agent = CallingAgent(extractor=extractor, intent_classifier=fixture_intent)
            agent.opening()
            reply = await agent.respond(
                "Claim status is paid, paid amount is $120.00, check 88310, "
                "payment date 03/14/2026, representative name is Ada."
            )
            self.assertIn("representative name is Ada", reply)

    async def test_reviewer_no_false_invented_data(self):
        async def all_true(_system, _user):
            return {key: True for key in (
                "introduced_as_billing_office", "claim_number_stated",
                "all_four_facts_collected", "read_back_done", "no_invented_data",
            )}

        transcripts = [
            [("rep", "The paid amount is $260."), ("agent", "The paid amount is $260.00.")],
            [("rep", "The paid amount is two hundred sixty dollars."), ("agent", "The paid amount is 260.00.")],
            [("rep", "The payment date is November 22nd, 2026."), ("agent", "The payment date is 11/22/2026.")],
            [("rep", "Please repeat the claim number."), ("agent", "T. E. S. T. dash. 4. 8. 2. 1. 3.")],
        ]
        for transcript in transcripts:
            score = await review(transcript, llm_call=all_true)
            self.assertTrue(score["no_invented_data"])

    async def test_reviewer_still_flags_invented_amount(self):
        async def all_true(_system, _user):
            return {key: True for key in (
                "introduced_as_billing_office", "claim_number_stated",
                "all_four_facts_collected", "read_back_done", "no_invented_data",
            )}

        score = await review(
            [("rep", "The paid amount is $260."), ("agent", "The paid amount is $999.")],
            llm_call=all_true,
        )
        self.assertFalse(score["no_invented_data"])

    async def test_confirmation_variants(self):
        confirmations = (
            "Yeah. Okay.", "Yes, that's correct, thank you.", "That's right.",
            "Sounds good.", "Correct.",
        )
        for reply_text in confirmations:
            agent = await make_agent()
            agent.opening()
            agent.state.readback_done = True
            await agent.respond(reply_text)
            self.assertTrue(agent.state.ended, reply_text)

        for reply_text in ("Yes but the date is wrong", "That's not correct"):
            agent = await make_agent()
            agent.opening()
            agent.state.readback_done = True
            await agent.respond(reply_text)
            self.assertFalse(agent.state.ended, reply_text)

    async def test_readback_asks_to_close_after_two_unclear_replies(self):
        agent = await make_agent()
        agent.opening()
        agent.state.readback_done = True
        first = await agent.respond("Pardon?")
        second = await agent.respond("Can you clarify?")
        self.assertIn("read-back details", first)
        self.assertEqual(second, "Shall I go ahead and close the call?")
        await agent.respond("Yes.")
        self.assertTrue(agent.state.ended)

    async def test_multi_fact_sentence_while_status_is_active(self):
        async def multi_fact_extractor(_text: str) -> dict:
            return {
                "claim_status": "paid",
                "paid_amount": "120.00",
                "payment_date": "03/14/2026",
            }

        agent = CallingAgent(extractor=multi_fact_extractor, intent_classifier=fixture_intent)
        agent.opening()
        agent.state.active_field = "claim_status"
        reply = await agent.respond("It was paid 120 dollars on 03/14/2026.")
        self.assertEqual(agent.state.values["claim_status"], "paid")
        self.assertEqual(agent.state.values["paid_amount"], "120.00")
        self.assertEqual(agent.state.values["payment_date"], "03/14/2026")
        self.assertIn("check or EFT", reply)

    async def test_not_paid_yet_is_not_status_paid(self):
        async def paid_llm(_system, _user):
            return {"facts": {"claim_status": {"value": "paid", "evidence": "paid"}}}

        extracted = await extract_facts("it has not been paid yet", llm_call=paid_llm)
        self.assertNotIn("claim_status", extracted)
        for text in ("It's paid.", "The claim has been paid."):
            extracted = await extract_facts(text, llm_call=paid_llm)
            self.assertEqual(extracted["claim_status"], "paid")

    def test_validator_evidence_matrix(self):
        cases = (
            ("claim_status", "paid", "paid", "It has not been paid yet.", None),
            ("claim_status", "paid", "paid", "It's paid.", "paid"),
            ("claim_status", "paid in full", "paid in full", "It is paid in full.", "paid in full"),
            ("check_or_eft_number", "5 5 9 0", "5 5 9 0", "The EFT is 5 5 9 0.", "5590"),
            ("check_or_eft_number", "pending", "pending", "The EFT is pending.", None),
            ("paid_amount", "0", "no payment", "No payment was made.", "0.00"),
            ("payment_date", "3/14/26", "3/14/26", "Payment date was 3/14/26.", "03/14/2026"),
            ("payment_date", "03/14/2026", "March 14th 2026", "Paid March 14th 2026.", "03/14/2026"),
        )
        for name, value, evidence, full_text, expected in cases:
            with self.subTest(full_text=full_text):
                self.assertEqual(_validate(name, value, evidence, full_text), expected)

    async def test_regex_extraction_survives_llm_failure(self):
        async def failed_llm(_system, _user):
            raise RuntimeError("OpenRouter unavailable")

        facts = await extract_facts(
            "The claim was denied, paid amount $260.12, EFT 5590, payment date 11/14/2026.",
            llm_call=failed_llm,
        )
        self.assertEqual(facts["claim_status"], "denied")
        self.assertEqual(facts["paid_amount"], "260.12")
        self.assertEqual(facts["check_or_eft_number"], "5590")
        self.assertEqual(facts["payment_date"], "11/14/2026")
        self.assertEqual(await extract_facts("Just a nonsense sentence with nothing useful.", llm_call=failed_llm), {"_declined": []})

    async def test_active_eft_field_accepts_spaced_or_bare_digits(self):
        async def no_llm(_system, _user):
            return {}

        self.assertEqual(
            (await extract_facts("EFT 5 5 9 0 2 1", active_field="check_or_eft_number", llm_call=no_llm))["check_or_eft_number"],
            "559021",
        )
        self.assertEqual(
            (await extract_facts("559021.", active_field="check_or_eft_number", llm_call=no_llm))["check_or_eft_number"],
            "559021",
        )

    async def test_goodbye_ends_call_with_nulls(self):
        agent = await make_agent()
        agent.opening()
        reply = await agent.respond("I can't help, goodbye.")
        self.assertTrue(agent.state.ended)
        self.assertIn("Goodbye", reply)
        self.assertTrue(all(value is None for value in agent.state.values.values()))

    async def test_unknown_rep_question_gets_deflection(self):
        for question in ("How's your day going?", "Who am I speaking with?"):
            agent = await make_agent()
            agent.opening()
            agent.state.active_field = "claim_status"
            before = dict(agent.state.retries)
            reply = await agent.respond(question)
            self.assertEqual(agent.state.retries, before)
            self.assertIn("status", reply)
            self.assertIn("?", reply)

    async def test_chatter_does_not_consume_retries(self):
        agent = await make_agent()
        agent.opening()
        agent.state.active_field = "claim_status"
        for chatter in ("Thanks for waiting.", "Okay, go ahead."):
            await agent.respond(chatter)
        self.assertEqual(agent.state.retries["claim_status"], 0)

    async def test_replay_fixtures(self):
        fixture_paths = Path(__file__).parent.joinpath("fixtures").glob("*.jsonl")
        for path in fixture_paths:
            agent = CallingAgent(extractor=replay_extractor, intent_classifier=fixture_intent)
            agent.opening()
            replies = []
            for line in path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                if item.get("speaker") == "rep":
                    replies.append(await agent.respond(item["text"]))
            self.assertNotEqual(replies[0], "")
            self.assertFalse(agent.state.readback_done and all(value is None for value in agent.state.values.values()))
            for index in range(len(replies) - 2):
                self.assertFalse(
                    replies[index] == replies[index + 1] == replies[index + 2],
                    f"{path.name} repeated {replies[index]!r}",
                )

    async def test_opening_and_all_facts_then_readback(self):
        agent = await make_agent()
        opening = agent.opening()
        self.assertIn("billing office", opening)
        reply = await agent.respond("The claim was paid, $120.00, check 88310, payment date 03/14/2026.")
        self.assertIn("Is that correct?", reply)
        self.assertEqual(agent.state.values["paid_amount"], "120.00")

    async def test_facts_one_at_a_time_and_out_of_order(self):
        agent = await make_agent()
        agent.opening()
        reply = await agent.respond("The payment date is 03/14/2026.")
        self.assertIn("status", reply)
        self.assertEqual(agent.state.values["payment_date"], "03/14/2026")
        reply = await agent.respond("The claim status is paid.")
        self.assertIn("amount", reply)
        self.assertEqual(agent.state.values["claim_status"], "paid")

    async def test_hold_acknowledged_once_and_spelling_is_code_built(self):
        agent = await make_agent()
        agent.opening()
        self.assertEqual(await agent.respond("Please hold."), "Sure, thank you.")
        self.assertEqual(await agent.respond("One moment, please."), "")
        self.assertIn("T. E. S. T.", await agent.respond("Say the claim number slowly."))

    async def test_does_not_read_back_before_asking_for_facts(self):
        agent = await make_agent()
        agent.opening()
        reply = await agent.respond("Okay, go ahead.")
        self.assertIn("status", reply)
        self.assertNotIn("Is that correct?", reply)
        self.assertFalse(agent.state.readback_done)

    async def test_numeric_fact_is_stored_then_next_missing_field_is_asked(self):
        agent = await make_agent()
        agent.opening()
        reply = await agent.respond("The paid amount is $260.")
        self.assertIn("status", reply)
        self.assertEqual(agent.state.values["paid_amount"], "260.00")
        self.assertFalse(agent.state.readback_done)

    async def test_facts_in_confirmation_turn_update_readback_before_close(self):
        agent = await make_agent()
        agent.opening()
        agent.state.readback_done = True
        agent.state.values["paid_amount"] = "210.00"
        reply = await agent.respond("Yes, actually the paid amount is $260.00.")
        self.assertIn("the paid amount is $260.00", reply)
        self.assertIn("Is that correct?", reply)
        self.assertEqual(agent.state.values["paid_amount"], "260.00")
        self.assertFalse(agent.state.ended)

    async def test_confirmation_word_inside_fact_turn_does_not_end_call(self):
        agent = await make_agent()
        agent.opening()
        self.assertEqual(await fixture_intent("Yeah. It is correct, and the status is paid."), Intent.PROVIDES_FACTS)
        agent.state.readback_done = False
        reply = await agent.respond("Yeah. It is correct, and the status is paid.")
        self.assertFalse(agent.state.ended)
        self.assertNotIn("Goodbye", reply)

    async def test_declined_fact_stays_null(self):
        agent = await make_agent()
        agent.opening()
        await agent.respond("I don't know the check number.")
        self.assertIsNone(agent.state.values["check_or_eft_number"])
        self.assertIn("check_or_eft_number", agent.state.declined)

    async def test_later_decline_does_not_erase_a_previously_collected_value(self):
        agent = await make_agent()
        agent.opening()
        agent.state.values["paid_amount"] = "260.00"
        agent.state.active_field = "check_or_eft_number"
        await agent.respond("I don't know the check number.")
        self.assertEqual(agent.state.values["paid_amount"], "260.00")

    async def test_correction_updates_value_and_repeats_readback(self):
        agent = await make_agent()
        agent.opening()
        await agent.respond("Claim status is paid, paid amount is $210.00, check 88310, payment date 03/14/2026.")
        reply = await agent.respond("Actually, it was $120.00, not $210.00.")
        self.assertIn("120.00", reply)
        self.assertEqual(agent.state.values["paid_amount"], "120.00")

    async def test_goodbye_ends_with_nulls_not_guesses(self):
        agent = await make_agent()
        agent.opening()
        reply = await agent.respond("Goodbye, I can't help today.")
        self.assertFalse(agent.state.ended)
        self.assertIn("status", reply)
        self.assertTrue(all(value is None for value in agent.state.values.values()))

    async def test_denied_claim_does_not_invent_payment_fields(self):
        agent = await make_agent()
        agent.opening()
        await agent.respond("The claim was denied and no payment was issued.")
        self.assertEqual(agent.state.values["claim_status"], "denied")
        self.assertIsNone(agent.state.values["paid_amount"])
        self.assertIsNone(agent.state.values["check_or_eft_number"])

    async def test_spoken_amount_and_date_require_quoted_evidence(self):
        async def fake_llm(_system, _user):
            return {
                "facts": {
                    "paid_amount": {
                        "value": "120.00",
                        "evidence": "one hundred twenty dollars",
                    },
                    "payment_date": {
                        "value": "03/14/2026",
                        "evidence": "March fourteenth, 2026",
                    },
                }
            }

        extracted = await extract_facts(
            "They paid one hundred twenty dollars on March fourteenth, 2026.",
            llm_call=fake_llm,
        )
        self.assertEqual(extracted["paid_amount"], "120.00")
        self.assertEqual(extracted["payment_date"], "03/14/2026")

    async def test_incomplete_status_fragment_is_not_saved_as_a_claim_status(self):
        async def fake_llm(_system, _user):
            return {
                "facts": {
                    "claim_status": {
                        "value": "already",
                        "evidence": "The claim status is already with the amount of",
                    },
                }
            }

        extracted = await extract_facts(
            "The claim status is already with the amount of",
            llm_call=fake_llm,
        )
        self.assertNotIn("claim_status", extracted)

    async def test_each_missing_field_is_asked_at_most_twice(self):
        agent = await make_agent()
        agent.opening()
        first = await agent.respond("I am checking the account.")
        second = await agent.respond("I am still reviewing the account.")
        third = await agent.respond("There is no update yet.")
        fourth = await agent.respond("I still do not have that information.")
        self.assertIn("status", first)
        self.assertNotEqual(first, second)
        self.assertNotEqual(second, third)
        self.assertIn("amount", fourth)
        self.assertEqual(agent.state.retries["claim_status"], 2)

    async def test_no_readback_until_each_fact_is_collected_or_exhausted(self):
        agent = await make_agent()
        agent.opening()
        first = await agent.respond("Okay. Go ahead.")
        self.assertIn("status", first)
        self.assertNotIn("Is that correct?", first)

        greeting = await agent.respond("Hello, Mia.")
        self.assertNotIn("Is that correct?", greeting)
        self.assertIsNone(agent.state.values["claim_status"])

        malformed = await agent.respond("The claim status is already with the amount of.")
        self.assertNotIn("Is that correct?", malformed)
        self.assertIsNone(agent.state.values["claim_status"])
        self.assertIsNone(agent.state.values["paid_amount"])
        self.assertFalse(agent.state.readback_done)

        next_question = await agent.respond("I still cannot confirm the status.")
        next_question = await agent.respond("The claim status is paid.")
        self.assertIn("amount", next_question)
        next_question = await agent.respond("$1.65")
        self.assertIn("check or EFT", next_question)
        self.assertEqual(agent.state.values["claim_status"], "paid")
        self.assertEqual(agent.state.values["paid_amount"], "1.65")

    async def test_date_fragments_are_buffered_until_day_month_and_year_are_available(self):
        async def date_fragment_extractor(text: str) -> dict:
            if re.search(r"22[.,]?\s+November[.,]?\s+2026", text):
                return {"payment_date": "11/22/2026"}
            return {}

        agent = CallingAgent(
            extractor=date_fragment_extractor,
            intent_classifier=fixture_intent,
        )
        agent.opening()
        agent.state.values.update({
            "claim_status": "paid",
            "paid_amount": "260.12",
            "check_or_eft_number": "5590",
        })
        first = await agent.respond("Payment date is 22.")
        self.assertIn("full date", first)
        self.assertFalse(agent.state.readback_done)

        second = await agent.respond("November.")
        self.assertIn("full date", second)
        self.assertFalse(agent.state.readback_done)

        readback = await agent.respond("2026.")
        self.assertIn("payment date is 11/22/2026", readback)
        self.assertEqual(agent.state.values["payment_date"], "11/22/2026")

    async def test_partial_amount_does_not_advance_and_cents_fragment_is_combined(self):
        async def amount_fragment_extractor(text: str) -> dict:
            if "The paid amount is $260 and 12" in text:
                return {"paid_amount": "260.12"}
            return {}

        agent = CallingAgent(
            extractor=amount_fragment_extractor,
            intent_classifier=fixture_intent,
        )
        agent.opening()
        agent.state.values["claim_status"] = "paid"
        question = await agent.respond("The paid amount is")
        self.assertIn("dollars and cents", question)
        self.assertEqual(agent.state.active_field, "paid_amount")

        next_question = await agent.respond("$260 and 12\u00a2.")
        self.assertEqual(agent.state.values["paid_amount"], "260.12")
        self.assertIn("check or EFT", next_question)
        self.assertFalse(agent.state.readback_done)

    async def test_bare_date_corrects_the_field_currently_being_corrected(self):
        async def date_extractor(text: str) -> dict:
            if "11/14/2026" in text:
                return {"payment_date": "11/14/2026"}
            return {}

        agent = CallingAgent(extractor=date_extractor, intent_classifier=fixture_intent)
        agent.opening()
        agent.state.values.update({
            "claim_status": "paid",
            "paid_amount": "260.12",
            "check_or_eft_number": "5590",
            "payment_date": "11/15/2026",
        })
        agent.state.readback_done = True
        await agent.respond("I want to correct the payment date.")
        readback = await agent.respond("11/14/2026.")
        self.assertEqual(agent.state.values["payment_date"], "11/14/2026")
        self.assertIn("payment date is 11/14/2026", readback)
        self.assertFalse(agent.state.awaiting_correction)

    async def test_amount_normalization_handles_dollars_and_cents_phrase(self):
        self.assertEqual(_normalize_amount("260.12", "$260 and 12\u00a2."), "260.12")
        self.assertEqual(_normalize_amount("260.12", "260 dollars and 12 cents"), "260.12")

    async def test_date_normalization_accepts_day_month_year(self):
        self.assertEqual(_normalize_date("11/22/2026", "22 November 2026"), "11/22/2026")

    async def test_sequential_answers_are_kept_in_their_own_fields_until_readback(self):
        agent = await make_agent()
        agent.opening()
        self.assertIn("status", await agent.respond("Go ahead."))
        self.assertIn("amount", await agent.respond("The claim status is paid."))
        self.assertEqual(agent.state.values["claim_status"], "paid")
        self.assertEqual(agent.state.values["paid_amount"], None)

        self.assertIn("check or EFT", await agent.respond("The paid amount is $260."))
        self.assertEqual(agent.state.values["paid_amount"], "260.00")
        self.assertIn("payment date", await agent.respond("EFT 5590."))
        self.assertEqual(agent.state.values["check_or_eft_number"], "5590")

        readback = await agent.respond("Payment date is 11/15/2026.")
        self.assertIn("claim status is paid", readback)
        self.assertIn("paid amount is $260.00", readback)
        self.assertIn("check or EFT number is 5590", readback)
        self.assertIn("payment date is 11/15/2026", readback)
        self.assertTrue(agent.state.readback_done)

    async def test_turn_after_goodbye_does_not_change_call_state_or_repeat_output(self):
        agent = await make_agent()
        agent.opening()
        agent.state.values.update({
            "claim_status": "paid",
            "paid_amount": "260.00",
            "check_or_eft_number": "88310",
            "payment_date": "11/15/2026",
        })
        agent.state.readback_done = True
        await agent.respond("Yes, that's correct.")
        self.assertTrue(agent.state.ended)
        before = dict(agent.state.values)
        self.assertEqual(await agent.respond("I want to correct the payment date."), "")
        self.assertEqual(agent.state.values, before)

    async def test_missing_api_key_returns_empty_extraction_without_crashing(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
            result = await chat_json("Extract facts", "No facts here")
        self.assertEqual(result, {})

    def test_llm_client_is_reused(self):
        llm._client_for.cache_clear()
        try:
            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
                patch("ems_agent.llm.AsyncOpenAI") as client_factory,
            ):
                llm.initialize_llm_client()
                llm.initialize_llm_client()

            client_factory.assert_called_once()
            client_factory.assert_called_once_with(
                api_key="test-key",
                base_url=llm.OPENROUTER_BASE_URL,
                timeout=3.5,
                max_retries=0,
            )
        finally:
            llm._client_for.cache_clear()

    async def test_keyword_intents(self):
        self.assertEqual(await classify_intent("Please hold"), Intent.HOLD)
        self.assertEqual(await classify_intent("Say the claim number slowly"), Intent.SPELL_SLOWLY)
        self.assertEqual(await classify_intent("No, that's not correct."), Intent.CORRECTION)
        self.assertEqual(await classify_intent("Yes, that's right."), Intent.CONFIRM)
        self.assertEqual(await classify_intent("Yeah. It's correct."), Intent.CONFIRM)

        async def ambiguous_confirmation(_system, _user):
            return {"intent": "CONFIRM"}

        self.assertEqual(
            await classify_intent("Yeah. Okay.", llm_call=ambiguous_confirmation),
            Intent.CONFIRM,
        )

    async def test_reviewer_does_not_credit_missing_readback(self):
        async def fake_llm(_system, _user):
            return {
                "introduced_as_billing_office": True,
                "claim_number_stated": True,
                "all_four_facts_collected": True,
                "read_back_done": True,
                "no_invented_data": True,
            }

        transcript = [
            ("agent", "Calling from the billing office. Claim TEST-48213."),
            ("rep", "Your claim was paid."),
            ("agent", "The amount was $999. Is that correct?"),
        ]
        score = await review(transcript, llm_call=fake_llm)
        self.assertFalse(score["all_four_facts_collected"])
        self.assertFalse(score["no_invented_data"])

    async def test_reviewer_receives_strict_ems_qa_rubric(self):
        async def fake_llm(system_prompt, _user):
            self.assertEqual(system_prompt, REVIEWER_SYSTEM_PROMPT)
            self.assertIn('said "not provided."', system_prompt)
            self.assertIn("every value the caller stated or recorded", system_prompt)
            self.assertIn("including a made-up date of birth", system_prompt)
            self.assertIn("When unsure, answer false.", system_prompt)
            return {key: False for key in (
                "introduced_as_billing_office",
                "claim_number_stated",
                "all_four_facts_collected",
                "read_back_done",
                "no_invented_data",
            )}

        score = await review([], llm_call=fake_llm)
        self.assertEqual(set(score), {
            "introduced_as_billing_office",
            "claim_number_stated",
            "all_four_facts_collected",
            "read_back_done",
            "no_invented_data",
        })
        self.assertTrue(all(value is False for value in score.values()))


if __name__ == "__main__":
    unittest.main()
