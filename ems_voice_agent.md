# PROJECT BRIEF: Voice Calling Agent (EMS Billing) + AI Reviewer

## 1. Goal
Build a VOICE agent that plays a caller from the billing office of an EMS agency, phoning an insurance company about a claim. A human assessor plays the insurance rep and speaks to it live through the laptop mic. After the call, a separate AI reviewer reads the transcript and prints a true/false scorecard. Time limit: 60 minutes total, so keep everything simple and explainable line by line.

## 2. Fixed test data
- Claim number: TEST-48213
- Date of service: 03/14/2026
- Patient name: John Smith
- Caller: billing office of an EMS agency

## 3. Stack (no SIP trunk, no phone number)
- Python 3.10+
- LiveKit Agents (v1.x) for the voice pipeline
- STT: Deepgram (streaming) via `livekit-plugins-deepgram`
- TTS: Deepgram Aura via `livekit-plugins-deepgram`
- LLM: OpenAI GPT-4o mini through OpenRouter. Model slug: `openai/gpt-4o-mini`. Use the official `openai` Python SDK with `base_url="https://openrouter.ai/api/v1"` and `api_key=OPENROUTER_API_KEY`.
- OpenRouter is pay-as-you-go (about $0.15 per 1M input and $0.60 per 1M output tokens), so a small credit top-up is needed. This project uses very few tokens, so a few dollars is plenty.
- The LLM is called directly from extractor.py, intents.py (fallback) and reviewer.py. Use `temperature=0`, non-streaming, and `response_format={"type": "json_object"}`. Always `json.loads` inside try/except, validate keys in code, and retry once on parse failure.
- Deepgram gives free signup credit, so one Deepgram key covers both speech-in and speech-out.
- Demo via LiveKit console mode (`python voice_agent.py console`) using the laptop mic and speaker, or the LiveKit playground as an alternative.
- .env holds OPENROUTER_API_KEY, DEEPGRAM_API_KEY, LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
- Check the current LiveKit Agents, OpenRouter and Deepgram docs for exact class names and parameters before coding.
- All model names live in config.py only (LLM_MODEL, LLM_FALLBACK_MODEL, STT_MODEL, TTS_VOICE), so any model can be swapped in one line. If `openai/gpt-4o-mini` is unavailable or errors, automatically retry with LLM_FALLBACK_MODEL (another OpenRouter model slug, for example a cheap or `:free` model).

## 4. Core design rule
CODE OWNS THE TRUTH. The LLM only extracts facts and classifies intent. Everything the agent says comes from code-controlled logic reading a state dict. Do not let the LiveKit LLM free-chat. Intercept each finished user turn, run our own logic, and speak exactly what our logic returns.

## 5. Agent behavior (all mandatory)
1. Speak first: introduce itself as calling from the billing office of an EMS agency, state claim number TEST-48213 and date of service.
2. Collect four facts: claim_status, paid_amount, check_or_eft_number, payment_date.
3. "Please hold" (or "one moment", etc.): say a short "Sure, I'll hold" ONCE, then stay completely silent on later hold or silence turns. Never hang up, never repeat itself.
4. If asked to say the claim number slowly: speak it one character at a time with pauses, e.g. "T. E. S. T. dash. 4. 8. 2. 1. 3." Build this string in code, not with the LLM.
5. Ask again (specifically) for any fact not given, max 2 retries per fact, then keep null.
6. Before ending, read back everything captured from state only ("not provided" for nulls), ask "Is that correct?", handle a correction by updating state and reading back again.
7. At the end, print JSON with EXACTLY these keys: claim_status, paid_amount, check_or_eft_number, payment_date. Use null for anything not obtained. NEVER invent a value.

## 6. File structure
```
ems_agent/
  config.py       # test data, FIELDS dict, model names, base URLs
  llm.py          # one helper: chat_json(system, user) -> dict (OpenRouter call, retry, fallback model)
  state.py        # CallState: values, retries, flags, transcript
  extractor.py    # LLM extraction + code validation
  intents.py      # HOLD, SPELL_SLOWLY, PROVIDES_FACTS, CORRECTION, DONT_KNOW, CONFIRM, GOODBYE, OTHER
  brain.py        # CallingAgent: opening() and respond(rep_text) -> str (pure logic, NO audio code)
  reviewer.py     # review(transcript) -> scorecard dict
  voice_agent.py  # LiveKit wrapper (audio in/out only)
  tests.py        # runs brain.py against scripted rep lines, no audio needed
  transcripts/    # one file per call
  .env
```

## 7. Component details

### config.py
`FIELDS` dict, one entry per fact: name, ask_prompt, validator type. The extractor, re-ask logic, read-back, final JSON, and reviewer check must all loop over FIELDS, so adding a new field live (e.g. representative_name, reference_number) is a ONE-LINE change.

### llm.py
- Single place that talks to OpenRouter using the `openai` SDK.
- `chat_json(system_prompt, user_text)`: temperature 0, JSON output, parse with json.loads, retry once on bad JSON, then try LLM_FALLBACK_MODEL, then return an empty dict (never crash the call).
- Set a short timeout (about 8 seconds) so the voice call never hangs.

### state.py
values (all None), retries per field, declined set, flags: introduced, on_hold, readback_done, ended; transcript list of (speaker, text). Methods: missing_fields(), all_done(), to_json() using json.dumps.

### extractor.py
- One `chat_json` call, input is ONLY the rep's latest message.
- Prompt: "Extract only values literally stated in this message. If a field is not stated, return null. Do not guess."
- Code validation: accept a value only if it actually appears in the rep's message (regex for amounts, dates, check numbers). Normalize spoken forms ("one hundred twenty dollars" to 120.00).
- Handle self-corrections ("actually 120, not 210"): newer value overwrites older.
- If the rep says they don't know a field, mark it declined, store null, stop asking.
- Speech transcripts are messy: tolerate ASR formats like "forty eight two one three".

### intents.py
Keyword rules first ("hold", "one moment", "slowly", "spell", "say the claim number"), LLM fallback through `chat_json`.

### brain.py (pure logic)
`respond(rep_text)`:
1. Log the rep turn.
2. HOLD: first time say "Sure, I'll hold", afterwards return "" (silence).
3. SPELL_SLOWLY: return the code-built character string.
4. Otherwise: clear on_hold, run extractor, update state.
5. If any field is missing and retries < 2, ask for that one field specifically.
6. Else read back from state only, ask "Is that correct?".
7. If confirmed: thank them, set ended. If corrected: update state, read back again.
8. Never speak a value that is not in state. Log the agent turn.

### reviewer.py
- Input: transcript only. One `chat_json` call, temperature 0.
- Keys, each true/false: introduced_as_billing_office, claim_number_stated, all_four_facts_collected, read_back_done, no_invented_data.
- Prompt: "Judge only what actually happened. Do not give credit for what should have happened."
- Code cross-checks override the LLM where verifiable: claim number appears in an agent turn; every number or date the agent spoke appears in rep turns or the test data; read-back turn contains all captured values.
- Validate output has exactly those five boolean keys.

### voice_agent.py (LiveKit wrapper, no business logic)
- Subclass `Agent`. On session start, speak `brain.opening()`.
- Override `on_user_turn_completed`: take the final transcript text, call `brain.respond(text)`, then `await session.say(reply)` only if reply is non-empty. Raise `StopResponse` so LiveKit's own LLM does not generate a second reply.
- Turn detection: use a slightly longer end-of-speech delay so the agent does not cut off the rep mid-sentence (reps speak claim numbers slowly).
- Deepgram STT settings: enable punctuation and smart formatting, add keyterms for "EFT", "claim", "TEST".
- Deepgram TTS: pick one Aura voice, keep speech rate normal. For spelling, send characters separated by commas or periods so each one is spoken clearly.
- If AgentSession requires an llm object, pass an OpenAI-compatible LLM pointed at OpenRouter, but never let it reply (StopResponse blocks it).
- Write each turn to the transcript file as it happens.
- On `brain.ended` or session close: print result JSON, run reviewer, print scorecard JSON.
- Add a `--text` fallback that runs the same brain from terminal `input()` if audio fails during the demo.

## 8. Tests (tests.py, no audio, run brain against scripted rep lines)
1. All four facts in one sentence.
2. Facts one at a time.
3. Out of order facts or unrelated chatter.
4. "Please hold", several silent turns, then answers (agent speaks once, then silent, never repeats).
5. "Say the claim number slowly" mid-call.
6. Rep doesn't know the check number (null, not invented).
7. Self-correction ("actually $120, not $210").
8. Claim denied, no payment (no invented amount or check number).
9. Amounts and dates in words.
10. Rep ends early or refuses (JSON with nulls).
11. Reviewer sanity: feed a bad transcript (no read-back, invented amount), confirm false for those keys.
12. LLM failure test: simulate an OpenRouter error and confirm the call continues without crashing and without invented values.

Then do ONE real voice test through the mic covering hold + slow spelling.

## 9. Build order (60 min)
- 0 to 10: write the 10-line plan (hand to assessor)
- 10 to 25: config, llm.py, state, extractor, intents, brain
- 25 to 35: voice_agent.py wrapper, first spoken call
- 35 to 42: reviewer + JSON output
- 42 to 52: edge case tests + fixes, live voice test
- 52 to 60: rehearse the live "add one field" change

If short on time, cut: extra logging, UI. Never cut: no-invented-data safeguards, read-back, honest reviewer.

## 10. Coding rules
- Short, readable functions with comments, no clever tricks.
- All structured LLM calls: temperature 0, non-streaming, JSON output, parsed and validated in code.
- Final JSON from json.dumps(state), never from LLM prose.
- Never speak or print a value not in state.
- Keep audio code only in voice_agent.py.
- Never hard-code API keys; read them from .env.

## 11. Pre-demo checklist
- OpenRouter key works and the account has a few dollars of credit.
- Deepgram key works (STT and TTS both tested).
- One full voice run completed end to end, including hold and slow spelling.
- `--text` fallback tested.
- Live change rehearsed: add one entry to FIELDS and confirm the extractor, re-ask, read-back, JSON and reviewer all pick it up.
