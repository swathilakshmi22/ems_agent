# EMS Claim Voice Agent

A controlled voice agent for a simulated EMS billing call to an insurance representative. It collects claim information, reads it back for confirmation, saves the call record, and produces a post-call quality scorecard. The demo uses synthetic data only.

## What it does

The caller introduces itself as an EMS billing office and collects:

1. Claim status
2. Paid amount
3. Check or EFT number
4. Payment date

It handles holds, claim-number spelling requests, partial answers, corrections, unavailable information, and early goodbyes. It reads back recorded values before closing.

## Design principles

- **Code owns the truth.** Spoken values come only from fixed claim data or recorded state.
- **Evidence is required.** Extracted values must be supported by the representative's turn.
- **No invented facts.** Unknown values stay `null` and are read back as `not provided`.
- **Graceful degradation.** Regex extraction and deterministic reviewer checks work when the LLM is unavailable.

## Fixed demo data

| Field | Value |
| --- | --- |
| Claim number | `TEST-48213` |
| Date of service | `03/14/2026` |
| Patient | John Smith |
| Caller | Priya, EMS billing office |

## Setup

Requirements: Python 3.10+, Deepgram key for live speech, LiveKit credentials for voice mode, and an OpenRouter key for optional LLM assistance.

```powershell
cd ems_agent
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set these values in `.env`:

```text
OPENROUTER_API_KEY=
DEEPGRAM_API_KEY=
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
```

Return to the repository root before running package commands:

```powershell
cd ..
```

## Run

### Text mode

```powershell
python -m ems_agent.voice_agent --text
```

Use this for the fastest functional check. Type representative turns after `Rep>`.

### LiveKit console mode

```powershell
python -m ems_agent.voice_agent console
```

Use headphones and wait for the agent to finish speaking before replying. If `nova-3-medical` mishears numbers or identifiers, change `STT_MODEL` in `config.py` to `nova-3` and retry.

## Results and transcripts

| Location | Contents |
| --- | --- |
| `transcripts/call-<id>.jsonl` | Turn-by-turn transcript |
| `results/call-<id>-result.json` | Final field values and reviewer scorecard |

The terminal prints `FINAL RESULT` and `SCORECARD` at the end of every call.

Review an existing transcript:

```powershell
python -m ems_agent.reviewer ems_agent\transcripts\call-<id>.jsonl
```

The reviewer uses deterministic transcript checks if the LLM fails. A warning indicates the fallback was used; the scorecard always contains the same five boolean keys.

## Test

Run from the repository root:

```powershell
python -m ems_agent.tests
```

The scripted suite requires no audio hardware or API credentials.

## Project map

| File | Responsibility |
| --- | --- |
| `config.py` | Claim data, extensible `FIELDS`, and settings |
| `state.py` | Call state, transcript, retries, and phase |
| `brain.py` | Deterministic call flow and replies |
| `extractor.py` | Evidence-backed extraction and validation |
| `intents.py` | Intent classification |
| `llm.py` | Bounded OpenRouter JSON client |
| `reviewer.py` | Transcript scorecard and JSONL CLI |
| `voice_agent.py` | LiveKit audio wrapper and text runner |
| `tests.py` | Offline regression suite |

## Add a field

Add one entry to `FIELDS` in `config.py`. Prompts, labels, read-back text, extraction schema, and reviewer field checks derive from this configuration. Provide `label`, `ask_prompt`, and `validator`; matching patterns have safe label-based defaults.
