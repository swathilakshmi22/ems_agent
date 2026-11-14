# EMS Claim Voice Agent

A small demo that calls an insurance representative about the fixed test claim. The caller role and conversation requirements are included in the LiveKit agent instructions, but the call flow and spoken values are controlled by Python code; the LLM only extracts facts, classifies unfamiliar intents, and produces the reviewer scorecard. The agent asks for one missing fact at a time, stores each in its own field, gives each missing field at most two retries, and does not start the final read-back until every field is obtained or marked unavailable.

## Setup

Use Python 3.10 or newer:

```powershell
cd ems_agent
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add an OpenRouter key, a Deepgram key for Nova-3 Medical STT and Aura TTS, plus LiveKit credentials to `.env`. STT uses Deepgram's realtime streaming API with EMS and insurance keyterms, interim results, and numeric formatting. The OpenRouter client is initialized once at startup and reused to avoid blocking audio handling during every turn. The sample claim data is synthetic. Transcripts are saved locally under `transcripts/`.

## Run

The setup commands leave you inside `ems_agent/`. Return to its parent before running the package commands:

```powershell
cd ..
```

Run the following commands from the workspace parent directory (the directory containing `ems_agent/`):

Text fallback:

```powershell
python -m ems_agent.voice_agent --text
```

LiveKit console mode (run from the workspace parent):

```powershell
python -m ems_agent.voice_agent console
```

Console mode pauses microphone input while the agent speaks, so wait for the prompt to finish before answering. Headphones are still recommended to prevent residual speaker feedback from being transcribed as caller speech.

Run the scripted tests without audio or API credentials:

```powershell
python -m ems_agent.tests
```

## Project map

- `config.py` holds the fixed test data, extensible field definitions, and model settings.
- `brain.py`, `state.py`, `extractor.py`, and `intents.py` implement deterministic call behavior.
- `voice_agent.py` is the LiveKit audio wrapper and terminal text fallback.
- `reviewer.py` checks the transcript and emits a boolean scorecard.