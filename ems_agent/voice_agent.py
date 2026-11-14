import argparse
import asyncio
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

from .brain import CallingAgent
from .config import (
    CALLER_SYSTEM_PROMPT,
    LLM_MODEL,
    OPENROUTER_BASE_URL,
    STT_MODEL,
    STT_KEYTERMS,
    RESULTS_DIR,
    TRANSCRIPTS_DIR,
    TTS_MODEL,
)
from .llm import initialize_llm_client
from .reviewer import review


def _new_brain() -> CallingAgent:
    path = TRANSCRIPTS_DIR / f"call-{uuid.uuid4().hex[:8]}.jsonl"
    return CallingAgent(transcript_path=path)


async def _print_results(brain: CallingAgent) -> None:
    result = json.loads(brain.state.to_json())
    scorecard = await review(brain.state.transcript)
    result_path = _save_results(brain, result, scorecard)
    print("FINAL RESULT")
    print(json.dumps(result, indent=2))
    print("SCORECARD")
    print(json.dumps(scorecard, indent=2))
    print(f"Saved results: {result_path}")


def _save_results(
    brain: CallingAgent, result: dict[str, object], scorecard: dict[str, bool],
) -> Path:
    """Write final call data separately from the turn-by-turn transcript."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = brain.transcript_path.stem if brain.transcript_path else f"call-{uuid.uuid4().hex[:8]}"
    path = RESULTS_DIR / f"{stem}-result.json"
    path.write_text(
        json.dumps({"result": result, "scorecard": scorecard}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


async def run_text() -> None:
    brain = _new_brain()
    print(brain.opening())
    while not brain.state.ended:
        try:
            rep_text = input("Rep> ").strip()
        except (EOFError, KeyboardInterrupt):
            brain.state.ended = True
            break
        if not rep_text:
            continue
        reply = await brain.respond(rep_text)
        if reply:
            print(f"Agent> {reply}")
    await _print_results(brain)


def run_voice() -> None:
    from livekit import agents
    from livekit.agents import Agent, AgentSession
    from livekit.agents.llm import StopResponse
    from livekit.plugins import deepgram, openai

    server = agents.AgentServer()

    @server.rtc_session(agent_name="ems-claim-agent")
    async def entrypoint(ctx: agents.JobContext) -> None:
        brain = _new_brain()
        results_printed = False
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY must be set before starting the voice agent.")
        session = AgentSession(
            stt=deepgram.STT(
                model=STT_MODEL,
                language="en-US",
                keyterm=STT_KEYTERMS,
                smart_format=True,
                numerals=True,
                interim_results=True,
                no_delay=True,
                endpointing_ms=400,
                utterance_end_ms=1000,
            ),
            llm=openai.LLM(
                model=LLM_MODEL,
                base_url=OPENROUTER_BASE_URL,
                api_key=openrouter_api_key,
            ),
            tts=deepgram.TTS(model=TTS_MODEL),
            turn_handling={
                "turn_detection": "stt",
                "preemptive_generation": {"enabled": False},
                "interruption": {"enabled": False},
            },
        )

        class ScriptedAgent(Agent):
            def __init__(self) -> None:
                super().__init__(instructions=CALLER_SYSTEM_PROMPT)

            async def on_enter(self) -> None:
                await self.session.say(brain.opening())

            async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
                nonlocal results_printed
                reply = await brain.respond(new_message.text_content or "")
                if reply:
                    await self.session.say(reply)
                if brain.state.ended and not results_printed:
                    results_printed = True
                    await _print_results(brain)
                raise StopResponse()

        @session.on("agent_state_changed")
        def on_agent_state_changed(event) -> None:
            session.input.set_audio_enabled(event.new_state != "speaking")

        await session.start(room=ctx.room, agent=ScriptedAgent())

        @session.on("close")
        def on_close(_event) -> None:
            if not brain.state.ended:
                print(brain.state.to_json())
                asyncio.create_task(_print_results(brain))

    agents.cli.run_app(server)


def main() -> None:
    load_dotenv(Path(__file__).parent / ".env")
    initialize_llm_client()
    parser = argparse.ArgumentParser(description="EMS claim billing voice agent")
    parser.add_argument("mode", choices=("console",), nargs="?", default="console")
    parser.add_argument("--text", action="store_true", help="Use terminal input instead of audio")
    args = parser.parse_args()
    if args.text:
        asyncio.run(run_text())
    else:
        run_voice()


if __name__ == "__main__":
    main()
