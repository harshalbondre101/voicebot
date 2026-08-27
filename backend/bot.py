from __future__ import annotations

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

from backend.services.faster_whisper_stt import FasterWhisperSTTService
from backend.services.orato_tts import OratoTTSService
from backend.settings import SYSTEM_PROMPT, BotSettings


settings = BotSettings()

transport_params = {
    "eval": lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True),
    "daily": lambda: DailyParams(audio_in_enabled=True, audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    logger.info("Starting {}", settings.bot_name)

    stt = FasterWhisperSTTService(
        model_name=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        language=settings.whisper_language,
    )

    llm = OpenAILLMService(
        api_key=settings.llama_api_key,
        base_url=settings.llama_base_url,
        settings=OpenAILLMService.Settings(
            model=settings.llama_model,
            system_instruction=SYSTEM_PROMPT,
            temperature=0.45,
            max_tokens=180,
        ),
    )

    tts = OratoTTSService(
        model_id=settings.orato_model_id,
        model_dir=settings.orato_model_dir or None,
        voice=settings.orato_voice,
        device=settings.orato_device,
        sample_rate=settings.orato_sample_rate,
        hf_token=settings.hf_token or None,
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=getattr(runner_args, "handle_sigint", True))
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client):
        logger.info("Client connected")
        context.add_message(
            {
                "role": "user",
                "content": "यूजर को हिंदी में बहुत छोटा परिचय दें और पूछें कि आप कैसे मदद कर सकते हैं।",
            }
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments) -> None:
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
