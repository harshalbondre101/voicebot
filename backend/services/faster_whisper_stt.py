from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

import numpy as np
from faster_whisper import WhisperModel
from loguru import logger

try:
    from pipecat.frames.frames import Frame, TranscriptionFrame
    from pipecat.services.stt_service import STTService
except Exception as exc:  # pragma: no cover - import-time guard for setup checks
    raise RuntimeError("Pipecat must be installed before importing FasterWhisperSTTService") from exc


class FasterWhisperSTTService(STTService):
    """Segmented faster-whisper adapter for Pipecat.

    Pipecat's STTService receives PCM chunks from the transport/VAD path. This adapter
    keeps the implementation simple and Colab-friendly: each committed chunk is decoded
    by faster-whisper and emitted as a final TranscriptionFrame. Lower latency comes
    from short VAD segments rather than a long batch transcription.
    """

    def __init__(
        self,
        *,
        model_name: str = "small",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "hi",
        sample_rate: int = 16000,
        user_id: str = "user",
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._user_id = user_id
        self._model: WhisperModel | None = None
        self._load_lock = asyncio.Lock()

    async def _ensure_model(self) -> WhisperModel:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is None:
                logger.info(
                    "Loading faster-whisper model={} device={} compute_type={}",
                    self._model_name,
                    self._device,
                    self._compute_type,
                )
                self._model = await asyncio.to_thread(
                    WhisperModel,
                    self._model_name,
                    device=self._device,
                    compute_type=self._compute_type,
                )
        return self._model

    def _pcm16_to_float32(self, audio: bytes) -> np.ndarray:
        pcm = np.frombuffer(audio, dtype=np.int16)
        if pcm.size == 0:
            return np.array([], dtype=np.float32)
        return (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        model = await self._ensure_model()
        samples = self._pcm16_to_float32(audio)
        if samples.size < int(self.sample_rate * 0.12):
            return

        def transcribe() -> str:
            segments, _info = model.transcribe(
                samples,
                language=self._language,
                vad_filter=False,
                beam_size=1,
                condition_on_previous_text=False,
                no_speech_threshold=0.55,
                compression_ratio_threshold=2.4,
            )
            return " ".join(segment.text.strip() for segment in segments).strip()

        text = await asyncio.to_thread(transcribe)
        if not text:
            return

        timestamp = datetime.now(timezone.utc).isoformat()
        yield TranscriptionFrame(
            text=text,
            user_id=self._user_id,
            timestamp=timestamp,
            language=None,
            result={"engine": "faster-whisper", "model": self._model_name},
            finalized=True,
        )
