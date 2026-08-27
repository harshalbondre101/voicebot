from __future__ import annotations

import asyncio
import json
import re
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download
from loguru import logger
from scipy.signal import resample_poly

try:
    from pipecat.frames.frames import Frame, TTSAudioRawFrame
    from pipecat.services.tts_service import TTSService
except Exception as exc:  # pragma: no cover - import-time guard for setup checks
    raise RuntimeError("Pipecat must be installed before importing OratoTTSService") from exc


PHRASE_SPLIT_RE = re.compile(r"([।.!?\n]+)")


class OratoTTSService(TTSService):
    """F5-TTS based Orato Hindi adapter.

    Orato's model card exposes F5-TTS weights, vocab, and reference voices. The
    current public instructions do not expose a native streaming endpoint, so this
    service streams at phrase granularity: Pipecat aggregates partial LLM text,
    this adapter synthesizes short chunks, and frames are emitted immediately.
    """

    def __init__(
        self,
        *,
        model_id: str = "tryorato/orato-tts-hindi-v1",
        model_dir: str | None = None,
        voice: str = "female",
        device: str = "cuda",
        sample_rate: int = 24000,
        hf_token: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._model_id = model_id
        self._model_dir = Path(model_dir).expanduser() if model_dir else None
        self._voice = voice
        self._device = device
        self._sample_rate = sample_rate
        self._hf_token = hf_token or None
        self._engine: Any | None = None
        self._ref_wav: Path | None = None
        self._ref_text: str = ""
        self._load_lock = asyncio.Lock()

    async def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        async with self._load_lock:
            if self._engine is not None:
                return self._engine

            snap = self._model_dir
            if snap is None:
                logger.info("Downloading Orato TTS snapshot: {}", self._model_id)
                snap = Path(
                    await asyncio.to_thread(
                        snapshot_download,
                        repo_id=self._model_id,
                        token=self._hf_token,
                    )
                )

            voices = json.loads((snap / "voices.json").read_text(encoding="utf-8"))
            if self._voice not in voices:
                raise ValueError(f"Unknown Orato voice '{self._voice}'. Available: {sorted(voices)}")

            self._ref_wav = snap / voices[self._voice]["wav"]
            self._ref_text = voices[self._voice]["ref_text"]
            ckpt = snap / "model.pt"
            vocab = snap / "vocab.txt"

            try:
                from f5_tts.api import F5TTS
            except Exception as exc:
                raise RuntimeError(
                    "F5-TTS is required for Orato TTS. In Colab run: "
                    "pip install git+https://github.com/SWivid/F5-TTS.git"
                ) from exc

            logger.info("Loading Orato/F5-TTS voice={} ckpt={}", self._voice, ckpt)
            self._engine = await asyncio.to_thread(
                F5TTS,
                model="F5TTS_Base",
                ckpt_file=str(ckpt),
                vocab_file=str(vocab),
                device=self._device,
            )
        return self._engine

    def _split_text(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        parts: list[str] = []
        buf = ""
        for piece in PHRASE_SPLIT_RE.split(text):
            buf += piece
            if PHRASE_SPLIT_RE.fullmatch(piece or "") and buf.strip():
                parts.append(buf.strip())
                buf = ""
        if buf.strip():
            parts.append(buf.strip())
        return parts

    def _to_pcm16(self, wav: np.ndarray, src_rate: int) -> bytes:
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if src_rate != self._sample_rate:
            wav = resample_poly(wav, self._sample_rate, src_rate)
        wav = np.clip(wav, -1.0, 1.0)
        return (wav * 32767.0).astype(np.int16).tobytes()

    async def _synthesize(self, engine: Any, text: str) -> bytes:
        if self._ref_wav is None:
            raise RuntimeError("Orato reference voice is not initialized")

        def infer() -> bytes:
            result = engine.infer(
                ref_file=str(self._ref_wav),
                ref_text=self._ref_text,
                gen_text=text,
            )
            if isinstance(result, tuple) and len(result) >= 2:
                wav, rate = result[0], int(result[1])
                return self._to_pcm16(np.asarray(wav, dtype=np.float32), rate)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                engine.infer(
                    ref_file=str(self._ref_wav),
                    ref_text=self._ref_text,
                    gen_text=text,
                    file_wave=tmp.name,
                )
                wav, rate = sf.read(tmp.name, dtype="float32")
                return self._to_pcm16(np.asarray(wav), int(rate))

        return await asyncio.to_thread(infer)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        engine = await self._ensure_engine()
        for phrase in self._split_text(text):
            audio = await self._synthesize(engine, phrase)
            if audio:
                yield TTSAudioRawFrame(
                    audio=audio,
                    sample_rate=self._sample_rate,
                    num_channels=1,
                    context_id=None,
                )
