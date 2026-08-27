from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
        self._vocoder: Any | None = None
        self._voices: dict[str, Any] = {}
        self._snapshot_dir: Path | None = None
        self._ref_cache: dict[str, tuple[Any, str]] = {}
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
                from f5_tts.infer.utils_infer import load_vocoder
                from f5_tts.model import CFM, DiT
                from f5_tts.model.utils import get_tokenizer
            except Exception as exc:
                raise RuntimeError(
                    "AI4Bharat IndicF5 is required for Orato TTS. In Colab run: "
                    "pip install git+https://github.com/AI4Bharat/IndicF5.git"
                ) from exc

            def load_model() -> tuple[Any, Any]:
                vocab_char_map, vocab_size = get_tokenizer(str(vocab), "custom")
                model = CFM(
                    transformer=DiT(
                        dim=1024,
                        depth=22,
                        heads=16,
                        ff_mult=2,
                        text_dim=512,
                        conv_layers=4,
                        text_num_embeds=vocab_size,
                        mel_dim=100,
                    ),
                    mel_spec_kwargs={
                        "n_fft": 1024,
                        "hop_length": 256,
                        "win_length": 1024,
                        "n_mel_channels": 100,
                        "target_sample_rate": 24000,
                        "mel_spec_type": "vocos",
                    },
                    odeint_kwargs={"method": "euler"},
                    vocab_char_map=vocab_char_map,
                )
                checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                model = model.to(self._device)
                model.eval()
                vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=self._device)
                return model, vocoder

            logger.info("Loading Orato/IndicF5 voice={} ckpt={}", self._voice, ckpt)
            self._engine, self._vocoder = await asyncio.to_thread(load_model)
            self._voices = voices
            self._snapshot_dir = snap
        return self._engine

    async def _get_reference(self, voice: str) -> tuple[Any, str]:
        if voice in self._ref_cache:
            return self._ref_cache[voice]
        await self._ensure_engine()
        if self._snapshot_dir is None:
            raise RuntimeError("Orato snapshot directory is not initialized")

        try:
            from f5_tts.infer.utils_infer import preprocess_ref_audio_text
        except Exception as exc:
            raise RuntimeError("AI4Bharat IndicF5 preprocessing utilities are unavailable") from exc

        if voice not in self._voices:
            raise ValueError(f"Unknown Orato voice '{voice}'. Available: {sorted(self._voices)}")

        ref_wav = self._snapshot_dir / self._voices[voice]["wav"]
        ref_text = self._voices[voice]["ref_text"]

        reference = await asyncio.to_thread(
            preprocess_ref_audio_text,
            str(ref_wav),
            ref_text,
        )
        self._ref_cache[voice] = reference
        return reference

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
        if self._vocoder is None:
            raise RuntimeError("Orato vocoder is not initialized")
        ref_audio, ref_text = await self._get_reference(self._voice)

        try:
            from f5_tts.infer.utils_infer import infer_process
        except Exception as exc:
            raise RuntimeError("AI4Bharat IndicF5 inference utilities are unavailable") from exc

        def infer() -> bytes:
            audio_output, sample_rate_output, _ = infer_process(
                ref_audio=ref_audio,
                ref_text=ref_text,
                gen_text=text,
                model_obj=engine,
                vocoder=self._vocoder,
                mel_spec_type="vocos",
                nfe_step=32,
                cfg_strength=2.0,
                speed=1.0,
                device=self._device,
            )
            return self._to_pcm16(np.asarray(audio_output, dtype=np.float32), int(sample_rate_output))

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
