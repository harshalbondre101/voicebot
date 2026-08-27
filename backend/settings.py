from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class BotSettings:
    bot_name: str = env("BOT_NAME", "Hindi Voicebot")
    llama_base_url: str = env("LLAMA_BASE_URL", "http://127.0.0.1:8080/v1")
    llama_model: str = env("LLAMA_MODEL", "qwen3-8b")
    llama_api_key: str = env("LLAMA_API_KEY", "not-needed")
    whisper_model: str = env("WHISPER_MODEL", "small")
    whisper_device: str = env("WHISPER_DEVICE", "cuda")
    whisper_compute_type: str = env("WHISPER_COMPUTE_TYPE", "float16")
    whisper_language: str = env("WHISPER_LANGUAGE", "hi")
    orato_model_id: str = env("ORATO_MODEL_ID", "tryorato/orato-tts-hindi-v1")
    orato_model_dir: str = env("ORATO_MODEL_DIR", "")
    orato_voice: str = env("ORATO_VOICE", "female")
    orato_device: str = env("ORATO_DEVICE", "cuda")
    orato_sample_rate: int = int(env("ORATO_SAMPLE_RATE", "24000"))
    hf_token: str = env("HF_TOKEN", "")
    log_level: str = env("LOG_LEVEL", "INFO")


SYSTEM_PROMPT = """
आप एक तेज, प्राकृतिक और मददगार हिंदी वॉयसबॉट हैं।
यूजर हिंदी या Hinglish में बात कर सकता है; जवाब मुख्यतः सरल हिंदी में दें।
आवाज़ में जवाब छोटा रखें: सामान्यतः 1-3 वाक्य।
अगर सवाल अस्पष्ट हो तो एक छोटा स्पष्टिकरण सवाल पूछें।
तकनीकी बातों में भी बोलचाल की हिंदी रखें।
""".strip()
