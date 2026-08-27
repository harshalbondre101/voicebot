# Hindi Voicebot with Pipecat, faster-whisper, Qwen3, and Orato TTS

Browser-based Hindi/Hinglish voice chatbot scaffold optimized for Colab GPU testing.

## Architecture

```
Browser mic
  -> Pipecat SmallWebRTC transport
  -> Silero VAD / turn detection
  -> faster-whisper Hindi ASR
  -> llama.cpp OpenAI-compatible Qwen3 8B GGUF server
  -> Orato Hindi F5-TTS adapter
  -> Pipecat SmallWebRTC audio output
```

The project is intentionally split into ordinary source files plus a Colab notebook:

- `backend/` contains the Pipecat bot and custom ASR/TTS adapters.
- `frontend/` contains a minimal browser voice console.
- `notebooks/colab_runner.ipynb` installs dependencies, starts llama.cpp, starts the bot, and exposes it with Ngrok.

## Colab Quick Start

1. Upload or clone this repo into Colab.
2. Open `notebooks/colab_runner.ipynb`.
3. Set these values in the notebook:
   - `NGROK_AUTHTOKEN`
   - `QWEN_GGUF_PATH` or `QWEN_GGUF_URL`
   - `HF_TOKEN` if needed for gated Orato/IndicF5 access
4. Run all notebook cells.
5. Open the printed Ngrok URL and use `/client` for Pipecat's built-in client or `/voice-console/` for this repo's minimal console.

## Local Dev

Python 3.11+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python backend/bot.py --transport webrtc --host 0.0.0.0 --port 7860
```

In a separate terminal, start `llama.cpp`:

```bash
./llama-server -m /path/to/qwen3-8b-quantized.gguf --host 0.0.0.0 --port 8080 -ngl 99 -c 4096
```

## Important Model Notes

- Orato TTS is based on IndicF5/F5-TTS and requires its own `vocab.txt`; using the default F5-TTS vocab will produce bad Hindi output.
- The Orato Hugging Face repo is gated at the time this scaffold was created. Accept the model terms and pass `HF_TOKEN` in Colab if download fails.
- Qwen3 8B should be a GGUF quantized file for `llama.cpp`; Q4_K_M is a practical default for Colab GPUs.

## Environment Variables

Copy `.env.example` and adjust values:

```bash
LLAMA_BASE_URL=http://127.0.0.1:8080/v1
LLAMA_MODEL=qwen3-8b
WHISPER_MODEL=small
ORATO_MODEL_ID=tryorato/orato-tts-hindi-v1
ORATO_VOICE=female
```

## Verification Checklist

- Browser opens through Ngrok.
- Microphone permission succeeds.
- Hindi/Hinglish speech produces transcripts.
- Bot begins speaking before the full LLM answer is complete.
- Speaking over the bot interrupts current TTS and starts a new user turn.
