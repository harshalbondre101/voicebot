from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import requests
from pyngrok import ngrok


ROOT = Path(__file__).resolve().parents[1]


def start_process(command: list[str], cwd: Path | None = None) -> subprocess.Popen:
    print("+", " ".join(command))
    return subprocess.Popen(command, cwd=str(cwd or ROOT), start_new_session=True)


def wait_for(url: str, label: str, timeout: int = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code < 500:
                print(f"{label} is up: {url}")
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"{label} did not start within {timeout}s: {url}")


def configure_ngrok() -> None:
    token = os.getenv("NGROK_AUTHTOKEN")
    if token:
        ngrok.set_auth_token(token)


def start_llama() -> subprocess.Popen | None:
    server = os.getenv("LLAMA_SERVER_BIN", "/content/llama.cpp/build/bin/llama-server")
    model_path = os.getenv("QWEN_GGUF_PATH", "")
    if not model_path:
        print("QWEN_GGUF_PATH is empty; assuming llama.cpp server is already running.")
        return None

    command = [
        server,
        "-m",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "-ngl",
        os.getenv("LLAMA_N_GPU_LAYERS", "99"),
        "-c",
        os.getenv("LLAMA_CONTEXT", "4096"),
    ]
    process = start_process(command)
    wait_for("http://127.0.0.1:8080/v1/models", "llama.cpp")
    return process


def main() -> None:
    configure_ngrok()

    llama_proc = start_llama()
    backend_proc = start_process(
        [
            "python",
            "backend/bot.py",
            "--transport",
            "webrtc",
            "--host",
            "0.0.0.0",
            "--port",
            "7860",
        ],
        cwd=ROOT,
    )
    frontend_proc = start_process(["npm", "run", "dev"], cwd=ROOT / "frontend")

    wait_for("http://127.0.0.1:7860/client", "Pipecat backend")
    wait_for("http://127.0.0.1:3000", "voice console")

    backend_tunnel = ngrok.connect(7860, bind_tls=True)
    frontend_tunnel = ngrok.connect(3000, bind_tls=True)
    backend_url = backend_tunnel.public_url
    frontend_url = frontend_tunnel.public_url

    print("\nBackend URL:", backend_url)
    print("Pipecat built-in client:", f"{backend_url}/client")
    print("Voice console:", f"{frontend_url}/?backend={backend_url}")
    print("\nPress Ctrl+C in the notebook cell to stop processes.")

    children = [p for p in [llama_proc, backend_proc, frontend_proc] if p is not None]
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        for proc in children:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.terminate()
        ngrok.kill()


if __name__ == "__main__":
    main()
