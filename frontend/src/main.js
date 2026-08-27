import { PipecatClient } from "@pipecat-ai/client-js";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";
import "./styles.css";

const backendUrlInput = document.querySelector("#backendUrl");
const connectBtn = document.querySelector("#connectBtn");
const disconnectBtn = document.querySelector("#disconnectBtn");
const muteBtn = document.querySelector("#muteBtn");
const stateText = document.querySelector("#stateText");
const statusDot = document.querySelector("#statusDot");
const turnState = document.querySelector("#turnState");
const latencyValue = document.querySelector("#latencyValue");
const transcript = document.querySelector("#transcript");

let client = null;
let muted = false;
let lastUserFinalAt = 0;
let firstAudioSeen = false;

const params = new URLSearchParams(window.location.search);
const urlFromQuery = params.get("backend");
if (urlFromQuery) {
  backendUrlInput.value = urlFromQuery;
}

function setState(text, connected = false) {
  stateText.textContent = text;
  statusDot.classList.toggle("connected", connected);
}

function addLine(role, text, transient = false) {
  if (!text) return;
  const row = document.createElement("div");
  row.className = `line ${role}${transient ? " transient" : ""}`;
  row.textContent = text;
  transcript.appendChild(row);
  transcript.scrollTop = transcript.scrollHeight;
}

function normalizeBackendUrl() {
  const raw = backendUrlInput.value.trim().replace(/\/+$/, "");
  if (!raw) throw new Error("Backend URL is required");
  return raw;
}

function handleBotAudio(track, participant) {
  if (participant?.local || track.kind !== "audio") return;
  const audio = document.createElement("audio");
  audio.autoplay = true;
  audio.playsInline = true;
  audio.srcObject = new MediaStream([track]);
  audio.addEventListener(
    "playing",
    () => {
      if (!firstAudioSeen && lastUserFinalAt > 0) {
        latencyValue.textContent = `${Date.now() - lastUserFinalAt} ms`;
        firstAudioSeen = true;
      }
    },
    { once: true },
  );
  document.body.appendChild(audio);
  audio.play().catch(() => {});
}

function createClient() {
  return new PipecatClient({
    transport: new SmallWebRTCTransport({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    }),
    enableMic: true,
    enableCam: false,
    timeout: 30000,
    callbacks: {
      onConnected: () => {
        setState("Connected", true);
        connectBtn.disabled = true;
        disconnectBtn.disabled = false;
        muteBtn.disabled = false;
      },
      onDisconnected: () => {
        setState("Disconnected", false);
        connectBtn.disabled = false;
        disconnectBtn.disabled = true;
        muteBtn.disabled = true;
        turnState.textContent = "Idle";
      },
      onTransportStateChanged: (state) => {
        setState(state, state === "ready" || state === "connected");
      },
      onBotReady: () => {
        setState("Bot ready", true);
      },
      onUserStartedSpeaking: () => {
        firstAudioSeen = false;
        turnState.textContent = "Listening";
      },
      onUserStoppedSpeaking: () => {
        lastUserFinalAt = Date.now();
        turnState.textContent = "Thinking";
      },
      onBotStartedSpeaking: () => {
        turnState.textContent = "Speaking";
      },
      onBotStoppedSpeaking: () => {
        turnState.textContent = "Idle";
      },
      onUserTranscript: (data) => {
        const text = data?.text || data?.transcript || "";
        if (data?.final ?? data?.finalized ?? true) {
          lastUserFinalAt = Date.now();
          addLine("user", text);
        }
      },
      onBotTranscript: (data) => {
        addLine("bot", data?.text || data?.transcript || "");
      },
      onTrackStarted: handleBotAudio,
      onError: (message) => {
        addLine("error", `Error: ${JSON.stringify(message)}`);
      },
    },
  });
}

connectBtn.addEventListener("click", () => {
  try {
    const backendUrl = normalizeBackendUrl();
    client = createClient();
    setState("Connecting...");
    client.connect({
      connection_url: `${backendUrl}/api/offer`,
    });
  } catch (error) {
    addLine("error", error.message);
  }
});

disconnectBtn.addEventListener("click", async () => {
  if (!client) return;
  await client.disconnect();
  client = null;
});

muteBtn.addEventListener("click", async () => {
  if (!client) return;
  muted = !muted;
  await client.enableMic(!muted);
  muteBtn.textContent = muted ? "Unmute mic" : "Mute mic";
});
