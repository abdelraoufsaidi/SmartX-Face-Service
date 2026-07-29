"""
SMARTX Face Service — container séparé, headless (pas de cv2.imshow).

Expose une API REST utilisée par l'intégration Home Assistant :
- GET  /health
- GET  /snapshot                 -> JPEG de la dernière frame (preview live)
- POST /config                   -> {vto: {ip, username, password, channel, subtype}}
- POST /enroll/start              {"name": "Abd El Raouf"}
- POST /enroll/capture             -> capture le visage de la frame courante
- POST /enroll/finish               -> sauvegarde le profil moyenné
- GET  /profiles
- DELETE /profiles/<name>
- POST /door/open                  -> appelle le VTO Dahua (Digest auth)
"""

import json
import time
import os
import threading
import logging
from pathlib import Path
from urllib.parse import quote

# Doit être fait AVANT l'import de cv2 pour que FFmpeg en tienne compte.
# Force le RTSP en TCP au lieu d'UDP : bien plus stable en WiFi faible/instable
# (moins de paquets perdus = moins d'artefacts et de coupures).
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_sock import Sock
import paho.mqtt.client as mqtt
from insightface.app import FaceAnalysis

from smartx_ha_discovery import publish_discovery, publish_presence, publish_availability
import sip_talk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CONFIG_PATH = Path("smartx_face_config.json")
PROFILES_DIR = Path("face_profiles")
PROFILES_DIR.mkdir(exist_ok=True)

MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883

THRESHOLD = 0.55
DETECTION_INTERVAL = 0.4   # secondes minimum entre deux détections (au lieu de compter les frames)
PRESENCE_TIMEOUT = 8
MIN_ENROLL_CAPTURES = 5
COHERENCE_THRESHOLD = 0.5

app = Flask(__name__)
CORS(app)
sock = Sock(app)

face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(320, 320))  # 320 au lieu de 640 : bien plus rapide sur CPU,
                                                   # suffisant pour un visage proche (porte d'entrée)

# ═══════════════════════════════════════════════════════════════
# ÉTAT PARTAGÉ (protégé par _lock)
# ═══════════════════════════════════════════════════════════════

_lock = threading.Lock()
_state = {
    "config": {
        # "camera" : source RTSP utilisée pour la reconnaissance (peut être le VTO
        # lui-même, ou une caméra IP séparée — décidé côté intégration HA).
        "camera": {"ip": "", "username": "", "password": "", "channel": "1", "subtype": "1"},
        # "vto" : toujours utilisé pour la commande d'ouverture de porte.
        "vto": {"ip": "", "username": "", "password": ""},
        # "talk" : identifiants SIP de l'extension VTS créée manuellement sur le VTO,
        # + IP locale de la machine qui héberge ce container (pour bind SIP/RTP).
        "talk": {"local_ip": "", "extension": "", "password": "", "vto_extension": "8001"},
    },
    "latest_frame": None,        # np.ndarray BGR
    "latest_faces": [],          # dernières détections (pour /snapshot annoté)
    "cap_ok": False,
    "enroll_session": None,      # {"name": str, "embeddings": [...]}
}

_current_call: "sip_talk.VtoCall | None" = None
_call_lock = threading.Lock()

profiles: dict[str, np.ndarray] = {
    p.stem: np.load(str(p)) for p in PROFILES_DIR.glob("*.npy")
}

def _on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        log.info("MQTT connecté")
        publish_discovery(client, list(profiles.keys()))
        publish_availability(client, online=True)
    else:
        log.warning(f"Échec connexion MQTT, code {rc}")


def _on_mqtt_disconnect(client, userdata, rc):
    log.warning(f"MQTT déconnecté (code {rc}), tentative de reconnexion en arrière-plan...")


mqtt_client = mqtt.Client(client_id="smartx_face_service")
mqtt_client.will_set("smartx/face_recognition/availability", "offline", retain=True)
mqtt_client.on_connect = _on_mqtt_connect
mqtt_client.on_disconnect = _on_mqtt_disconnect


def _load_saved_config():
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            with _lock:
                # Fusion défensive : si l'ancien fichier date d'un format précédent
                # (ex: sans clé "camera"), on garde les valeurs par défaut pour
                # les clés manquantes au lieu de planter.
                if "camera" in loaded:
                    _state["config"]["camera"] = loaded["camera"]
                if "vto" in loaded:
                    _state["config"]["vto"] = loaded["vto"]
                if "talk" in loaded:
                    _state["config"]["talk"] = loaded["talk"]
        except Exception as e:
            log.warning(f"Config existante illisible : {e}")


def _save_config():
    with _lock:
        cfg = _state["config"]
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def _build_rtsp_url() -> str:
    with _lock:
        cam = _state["config"]["camera"]
    pwd = quote(cam.get("password", ""), safe="")
    return (
        f"rtsp://{cam.get('username','')}:{pwd}@{cam.get('ip','')}:554"
        f"/cam/realmonitor?channel={cam.get('channel','1')}&subtype={cam.get('subtype','1')}"
    )


def cosine_similarity(a, b) -> float:
    return float(np.dot(a, b))

# ═══════════════════════════════════════════════════════════════
# THREAD DE CAPTURE (RTSP + reconnaissance + MQTT présence)
# ═══════════════════════════════════════════════════════════════

def capture_loop():
    cap = None
    last_detection_time = 0.0
    last_seen = {}
    currently_present = set()

    while True:
        with _lock:
            cam_cfg = dict(_state["config"]["camera"])

        if not cam_cfg.get("ip"):
            time.sleep(2)
            continue

        if cap is None:
            rtsp_url = _build_rtsp_url()
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            with _lock:
                _state["cap_ok"] = cap.isOpened()

        ret, frame = cap.read()
        if not ret:
            with _lock:
                _state["cap_ok"] = False
            cap.release()
            cap = None
            time.sleep(2)
            continue

        with _lock:
            _state["latest_frame"] = frame
            _state["cap_ok"] = True

        now = time.time()
        if (now - last_detection_time) < DETECTION_INTERVAL:
            continue  # on saute la détection, mais la frame reste dispo pour /snapshot
        last_detection_time = now

        faces = face_app.get(frame)
        annotated = []
        recognized_this_frame = None

        for f in faces:
            emb = f.normed_embedding
            best_score, best_name = 0.0, None
            for name, profile in profiles.items():
                score = cosine_similarity(emb, profile)
                if score > best_score:
                    best_score, best_name = score, name

            box = f.bbox.astype(int).tolist()
            recognized = best_score >= THRESHOLD
            annotated.append({
                "box": box,
                "name": best_name if recognized else None,
                "score": round(best_score, 3),
            })
            if recognized:
                last_seen[best_name] = now
                recognized_this_frame = best_name

        with _lock:
            _state["latest_faces"] = annotated

        for person, ts in last_seen.items():
            if (now - ts) < PRESENCE_TIMEOUT and person not in currently_present:
                publish_presence(mqtt_client, person, detected=True)
                currently_present.add(person)

        for person in list(currently_present):
            if (now - last_seen.get(person, 0)) >= PRESENCE_TIMEOUT:
                publish_presence(mqtt_client, person, detected=False)
                currently_present.discard(person)


# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════


@app.route("/enroll_ui", methods=["GET"])
def enroll_ui():
    """Page HTML servie directement par le container, affichée dans le panel HA (iframe)."""
    return Response(_ENROLL_HTML, mimetype="text/html")


_ENROLL_HTML = """
<!DOCTYPE html>

<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SMARTX — Enrôlement facial</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0c0f14;
    --surface: #151a21;
    --surface-2: #1b212a;
    --border: #262d37;
    --text: #eef1f5;
    --muted: #8b93a1;
    --accent: #0090e6;
    --accent-light: #29c2f7;
    --accent-ink: #ffffff;
    --ok: #2fa66a;
    --danger: #e0524c;
  }
  * { box-sizing:border-box; }
  body {
    background: radial-gradient(120% 140% at 50% -10%, #171d26 0%, var(--bg) 55%);
    color: var(--text);
    font-family: 'Manrope', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    text-align:center; padding:24px 16px 32px; margin:0; min-height:100vh;
  }
  .brand { font-size:12px; font-weight:700; letter-spacing:.16em; color:var(--muted); text-transform:uppercase; margin-bottom:16px; }
  .brand span {
    background:linear-gradient(135deg, var(--accent-light), var(--accent));
    -webkit-background-clip:text; background-clip:text; color:transparent;
  }

  .card {
    max-width:480px; margin:0 auto; text-align:left;
    background:var(--surface); border:1px solid var(--border); border-radius:20px;
    padding:16px; box-shadow:0 20px 40px -20px rgba(0,0,0,.6);
  }

  .monitor { position:relative; border-radius:13px; overflow:hidden; background:#000; }
  .monitor::before {
    content:"⏺ ENRÔLEMENT"; position:absolute; top:12px; left:12px; z-index:2;
    font-size:11px; font-weight:800; letter-spacing:.06em; color:var(--accent);
    background:rgba(10,12,16,.6); padding:4px 8px; border-radius:6px;
    -webkit-backdrop-filter:blur(4px); backdrop-filter:blur(4px);
  }
  img#preview { width:100%; display:block; }

  .field { margin-top:16px; }
  .field label { display:block; font-size:12px; font-weight:700; color:var(--muted); margin-bottom:6px; letter-spacing:.03em; }
  .input-wrap { position:relative; }
  .input-wrap svg {
    position:absolute; left:13px; top:50%; transform:translateY(-50%);
    width:17px; height:17px; color:var(--muted); pointer-events:none;
  }
  input#nameInput {
    width:100%; font-family:inherit; font-size:15px; font-weight:600; color:var(--text);
    background:var(--surface-2); border:1px solid var(--border); border-radius:12px;
    padding:12px 14px 12px 40px; outline:none;
    transition: border-color .15s ease, box-shadow .15s ease;
  }
  input#nameInput::placeholder { color:var(--muted); font-weight:500; }
  input#nameInput:focus { border-color:var(--accent); box-shadow:0 0 0 3px rgba(0,144,230,.22); }

  button.primary {
    width:100%; margin-top:10px; font-family:inherit; font-size:15px; font-weight:800;
    color:var(--accent-ink); background:var(--accent); border:none; border-radius:12px;
    padding:13px; cursor:pointer; display:flex; align-items:center; justify-content:center; gap:8px;
    box-shadow:0 8px 20px -8px rgba(0,144,230,.5);
    transition: transform .12s ease, opacity .15s ease;
  }
  button.primary svg { width:18px; height:18px; flex-shrink:0; }
  button.primary:active { transform:scale(.98); }
  button.primary:disabled { background:var(--surface-2); color:var(--muted); box-shadow:none; cursor:not-allowed; }

  .progress { display:flex; align-items:center; gap:8px; margin:16px 2px 4px; }
  .progress .dot { flex:1; height:6px; border-radius:999px; background:var(--surface-2); border:1px solid var(--border); transition:background .2s ease, border-color .2s ease; }
  .progress .dot.filled { background:var(--ok); border-color:var(--ok); }
  .progress-label { font-size:12px; font-weight:700; color:var(--muted); margin-bottom:2px; }

  .actions { display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; margin-top:14px; }
  button.action {
    font-family:inherit; font-size:13px; font-weight:700; color:var(--text);
    background:var(--surface-2); border:1px solid var(--border); border-radius:11px;
    padding:11px 6px; cursor:pointer; display:flex; flex-direction:column; align-items:center; gap:5px;
    transition: transform .12s ease, background-color .15s ease, border-color .15s ease, opacity .15s ease;
  }
  button.action svg { width:18px; height:18px; }
  button.action:active:not(:disabled) { transform:scale(.96); }
  button.action:disabled { opacity:.4; cursor:not-allowed; }
  #captureBtn:not(:disabled) { border-color:var(--accent); color:var(--accent); }
  #finishBtn:not(:disabled) { border-color:var(--ok); color:var(--ok); }
  #cancelBtn:not(:disabled) { border-color:var(--danger); color:var(--danger); }

  #status {
    margin-top:16px; font-size:13px; font-weight:600; color:var(--muted);
    display:flex; align-items:center; gap:8px; min-height:20px;
  }
  #status::before { content:""; width:7px; height:7px; border-radius:50%; background:var(--muted); flex-shrink:0; }
  #status.ok { color:var(--ok); } #status.ok::before { background:var(--ok); }
  #status.warn { color:var(--accent); } #status.warn::before { background:var(--accent); }
  #status.err { color:var(--danger); } #status.err::before { background:var(--danger); }
</style>
</head>
<body>
  <div class="brand">SMART<span>X</span> · Enrôlement facial</div>
  <div class="card">
    <div class="monitor">
      <img id="preview" src="stream" alt="Flux caméra">
    </div>

    <div class="field">
      <label for="nameInput">Nom de la personne</label>
      <div class="input-wrap">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
        <input id="nameInput" type="text" placeholder="ex. Abd El Raouf">
      </div>
      <button id="startBtn" class="primary" onclick="startEnroll()">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        Démarrer
      </button>
    </div>

    <div class="progress-label">Captures</div>
    <div class="progress" id="progressDots"></div>

    <div class="actions">
      <button id="captureBtn" class="action" onclick="capture()" disabled>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
        Capturer
      </button>
      <button id="finishBtn" class="action" onclick="finish()" disabled>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
        Terminer
      </button>
      <button id="cancelBtn" class="action" onclick="cancelEnroll()" disabled>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        Annuler
      </button>
    </div>

    <div id="status">Prêt à démarrer.</div>
  </div>

<script>
let captureCount = 0;
const MIN_CAPTURES = 5;

const dotsEl = document.getElementById("progressDots");
function renderDots(count) {
  dotsEl.innerHTML = "";
  for (let i = 0; i < MIN_CAPTURES; i++) {
    const d = document.createElement("div");
    d.className = "dot" + (i < count ? " filled" : "");
    dotsEl.appendChild(d);
  }
}
renderDots(0);

function setStatus(msg, cls) {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.className = cls || "";
}

// Le flux /stream est du MJPEG continu (multipart) : le navigateur l'affiche
// nativement dans la balise <img>, pas besoin de rafraîchir manuellement.

async function startEnroll() {
  const name = document.getElementById("nameInput").value.trim();
  if (!name) { setStatus("Entre un nom d'abord.", "err"); return; }

  const resp = await fetch("enroll/start", {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({name})
  });
  const data = await resp.json();
  if (data.started) {
    captureCount = 0;
    renderDots(0);
    setStatus(`Session démarrée pour '${name}'. Capture 0/${MIN_CAPTURES}.`, "ok");
    document.getElementById("captureBtn").disabled = false;
    document.getElementById("cancelBtn").disabled = false;
    document.getElementById("finishBtn").disabled = true;
    document.getElementById("startBtn").disabled = true;
  } else {
    setStatus(data.error || "Erreur au démarrage", "err");
  }
}

async function capture() {
  const resp = await fetch("enroll/capture", { method: "POST" });
  const data = await resp.json();
  if (data.accepted) {
    captureCount = data.count;
    renderDots(captureCount);
    setStatus(`✓ Capture ${data.count}/${data.min_required}`, "ok");
    if (data.ready) {
      document.getElementById("finishBtn").disabled = false;
    }
  } else {
    setStatus(`⚠ ${data.reason}`, "warn");
  }
}

async function finish() {
  const resp = await fetch("enroll/finish", { method: "POST" });
  const data = await resp.json();
  if (data.saved) {
    setStatus(`✓ Profil '${data.name}' enregistré (${data.captures} captures).`, "ok");
    resetUI();
  } else {
    setStatus(data.reason || "Erreur à la sauvegarde", "err");
  }
}

async function cancelEnroll() {
  await fetch("enroll/cancel", { method: "POST" });
  setStatus("Session annulée.", "warn");
  resetUI();
}

function resetUI() {
  document.getElementById("nameInput").value = "";
  document.getElementById("startBtn").disabled = false;
  document.getElementById("captureBtn").disabled = true;
  document.getElementById("finishBtn").disabled = true;
  document.getElementById("cancelBtn").disabled = true;
  captureCount = 0;
  renderDots(0);
}
</script>
</body>
</html>
"""


_TALK_HTML = """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SMARTX — Interphone</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0c0f14;
    --surface: #151a21;
    --border: #262d37;
    --text: #eef1f5;
    --muted: #8b93a1;
    --accent: #0090e6;
    --accent-light: #29c2f7;
    --accent-ink: #ffffff;
    --call: #2fa66a;
    --call-ink: #08150e;
    --hangup: #e0524c;
    --focus: #29c2f7;
  }
  * { box-sizing:border-box; }
  body {
    background: radial-gradient(120% 140% at 50% -10%, #171d26 0%, var(--bg) 55%);
    color: var(--text);
    font-family: 'Manrope', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    text-align:center; margin:0; padding:20px 16px 28px;
    min-height:100vh;
  }
  .brand { font-size:12px; font-weight:700; letter-spacing:.16em; color:var(--muted); text-transform:uppercase; margin-bottom:14px; }
  .brand span {
    background:linear-gradient(135deg, var(--accent-light), var(--accent));
    -webkit-background-clip:text; background-clip:text; color:transparent;
  }

  .monitor {
    position:relative; max-width:480px; margin:0 auto;
    border-radius:18px; padding:6px;
    background:linear-gradient(160deg,#232a34,#11151b);
    border:1px solid var(--border);
    box-shadow:0 20px 40px -20px rgba(0,0,0,.6), inset 0 0 0 1px rgba(255,255,255,.02);
  }
  .monitor::before {
    content:"● LIVE"; position:absolute; top:16px; left:16px; z-index:2;
    font-size:11px; font-weight:800; letter-spacing:.06em; color:#ff5f5f;
    background:rgba(10,12,16,.55); padding:4px 8px; border-radius:6px;
    -webkit-backdrop-filter:blur(4px); backdrop-filter:blur(4px);
  }
  img#stream { width:100%; display:block; border-radius:13px; background:#000; }

  .row { margin-top:22px; display:flex; gap:14px; justify-content:center; flex-wrap:wrap; }

  button.ctrl {
    font-family:inherit; font-size:15px; font-weight:700; color:var(--text);
    border:none; border-radius:16px; cursor:pointer;
    padding:14px 24px 14px 18px; min-width:168px;
    display:inline-flex; align-items:center; gap:10px; justify-content:center;
    transition: transform .12s ease, box-shadow .12s ease, background-color .15s ease;
  }
  button.ctrl:active { transform:scale(.96); }
  button.ctrl svg { width:20px; height:20px; flex-shrink:0; }

  #callBtn {
    background:var(--call); color:var(--call-ink);
    box-shadow:0 8px 20px -8px rgba(47,166,106,.55);
  }
  #callBtn:hover { box-shadow:0 10px 24px -8px rgba(47,166,106,.7); }
  #callBtn.active {
    background:var(--hangup); color:#1a0908;
    box-shadow:0 8px 20px -8px rgba(224,82,76,.6);
    animation: pulseRing 1.6s ease-out infinite;
  }
  #doorBtn {
    background:var(--accent); color:var(--accent-ink);
    box-shadow:0 8px 20px -8px rgba(0,144,230,.5);
  }
  #doorBtn:hover { box-shadow:0 10px 24px -8px rgba(0,144,230,.65); }
  #doorBtn.busy { opacity:.7; cursor:default; animation:none; }

  @keyframes pulseRing {
    0%   { box-shadow:0 8px 20px -8px rgba(224,82,76,.6), 0 0 0 0 rgba(224,82,76,.45); }
    70%  { box-shadow:0 8px 20px -8px rgba(224,82,76,.6), 0 0 0 14px rgba(224,82,76,0); }
    100% { box-shadow:0 8px 20px -8px rgba(224,82,76,.6), 0 0 0 0 rgba(224,82,76,0); }
  }

  #status {
    margin-top:18px; font-size:13px; color:var(--muted); font-weight:600;
    display:inline-flex; align-items:center; gap:8px;
    background:var(--surface); border:1px solid var(--border);
    padding:8px 16px; border-radius:999px;
  }
  #status::before { content:""; width:7px; height:7px; border-radius:50%; background:var(--muted); }
  #status.ok::before { background:var(--call); }
  #status.warn::before { background:var(--accent); }
  #status.err::before { background:var(--hangup); }
</style>
</head>
<body>
  <div class="brand">SMART<span>X</span> · Interphone</div>
  <div class="monitor">
    <img id="stream" src="stream" alt="Flux caméra porte">
  </div>
  <div class="row">
    <button id="callBtn" class="ctrl" onclick="toggleCall()">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z"/></svg>
      <span id="callLabel">Appeler</span>
    </button>
    <button id="doorBtn" class="ctrl" onclick="openDoor()">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 21h18M6 21V4a1 1 0 0 1 1-1h7l5 3v15M13 12v.01"/></svg>
      <span>Ouvrir la porte</span>
    </button>
  </div>
  <div id="status">Prêt.</div>

<script>
const statusEl = document.getElementById('status');
const callBtn = document.getElementById('callBtn');
const callLabel = document.getElementById('callLabel');
const doorBtn = document.getElementById('doorBtn');

function setCallState(active) {
  callLabel.textContent = active ? "Raccrocher" : "Appeler";
  callBtn.classList.toggle("active", active);
}

let ws = null;
let audioCtx = null;
let micStream = null;
let micNode = null;
let playTime = 0;

function setStatus(msg, cls) {
  statusEl.textContent = msg;
  statusEl.className = cls || "";
}

async function openDoor() {
  if (doorBtn.classList.contains("busy")) return;
  doorBtn.classList.add("busy");
  setStatus("Ouverture de la porte...", "warn");
  try {
    const resp = await fetch('door/open', { method: 'POST' });
    const data = await resp.json();
    setStatus(data.opened ? "Porte ouverte." : "Échec ouverture porte.", data.opened ? "ok" : "err");
  } catch (e) {
    setStatus("Erreur réseau (ouverture porte).", "err");
  } finally {
    setTimeout(() => doorBtn.classList.remove("busy"), 1200);
  }
}

function toggleCall() {
  if (ws) {
    stopCall();
  } else {
    startCall();
  }
}

async function startCall() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("Micro indisponible : la page doit être servie en HTTPS (ou via localhost).", "err");
    return;
  }

  setStatus("Connexion...", "warn");
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    setStatus("Micro refusé par le navigateur.", "err");
    return;
  }

  const proto = location.protocol === "https:" ? "wss" : "ws";
  // Construit le chemin du WebSocket à partir de la page courante :
  // fonctionne en accès direct (/talk_ui -> /talk/ws) comme via le proxy HA
  // (/api/smartx_face/<entry_id>/talk_ui -> /api/smartx_face/<entry_id>/talk/ws).
  const wsPath = location.pathname.endsWith("/talk_ui")
    ? location.pathname.slice(0, -("/talk_ui".length)) + "/talk/ws"
    : "/talk/ws";
  ws = new WebSocket(`${proto}://${location.host}${wsPath}`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    playTime = audioCtx.currentTime;

    const source = audioCtx.createMediaStreamSource(micStream);
    // ScriptProcessorNode : simple et largement supporté (suffisant ici, pas besoin
    // de la latence ultra-faible d'un AudioWorklet pour un interphone).
    micNode = audioCtx.createScriptProcessor(2048, 1, 1);
    source.connect(micNode);
    micNode.connect(audioCtx.destination);  // requis par certains navigateurs pour tourner le graphe

    const nativeRate = audioCtx.sampleRate;
    const ratio = nativeRate / 8000;

    micNode.onaudioprocess = (e) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const input = e.inputBuffer.getChannelData(0);
      const outLen = Math.floor(input.length / ratio);
      const pcm16 = new Int16Array(outLen);
      for (let i = 0; i < outLen; i++) {
        const srcIdx = Math.floor(i * ratio);
        let s = input[srcIdx];
        s = Math.max(-1, Math.min(1, s));
        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      ws.send(pcm16.buffer);
    };

    setCallState(true);
    setStatus("Appel en cours...", "ok");
  };

  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      const msg = JSON.parse(event.data);
      if (msg.type === "error") setStatus("Erreur : " + msg.message, "err");
      if (msg.type === "connected") setStatus("Appel en cours...", "ok");
      return;
    }
    // PCM16 mono 8kHz venant du VTO -> lecture via un petit buffer audio.
    const pcm16 = new Int16Array(event.data);
    const float32 = new Float32Array(pcm16.length);
    for (let i = 0; i < pcm16.length; i++) float32[i] = pcm16[i] / 0x8000;

    const buffer = audioCtx.createBuffer(1, float32.length, 8000);
    buffer.copyToChannel(float32, 0);
    const src = audioCtx.createBufferSource();
    src.buffer = buffer;
    src.connect(audioCtx.destination);

    const now = audioCtx.currentTime;
    if (playTime < now) playTime = now + 0.05;  // petit jitter buffer
    src.start(playTime);
    playTime += buffer.duration;
  };

  ws.onclose = () => stopCall();
  ws.onerror = () => setStatus("Erreur WebSocket.", "err");
}

function stopCall() {
  if (micNode) { micNode.disconnect(); micNode = null; }
  if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  if (ws) { try { ws.close(); } catch(e) {} ws = null; }
  setCallState(false);
  setStatus("Appel terminé.", "warn");
}
</script>
</body>
</html>
"""


@app.route("/health", methods=["GET"])
def health():
    with _lock:
        return jsonify({
            "camera_connected": _state["cap_ok"],
            "profiles": list(profiles.keys()),
        })


def _generate_mjpeg():
    """Générateur MJPEG : flux vidéo continu (multipart), bien plus fluide qu'un polling d'images."""
    while True:
        with _lock:
            frame = _state["latest_frame"]
            faces = _state["latest_faces"]

        if frame is None:
            time.sleep(0.1)
            continue

        display = frame.copy()
        for f in faces:
            x1, y1, x2, y2 = f["box"]
            color = (0, 255, 0) if f["name"] else (0, 0, 255)
            label = f"{f['name']} ({f['score']:.2f})" if f["name"] else f"? ({f['score']:.2f})"
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ok, jpeg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )
        time.sleep(0.08)  # ~12 fps, largement suffisant pour une preview d'enrôlement


@app.route("/stream", methods=["GET"])
def stream():
    return Response(_generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/snapshot", methods=["GET"])
def snapshot():
    """Retourne la dernière frame en JPEG, avec les rectangles de détection dessinés."""
    with _lock:
        frame = _state["latest_frame"]
        faces = _state["latest_faces"]

    if frame is None:
        return jsonify({"error": "Pas encore de frame disponible"}), 503

    display = frame.copy()
    for f in faces:
        x1, y1, x2, y2 = f["box"]
        color = (0, 255, 0) if f["name"] else (0, 0, 255)
        label = f"{f['name']} ({f['score']:.2f})" if f["name"] else f"? ({f['score']:.2f})"
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    ok, jpeg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return jsonify({"error": "Encodage JPEG échoué"}), 500
    return Response(jpeg.tobytes(), mimetype="image/jpeg")


@app.route("/config", methods=["POST"])
def set_config():
    """Reçoit la config depuis l'intégration HA :
    {camera: {ip, username, password, channel, subtype}, vto: {ip, username, password}}"""
    data = request.get_json(silent=True) or {}
    camera = data.get("camera")
    vto = data.get("vto")
    talk = data.get("talk")

    with _lock:
        if camera:
            _state["config"]["camera"] = camera
            _state["cap_ok"] = False  # force la reconnexion RTSP au prochain tour
        if vto:
            _state["config"]["vto"] = vto
        if talk:
            _state["config"]["talk"] = talk

    _save_config()
    log.info("Configuration mise à jour depuis HA")
    return jsonify({"saved": True})


@app.route("/config", methods=["GET"])
def get_config():
    with _lock:
        cfg = json.loads(json.dumps(_state["config"]))  # copie profonde
    cfg["camera"]["password"] = "***" if cfg["camera"].get("password") else ""
    cfg["vto"]["password"] = "***" if cfg["vto"].get("password") else ""
    cfg["talk"]["password"] = "***" if cfg["talk"].get("password") else ""
    return jsonify(cfg)


@sock.route("/talk/ws")
def talk_ws(ws):
    """
    Pont audio bidirectionnel pour la fonction interphone.
    Le cycle de vie de l'appel SIP est calé sur celui de la connexion WebSocket :
    - connexion ouverte -> REGISTER + INVITE vers le VTO
    - messages binaires reçus du navigateur (PCM16 mono 8kHz) -> envoyés en RTP au VTO
    - audio RTP reçu du VTO -> renvoyé en binaire au navigateur (thread dédié)
    - connexion fermée (ou VTO raccroche) -> BYE + libération des sockets
    """
    global _current_call

    with _lock:
        talk_cfg = dict(_state["config"]["talk"])

    if not talk_cfg.get("local_ip") or not talk_cfg.get("extension"):
        ws.send(json.dumps({"type": "error", "message": "Configuration interphone incomplète (IP locale / extension VTS manquante)"}))
        return

    with _call_lock:
        if _current_call is not None and _current_call.active:
            ws.send(json.dumps({"type": "error", "message": "Un appel est déjà en cours"}))
            return
        call = sip_talk.VtoCall(
            local_ip=talk_cfg["local_ip"],
            vto_ip=_state["config"]["vto"].get("ip", ""),
            my_extension=talk_cfg["extension"],
            my_password=talk_cfg["password"],
            vto_extension=talk_cfg.get("vto_extension", "8001"),
        )
        _current_call = call

    ok, message = call.start()
    if not ok:
        ws.send(json.dumps({"type": "error", "message": message}))
        with _call_lock:
            _current_call = None
        return

    ws.send(json.dumps({"type": "connected"}))
    log.info("Appel interphone établi (WebSocket connecté)")

    def _forward_vto_audio():
        while call.active:
            chunk = call.pull_audio(timeout=1.0)
            if chunk is None:
                continue
            try:
                ws.send(chunk)
            except Exception:
                break

    sender_thread = threading.Thread(target=_forward_vto_audio, daemon=True)
    sender_thread.start()

    try:
        while call.active:
            data = ws.receive(timeout=1.0)
            if data is None:
                continue
            if isinstance(data, (bytes, bytearray)):
                call.push_audio(bytes(data))
    except Exception as e:
        log.info(f"WebSocket interphone fermé : {e}")
    finally:
        call.stop()
        with _call_lock:
            _current_call = None
        log.info("Appel interphone terminé")


@app.route("/talk_ui", methods=["GET"])
def talk_ui():
    """Page HTML servie par le container, affichée dans le panel HA (iframe)."""
    return Response(_TALK_HTML, mimetype="text/html")


@app.route("/door/open", methods=["POST"])
def open_door():
    with _lock:
        vto = dict(_state["config"]["vto"])

    if not vto.get("ip"):
        return jsonify({"error": "VTO non configuré"}), 400

    url = (
        f"http://{vto['ip']}/cgi-bin/accessControl.cgi"
        f"?action=openDoor&channel=1&UserID=101&Type=Remote"
    )
    try:
        resp = requests.get(url, auth=HTTPDigestAuth(vto["username"], vto["password"]), timeout=5)
        ok = resp.status_code == 200
        log.info(f"Ouverture porte VTO -> {'OK' if ok else 'ÉCHEC'} ({resp.status_code})")
        return jsonify({"opened": ok, "status_code": resp.status_code})
    except Exception as e:
        log.error(f"Erreur ouverture porte : {e}")
        return jsonify({"opened": False, "error": str(e)}), 500


# ── Enrôlement piloté par API (remplace les touches clavier) ──

@app.route("/enroll/start", methods=["POST"])
def enroll_start():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip().lower()
    if not name:
        return jsonify({"error": "Nom manquant"}), 400

    with _lock:
        _state["enroll_session"] = {"name": name, "embeddings": []}

    return jsonify({"started": True, "name": name})


@app.route("/enroll/capture", methods=["POST"])
def enroll_capture():
    with _lock:
        session = _state["enroll_session"]
        frame = _state["latest_frame"]

    if session is None:
        return jsonify({"error": "Aucune session d'enrôlement en cours"}), 400
    if frame is None:
        return jsonify({"error": "Pas encore de frame disponible"}), 503

    faces = face_app.get(frame)
    if not faces:
        return jsonify({"accepted": False, "reason": "Aucun visage détecté"}), 200
    if len(faces) > 1:
        return jsonify({"accepted": False, "reason": "Plusieurs visages détectés"}), 200

    emb = faces[0].normed_embedding

    with _lock:
        embeddings = session["embeddings"]
        if embeddings:
            centroid = np.mean(embeddings, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            score = cosine_similarity(emb, centroid)
            if score < COHERENCE_THRESHOLD:
                return jsonify({
                    "accepted": False,
                    "reason": f"Capture incohérente (score={score:.3f})",
                    "count": len(embeddings),
                }), 200

        embeddings.append(emb)
        count = len(embeddings)

    return jsonify({
        "accepted": True,
        "count": count,
        "ready": count >= MIN_ENROLL_CAPTURES,
        "min_required": MIN_ENROLL_CAPTURES,
    })


@app.route("/enroll/finish", methods=["POST"])
def enroll_finish():
    with _lock:
        session = _state["enroll_session"]

    if session is None:
        return jsonify({"error": "Aucune session d'enrôlement en cours"}), 400

    embeddings = session["embeddings"]
    if len(embeddings) < MIN_ENROLL_CAPTURES:
        return jsonify({
            "saved": False,
            "reason": f"Pas assez de captures ({len(embeddings)}/{MIN_ENROLL_CAPTURES})",
        }), 400

    profile = np.mean(embeddings, axis=0)
    profile = profile / np.linalg.norm(profile)

    name = session["name"]
    profile_path = PROFILES_DIR / f"{name}.npy"
    np.save(str(profile_path), profile)
    profiles[name] = profile

    with _lock:
        _state["enroll_session"] = None

    publish_discovery(mqtt_client, list(profiles.keys()))
    log.info(f"Profil '{name}' enregistré avec {len(embeddings)} captures")

    return jsonify({"saved": True, "name": name, "captures": len(embeddings)})


@app.route("/enroll/cancel", methods=["POST"])
def enroll_cancel():
    with _lock:
        _state["enroll_session"] = None
    return jsonify({"cancelled": True})


@app.route("/profiles", methods=["GET"])
def list_profiles():
    return jsonify({"profiles": list(profiles.keys())})


@app.route("/profiles/<name>", methods=["DELETE"])
def delete_profile(name):
    path = PROFILES_DIR / f"{name}.npy"
    if path.exists():
        path.unlink()
        profiles.pop(name, None)
        publish_discovery(mqtt_client, list(profiles.keys()))
        return jsonify({"deleted": True, "name": name})
    return jsonify({"error": "Profil introuvable"}), 404


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _load_saved_config()

    # connect_async + reconnect_delay_set : le service démarre même si le broker
    # MQTT n'est pas encore joignable, et se reconnecte tout seul en arrière-plan.
    # publish_discovery/availability sont déclenchés via le callback on_connect,
    # pas ici, pour être republiés à chaque (re)connexion.
    mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)
    try:
        mqtt_client.connect_async(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)
    except Exception as e:
        log.warning(f"Connexion MQTT initiale échouée ({e}), nouvelle tentative en arrière-plan")
    mqtt_client.loop_start()

    threading.Thread(target=capture_loop, daemon=True).start()

    log.info("Démarrage du service SMARTX Face sur 0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
