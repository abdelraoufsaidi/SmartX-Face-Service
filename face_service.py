"""
SMARTX Face Service — container séparé, headless (pas de cv2.imshow).

Expose une API REST utilisée par l'intégration Home Assistant :
- GET  /health
- GET  /snapshot                 -> JPEG de la dernière frame (preview live)
- POST /config                   -> {vto: {ip, username, password, channel, subtype}}
- POST /enroll/start              {"name": "nabil"}
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
<title>SMARTX — Enrôlement facial</title>
<style>
  body { background:#1a1a1a; color:#eee; font-family: sans-serif; text-align:center; padding:20px; }
  img { max-width:100%; border-radius:8px; border:2px solid #F5A623; }
  input, button {
    font-size:16px; padding:8px 12px; margin:6px; border-radius:6px; border:none;
  }
  button { background:#F5A623; color:#1a1a1a; cursor:pointer; font-weight:bold; }
  button:disabled { background:#555; color:#999; cursor:not-allowed; }
  #status { margin-top:10px; min-height:24px; }
  .ok { color:#4caf50; }
  .warn { color:#ff9800; }
  .err { color:#f44336; }
</style>
</head>
<body>
  <h2>SMARTX — Enrôlement facial</h2>
  <img id="preview" src="stream" alt="Flux caméra">
  <div>
    <input id="nameInput" type="text" placeholder="Nom de la personne">
    <button id="startBtn" onclick="startEnroll()">Démarrer</button>
  </div>
  <div>
    <button id="captureBtn" onclick="capture()" disabled>Capturer</button>
    <button id="finishBtn" onclick="finish()" disabled>Terminer</button>
    <button id="cancelBtn" onclick="cancelEnroll()" disabled>Annuler</button>
  </div>
  <div id="status"></div>

<script>
let captureCount = 0;
const MIN_CAPTURES = 5;

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
<title>SMARTX — Interphone</title>
<style>
  body { background:#111; color:#eee; font-family: sans-serif; text-align:center; margin:0; padding:16px; }
  img#stream { width:100%; max-width:480px; border-radius:12px; border:2px solid #333; }
  .row { margin-top:16px; display:flex; gap:12px; justify-content:center; flex-wrap:wrap; }
  button { font-size:16px; padding:14px 22px; border:none; border-radius:10px; cursor:pointer; }
  #callBtn { background:#2b8a3e; color:white; }
  #callBtn.active { background:#c0392b; }
  #doorBtn { background:#1e6fd9; color:white; }
  #status { margin-top:10px; font-size:14px; color:#aaa; }
</style>
</head>
<body>
  <img id="stream" src="stream" alt="Flux caméra porte">
  <div class="row">
    <button id="callBtn" onclick="toggleCall()">📞 Appeler</button>
    <button id="doorBtn" onclick="openDoor()">🚪 Ouvrir la porte</button>
  </div>
  <div id="status">Prêt.</div>

<script>
const statusEl = document.getElementById('status');
const callBtn = document.getElementById('callBtn');

let ws = null;
let audioCtx = null;
let micStream = null;
let micNode = null;
let playTime = 0;

function setStatus(msg) { statusEl.textContent = msg; }

async function openDoor() {
  setStatus("Ouverture de la porte...");
  try {
    const resp = await fetch('door/open', { method: 'POST' });
    const data = await resp.json();
    setStatus(data.opened ? "Porte ouverte." : "Échec ouverture porte.");
  } catch (e) {
    setStatus("Erreur réseau (ouverture porte).");
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
    setStatus("Micro indisponible : la page doit être servie en HTTPS (ou via localhost).");
    return;
  }

  setStatus("Connexion...");
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    setStatus("Micro refusé par le navigateur.");
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

    callBtn.textContent = "📴 Raccrocher";
    callBtn.classList.add("active");
    setStatus("Appel en cours...");
  };

  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      const msg = JSON.parse(event.data);
      if (msg.type === "error") setStatus("Erreur : " + msg.message);
      if (msg.type === "connected") setStatus("Appel en cours...");
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
  ws.onerror = () => setStatus("Erreur WebSocket.");
}

function stopCall() {
  if (micNode) { micNode.disconnect(); micNode = null; }
  if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  if (ws) { try { ws.close(); } catch(e) {} ws = null; }
  callBtn.textContent = "📞 Appeler";
  callBtn.classList.remove("active");
  setStatus("Appel terminé.");
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
