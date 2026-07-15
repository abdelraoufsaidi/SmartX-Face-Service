# SMARTX Face Service

Service headless (RTSP + InsightFace + MQTT) qui exécute la reconnaissance
faciale pour SMARTX. Tourne dans son propre container, séparé de Home Assistant.

Voir l'intégration HA qui pilote ce service :
[SmartX-Face-HA](https://github.com/abdelraoufsaidi/SmartX-Face-HA)

## Ce que fait ce service

- Capture en continu le flux RTSP d'une caméra VTO Dahua
- Détecte et reconnaît les visages via InsightFace (`buffalo_l`)
- Publie la présence par personne en MQTT Discovery (compatible Home Assistant)
- Expose une API REST pour l'enrôlement (piloté depuis un navigateur, pas de clavier requis)
- Déclenche l'ouverture de porte via l'API CGI du VTO (Digest auth)

## Lancer avec Docker (recommandé)

```bash
git clone https://github.com/abdelraoufsaidi/SmartX-Face-Service.git
cd SmartX-Face-Service
mkdir -p data/face_profiles
touch data/smartx_face_config.json
docker compose up -d --build
```

⚠️ **Le `touch` est obligatoire avant le premier `docker compose up`.** Si le
fichier `data/smartx_face_config.json` n'existe pas encore sur l'hôte au
moment du bind-mount, Docker crée un **dossier** à sa place au lieu d'un
fichier, et le service ne pourra jamais lire/écrire sa configuration.

Le service écoute sur le port **5001**.

## Lancer sans Docker

```bash
pip install -r requirements.txt
python face_service.py
```

## Configuration

La configuration (IP/user/password VTO) se fait via l'intégration Home Assistant
(config flow), qui appelle `POST /config` sur ce service. Elle peut aussi être
poussée manuellement :

```bash
curl -X POST http://localhost:5001/config \
  -H "Content-Type: application/json" \
  -d '{"vto": {"ip": "192.168.1.X", "username": "admin", "password": "...", "channel": "1", "subtype": "1"}}'
```

## Endpoints principaux

| Endpoint | Méthode | Description |
|---|---|---|
| `/health` | GET | État de la connexion caméra + profils chargés |
| `/stream` | GET | Flux vidéo MJPEG continu (preview live) |
| `/snapshot` | GET | Image JPEG unique annotée |
| `/enroll_ui` | GET | Page web d'enrôlement (utilisée par le panel HA) |
| `/enroll/start` | POST | Démarre une session d'enrôlement `{"name": "..."}` |
| `/enroll/capture` | POST | Capture le visage de la frame courante |
| `/enroll/finish` | POST | Sauvegarde le profil moyenné |
| `/door/open` | POST | Ouvre la porte via le VTO |
| `/profiles` | GET | Liste des profils enrôlés |

## Prérequis matériel

- Caméra/VTO Dahua avec flux RTSP accessible (`/cam/realmonitor?channel=X&subtype=Y`)
- Broker MQTT accessible (ex: Mosquitto sur le même réseau que Home Assistant)
- CPU suffisant pour InsightFace en temps réel (testé sur Raspberry Pi 5)

## Licence

MIT
