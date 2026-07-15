"""
SMARTX Face Recognition — Intégration Home Assistant via MQTT Discovery.

Publie un device HA unique ("SMARTX Face Recognition") avec un binary_sensor
par personne enrôlée, sans rien coder en dur.
"""

import json
import paho.mqtt.client as mqtt

BROKER_HOST = "localhost"
BROKER_PORT = 1883

# Identifiant unique du "device" — sert à grouper tous les capteurs dans HA
DEVICE_ID = "smartx_face_recognition"

# ── Topics ──
STATE_TOPIC_TEMPLATE = "smartx/face_recognition/{slug}/state"
AVAILABILITY_TOPIC = "smartx/face_recognition/availability"
DISCOVERY_TOPIC_TEMPLATE = "homeassistant/binary_sensor/{device_id}/{slug}/config"


def _slugify(name: str) -> str:
    """Normalise un nom en identifiant sûr pour les topics MQTT (ex: 'Raouf S' -> 'Raouf_s')."""
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def build_discovery_payload(person_name: str) -> dict:
    """
    Construit le payload de découverte HA pour une personne donnée.
    Le champ "device" regroupe tous les capteurs sous une seule fiche
    appareil dans Home Assistant ("SMARTX Face Recognition").
    """
    slug = _slugify(person_name)
    return {
        "name": f"Présence {person_name.capitalize()}",
        "unique_id": f"{DEVICE_ID}_{slug}_presence",
        "state_topic": STATE_TOPIC_TEMPLATE.format(slug=slug),
        "payload_on": "detected",
        "payload_off": "clear",
        "device_class": "occupancy",
        "availability_topic": AVAILABILITY_TOPIC,
        "payload_available": "online",
        "payload_not_available": "offline",
        "device": {
            "identifiers": [DEVICE_ID],
            "name": "SMARTX Face Recognition",
            "manufacturer": "SMARTX",
            "model": "InsightFace RTSP Node",
            "sw_version": "1.0",
        },
    }


def publish_discovery(client: mqtt.Client, person_names: list[str]):
    """À appeler une fois au démarrage, pour chaque personne enrôlée."""
    for name in person_names:
        slug = _slugify(name)
        topic = DISCOVERY_TOPIC_TEMPLATE.format(device_id=DEVICE_ID, slug=slug)
        payload = build_discovery_payload(name)
        client.publish(topic, json.dumps(payload), retain=True)


def publish_presence(client: mqtt.Client, person_name: str, detected: bool):
    """À appeler à chaque reconnaissance (ou perte de reconnaissance)."""
    slug = _slugify(person_name)
    topic = STATE_TOPIC_TEMPLATE.format(slug=slug)
    client.publish(topic, "detected" if detected else "clear", retain=False)


def publish_availability(client: mqtt.Client, online: bool):
    client.publish(AVAILABILITY_TOPIC, "online" if online else "offline", retain=True)


if __name__ == "__main__":
    # Exemple d'usage autonome pour tester la découverte
    client = mqtt.Client(client_id="smartx_face_recognition")
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_start()

    example_people = ["ARS", "abd el raouf"]
    publish_discovery(client, example_people)
    publish_availability(client, online=True)

    print("Découverte publiée pour :", example_people)
