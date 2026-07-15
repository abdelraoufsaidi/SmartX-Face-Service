"""
Pont SIP <-> WebSocket pour la fonction "Interphone" (talk) depuis Home Assistant.

Reprend la logique SIP/RTP validée manuellement (REGISTER + INVITE + RTP G.711
+ écoute du BYE), mais sans dépendance à une carte son locale : l'audio circule
via des files Python, alimentées/consommées par les routes Flask/WebSocket de
face_service.py.

Le VTO doit avoir une entrée "VTS" créée dans sa liste d'appareils (Système ->
Ajouter -> Type VTS) avec un numéro et un mot de passe, qui doivent être
renseignés côté HA (config de l'intégration) et transmis ici via /config.
"""

import socket
import re
import hashlib
import random
import string
import time
import threading
import queue
import audioop
import logging

log = logging.getLogger(__name__)

SAMPLE_RATE = 8000
SAMPLES_PER_FRAME = 160  # 20 ms @ 8kHz


def _rand_str(n=10):
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(n))


def _md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def _build_digest_auth(username, realm, password, nonce, method, uri):
    ha1 = _md5(f"{username}:{realm}:{password}")
    ha2 = _md5(f"{method}:{uri}")
    response = _md5(f"{ha1}:{nonce}:{ha2}")
    return (
        f'Digest username="{username}", realm="{realm}", nonce="{nonce}", '
        f'uri="{uri}", response="{response}", algorithm=MD5'
    )


def _parse_auth_header(text: str):
    m = re.search(r'(?:WWW|Proxy)-Authenticate:\s*Digest\s+(.*)', text, re.IGNORECASE)
    if not m:
        return None
    return dict(re.findall(r'(\w+)="?([^",]+)"?', m.group(1)))


class VtoCall:
    """Une session d'appel SIP/RTP vers le VTO, pilotée par le service HA."""

    def __init__(self, local_ip, vto_ip, my_extension, my_password,
                 vto_extension="8001", sip_port=5070, rtp_port=16010):
        self.local_ip = local_ip
        self.vto_ip = vto_ip
        self.my_extension = str(my_extension)
        self.my_password = my_password
        self.vto_extension = str(vto_extension)
        self.sip_port = sip_port
        self.rtp_port = rtp_port

        self.sip_sock = None
        self.rtp_sock = None
        self.call_id = None
        self.cseq = 1
        self.tag = _rand_str(8)
        self.remote_ip = None
        self.remote_rtp_port = None

        self.active = False
        self.in_queue = queue.Queue()    # PCM16 8k venant du navigateur -> à envoyer au VTO
        self.out_queue = queue.Queue()   # PCM16 8k venant du VTO -> à envoyer au navigateur

    # ---------------- SIP ----------------

    def _send_sip(self, msg: str):
        self.sip_sock.sendto(msg.encode(), (self.vto_ip, self.sip_port))

    def _recv_sip(self, timeout=5):
        self.sip_sock.settimeout(timeout)
        data, _ = self.sip_sock.recvfrom(65535)
        return data.decode(errors="ignore")

    def _register(self):
        self.call_id = _rand_str(16) + "@" + self.local_ip
        auth_header = ""
        for _ in range(2):
            branch = "z9hG4bK" + _rand_str(10)
            msg = (
                f"REGISTER sip:{self.vto_ip} SIP/2.0\r\n"
                f"Via: SIP/2.0/UDP {self.local_ip}:{self.sip_port};branch={branch}\r\n"
                f"Max-Forwards: 70\r\n"
                f"From: <sip:{self.my_extension}@{self.vto_ip}>;tag={self.tag}\r\n"
                f"To: <sip:{self.my_extension}@{self.vto_ip}>\r\n"
                f"Call-ID: {self.call_id}\r\n"
                f"CSeq: {self.cseq} REGISTER\r\n"
                f"Contact: <sip:{self.my_extension}@{self.local_ip}:{self.sip_port}>\r\n"
                f"Expires: 3600\r\n"
                f"{auth_header}"
                f"Content-Length: 0\r\n\r\n"
            )
            self.cseq += 1
            self._send_sip(msg)
            try:
                resp = self._recv_sip()
            except socket.timeout:
                log.warning("REGISTER : pas de réponse (mot de passe VTS erroné + anti-bruteforce VTO ?)")
                return False
            status = resp.splitlines()[0]
            if " 200 " in status:
                return True
            if " 401 " in status or " 407 " in status:
                params = _parse_auth_header(resp)
                if not params:
                    return False
                digest = _build_digest_auth(
                    self.my_extension, params["realm"], self.my_password,
                    params["nonce"], "REGISTER", f"sip:{self.vto_ip}"
                )
                auth_header = f"Authorization: {digest}\r\n"
                continue
            log.warning(f"REGISTER échoué : {status}")
            return False
        return False

    def _invite(self):
        sdp = (
            f"v=0\r\no=- {int(time.time())} {int(time.time())} IN IP4 {self.local_ip}\r\n"
            f"s=call\r\nc=IN IP4 {self.local_ip}\r\nt=0 0\r\n"
            f"m=audio {self.rtp_port} RTP/AVP 0\r\na=rtpmap:0 PCMU/8000\r\na=sendrecv\r\n"
        )
        auth_header = ""
        for _ in range(2):
            branch = "z9hG4bK" + _rand_str(10)
            msg = (
                f"INVITE sip:{self.vto_extension}@{self.vto_ip} SIP/2.0\r\n"
                f"Via: SIP/2.0/UDP {self.local_ip}:{self.sip_port};branch={branch}\r\n"
                f"Max-Forwards: 70\r\n"
                f"From: <sip:{self.my_extension}@{self.vto_ip}>;tag={self.tag}\r\n"
                f"To: <sip:{self.vto_extension}@{self.vto_ip}>\r\n"
                f"Call-ID: {self.call_id}-call\r\n"
                f"CSeq: {self.cseq} INVITE\r\n"
                f"Contact: <sip:{self.my_extension}@{self.local_ip}:{self.sip_port}>\r\n"
                f"Content-Type: application/sdp\r\n"
                f"{auth_header}"
                f"Content-Length: {len(sdp)}\r\n\r\n{sdp}"
            )
            self.cseq += 1
            self._send_sip(msg)

            try:
                resp = self._recv_sip(timeout=30)
                while int(resp.splitlines()[0].split()[1]) < 200:
                    resp = self._recv_sip(timeout=30)
            except socket.timeout:
                log.warning("INVITE : pas de réponse (VTO ne sonne pas / pas décroché à temps)")
                return False

            status = resp.splitlines()[0]
            if " 401 " in status or " 407 " in status:
                params = _parse_auth_header(resp)
                if not params:
                    return False
                digest = _build_digest_auth(
                    self.my_extension, params["realm"], self.my_password,
                    params["nonce"], "INVITE", f"sip:{self.vto_extension}@{self.vto_ip}"
                )
                auth_header = f"Authorization: {digest}\r\n"
                continue
            if " 200 " in status:
                body = resp.split("\r\n\r\n", 1)[1]
                m_ip = re.search(r"c=IN IP4 ([\d.]+)", body)
                m_port = re.search(r"m=audio (\d+)", body)
                self.remote_ip = m_ip.group(1) if m_ip else self.vto_ip
                self.remote_rtp_port = int(m_port.group(1)) if m_port else None

                to_line = next(l for l in resp.splitlines() if l.startswith("To:"))
                ack = (
                    f"ACK sip:{self.vto_extension}@{self.vto_ip} SIP/2.0\r\n"
                    f"Via: SIP/2.0/UDP {self.local_ip}:{self.sip_port};branch=z9hG4bK{_rand_str(10)}\r\n"
                    f"Max-Forwards: 70\r\n"
                    f"From: <sip:{self.my_extension}@{self.vto_ip}>;tag={self.tag}\r\n"
                    f"{to_line}\r\n"
                    f"Call-ID: {self.call_id}-call\r\n"
                    f"CSeq: {self.cseq - 1} ACK\r\n"
                    f"Content-Length: 0\r\n\r\n"
                )
                self._send_sip(ack)
                return True
            log.warning(f"INVITE échoué : {status}")
            return False
        return False

    def _bye(self):
        branch = "z9hG4bK" + _rand_str(10)
        msg = (
            f"BYE sip:{self.vto_extension}@{self.vto_ip} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {self.local_ip}:{self.sip_port};branch={branch}\r\n"
            f"Max-Forwards: 70\r\n"
            f"From: <sip:{self.my_extension}@{self.vto_ip}>;tag={self.tag}\r\n"
            f"To: <sip:{self.vto_extension}@{self.vto_ip}>\r\n"
            f"Call-ID: {self.call_id}-call\r\n"
            f"CSeq: {self.cseq} BYE\r\nContent-Length: 0\r\n\r\n"
        )
        self.cseq += 1
        try:
            self._send_sip(msg)
            self.sip_sock.settimeout(2)
            self.sip_sock.recvfrom(65535)
        except OSError:
            pass

    def _listen_for_bye(self):
        self.sip_sock.settimeout(1.0)
        while self.active:
            try:
                data, _ = self.sip_sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            text = data.decode(errors="ignore")
            if text.startswith("BYE "):
                log.info("Le VTO a raccroché (BYE reçu)")
                lines = text.split("\r\n")
                try:
                    via = next(l for l in lines if l.startswith("Via:"))
                    from_h = next(l for l in lines if l.startswith("From:"))
                    to_h = next(l for l in lines if l.startswith("To:"))
                    call_id = next(l for l in lines if l.startswith("Call-ID:"))
                    cseq = next(l for l in lines if l.startswith("CSeq:"))
                    resp = (f"SIP/2.0 200 OK\r\n{via}\r\n{from_h}\r\n{to_h}\r\n"
                            f"{call_id}\r\n{cseq}\r\nContent-Length: 0\r\n\r\n")
                    self._send_sip(resp)
                except StopIteration:
                    pass
                self.active = False
                break

    # ---------------- RTP ----------------

    def _rtp_sender_loop(self):
        seq = random.randint(0, 65535)
        ts = random.randint(0, 0xFFFFFFFF)
        ssrc = random.randint(0, 0xFFFFFFFF)
        buf = b""
        next_tick = time.monotonic()
        while self.active:
            next_tick += 0.02  # 20 ms
            try:
                while len(buf) < SAMPLES_PER_FRAME * 2:
                    buf += self.in_queue.get_nowait()
            except queue.Empty:
                pass

            if len(buf) >= SAMPLES_PER_FRAME * 2:
                frame, buf = buf[:SAMPLES_PER_FRAME * 2], buf[SAMPLES_PER_FRAME * 2:]
            else:
                frame = b"\x00\x00" * SAMPLES_PER_FRAME  # silence si rien à envoyer

            payload = audioop.lin2ulaw(frame, 2)
            header = (
                bytes([0x80, 0])
                + seq.to_bytes(2, "big")
                + ts.to_bytes(4, "big")
                + ssrc.to_bytes(4, "big")
            )
            try:
                self.rtp_sock.sendto(header + payload, (self.remote_ip, self.remote_rtp_port))
            except OSError:
                break
            seq = (seq + 1) & 0xFFFF
            ts = (ts + SAMPLES_PER_FRAME) & 0xFFFFFFFF

            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _rtp_receiver_loop(self):
        self.rtp_sock.settimeout(1.0)
        while self.active:
            try:
                data, _ = self.rtp_sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            payload = data[12:]
            pcm16 = audioop.ulaw2lin(payload, 2)
            self.out_queue.put(pcm16)

    # ---------------- API publique ----------------

    def start(self):
        """Enregistre + appelle le VTO. Retourne (ok: bool, message: str)."""
        self.sip_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sip_sock.bind((self.local_ip, self.sip_port))
        self.rtp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rtp_sock.bind((self.local_ip, self.rtp_port))

        if not self._register():
            return False, "Échec de l'enregistrement SIP (vérifier extension/mot de passe VTS sur le VTO)"
        if not self._invite():
            return False, "Échec de l'établissement de l'appel (INVITE refusé ou VTO ne répond pas)"

        self.active = True
        threading.Thread(target=self._rtp_sender_loop, daemon=True).start()
        threading.Thread(target=self._rtp_receiver_loop, daemon=True).start()
        threading.Thread(target=self._listen_for_bye, daemon=True).start()
        return True, "ok"

    def push_audio(self, pcm16_bytes: bytes):
        """Audio PCM16 mono 8kHz venant du navigateur, à envoyer au VTO."""
        if self.active:
            self.in_queue.put(pcm16_bytes)

    def pull_audio(self, timeout=1.0):
        """Renvoie un chunk PCM16 mono 8kHz venant du VTO (ou None si rien de nouveau)."""
        try:
            return self.out_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        was_active = self.active
        self.active = False
        time.sleep(0.1)
        if was_active:
            self._bye()
        for s in (self.sip_sock, self.rtp_sock):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
