FROM python:3.11-slim

# Dépendances système nécessaires à OpenCV + décodage RTSP
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY face_service.py .
COPY smartx_ha_discovery.py .
COPY sip_talk.py .

# Persistance : profils enrôlés + config VTO
VOLUME ["/app/face_profiles"]

EXPOSE 5001

CMD ["python", "face_service.py"]
