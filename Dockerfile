FROM python:3.8-slim-bullseye

WORKDIR /app
 
RUN --mount=type=cache,target=/root/.cache apt-get update && \
    apt-get install -y -qq \
    ffmpeg \
    espeak \
    libportaudio2 \
    build-essential \
    cmake \
    python3-dev \
    portaudio19-dev \
    python3-pyaudio

COPY ./requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt
COPY . .

VOLUME ["/app/models", "/app/output", "/app/datasets", "/app/logs", "/app/songs", "/app/.cache" ]

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
# RVC server
EXPOSE 5555

# Webui server
EXPOSE 8501

# Tensorboard server
EXPOSE 6006

ENV PATH="$PATH:/app/:/"

ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]