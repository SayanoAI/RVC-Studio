# syntax=docker/dockerfile:1

FROM python:3.8-bullseye

EXPOSE 8501
EXPOSE 6006

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache apt update && apt install -y -qq ffmpeg aria2 espeak libportaudio2

COPY ./requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt
COPY . .

VOLUME ["/app/models", "/app/output", "/app/datasets", "/app/logs", "/app/songs", "/app/.cache" ]

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]