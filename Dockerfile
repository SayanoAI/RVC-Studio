# syntax=docker/dockerfile:1

FROM python:3.8-bullseye

EXPOSE 8501
EXPOSE 6006

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache apt update && apt install -y -qq ffmpeg aria2 espeak
# RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D40k.pth -d /app/models/pretrained_v2/ -o D40k.pth
# RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G40k.pth -d /app/models/pretrained_v2/ -o G40k.pth
# RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth -d /app/models/pretrained_v2/ -o f0D40k.pth
# RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth -d /app/models/pretrained_v2/ -o f0G40k.pth
# RUN --mount=type=cache,target=/root/.cache aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP2-人声vocals+非人声instrumentals.pth -d models/uvr5_weights/ -o HP2-vocals+instrumentals.pth
# RUN --mount=type=cache,target=/root/.cache aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth -d models/uvr5_weights/ -o HP5-vocals+instrumentals.pth
# RUN --mount=type=cache,target=/root/.cache aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt -o models/hubert_base.pt
# RUN --mount=type=cache,target=/root/.cache aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt -o models/rmvpe.pt

COPY ./requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt
COPY . .

VOLUME ["/app/models", "/app/output", "/app/datasets", "/app/logs", "/app/songs", "/app/.cache" ]

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]