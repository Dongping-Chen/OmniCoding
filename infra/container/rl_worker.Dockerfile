FROM nvcr.io/nvidia/cuda:12.8.0-base-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        ffmpeg \
        file \
        git \
        imagemagick \
        jq \
        libgl1 \
        libglib2.0-0 \
        libmagic1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        mediainfo \
        poppler-utils \
        python3 \
        python3-pip \
        python3-venv \
        sox \
        tesseract-ocr \
        unzip \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/omnicoding
COPY pyproject.toml README.md LICENSE ./
COPY LICENSES ./LICENSES
COPY src ./src
COPY infra/container/whisper_cli.py /usr/local/bin/whisper

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir \
        "faster-whisper==1.2.1" \
        "librosa==0.11.0" \
        "litellm==1.93.0" \
        "matplotlib==3.10.3" \
        "opencv-python-headless==4.11.0.86" \
        "pillow==11.3.0" \
        "pydantic==2.13.4" \
        "requests==2.34.2" \
        "soundfile==0.13.1" \
        "tenacity==9.1.4" \
    && python3 -m pip install --no-cache-dir --no-deps . \
    && python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-small', local_dir='/opt/whisper-small')" \
    && chmod 0755 /usr/local/bin/whisper

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/tmp/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    XDG_CACHE_HOME=/tmp/cache \
    PATH=/usr/local/bin:/usr/bin:/bin

ENTRYPOINT []
CMD ["python3", "-m", "omnicoding.rl.run_trajectory", "--help"]
