FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    python3-pip \
    make \
    wget \
    ffmpeg \
    libsm6 \
    libxext6

WORKDIR /planet_service

RUN pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 5039

CMD make run_app
