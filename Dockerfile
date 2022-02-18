FROM nvcr.io/nvidia/pytorch:22.01-py3
RUN apt-get update && apt-get install -y \
    libcairo2-dev \
    pkg-config \
    build-essential \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /factorySim
COPY requirements.txt .
RUN python -m pip install -r requirements.txt
COPY . .
