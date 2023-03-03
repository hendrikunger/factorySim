# syntax=docker/dockerfile:1

FROM rayproject/ray-ml:latest-gpu

EXPOSE 8265
#Tensorboard  run tensorboard --bind_all --logdir .
EXPOSE 6006 

COPY requirements_factorySim.txt .
USER root

RUN  apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libc6 \
        libcairo2-dev \
        pkg-config \
        python3-dev \
        ffmpeg \
    && runuser -l  ray -c '$HOME/anaconda3/bin/pip --no-cache-dir install -U -r requirements_factorySim.txt' \
    && apt-get clean \
    && rm -rf /tmp/* \
    && rm -rf /var/lib/apt/lists/*
 
WORKDIR $HOME/factorySim

COPY . .


WORKDIR $HOME/factorySim/env
RUN $HOME/anaconda3/bin/pip install .
USER ray
WORKDIR $HOME/factorySim

