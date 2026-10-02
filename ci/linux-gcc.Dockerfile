FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

COPY cmake-version.txt /tmp/cmake-version.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ccache \
    g++ \
    git \
    libssl-dev \
    ninja-build \
    openssl \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/cmake \
    && /opt/cmake/bin/python -m pip install --no-cache-dir "cmake==$(cat /tmp/cmake-version.txt)"

ENV PATH="/opt/cmake/bin:${PATH}"
