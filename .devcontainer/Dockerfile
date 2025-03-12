FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    curl \
    git \
    && apt-get clean

RUN curl -sSL https://pdm-project.org/install-pdm.py | python3 -

ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace
