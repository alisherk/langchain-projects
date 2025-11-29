FROM mcr.microsoft.com/devcontainers/python:3.11

ENV SHELL /bin/bash
SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    nano \
    procps \
    build-essential \
    bash-completion \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=vscode

# Install UV for faster package installations
RUN mkdir -p /home/${USERNAME}/.local/bin && \
    chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.local && \
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/home/${USERNAME}/.local/bin sh && \
    chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.local

ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
ENV UV_PROJECT_ENVIRONMENT=".venv"

WORKDIR /home/${USERNAME}/workspace
