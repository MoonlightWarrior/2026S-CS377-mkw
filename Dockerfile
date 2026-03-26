FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# install python 3.12
RUN apt-get update && apt-get install -y \
    software-properties-common ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3.12-venv curl \
    && curl https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.12 1

# install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# dolphin runtime deps + xvfb for headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    git unzip tmux \
    libqt6widgets6 libqt6gui6 libqt6core6 libqt6dbus6 qt6-qpa-plugins \
    libasound2 libpulse0 libxi6 libevdev2 libbluetooth3 libudev1 \
    libavformat-dev libavcodec-dev libswscale-dev libavutil-dev \
    libcurl4 libegl1 libglx0 libxrandr2 libx11-6 \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
    libxcb-render-util0 libxcb-xinerama0 libxcb-xinput0 libxcb-xkb1 \
    libxkbcommon-x11-0 \
    xvfb libxtst6 libxv1 libglu1-mesa wget \
    && wget -q https://github.com/VirtualGL/virtualgl/releases/download/3.1.2/virtualgl_3.1.2_amd64.deb -O /tmp/virtualgl.deb \
    && dpkg -i /tmp/virtualgl.deb && rm /tmp/virtualgl.deb \
    && rm -rf /var/lib/apt/lists/*

# extract prebuilt dolphin binaries
COPY HEREISFILE/dolphin-emu.tar.gz /tmp/
RUN mkdir -p /opt/dolphin-base && \
    tar -xzf /tmp/dolphin-emu.tar.gz -C /opt/dolphin-base && \
    rm /tmp/dolphin-emu.tar.gz

# download savestates
RUN cd /opt && \
    curl -LO https://github.com/VIPTankz/Wii-RL/releases/download/savestates/MarioKartSaveStates.zip && \
    unzip MarioKartSaveStates.zip && \
    rm MarioKartSaveStates.zip

# install python requirements via uv (copy manifests for layer caching)
COPY pyproject.toml uv.lock /tmp/
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /tmp && uv sync --frozen --no-dev && rm pyproject.toml uv.lock

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /src/Vlab-WiiRL
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
