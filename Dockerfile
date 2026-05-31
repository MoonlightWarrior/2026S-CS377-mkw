# Self-contained image for CS377 Team 3 MARL (2v2 MAPPO self-play) on Mario Kart Wii.
# Builds the Vlab-dolphin scripting fork from source, then installs the vendored
# kart_env environment + the marl package. Clone-and-go: `docker compose -f
# compose.marl.yml up -d --build` (see STANDALONE.md). First build is ~20-30 min
# (Dolphin compile) and the image is large (~35 GB).
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# ── Dolphin build deps + headless display (xvfb/x11vnc/novnc) + VirtualGL ──────
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales wget curl git unzip tmux ca-certificates \
    pkg-config \
    libgl1-mesa-dev libx11-dev libxrandr-dev libxi-dev libegl1-mesa-dev \
    libavcodec-dev libavformat-dev libavutil-dev libswresample-dev libswscale-dev \
    libudev-dev libevdev-dev glslang-dev glslang-tools libpugixml-dev libxxhash-dev \
    libbz2-dev liblzma-dev libzstd-dev liblzo2-dev liblz4-dev libspng-dev libcubeb-dev \
    libusb-1.0-0-dev libminiupnpc-dev libcurl4-openssl-dev libhidapi-dev libsystemd-dev \
    libgtest-dev libasound2-dev libpulse-dev llvm-dev libbluetooth-dev \
    qt6-base-dev qt6-base-private-dev qt6-svg-dev gettext \
    xvfb x11vnc novnc websockify \
    && wget -q https://github.com/VirtualGL/virtualgl/releases/download/3.1.4/virtualgl_3.1.4_amd64.deb \
    && apt-get install -y --no-install-recommends ./virtualgl_3.1.4_amd64.deb \
    && rm virtualgl_3.1.4_amd64.deb \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && locale-gen en_US.UTF-8

# ── Build the Vlab-dolphin scripting fork and install to /usr/local/bin ────────
RUN git clone https://github.com/vlab-kaist/Vlab-dolphin.git /dolphin-src && \
    cd /dolphin-src && \
    git -c submodule."Externals/Qt".update=none \
        -c submodule."Externals/FFmpeg-bin".update=none \
        -c submodule."Externals/libadrenotools".update=none \
        submodule update --init --recursive && \
    mkdir build && cd build && \
    cmake .. \
        -DUSE_SYSTEM_MINIZIP-NG=OFF \
        -DUSE_SYSTEM_SFML=OFF \
        -DUSE_SYSTEM_MBEDTLS=OFF \
        -DUSE_SYSTEM_LIBMGBA=OFF \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && \
    make -j"$(nproc)" && make install && \
    cd / && rm -rf /dolphin-src

ENV NVIDIA_DRIVER_CAPABILITIES=all

# ── Install the repo (vendored kart_env + marl) and its Python deps ───────────
WORKDIR /workspace
COPY pyproject.toml ./
COPY kart_env ./kart_env
COPY marl ./marl
RUN pip install --no-cache-dir --break-system-packages -e .

# The env launches dolphin with `-e MarioKartWii.iso` relative to the CWD, so run
# training from /workspace with the ISO mounted at /workspace/MarioKartWii.iso.
CMD ["bash"]
