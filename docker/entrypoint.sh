#!/bin/bash
set -e

WORKDIR="/src/Vlab-WiiRL"
DOLPHIN_BASE="/opt/dolphin-base"
SAVESTATES_BASE="/opt/MarioKartSaveStates"
NUM_CLONES="${DOLPHIN_CLONES:-7}"

# --- display setup ---
if [ "${HEADLESS:-0}" = "1" ]; then
    Xvfb :99 -screen 0 640x480x24 +extension GLX &
    sleep 1
    export DISPLAY=:99
    export USE_VGLRUN=1
fi

# --- provision dolphin binaries if not present ---
if [ ! -d "$WORKDIR/dolphin0" ]; then
    echo "[entrypoint] copying dolphin0 from image..."
    cp -r "$DOLPHIN_BASE" "$WORKDIR/dolphin0"
fi

for i in $(seq 1 "$NUM_CLONES"); do
    if [ ! -d "$WORKDIR/dolphin${i}" ]; then
        echo "[entrypoint] cloning dolphin${i}..."
        cp -r "$WORKDIR/dolphin0" "$WORKDIR/dolphin${i}"
    fi
done

# --- provision savestates if not present ---
if [ ! -d "$WORKDIR/MarioKartSaveStates" ]; then
    echo "[entrypoint] copying savestates from image..."
    cp -r "$SAVESTATES_BASE" "$WORKDIR/MarioKartSaveStates"
fi

exec "$@"
