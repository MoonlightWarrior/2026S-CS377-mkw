#!/bin/bash
set -e

WORKDIR="/src/Vlab-WiiRL"
DOLPHIN_BASE="/opt/dolphin-base"
SAVESTATES_BASE="/opt/MarioKartSaveStates"
NUM_ENVS="${NUM_ENVS:-8}"

# --- display setup ---
if [ "${HEADLESS:-0}" = "1" ]; then
    Xvfb :99 -screen 0 640x480x24 +extension GLX &
    sleep 1
    export DISPLAY=:99
    export USE_VGLRUN=1
fi

# --- provision single dolphin binary (shared by all envs) ---
if [ ! -d "$WORKDIR/dolphin0" ]; then
    echo "[entrypoint] copying dolphin0 from image..."
    cp -r "$DOLPHIN_BASE" "$WORKDIR/dolphin0"
fi

# --- create per-instance user directories (config only, no binary duplication) ---
for i in $(seq 0 $((NUM_ENVS - 1))); do
    USER_DIR="/tmp/dolphin_user${i}"
    mkdir -p "$USER_DIR"
    # copy base config if dolphin0 has a User/Config directory
    if [ -d "$WORKDIR/dolphin0/User/Config" ] && [ ! -d "$USER_DIR/Config" ]; then
        cp -r "$WORKDIR/dolphin0/User/Config" "$USER_DIR/Config"
    fi
done

# --- provision savestates if not present ---
if [ ! -d "$WORKDIR/MarioKartSaveStates" ]; then
    echo "[entrypoint] copying savestates from image..."
    cp -r "$SAVESTATES_BASE" "$WORKDIR/MarioKartSaveStates"
fi

exec "$@"
