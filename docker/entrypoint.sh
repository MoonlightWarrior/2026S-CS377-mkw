#!/bin/bash
set -e

WORKDIR="/src/Vlab-WiiRL"
DOLPHIN_BASE="/opt/dolphin-base"
NUM_ENVS="${NUM_ENVS:-4}"

# --- headless virtual display with GPU rendering ---
Xvfb :99 -screen 0 1280x720x24 +extension GLX &
sleep 1
export DISPLAY=:99
export USE_VGLRUN=1

# --- clean up runtime state from any previous run ---
rm -f "$WORKDIR/alive.txt" "$WORKDIR/shared_value.txt" "$WORKDIR/shared_site.txt"
rm -rf "$WORKDIR/instance_info"

# --- python venv (stored in volume, survives container restarts) ---
if [ ! -d "$WORKDIR/.venv" ]; then
    echo "[setup] Creating Python venv and installing dependencies..."
    python3 -m venv "$WORKDIR/.venv"
    # requirements.txt is UTF-16; convert it before passing to pip
    python3 -c "
import codecs
with codecs.open('$WORKDIR/requirements.txt', 'r', 'utf-16') as f:
    data = f.read()
with open('/tmp/req.txt', 'w') as f:
    f.write(data)
"
    "$WORKDIR/.venv/bin/pip" install --quiet -r /tmp/req.txt
    echo "[setup] Python deps installed."
fi
source "$WORKDIR/.venv/bin/activate"

# --- dolphin0: copy from image if not in workdir ---
if [ ! -d "$WORKDIR/dolphin0" ]; then
    echo "[setup] Copying dolphin0 from image..."
    cp -r "$DOLPHIN_BASE" "$WORKDIR/dolphin0"
    touch "$WORKDIR/dolphin0/portable.txt"
fi

# --- dolphin1..N: hardlink binaries, copy Sys (saves ~90MB per env) ---
for i in $(seq 1 $((NUM_ENVS - 1))); do
    if [ ! -d "$WORKDIR/dolphin${i}" ]; then
        echo "[setup] Creating dolphin${i}..."
        mkdir -p "$WORKDIR/dolphin${i}"
        for bin in dolphin-emu dolphin-emu-nogui dolphin-tool traversal_server; do
            if [ -f "$WORKDIR/dolphin0/$bin" ]; then
                ln "$WORKDIR/dolphin0/$bin" "$WORKDIR/dolphin${i}/$bin" 2>/dev/null || \
                    cp "$WORKDIR/dolphin0/$bin" "$WORKDIR/dolphin${i}/$bin"
            fi
        done
        cp -r "$WORKDIR/dolphin0/Sys" "$WORKDIR/dolphin${i}/Sys"
        touch "$WORKDIR/dolphin${i}/portable.txt"
    fi
done

cd "$WORKDIR"
exec "$@"
