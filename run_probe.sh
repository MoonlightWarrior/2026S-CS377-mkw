#!/bin/bash
# In-container runner for probe_4p_control.py. One Dolphin instance, headless.
# Driven by PROBE_CHANNEL / PROBE_PORT / PROBE_STICK env vars.
set -u
cd /src/Vlab-WiiRL
exec ./dolphin0/dolphin-emu \
  -v Null \
  -C Dolphin.Core.EmulationSpeed=0.0 \
  -C Dolphin.Core.FastDiscSpeed=True \
  -C "Dolphin.DSP.Backend=No audio output" \
  -C Dolphin.Core.SIDevice0=6 \
  -C Dolphin.Core.SIDevice1=6 \
  -C Dolphin.Core.SIDevice2=6 \
  -C Dolphin.Core.SIDevice3=6 \
  -u /tmp/dolphin_user0 \
  --no-python-subinterpreters \
  --script /src/Vlab-WiiRL/probe_4p_control.py \
  $'\b' \
  --exec=/src/Vlab-WiiRL/game/mkw.iso
