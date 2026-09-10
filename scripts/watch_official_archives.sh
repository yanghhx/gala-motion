#!/usr/bin/env bash
# Extract official archives as soon as Google Drive downloads finish.
set -euo pipefail
ROOT=/home/qinyang/桌面/project
PY=/home/qinyang/.local/share/mamba/envs/gala-motion/bin/python
SEVEN=/home/qinyang/.local/share/mamba/envs/gala-motion/bin/7z

extract_when_ready() {
  local archive="$1"
  local dest="$2"
  local marker="$3"
  if [[ -f "$marker" ]]; then
    return 0
  fi
  if [[ -f "$archive" ]]; then
    echo "[$(date)] extracting $archive -> $dest"
    mkdir -p "$dest"
    "$SEVEN" x -y "-o$dest" "$archive"
    touch "$marker"
    echo "[$(date)] done $archive"
    # Stop the folder download so it does not start new_joints.rar.
    pkill -f "gdown.download_folder" 2>/dev/null || true
  fi
}

while true; do
  extract_when_ready \
    "$ROOT/data/downloads/KIT-ML-official/new_joint_vecs.rar" \
    "$ROOT/data/KIT-ML-official" \
    "$ROOT/data/KIT-ML-official/.vecs_extracted"
  extract_when_ready \
    "$ROOT/data/downloads/HumanML3D-official/new_joint_vecs.rar" \
    "$ROOT/research_sources/HumanML3D/HumanML3D" \
    "$ROOT/research_sources/HumanML3D/HumanML3D/.vecs_extracted"
  if [[ -f "$ROOT/data/KIT-ML-official/.vecs_extracted" && ! -f "$ROOT/data/KIT-ML-official/.config_switched" ]]; then
    "$PY" "$ROOT/scripts/switch_to_official_data.py"
    touch "$ROOT/data/KIT-ML-official/.config_switched"
  fi
  if [[ -f "$ROOT/research_sources/HumanML3D/HumanML3D/.vecs_extracted" && ! -f "$ROOT/data/HumanML3D/.config_switched" ]]; then
    "$PY" "$ROOT/scripts/switch_to_official_data.py"
    touch "$ROOT/data/HumanML3D/.config_switched"
  fi
  sleep 30
done
