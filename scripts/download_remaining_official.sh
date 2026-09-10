#!/usr/bin/env bash
# After the in-flight KIT vecs download finishes, pull HumanML vecs and evaluators one-by-one.
set -euo pipefail
ROOT=/home/qinyang/桌面/project
KIT_PART="$ROOT/data/downloads/KIT-ML-official/new_joint_vecs.rar8cki4tsl.part"
KIT_DONE="$ROOT/data/downloads/KIT-ML-official/new_joint_vecs.rar"
HML_DIR="$ROOT/data/downloads/HumanML3D-official"
EVAL="$ROOT/checkpoints/t2m_evaluators"
mkdir -p "$HML_DIR" "$EVAL"

wait_for_kit() {
  echo "[$(date)] waiting for official KIT new_joint_vecs.rar"
  while [[ ! -f "$KIT_DONE" ]]; do
    if [[ -f "$KIT_PART" ]]; then
      ls -lh "$KIT_PART" | awk '{print "[wait] kit part",$5}'
    fi
    sleep 60
  done
  echo "[$(date)] KIT vecs archive ready"
}

drive() {
  local id="$1" out="$2"
  if [[ -f "$out" && ! -f "${out}.part" ]]; then
    echo "skip existing $out"
    return 0
  fi
  echo "[$(date)] curl $out"
  curl -L --retry 8 --retry-delay 5 --continue-at - \
    -o "$out" \
    "https://drive.google.com/uc?export=download&id=${id}&confirm=t"
  ls -lh "$out"
}

wait_for_kit
drive "14HIdUa_LPGEh-M1ICDZJp0tLJftewnAX" "$HML_DIR/new_joint_vecs.rar"
drive "1O_GUHgjDbl2tgbyfSwZOUYXDACnk25Kb" "$EVAL/t2m.zip"
drive "12liZW5iyvoybXD8eOw4VanTgsMtynCuU" "$EVAL/kit.zip"
drive "1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n" "$EVAL/glove.zip"
echo "[$(date)] remaining official downloads finished"
