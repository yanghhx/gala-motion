#!/usr/bin/env python3
"""Download and extract official HumanML3D / KIT-ML features and Guo evaluators."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

PROJECT = Path("/home/qinyang/桌面/project")
DATA = PROJECT / "data"
EVAL = PROJECT / "checkpoints" / "t2m_evaluators"
HML_RAR = DATA / "downloads" / "HumanML3D-official" / "new_joint_vecs.rar"
HML_DEST = DATA / "HumanML3D"
KIT_DEST = DATA / "KIT-ML-official"

T2M_EVAL = "https://drive.google.com/uc?id=1O_GUHgjDbl2tgbyfSwZOUYXDACnk25Kb"
KIT_EVAL = "https://drive.google.com/uc?id=12liZW5iyvoybXD8eOw4VanTgsMtynCuU"
GLOVE = "https://drive.google.com/uc?id=1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n"
HML_VEC_ID = "14HIdUa_LPGEh-M1ICDZJp0tLJftewnAX"


def unzip(archive: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)


def extract_rar(archive: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    for cmd in (
        ["unrar", "x", "-o+", str(archive), str(dest) + "/"],
        ["unar", "-f", "-o", str(dest), str(archive)],
    ):
        if shutil.which(cmd[0]):
            subprocess.check_call(cmd)
            return
    raise RuntimeError("Need unrar or unar to extract official HumanML3D new_joint_vecs.rar")


def maybe_gdown(file_id: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"skip existing {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    import gdown

    gdown.download(id=file_id, output=str(dest), quiet=False)
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--humanml", action="store_true")
    parser.add_argument("--evaluators", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all or not (args.humanml or args.evaluators):
        args.humanml = args.evaluators = True

    if args.evaluators:
        EVAL.mkdir(parents=True, exist_ok=True)
        for name, url in (("t2m.zip", T2M_EVAL), ("kit.zip", KIT_EVAL), ("glove.zip", GLOVE)):
            archive = EVAL / name
            if not archive.exists() or archive.stat().st_size < 10_000:
                import gdown

                print(f"download {name}")
                gdown.download(url, str(archive), quiet=False, fuzzy=True)
            if name == "glove.zip" and not (EVAL / "glove" / "our_vab_data.npy").exists():
                unzip(archive, EVAL)
            if name == "t2m.zip" and not (EVAL / "t2m" / "text_mot_match" / "model" / "finest.tar").exists():
                unzip(archive, EVAL)
            if name == "kit.zip" and not (EVAL / "kit" / "text_mot_match" / "model" / "finest.tar").exists():
                unzip(archive, EVAL)
        print("evaluators ready:", EVAL)

    if args.humanml:
        if not HML_RAR.exists() or HML_RAR.stat().st_size < 1_000_000_000:
            print("download HumanML3D new_joint_vecs.rar")
            maybe_gdown(HML_VEC_ID, HML_RAR)
        vec_dir = HML_DEST / "new_joint_vecs"
        npy_count = len(list(vec_dir.glob("*.npy"))) if vec_dir.exists() else 0
        if npy_count < 1000:
            print(f"extract {HML_RAR} -> {HML_DEST}")
            extract_rar(HML_RAR, HML_DEST)
        for name in ("Mean.npy", "Std.npy", "train.txt", "val.txt", "test.txt", "all.txt"):
            src = DATA / "downloads" / "HumanML3D-official" / name
            dst = HML_DEST / name
            if src.exists() and (not dst.exists() or dst.stat().st_size < src.stat().st_size):
                shutil.copy2(src, dst)
        print("HumanML3D vecs:", len(list((HML_DEST / "new_joint_vecs").glob("*.npy"))))
        print("KIT official vecs:", len(list((KIT_DEST / "new_joint_vecs").glob("*.npy"))))

    import sys
    sys.path.insert(0, str(PROJECT))
    from evaluation.guo_evaluator import evaluator_ready, install_mdm_eval_stats

    if evaluator_ready(EVAL):
        install_mdm_eval_stats(EVAL)
        print("MDM eval_humanml.py stats installed under research_sources/motion-diffusion-model/dataset")


if __name__ == "__main__":
    main()
