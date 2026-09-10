import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--motion-dim", type=int, required=True)
    args = parser.parse_args()
    required = ["train.txt", "val.txt", "test.txt", "new_joint_vecs", "texts"]
    missing = [name for name in required if not (args.root / name).exists()]
    if missing:
        raise SystemExit(f"Missing required entries: {missing}")
    ids = [line.strip() for line in (args.root / "train.txt").read_text().splitlines() if line.strip()]
    bad = []
    for sample_id in ids:
        path = args.root / "new_joint_vecs" / f"{sample_id}.npy"
        if not path.exists():
            bad.append(f"{sample_id}: missing motion")
            continue
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2 or array.shape[1] != args.motion_dim or not np.isfinite(array).all():
            bad.append(f"{sample_id}: invalid shape/values {array.shape}")
    if bad:
        raise SystemExit("\n".join(bad[:50]))
    print(f"Validated {len(ids)} training motions in {args.root}")


if __name__ == "__main__":
    main()

