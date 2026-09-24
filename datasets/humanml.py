from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.kit_raw import hashed_token_sequence


def parse_text_file(path: Path) -> list[dict]:
    """Parse HumanML3D / KIT-ML caption files: caption#tokens#f_tag#to_tag."""
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split("#")
        if not parts or not parts[0].strip():
            continue
        caption = parts[0].strip()
        tokens = parts[1].split() if len(parts) > 1 else [f"{word}/OTHER" for word in caption.split()]
        try:
            f_tag = float(parts[2]) if len(parts) > 2 else 0.0
            to_tag = float(parts[3]) if len(parts) > 3 else 0.0
        except ValueError:
            f_tag = to_tag = 0.0
        if np.isnan(f_tag):
            f_tag = 0.0
        if np.isnan(to_tag):
            to_tag = 0.0
        entries.append({"caption": caption, "tokens": tokens, "f_tag": f_tag, "to_tag": to_tag})
    return entries


class HumanMLDataset(Dataset):
    """Official HumanML3D / KIT-ML new_joint_vecs loader (Guo / MDM protocol).

    Training uses dataset Mean.npy / Std.npy. Guo evaluation re-normalizes with
    the T2M evaluator statistics separately.
    """

    def __init__(
        self,
        root,
        split="train",
        text_embedding_dir="text_embeddings",
        max_frames=196,
        text_dim=512,
        hash_text=False,
        mode="train",
        unit_length=4,
    ):
        self.root = Path(root)
        split_file = self.root / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        requested = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        self.motion_dir = self.root / "new_joint_vecs"
        self.caption_dir = self.root / "texts"
        self.text_dir = self.root / text_embedding_dir
        self.max_frames = max_frames
        self.text_dim = text_dim
        self.hash_text = hash_text
        self.mode = mode
        self.unit_length = unit_length
        mean_path, std_path = self.root / "Mean.npy", self.root / "Std.npy"
        if not mean_path.exists() or not std_path.exists():
            raise FileNotFoundError(f"Missing official Mean.npy / Std.npy under {self.root}")
        self.mean = np.load(mean_path).astype(np.float32)
        self.std = np.maximum(np.load(std_path).astype(np.float32), 1e-6)
        self.motion_dim = int(self.mean.shape[-1])
        self.min_motion_len = 40 if self.motion_dim == 263 else 24
        self.items = self._index_split(requested, split)
        if not self.items:
            raise FileNotFoundError(f"No official motions for split {split} under {self.motion_dir}")
        self.ids = [item["id"] for item in self.items]

    def _index_split(self, requested, split):
        import pickle

        cache_path = self.root / f".gala_index_{split}_{self.min_motion_len}_{self.max_frames}.pkl"
        split_mtime = (self.root / f"{split}.txt").stat().st_mtime
        if cache_path.exists():
            cached = pickle.loads(cache_path.read_bytes())
            if cached.get("mtime") == split_mtime and cached.get("count") == len(requested):
                return cached["items"]
        items = []
        max_raw = 200
        for sample_id in requested:
            motion_path = self.motion_dir / f"{sample_id}.npy"
            if not motion_path.exists():
                continue
            motion = np.load(motion_path, mmap_mode="r")
            n_frames = int(motion.shape[0])
            caption_path = self.caption_dir / f"{sample_id}.txt"
            entries = parse_text_file(caption_path)
            if not entries:
                entries = [{"caption": sample_id, "tokens": [f"{sample_id}/OTHER"], "f_tag": 0.0, "to_tag": 0.0}]
            whole = []
            for entry in entries:
                if entry["f_tag"] == 0.0 and entry["to_tag"] == 0.0:
                    whole.append(entry)
                    continue
                start = int(entry["f_tag"] * 20)
                end = int(entry["to_tag"] * 20)
                length = end - start
                if length < self.min_motion_len or length >= max_raw:
                    continue
                items.append(
                    {
                        "id": sample_id,
                        "start": start,
                        "end": end,
                        "captions": [entry],
                        "caption_path": str(caption_path),
                    }
                )
            if whole and self.min_motion_len <= n_frames < max_raw:
                items.append(
                    {
                        "id": sample_id,
                        "start": 0,
                        "end": n_frames,
                        "captions": whole,
                        "caption_path": str(caption_path),
                    }
                )
        cache_path.write_bytes(pickle.dumps({"mtime": split_mtime, "count": len(requested), "items": items}))
        return items

    def __len__(self):
        return len(self.items)

    def denormalize(self, motion):
        mean = torch.as_tensor(self.mean, device=motion.device, dtype=motion.dtype)
        std = torch.as_tensor(self.std, device=motion.device, dtype=motion.dtype)
        return motion * std + mean

    def _crop_eval_length(self, n_frames):
        length = min(n_frames, self.max_frames)
        length = (length // self.unit_length) * self.unit_length
        return max(length, self.unit_length)

    def __getitem__(self, index):
        item = self.items[index]
        motion = np.load(self.motion_dir / f"{item['id']}.npy").astype(np.float32)
        motion = motion[item["start"]: item["end"]][: self.max_frames]
        if self.mode in {"eval", "eval_fixed"}:
            length = self._crop_eval_length(len(motion))
            if self.mode == "eval_fixed" or len(motion) == length:
                start = 0
            else:
                start = int(np.random.randint(0, len(motion) - length + 1))
            motion = motion[start: start + length]
        if self.mode == "eval_fixed":
            text_entry = item["captions"][0]
        else:
            text_entry = item["captions"][int(np.random.randint(0, len(item["captions"])))]
        caption, tokens = text_entry["caption"], text_entry["tokens"]
        if self.hash_text:
            embeddings, mask = hashed_token_sequence(caption, self.text_dim)
            text, text_mask = torch.from_numpy(embeddings), torch.from_numpy(mask)
        else:
            text = torch.zeros(1, self.text_dim)
            text_mask = torch.ones(1, dtype=torch.bool)
        motion = (motion - self.mean) / self.std
        return {
            "id": item["id"],
            "motion": torch.from_numpy(motion),
            "length": len(motion),
            "caption": caption,
            "tokens": tokens,
            "caption_path": item["caption_path"],
            "text": text,
            "text_mask": text_mask,
        }


def collate_motion_text(batch):
    batch_size = len(batch)
    frames = max(item["length"] for item in batch)
    motion_dim = batch[0]["motion"].shape[-1]
    text_tokens = max(item["text"].shape[0] for item in batch)
    text_dim = batch[0]["text"].shape[-1]
    motion = torch.zeros(batch_size, frames, motion_dim)
    text = torch.zeros(batch_size, text_tokens, text_dim)
    text_mask = torch.zeros(batch_size, text_tokens, dtype=torch.bool)
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    for i, item in enumerate(batch):
        motion[i, : item["length"]] = item["motion"]
        tokens = item["text"].shape[0]
        text[i, :tokens] = item["text"]
        text_mask[i, :tokens] = item["text_mask"]
    return {
        "ids": [item["id"] for item in batch],
        "captions": [item["caption"] for item in batch],
        "tokens": [item["tokens"] for item in batch],
        "caption_paths": [item["caption_path"] for item in batch],
        "motion": motion,
        "lengths": lengths,
        "text": text,
        "text_mask": text_mask,
    }
