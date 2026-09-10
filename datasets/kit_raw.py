from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch.utils.data import Dataset


def hashed_text_embedding(text: str, dim: int = 512) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = np.linalg.norm(vector)
    return vector / max(norm, 1e-6)


def hashed_token_sequence(text: str, dim: int = 512, max_tokens: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """One hashed vector per word so the text encoder can attend over tokens."""
    words = [token for token in text.lower().replace(",", " ").replace(".", " ").split() if token]
    words = words[:max_tokens] or ["none"]
    embeddings = np.stack([hashed_text_embedding(word, dim) for word in words], axis=0)
    return embeddings, np.ones((len(words),), dtype=bool)


def motion_duration_frames(xml_path: Path, target_fps: int = 20) -> int:
    root = ET.parse(xml_path).getroot()
    times = [float(node.text) for node in root.findall(".//MotionFrame/Timestep")]
    if len(times) < 2:
        return 1
    duration = max(times[-1] - times[0], 1.0 / target_fps)
    return max(1, int(round(duration * target_fps)) + 1)


def build_length_cache(raw_root: Path, cache_path: Path, max_frames: int = 196):
    rows = []
    for annotation_path in sorted(raw_root.rglob("*_annotations.json")):
        sample_id = annotation_path.name.split("_")[0]
        xml_path = annotation_path.with_name(f"{sample_id}_mmm.xml")
        if not xml_path.exists():
            continue
        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not annotations:
            continue
        frames = min(motion_duration_frames(xml_path), max_frames)
        for annotation in annotations:
            rows.append((sample_id, annotation, frames))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        ids=np.asarray([row[0] for row in rows]),
        texts=np.asarray([row[1] for row in rows]),
        frames=np.asarray([row[2] for row in rows], dtype=np.int64),
    )
    return rows


class KITLengthDataset(Dataset):
    def __init__(self, cache_path, split="train", bins=50, embedding_dim=512):
        data = np.load(cache_path)
        ids, texts, frames = data["ids"], data["texts"], data["frames"]
        selector = np.asarray([int(hashlib.md5(value.encode()).hexdigest(), 16) % 10 for value in ids])
        keep = selector != 0 if split == "train" else selector == 0
        self.texts, self.frames = texts[keep], frames[keep]
        self.bins, self.embedding_dim = bins, embedding_dim

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        embedding = hashed_text_embedding(str(self.texts[index]), self.embedding_dim)
        length_class = min(self.bins - 1, max(0, int(self.frames[index]) // 4))
        return torch.from_numpy(embedding), torch.tensor(length_class), torch.tensor(int(self.frames[index]))

