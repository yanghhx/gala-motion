from pathlib import Path

import numpy as np
import torch

from datasets.humanml import HumanMLDataset, parse_text_file
from evaluation.guo_evaluator import (
    evaluator_ready,
    official_replication_metrics,
    to_evaluator_space,
    vectorize_tokens,
)


def test_parse_official_caption_file():
    path = Path("/home/qinyang/桌面/project/research_sources/HumanML3D/HumanML3D/texts/000000.txt")
    entries = parse_text_file(path)
    assert entries
    assert entries[0]["caption"]
    assert any("/" in token for token in entries[0]["tokens"])


def test_humanml_official_item_is_263d():
    dataset = HumanMLDataset("/home/qinyang/桌面/project/data/HumanML3D", split="train", hash_text=False)
    item = dataset[0]
    assert item["motion"].shape[-1] == 263
    assert item["tokens"]
    assert item["caption"]
    restored = dataset.denormalize(item["motion"].unsqueeze(0))[0]
    raw = np.load(dataset.motion_dir / f"{item['id']}.npy").astype(np.float32)
    raw = raw[dataset.items[0]["start"]: dataset.items[0]["end"]][: dataset.max_frames]
    np.testing.assert_allclose(restored.numpy()[: len(raw)], raw[: restored.shape[0]], atol=1e-4)


def test_kit_official_item_is_251d():
    dataset = HumanMLDataset("/home/qinyang/桌面/project/data/KIT-ML-official", split="train", hash_text=False)
    item = dataset[0]
    assert item["motion"].shape[-1] == 251
    assert dataset.mean.shape == (251,)


def test_t2m_renorm_uses_evaluator_std():
    motion = torch.zeros(1, 8, 263)
    dataset_mean = torch.zeros(263)
    dataset_std = torch.ones(263)
    eval_mean = torch.zeros(263)
    eval_std = torch.ones(263)
    eval_std[:4] = 0.04
    converted = to_evaluator_space(motion, dataset_mean, dataset_std, eval_mean, eval_std)
    assert converted.shape == motion.shape
    assert torch.allclose(converted[..., :4], torch.zeros(1, 8, 4))


def test_official_metrics_use_batch_32():
    rng = np.random.default_rng(0)
    real = rng.normal(size=(64, 16)).astype(np.float32)
    gen = real + rng.normal(scale=0.01, size=real.shape).astype(np.float32)
    text = gen + rng.normal(scale=0.01, size=gen.shape).astype(np.float32)
    metrics = official_replication_metrics(real, gen, text, batch_size=32, diversity_times=20)
    assert metrics["rprecision_samples"] == 64
    assert 0 <= metrics["R@3"] <= 1
    assert metrics["FID"] >= 0
    assert metrics["Diversity"] > 0


def test_guo_evaluator_files_are_ready():
    assert evaluator_ready()


def test_vectorize_tokens_adds_sos_eos():
    class Dummy:
        def __getitem__(self, token):
            return np.ones(300, dtype=np.float32), np.zeros(15, dtype=np.float32)

    words, pos, sent_len = vectorize_tokens(Dummy(), ["a/DET", "man/NOUN"])
    assert words.shape[0] == 22
    assert sent_len == 4
