"""Official Guo T2M evaluator — same protocol as MDM eval/eval_humanml.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MDM_ROOT = PROJECT_ROOT / "research_sources" / "motion-diffusion-model"
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "checkpoints" / "t2m_evaluators"
MAX_TEXT_LEN = 20
MAX_MOTION_LEN = 196

if str(MDM_ROOT) not in sys.path:
    sys.path.append(str(MDM_ROOT))


def evaluator_stat_paths(dataset_name: str, eval_root: Path | None = None) -> tuple[Path, Path]:
    root = Path(eval_root) if eval_root else DEFAULT_EVAL_ROOT
    if dataset_name in {"humanml", "humanml3d", "t2m"}:
        meta = root / "t2m" / "Comp_v6_KLD01" / "meta"
    else:
        meta = root / "kit" / "Comp_v6_KLD005" / "meta"
    return meta / "mean.npy", meta / "std.npy"


def install_mdm_eval_stats(eval_root: Path | None = None) -> None:
    """Copy Guo meta Mean/Std into the MDM dataset folder used by eval_humanml.py."""
    dest = MDM_ROOT / "dataset"
    dest.mkdir(parents=True, exist_ok=True)
    mapping = {
        "t2m": evaluator_stat_paths("t2m", eval_root),
        "kit": evaluator_stat_paths("kit", eval_root),
    }
    for name, (mean_src, std_src) in mapping.items():
        if mean_src.exists() and std_src.exists():
            np.save(dest / f"{name}_mean.npy", np.load(mean_src))
            np.save(dest / f"{name}_std.npy", np.load(std_src))
    glove = (Path(eval_root) if eval_root else DEFAULT_EVAL_ROOT) / "glove"
    link = dest.parent / "glove"
    if glove.exists() and not link.exists():
        try:
            link.symlink_to(glove)
        except OSError:
            pass


def vectorize_tokens(vectorizer, tokens: list[str], max_text_len: int = MAX_TEXT_LEN):
    tokens = [token for token in tokens if "/" in token][:max_text_len]
    if not tokens:
        tokens = ["unk/OTHER"]
    tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
    sent_len = len(tokens)
    tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
    word_dim, pos_dim = 300, 15
    words = np.zeros((len(tokens), word_dim), dtype=np.float32)
    pos = np.zeros((len(tokens), pos_dim), dtype=np.float32)
    for index, token in enumerate(tokens):
        word_vec, pos_vec = vectorizer[token]
        words[index] = word_vec
        pos[index] = pos_vec
    return words, pos, sent_len


def to_evaluator_space(motion, dataset_mean, dataset_std, eval_mean, eval_std):
    original = motion * dataset_std + dataset_mean
    return (original - eval_mean) / eval_std


def pad_motions(motion: torch.Tensor, max_len: int = MAX_MOTION_LEN) -> torch.Tensor:
    if motion.shape[1] >= max_len:
        return motion[:, :max_len]
    return F.pad(motion, (0, 0, 0, max_len - motion.shape[1]))


def _metric_statistics(values, replications):
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    conf = 1.96 * std / np.sqrt(max(replications, 1))
    return mean, conf


class GuoEvaluator:
    def __init__(self, dataset_name: str, device, eval_root: Path | None = None):
        from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
        from data_loaders.humanml.utils.word_vectorizer import WordVectorizer

        self.dataset_name = "humanml" if dataset_name in {"humanml", "humanml3d", "t2m"} else "kit"
        self.device = device
        self.eval_root = Path(eval_root) if eval_root else DEFAULT_EVAL_ROOT
        ckpt_name = "t2m" if self.dataset_name == "humanml" else "kit"
        finest = self.eval_root / ckpt_name / "text_mot_match" / "model" / "finest.tar"
        glove = self.eval_root / "glove"
        if not finest.exists():
            raise FileNotFoundError(f"Missing Guo evaluator: {finest}")
        if not (glove / "our_vab_data.npy").exists():
            raise FileNotFoundError(f"Missing GloVe vectors under {glove}")
        mean_path, std_path = evaluator_stat_paths(self.dataset_name, self.eval_root)
        if not mean_path.exists() or not std_path.exists():
            raise FileNotFoundError(f"Missing T2M evaluator Mean/Std: {mean_path}")
        self.eval_mean = torch.from_numpy(np.load(mean_path).astype(np.float32)).to(device)
        self.eval_std = torch.from_numpy(np.maximum(np.load(std_path).astype(np.float32), 1e-6)).to(device)

        import os

        cwd = Path.cwd()
        try:
            os.chdir(self.eval_root)
            self.wrapper = EvaluatorMDMWrapper(self.dataset_name, device)
        finally:
            os.chdir(cwd)
        self.vectorizer = WordVectorizer(str(glove), "our_vab")

    def prepare_motion(self, motion, lengths, dataset_mean, dataset_std):
        mean = torch.as_tensor(dataset_mean, device=motion.device, dtype=motion.dtype)
        std = torch.as_tensor(dataset_std, device=motion.device, dtype=motion.dtype)
        eval_mean = self.eval_mean.to(device=motion.device, dtype=motion.dtype)
        eval_std = self.eval_std.to(device=motion.device, dtype=motion.dtype)
        converted = to_evaluator_space(motion, mean, std, eval_mean, eval_std)
        return pad_motions(converted), lengths.clamp(max=MAX_MOTION_LEN)

    def _text_tensors(self, tokens_batch):
        max_text = MAX_TEXT_LEN + 2
        words = np.zeros((len(tokens_batch), max_text, 300), dtype=np.float32)
        pos = np.zeros((len(tokens_batch), max_text, 15), dtype=np.float32)
        cap_lens = []
        for i, tokens in enumerate(tokens_batch):
            word, pos_ohot, cap_len = vectorize_tokens(self.vectorizer, tokens)
            words[i] = word
            pos[i] = pos_ohot
            cap_lens.append(cap_len)
        return (
            torch.from_numpy(words).to(self.device),
            torch.from_numpy(pos).to(self.device),
            torch.tensor(cap_lens, device=self.device),
        )

    @torch.no_grad()
    def embed_batch(self, motions, lengths, tokens_batch):
        word_t, pos_t, cap_t = self._text_tensors(tokens_batch)
        # MDM collate_fn sorts by caption length; Guo's text GRU requires it.
        order = torch.argsort(cap_t, descending=True)
        word_t, pos_t, cap_t = word_t[order], pos_t[order], cap_t[order]
        motions, lengths = motions[order], lengths[order]
        text_emb, motion_emb = self.wrapper.get_co_embeddings(word_t, pos_t, cap_t, motions, lengths)
        return text_emb.detach().cpu().numpy(), motion_emb.detach().cpu().numpy()

    @torch.no_grad()
    def embed_motion(self, motions, lengths):
        return self.wrapper.get_motion_embeddings(motions, lengths).detach().cpu().numpy()


def official_replication_metrics(real_motion, gen_motion, text_emb, batch_size=32, diversity_times=300):
    from data_loaders.humanml.utils.metrics import (
        calculate_activation_statistics,
        calculate_diversity,
        calculate_frechet_distance,
        calculate_top_k,
        euclidean_distance_matrix,
    )

    matching_sum = 0.0
    top_k_count = np.zeros(3, dtype=np.float64)
    counted = 0
    for start in range(0, len(gen_motion) - batch_size + 1, batch_size):
        text = text_emb[start: start + batch_size]
        motion = gen_motion[start: start + batch_size]
        dist = euclidean_distance_matrix(text, motion)
        matching_sum += np.trace(dist)
        argsmax = np.argsort(dist, axis=1)
        top_k_count += calculate_top_k(argsmax, top_k=3).sum(axis=0)
        counted += batch_size
    if counted == 0:
        raise ValueError("Need at least one full batch of 32 for official R-Precision")
    gt_mu, gt_cov = calculate_activation_statistics(real_motion)
    gen_mu, gen_cov = calculate_activation_statistics(gen_motion)
    pairs = min(diversity_times, max(len(gen_motion) - 1, 1))
    return {
        "R@1": float(top_k_count[0] / counted),
        "R@2": float(top_k_count[1] / counted),
        "R@3": float(top_k_count[2] / counted),
        "FID": float(calculate_frechet_distance(gt_mu, gt_cov, gen_mu, gen_cov)),
        "MM Dist": float(matching_sum / counted),
        "Diversity": float(calculate_diversity(gen_motion, pairs)),
        "num_samples": int(len(gen_motion)),
        "rprecision_samples": int(counted),
    }


def summarize_replications(runs):
    keys = ["R@1", "R@2", "R@3", "FID", "MM Dist", "Diversity"]
    extra = sorted({key for run in runs for key in run if str(key).startswith("physical/")})
    summary = {"protocol": "official_guo_t2m", "replication_times": len(runs)}
    for key in keys + extra:
        mean, conf = _metric_statistics([run[key] for run in runs], len(runs))
        summary[key] = float(mean)
        summary[f"{key}_ci"] = float(conf)
    summary["num_samples"] = int(np.mean([run["num_samples"] for run in runs]))
    return summary


def evaluator_ready(eval_root: Path | None = None) -> bool:
    root = Path(eval_root) if eval_root else DEFAULT_EVAL_ROOT
    t2m_mean, t2m_std = evaluator_stat_paths("t2m", root)
    return (
        (root / "t2m" / "text_mot_match" / "model" / "finest.tar").exists()
        and (root / "glove" / "our_vab_data.npy").exists()
        and t2m_mean.exists()
        and t2m_std.exists()
    )
