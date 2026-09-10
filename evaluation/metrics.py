import numpy as np


def _sqrt_psd(matrix):
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return (vectors * np.sqrt(np.clip(values, 0, None))) @ vectors.T


def frechet_distance(real, generated, eps=1e-6):
    real, generated = np.asarray(real), np.asarray(generated)
    mu_r, mu_g = real.mean(0), generated.mean(0)
    cov_r, cov_g = np.cov(real, rowvar=False), np.cov(generated, rowvar=False)
    sqrt_r = _sqrt_psd(cov_r)
    middle = _sqrt_psd(sqrt_r @ cov_g @ sqrt_r)
    value = (mu_r - mu_g) @ (mu_r - mu_g) + np.trace(cov_r + cov_g - 2 * middle)
    return float(max(value, 0.0))


def r_precision(text_embeddings, motion_embeddings, top_k=(1, 2, 3)):
    text = np.asarray(text_embeddings)
    motion = np.asarray(motion_embeddings)
    distances = ((text[:, None] - motion[None]) ** 2).sum(-1)
    ranking = distances.argsort(1)
    target = np.arange(len(text))[:, None]
    return {f"top_{k}": float((ranking[:, :k] == target).any(1).mean()) for k in top_k}


def diversity(embeddings, pairs=300, seed=0):
    embeddings = np.asarray(embeddings)
    rng = np.random.default_rng(seed)
    a = rng.integers(0, len(embeddings), pairs)
    b = rng.integers(0, len(embeddings), pairs)
    return float(np.linalg.norm(embeddings[a] - embeddings[b], axis=-1).mean())


def multimodality(grouped_embeddings, pairs=20, seed=0):
    grouped = np.asarray(grouped_embeddings)
    if grouped.ndim != 3:
        raise ValueError("Expected [prompts, samples, embedding_dim]")
    rng = np.random.default_rng(seed)
    a = rng.integers(0, grouped.shape[1], (grouped.shape[0], pairs))
    b = rng.integers(0, grouped.shape[1], (grouped.shape[0], pairs))
    rows = np.arange(grouped.shape[0])[:, None]
    return float(np.linalg.norm(grouped[rows, a] - grouped[rows, b], axis=-1).mean())
