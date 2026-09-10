import numpy as np

from evaluation.metrics import diversity, frechet_distance, multimodality, r_precision


def test_evaluation_metrics_sanity():
    embeddings = np.eye(8)
    assert abs(frechet_distance(embeddings, embeddings)) < 1e-6
    assert r_precision(embeddings, embeddings) == {"top_1": 1.0, "top_2": 1.0, "top_3": 1.0}
    assert diversity(embeddings, pairs=20) >= 0
    assert multimodality(np.stack([embeddings[:4], embeddings[4:]], axis=0), pairs=10) >= 0

