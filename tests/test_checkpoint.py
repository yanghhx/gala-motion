import torch

from trainers.checkpoint import CheckpointManager


def test_checkpoint_roundtrip(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager = CheckpointManager(tmp_path, mode="min")
    manager.save(model, optimizer, epoch=4, step=17, metric=0.4, config={"name": "test"})
    with torch.no_grad():
        model.weight.zero_()
    state = manager.resume(model, optimizer, tmp_path / "last.pt")
    assert state["epoch"] == 4 and state["step"] == 17
    assert torch.count_nonzero(model.weight) > 0
    assert (tmp_path / "best.pt").exists()
