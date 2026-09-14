from types import SimpleNamespace

import torch

from trainers.checkpoint import CheckpointManager
from trainers.train import _prefer_global_if_part_regressed


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


def test_v2_init_falls_back_when_part_fid_regresses(tmp_path):
    part_dir = tmp_path / "gala_humanml3d_flow_part"
    global_dir = tmp_path / "gala_humanml3d_flow"
    part_dir.mkdir()
    global_dir.mkdir()
    torch.save({"metric": 0.433, "model": {}}, part_dir / "best.pt")
    torch.save({"metric": 0.330, "model": {}}, global_dir / "best.pt")
    cfg = SimpleNamespace(use_kinematic_flow=True)
    chosen = _prefer_global_if_part_regressed(str(part_dir / "best.pt"), cfg, local_rank=1)
    assert chosen == str(global_dir / "best.pt")
    cfg.use_kinematic_flow = False
    chosen = _prefer_global_if_part_regressed(str(part_dir / "best.pt"), cfg, local_rank=1)
    assert chosen == str(part_dir / "best.pt")
