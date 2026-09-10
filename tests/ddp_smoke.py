import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from models.gala_motion import GALAMotion, GALAMotionConfig


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    cfg = GALAMotionConfig(
        motion_dim=263, num_joints=22, latent_dim=32, model_dim=32,
        text_dim=24, num_heads=4, graph_layers=1, dit_layers=1, max_frames=20,
    )
    model = DistributedDataParallel(GALAMotion(cfg).cuda(), device_ids=[local_rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(2):
        losses = model(
            torch.randn(2, 20, 263, device="cuda"), torch.tensor([20, 16], device="cuda"),
            torch.randn(2, 6, 24, device="cuda"), torch.ones(2, 6, dtype=torch.bool, device="cuda"),
        )
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
    dist.barrier()
    if local_rank == 0:
        print(f"DDP_SMOKE_OK loss={losses['total'].item():.6f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
