import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

from torch.utils.tensorboard import SummaryWriter

from datasets.kit_raw import KITLengthDataset, build_length_cache
from trainers.checkpoint import CheckpointManager


class KITLengthEstimator(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, bins=50):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(hidden_dim, bins),
        )

    def forward(self, text_embedding):
        return self.network(text_embedding)


def evaluate(model, loader, device):
    model.eval(); correct = count = 0; absolute_error = 0.0
    with torch.no_grad():
        for text, target, frames in loader:
            prediction = model(text.to(device)).argmax(-1).cpu()
            correct += (prediction == target).sum().item()
            absolute_error += ((prediction * 4 - frames).abs()).sum().item()
            count += len(target)
    return correct / max(count, 1), absolute_error / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="/data/qinyang.z/project/data/KIT-ML-raw")
    parser.add_argument("--cache", default="/data/qinyang.z/project/data/KIT-ML-length-cache.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    cache = Path(args.cache)
    if not cache.exists():
        rows = build_length_cache(Path(args.raw_root), cache)
        print(f"Built cache with {len(rows)} text-motion pairs")
    train_set = KITLengthDataset(cache, "train")
    val_set = KITLengthDataset(cache, "val")
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, args.batch_size, num_workers=4, pin_memory=True)
    device = torch.device("cuda:0")
    model = KITLengthEstimator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    manager = CheckpointManager("/data/qinyang.z/project/checkpoints/kit_length", mode="min")
    writer = SummaryWriter("/data/qinyang.z/project/runs/kit_length")
    step = 0
    for epoch in range(args.epochs):
        model.train(); running = samples = 0
        for text, target, _ in train_loader:
            text, target = text.to(device, non_blocking=True), target.to(device, non_blocking=True)
            loss = nn.functional.cross_entropy(model(text), target)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            running += loss.item() * len(target); samples += len(target); step += 1
        accuracy, frame_mae = evaluate(model, val_loader, device)
        train_loss = running / max(samples, 1)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/accuracy", accuracy, epoch)
        writer.add_scalar("val/frame_mae", frame_mae, epoch)
        manager.save(model, optimizer, epoch, step, frame_mae, vars(args))
        print(f"epoch={epoch:03d} loss={train_loss:.4f} val_acc={accuracy:.4f} frame_mae={frame_mae:.2f}", flush=True)
    writer.close()


if __name__ == "__main__":
    main()
