from pathlib import Path

import torch


class CheckpointManager:
    def __init__(self, directory, mode="min"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")

    def _is_better(self, metric):
        return metric < self.best if self.mode == "min" else metric > self.best

    def save(
        self, model, optimizer, epoch, step, metric, config, scheduler=None, scaler=None,
        ema=None, update_best=True,
    ):
        raw_model = model.module if hasattr(model, "module") else model
        state = {
            "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "step": step, "metric": metric, "config": config,
            "optimizer_step": config.get("optimizer_step", step) if isinstance(config, dict) else step,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "ema": ema.state_dict() if ema is not None else None,
        }
        torch.save(state, self.directory / "last.pt")
        if update_best and self._is_better(metric):
            self.best = metric
            torch.save(state, self.directory / "best.pt")

    def resume(self, model, optimizer, path, scheduler=None, scaler=None, ema=None, map_location="cpu"):
        state = torch.load(path, map_location=map_location, weights_only=False)
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        if scaler and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        if ema is not None and state.get("ema"):
            ema.load_state_dict(state["ema"])
        self.best = state.get("metric", self.best)
        return state

