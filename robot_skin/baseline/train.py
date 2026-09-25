"""Small training loop for :class:`BaselinePredictor` (CPU-friendly, deterministic)."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .dataset import NoContactSession, NoContactWindowDataset
from .model import BaselinePredictor


def train_baseline(model: BaselinePredictor, dataset: NoContactWindowDataset, *, epochs: int = 50,
                   batch_size: int = 256, lr: float = 1e-3, weight_decay: float = 1e-5,
                   val_frac: float = 0.2, seed: int = 0, device: str = "cpu") -> dict[str, list[float]]:
    """Fit with Huber loss; returns ``{"train": [...], "val": [...]}`` per-epoch losses.

    The split is a contiguous tail (``val_frac``) — windows overlap in time, so a random split
    would leak.
    """
    torch.manual_seed(seed)
    n = len(dataset)
    n_val = int(round(n * val_frac)) if n > 1 else 0
    tr = Subset(dataset, range(0, n - n_val))
    va = Subset(dataset, range(n - n_val, n))
    g = torch.Generator().manual_seed(seed)
    dl_tr = DataLoader(tr, batch_size=batch_size, shuffle=True, generator=g)
    dl_va = DataLoader(va, batch_size=batch_size) if n_val else None
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.HuberLoss(delta=1.0)
    hist: dict[str, list[float]] = {"train": [], "val": []}

    def _run(dl, train: bool) -> float:
        model.train(train)
        tot, cnt = 0.0, 0
        with torch.set_grad_enabled(train):
            for b in dl:
                b = {k: v.to(device) for k, v in b.items()}
                loss = loss_fn(model(b["pos"], b["nrm"], b["q"], b["qd"]), b["y"])
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                tot += float(loss.detach()) * b["y"].shape[0]
                cnt += b["y"].shape[0]
        return tot / max(cnt, 1)

    for _ in range(epochs):
        hist["train"].append(_run(dl_tr, True))
        if dl_va is not None:
            hist["val"].append(_run(dl_va, False))
    return hist


@torch.no_grad()
def predict_session(model: BaselinePredictor, session: NoContactSession,
                    joint_norm, device: str = "cpu") -> np.ndarray:
    """Frame-wise prediction ``[T, N]`` for any (contact or not) aligned session."""
    model.eval().to(device)
    q = torch.from_numpy(joint_norm[0].apply(session.q)).to(device)
    qd = torch.from_numpy(joint_norm[1].apply(session.qd)).to(device)
    pos = torch.as_tensor(session.pos, dtype=torch.float32, device=device)
    nrm = torch.as_tensor(session.nrm, dtype=torch.float32, device=device)
    return model(pos, nrm, q, qd).cpu().numpy()
