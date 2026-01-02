# src/lr_finder.py
"""
Recherche de taux d'apprentissage (LR finder).

Exécution :
    python -m src.lr_finder --config configs/config.yaml

Logs TensorBoard (onglet Scalars) :
- lr_finder/lr
- lr_finder/loss
"""

import argparse
import os
import time
import random

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model


def lr_finder(config, num_steps=200, initial_lr=1e-7, final_lr=1.0):
    # Seed fixe (pas besoin de l'ajouter dans la config)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    prefer = config.get("train", {}).get("device", "auto")
    if prefer == "cpu":
        device = "cpu"
    elif prefer == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # TensorBoard
    runs_dir = config.get("paths", {}).get("runs_dir", "./runs")
    os.makedirs(runs_dir, exist_ok=True)
    log_dir = os.path.join(runs_dir, f"lr_finder_{time.strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir=log_dir)
    # Data / Model
    train_loader, _, _, meta = get_dataloaders(config)
    model = build_model(config).to(device)
    model.train()
    # Loss (Tiny ImageNet = multiclass)
    criterion = nn.CrossEntropyLoss()
    # Optimizer (utilise ce qui est déjà dans train.optimizer si présent)
    opt_cfg = config.get("train", {}).get("optimizer", {})
    opt_name = str(opt_cfg.get("name", "adam")).lower()
    weight_decay = float(opt_cfg.get("weight_decay", 1e-5))
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    gamma = (final_lr / initial_lr) ** (1.0 / (num_steps - 1))
    it = iter(train_loader)
    best_loss = float("inf")
    for step in range(num_steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        lr = initial_lr * (gamma ** step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            print(f"[lr_finder] loss non-finie à step={step}, arrêt.")
            break
        loss_val = float(loss.item())
        loss.backward()
        optimizer.step()
        # logs
        writer.add_scalar("lr_finder/lr", lr, step)
        writer.add_scalar("lr_finder/loss", loss_val, step)
        # stop
        best_loss = min(best_loss, loss_val)
        if step > 10 and loss_val > 4.0 * best_loss:
            print(f"[lr_finder] divergence détectée à step={step} (loss={loss_val:.4f}), arrêt.")
            break
        if step % 20 == 0:
            print(f"[lr_finder] step={step:03d} lr={lr:.2e} loss={loss_val:.4f}")
    writer.close()
    print(f"[lr_finder] terminé. logs: {log_dir}")
    print(f"[lr_finder] device={device} weight_decay={weight_decay} meta={meta}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r"))

    # Hyperparamètres
    NUM_STEPS = 200
    INITIAL_LR = 1e-7
    FINAL_LR = 1.0

    lr_finder(
        config,
        num_steps=NUM_STEPS,
        initial_lr=INITIAL_LR,
        final_lr=FINAL_LR
    )


if __name__ == "__main__":
    main()
