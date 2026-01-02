"""
Évaluation — à implémenter.

Exécution :
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt

Logs TensorBoard :
- test/loss
- test/accuracy
"""

import argparse
import os
import time
import yaml

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import get_device, set_seed


@torch.no_grad()
def evaluate_loss_accuracy(model: torch.nn.Module, loader, device: str):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")  # somme pour faire une moyenne propre ensuite

    loss_sum = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss_sum += float(loss_fn(logits, y).item())

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()

    total = max(1, total)
    avg_loss = loss_sum / total
    acc = correct / total
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")  # optionnel
    parser.add_argument("--seed", type=int, default=None)      # optionnel
    args = parser.parse_args()

    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_cfg = config.get("train", {})
    paths_cfg = config.get("paths", {})

    
    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 42))
    set_seed(seed)

    
    prefer_device = args.device if args.device else train_cfg.get("device", "auto")
    device = get_device(prefer_device)

    
    config = dict(config)
    config.setdefault("augment", {})
    config["augment"]["random_flip"] = False
    config["augment"]["random_crop"] = None
    config["augment"]["color_jitter"] = None
    _train_loader, _val_loader, test_loader, meta = get_dataloaders(config)
    model = build_model(config).to(device)
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint introuvable: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    test_loss, test_acc = evaluate_loss_accuracy(model, test_loader, device)
    runs_dir = paths_cfg.get("runs_dir", "./runs")
    eval_dir = os.path.join(runs_dir, "evaluate")
    os.makedirs(eval_dir, exist_ok=True)

    log_dir = os.path.join(eval_dir, f"TEST_{time.strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_scalar("test/loss", float(test_loss), 0)
    writer.add_scalar("test/accuracy", float(test_acc), 0)
    writer.close()
    print("========== TEST EVAL ==========")
    print(f"device        : {device}")
    print(f"seed          : {seed}")
    print(f"checkpoint    : {args.checkpoint}")
    print(f"test_loss     : {test_loss:.4f}")
    print(f"test_accuracy : {test_acc:.4f}")
    print("================================")


if __name__ == "__main__":
    main()

