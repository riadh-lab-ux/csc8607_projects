"""
M7 - Comparaisons de courbes (LR / WD / hyperparams modèle) en un seul script.

Exécution :
    python -m src.m7_compare --config configs/config.yaml
"""

import argparse
import copy
import os
import time
from typing import Any, Dict
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import set_seed, get_device, count_parameters, save_config_snapshot

def set_by_path(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Ex: dotted_key='train.optimizer.lr' -> cfg['train']['optimizer']['lr'] = value"""
    keys = dotted_key.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    for k, v in overrides.items():
        set_by_path(cfg, k, v)

def build_optimizer(model: torch.nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    opt_cfg = train_cfg["optimizer"]
    name = str(opt_cfg.get("name", "adam")).lower()
    lr = float(opt_cfg["lr"])
    wd = float(opt_cfg.get("weight_decay", 0.0))
    if name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, loss_fn, device: str):
    model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_acc += float(accuracy_from_logits(logits, y)) * bs
        total_n += bs
    total_n = max(1, total_n)
    return total_loss / total_n, total_acc / total_n


def train_one_run(cfg: Dict[str, Any], run_dir: str, artifacts_dir: str) -> Dict[str, float]:
    train_cfg = cfg["train"]
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)
    device = get_device(train_cfg.get("device", "auto"))
    train_loader, val_loader, _, meta = get_dataloaders(cfg)
    # model
    model = build_model(cfg).to(device)
    n_params = count_parameters(model)
    print(f"[run] device={device} seed={seed} params={n_params}")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, train_cfg)
    epochs = int(train_cfg.get("epochs", 30))
    # TensorBoard
    writer = SummaryWriter(run_dir)
    writer.add_text("run/seed", str(seed), 0)
    writer.add_text("run/device", str(device), 0)
    writer.add_text("run/model", str(cfg.get("model", {})), 0)
    writer.add_text("run/train_cfg", str(train_cfg), 0)
    writer.add_text("run/meta", str(meta), 0)
    # artifacts par run
    os.makedirs(artifacts_dir, exist_ok=True)
    save_config_snapshot(cfg, artifacts_dir)
    best_ckpt_path = os.path.join(artifacts_dir, "best.ckpt")
    best_val_acc = -1.0
    best_val_loss = float("inf")
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_acc_sum = 0.0
        n_seen = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            bs = y.size(0)
            acc = accuracy_from_logits(logits, y)
            epoch_loss_sum += float(loss.item()) * bs
            epoch_acc_sum += float(acc) * bs
            n_seen += bs
        n_seen = max(1, n_seen)
        train_loss_epoch = epoch_loss_sum / n_seen
        train_acc_epoch = epoch_acc_sum / n_seen

        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        writer.add_scalar("train/loss", float(train_loss_epoch), epoch)
        writer.add_scalar("train/accuracy", float(train_acc_epoch), epoch)
        writer.add_scalar("val/loss", float(val_loss), epoch)
        writer.add_scalar("val/accuracy", float(val_acc), epoch)
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss_epoch:.4f} train_acc={train_acc_epoch:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        # best ckpt 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_loss": float(val_loss),
                    "val_accuracy": float(val_acc),
                },
                best_ckpt_path,
            )
    elapsed = time.time() - start
    writer.add_text("run/time_sec", str(elapsed), 0)
    writer.close()
    return {
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "time_sec": float(elapsed),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        base_cfg = yaml.safe_load(f)
    runs_root = base_cfg.get("paths", {}).get("runs_dir", "./runs")
    artifacts_root = base_cfg.get("paths", {}).get("artifacts_dir", "./artifacts")
    os.makedirs(runs_root, exist_ok=True)
    os.makedirs(artifacts_root, exist_ok=True)
    group_name = f"M7_compare_{time.strftime('%Y%m%d-%H%M%S')}"
    group_runs_dir = os.path.join(runs_root, group_name)
    group_artifacts_dir = os.path.join(artifacts_root, group_name)
    os.makedirs(group_runs_dir, exist_ok=True)
    os.makedirs(group_artifacts_dir, exist_ok=True)
    base_lr = float(base_cfg["train"]["optimizer"]["lr"])
    run_specs = [
        ("LR_HALF", {"train.optimizer.lr": base_lr * 0.5}),
        ("WD_HIGH", {"train.optimizer.weight_decay": 1e-4}),
        ("HP_M4", {"model.num_modules": 4}),
        ("HP_BC_48-72-72", {"model.branch_channels": [48, 72, 72]}),
    ]
    summary = []
    for name, overrides in run_specs:
        cfg_run = copy.deepcopy(base_cfg)
        cfg_run["train"]["epochs"] = 30
        apply_overrides(cfg_run, overrides)
        m = cfg_run["model"]["num_modules"]
        bc = cfg_run["model"]["branch_channels"]
        lr = float(cfg_run["train"]["optimizer"]["lr"])
        wd = float(cfg_run["train"]["optimizer"]["weight_decay"])
        bs = int(cfg_run["train"]["batch_size"])
        run_name = f"{name}_M={m}_bc={bc[0]}-{bc[1]}-{bc[2]}_bs={bs}_lr={lr:.2e}_wd={wd:.0e}"
        run_dir = os.path.join(group_runs_dir, run_name)
        art_dir = os.path.join(group_artifacts_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(art_dir, exist_ok=True)
        print(f"\n===== RUN {name} -> {run_name} =====")
        metrics = train_one_run(cfg_run, run_dir, art_dir)
        summary.append({
            "run": run_name,
            "lr": lr,
            "wd": wd,
            "bs": bs,
            "M": m,
            "bc": f"{bc[0]}-{bc[1]}-{bc[2]}",
            **metrics
        })

        print(f"[done] {run_name} | best_val_acc={metrics['best_val_acc']:.4f} best_val_loss={metrics['best_val_loss']:.4f}")
    summary.sort(key=lambda d: d["best_val_acc"], reverse=True)
    print("\n========== SUMMARY (sorted by best_val_acc) ==========")
    for s in summary:
        print(f"{s['best_val_acc']:.4f} | {s['run']}")
    print("======================================================")
    print(f"[M7] All TB logs under: {group_runs_dir}")
    print(f"[M7] All artifacts under: {group_artifacts_dir}")


if __name__ == "__main__":
    main()
