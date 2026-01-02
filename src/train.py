"""
Entraînement principal (à implémenter par l'étudiant·e).

Exécution :
    python -m src.train --config configs/config.yaml [--seed 42] [--overfit_small]
                        [--max_epochs 20] [--max_steps 1000]

"""

import argparse
import os
import time
import copy
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import set_seed, get_device, count_parameters, save_config_snapshot

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, loss_fn, device: str):
    model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits, y) * bs
        total_n += bs
    total_n = max(1, total_n)
    return total_loss / total_n, total_acc / total_n


def build_optimizer(model: torch.nn.Module, train_cfg: dict):
    opt_cfg = train_cfg["optimizer"]
    name = str(opt_cfg["name"]).lower()
    lr = float(opt_cfg["lr"])
    wd = float(opt_cfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)



def make_run_tag(config: dict, seed: int, overfit_mode: bool) -> str:
    m = config["model"]["num_modules"]
    bc = config["model"]["branch_channels"]
    tag = f"M{m}_ch{bc}_seed{seed}"
    return ("OVERFIT_" + tag) if overfit_mode else ("TRAIN_" + tag)


def init_writer(runs_dir: str, run_tag: str):
    log_dir = os.path.join(runs_dir, f"{run_tag}_{time.strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir=log_dir)
    return writer, log_dir


def parse_epochs_steps(train_cfg: dict, args):
    epochs = int(train_cfg.get("epochs", 20))
    if args.max_epochs is not None:
        epochs = int(args.max_epochs)
    max_steps = train_cfg.get("max_steps", None)
    if args.max_steps is not None:
        max_steps = int(args.max_steps)
    if max_steps in (None, "null"):
        max_steps = None

    return epochs, max_steps

def run_overfit_small(base_config: dict, train_cfg: dict, seed: int, device: str,
                      runs_dir: str, artifacts_dir: str, args):
    """
    M3 : sur-apprendre sur un tout petit subset (k exemples).
    """
    config = copy.deepcopy(base_config)

    # Désactiver augmentations aléatoires
    config.setdefault("augment", {})
    config["augment"]["random_flip"] = False
    config["augment"]["random_crop"] = None
    config["augment"]["color_jitter"] = None
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    k = int(train_cfg.get("overfit_size", 32))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(train_loader.dataset), generator=g)[:k].tolist()
    small_train_ds = Subset(train_loader.dataset, idx)
    bs = min(int(train_cfg.get("batch_size", 64)), k)
    pin = (device == "cuda")
    train_loader = DataLoader(small_train_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(small_train_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=pin)

    print(f"[overfit_small] k={k} batch_size={bs} (val = même subset)")

    model = build_model(config).to(device)
    n_params = count_parameters(model)
    print(f"[model] params={n_params} device={device} seed={seed}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, train_cfg)
    epochs, max_steps = parse_epochs_steps(train_cfg, args)

    run_tag = make_run_tag(config, seed, overfit_mode=True)
    writer, log_dir = init_writer(runs_dir, run_tag)
    writer.add_text("run/mode", "overfit_small", 0)
    writer.add_text("run/seed", str(seed), 0)
    writer.add_text("run/device", str(device), 0)
    writer.add_text("run/model", str(config.get("model", {})), 0)
    writer.add_text("run/train_cfg", str(train_cfg), 0)
    writer.add_text("run/meta", str(meta), 0)
    global_step = 0
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_acc, total_n = 0.0, 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            bs_ = y.size(0)
            acc = accuracy_from_logits(logits, y)
            total_loss += loss.item() * bs_
            total_acc += acc * bs_
            total_n += bs_
            writer.add_scalar("train/loss", float(loss.item()), global_step)
            writer.add_scalar("train/acc", float(acc), global_step)
            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                break
        total_n = max(1, total_n)
        train_loss = total_loss / total_n
        train_acc = total_acc / total_n
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        writer.add_scalar("val/loss", float(val_loss), epoch)
        writer.add_scalar("val/acc", float(val_acc), epoch)
        print(
            f"[OVERFIT] Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if max_steps is not None and global_step >= max_steps:
            break
    elapsed = time.time() - start
    writer.add_text("run/time_sec", str(elapsed), 0)
    writer.close()
    print(f"[overfit_small] logs: {log_dir}")
    print(f"[overfit_small] elapsed: {elapsed:.1f}s")

def run_train_full(config: dict, train_cfg: dict, seed: int, device: str,
                   runs_dir: str, artifacts_dir: str, args):
    """
    M6 : entraînement “complet” (10–20 epochs typiquement),
    """
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    model = build_model(config).to(device)
    n_params = count_parameters(model)
    print(f"[model] params={n_params} device={device} seed={seed}")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, train_cfg)
    epochs, max_steps = parse_epochs_steps(train_cfg, args)
    run_tag = make_run_tag(config, seed, overfit_mode=False)
    writer, log_dir = init_writer(runs_dir, run_tag)

    writer.add_text("run/mode", "train_full", 0)
    writer.add_text("run/seed", str(seed), 0)
    writer.add_text("run/device", str(device), 0)
    writer.add_text("run/model", str(config.get("model", {})), 0)
    writer.add_text("run/train_cfg", str(train_cfg), 0)
    writer.add_text("run/meta", str(meta), 0)

    # checkpoint best
    best_path = os.path.join(artifacts_dir, "best.ckpt")
    best_val_acc = -1.0
    global_step = 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_n = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            bs = y.size(0)
            total_train_loss += float(loss.item()) * bs
            total_train_n += bs
            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                break
        total_train_n = max(1, total_train_n)
        train_loss = total_train_loss / total_train_n
        writer.add_scalar("train/loss", float(train_loss), epoch)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        writer.add_scalar("val/loss", float(val_loss), epoch)
        writer.add_scalar("val/accuracy", float(val_acc), epoch)
        print(
            f"[TRAIN] Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "meta": meta,
                },
                best_path,
            )
            print(f"  -> saved BEST: {best_path} (val_acc={best_val_acc:.4f})")

        if max_steps is not None and global_step >= max_steps:
            break

    elapsed = time.time() - start
    writer.add_text("run/time_sec", str(elapsed), 0)
    writer.close()

    print(f"[train_full] logs: {log_dir}")
    print(f"[train_full] best checkpoint saved: {best_path}")
    print(f"[train_full] elapsed: {elapsed:.1f}s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overfit_small", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()
    base_config = yaml.safe_load(open(args.config, "r"))
    train_cfg = base_config.get("train", {})
    paths_cfg = base_config.get("paths", {})
    runs_dir = paths_cfg.get("runs_dir", "./runs")
    artifacts_dir = paths_cfg.get("artifacts_dir", "./artifacts")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 42))
    set_seed(seed)
    device = get_device(train_cfg.get("device", "auto"))
    overfit_mode = bool(args.overfit_small) or bool(train_cfg.get("overfit_small", False))
    save_config_snapshot(base_config, artifacts_dir)
    if overfit_mode:
        run_overfit_small(base_config, train_cfg, seed, device, runs_dir, artifacts_dir, args)
    else:
        run_train_full(base_config, train_cfg, seed, device, runs_dir, artifacts_dir, args)


if __name__ == "__main__":
    main()
