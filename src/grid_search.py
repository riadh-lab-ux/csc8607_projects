"""
Mini grid search (rapide) autour des paramètres trouvés.

Exécution :
    python -m src.grid_search --config configs/config.yaml
    python -m src.grid_search --config configs/config.yaml --refined

Logs TensorBoard :
- Scalars: train/loss, val/loss, val/acc
- HParams: hparams + best_val_acc
"""

import argparse
import os
import time
import copy
import itertools

import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import set_seed, get_device, save_config_snapshot  # snapshot


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
        loss_sum += float(loss.item()) * y.size(0)

    avg_loss = loss_sum / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def train_one_run(cfg, run_dir, return_best_state=False):
    # seed stable pour comparer proprement les configs
    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed)

    device = get_device(cfg.get("train", {}).get("device", "auto"))

    train_loader, val_loader, _, meta = get_dataloaders(cfg)
    model = build_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss()

    opt_cfg = cfg["train"]["optimizer"]
    lr = float(opt_cfg["lr"])
    wd = float(opt_cfg["weight_decay"])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    epochs = int(cfg["hparams"]["epochs"])
    max_steps = cfg["train"].get("max_steps", None)
    if max_steps is not None:
        max_steps = int(max_steps)

    writer = SummaryWriter(run_dir)

    global_step = 0
    best_val_acc = 0.0
    best_val_loss = float("inf")

    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_seen = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            epoch_loss += float(loss.item()) * bs
            n_seen += bs

            # (inchangé) train/loss par step (utile pour "début d'entraînement")
            writer.add_scalar("train/loss", float(loss.item()), global_step)
            global_step += 1

            if max_steps is not None and global_step >= max_steps:
                break

        train_loss_epoch = epoch_loss / max(1, n_seen)
        writer.add_scalar("train/loss_epoch", float(train_loss_epoch), epoch)

        val_loss, val_acc = evaluate(model, val_loader, device)
        writer.add_scalar("val/loss", float(val_loss), epoch)
        writer.add_scalar("val/acc", float(val_acc), epoch)

        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            if return_best_state:
                best_state = copy.deepcopy(model.state_dict())

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)

        if max_steps is not None and global_step >= max_steps:
            break

    # Log HParams
    hparams_dict = {
        "lr": float(cfg["train"]["optimizer"]["lr"]),
        "weight_decay": float(cfg["train"]["optimizer"]["weight_decay"]),
        "batch_size": int(cfg["train"]["batch_size"]),
        "num_modules": int(cfg["model"]["num_modules"]),
        "branch_channels": "-".join(map(str, cfg["model"]["branch_channels"])),
        "seed": int(cfg.get("train", {}).get("seed", 42)),
    }
    metrics_dict = {
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
    }
    writer.add_hparams(hparams_dict, metrics_dict)
    writer.close()

    if return_best_state:
        return metrics_dict, meta, best_state
    return metrics_dict, meta


def grid_search_base(base_cfg, runs_dir, artifacts_dir):
    hp = base_cfg["hparams"]

    grid_lr = hp["lr"]
    grid_bs = hp["batch_size"]
    grid_wd = hp["weight_decay"]
    grid_m = hp["num_modules"]
    grid_bc = hp["branch_channels"]
    tag = hp.get("tag", "grid_search")

    summary = []

    best_overall_acc = -1.0
    best_overall_cfg = None
    best_overall_run = None

    combos = list(itertools.product(grid_lr, grid_bs, grid_wd, grid_m, grid_bc))
    print(f"[grid_search] {len(combos)} combinaisons à tester")

    for k, (lr, bs, wd, m, bc) in enumerate(combos):
        cfg = copy.deepcopy(base_cfg)

        cfg["train"]["batch_size"] = int(bs)
        cfg["train"]["optimizer"]["lr"] = float(lr)
        cfg["train"]["optimizer"]["weight_decay"] = float(wd)
        cfg["model"]["num_modules"] = int(m)
        cfg["model"]["branch_channels"] = list(map(int, bc))

        run_name = (
            f"{tag}_k={k:03d}_M={m}_bc={bc[0]}-{bc[1]}-{bc[2]}_"
            f"bs={bs}_lr={lr:.2e}_wd={wd:.0e}_{time.strftime('%H%M%S')}"
        )
        run_dir = os.path.join(runs_dir, run_name)
        print(f"\n[grid_search] RUN: {run_name}")
        metrics, _meta = train_one_run(cfg, run_dir)
        best_val_acc = float(metrics["best_val_acc"])
        best_val_loss = float(metrics["best_val_loss"])
        summary.append({
            "run": run_name,
            "M": int(m),
            "bc": tuple(map(int, bc)),
            "bs": int(bs),
            "lr": float(lr),
            "wd": float(wd),
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
        })

        print(f"  -> best_val_acc={best_val_acc:.4f} | best_val_loss={best_val_loss:.4f}")
        if best_val_acc > best_overall_acc:
            best_overall_acc = best_val_acc
            best_overall_cfg = copy.deepcopy(cfg)
            best_overall_run = run_name
    summary.sort(key=lambda d: d["best_val_acc"], reverse=True)
    print("\n========== RÉSUMÉ (trié par best_val_acc) ==========")
    for r in summary:
        print(
            f"{r['best_val_acc']:.4f} | "
            f"M={r['M']} bc={r['bc']} bs={r['bs']} "
            f"lr={r['lr']:.2e} wd={r['wd']:.0e} | {r['run']}"
        )
    print("====================================================")
    best = summary[0]
    print("\n[grid_search] MEILLEUR CHOIX:")
    print(best)
    if best_overall_cfg is not None:
        out_dir = os.path.join(artifacts_dir, "grid_search_best")
        save_config_snapshot(best_overall_cfg, out_dir)
        print(f"[grid_search] snapshot sauvegardé dans: {out_dir}/config_snapshot.yaml")
        print(f"[grid_search] meilleur run snapshot: {best_overall_run}")


def grid_search_refined(base_cfg, runs_dir, artifacts_dir):
    """
    M8: mini grid search resserrée autour de la meilleure config.
    - 8 epochs par run
    - 3 LR autour de 2e-3
    - 3 WD: 1e-5 + 2 plus petits
    """
    refined_lrs = [1.5e-3, 2.0e-3, 2.5e-3]
    refined_wds = [1e-5, 5e-6, 1e-6]
    refined_epochs = 8
    tag = "mini_gridsearch_m8"
    runs_subdir = os.path.join(runs_dir, tag)
    artifacts_subdir = os.path.join(artifacts_dir, tag)
    os.makedirs(runs_subdir, exist_ok=True)
    os.makedirs(artifacts_subdir, exist_ok=True)

    combos = list(itertools.product(refined_lrs, refined_wds))
    print(f"[{tag}] {len(combos)} combinaisons à tester (LR x WD)")

    summary = []
    best_overall_acc = -1.0
    best_overall_cfg = None
    best_overall_run = None
    best_overall_state = None

    for k, (lr, wd) in enumerate(combos):
        cfg = copy.deepcopy(base_cfg)
        cfg["train"]["optimizer"]["lr"] = float(lr)
        cfg["train"]["optimizer"]["weight_decay"] = float(wd)
        cfg.setdefault("hparams", {})
        cfg["hparams"]["epochs"] = int(refined_epochs)
        run_name = f"{tag}_k={k:03d}_lr={lr:.2e}_wd={wd:.0e}_{time.strftime('%H%M%S')}"
        run_dir = os.path.join(runs_subdir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        # snapshot
        snap_dir = os.path.join(artifacts_subdir, run_name)
        save_config_snapshot(cfg, snap_dir)
        print(f"\n[{tag}] RUN: {run_name}")
        metrics, _meta, best_state = train_one_run(cfg, run_dir, return_best_state=True)
        best_val_acc = float(metrics["best_val_acc"])
        best_val_loss = float(metrics["best_val_loss"])
        print(f"  -> best_val_acc={best_val_acc:.4f} | best_val_loss={best_val_loss:.4f}")
        summary.append({
            "run": run_name,
            "lr": float(lr),
            "wd": float(wd),
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
        })
        if best_val_acc > best_overall_acc:
            best_overall_acc = best_val_acc
            best_overall_cfg = copy.deepcopy(cfg)
            best_overall_run = run_name
            best_overall_state = best_state  # state dict du meilleur modèle sur ce run
    summary.sort(key=lambda d: d["best_val_acc"], reverse=True)
    print(f"\n========== RÉSUMÉ {tag} (trié par best_val_acc) ==========")
    for r in summary:
        print(
            f"{r['best_val_acc']:.4f} | "
            f"lr={r['lr']:.2e} wd={r['wd']:.0e} | {r['run']}"
        )
    print("====================================================")
    if best_overall_cfg is not None and best_overall_state is not None:
        best_dir = os.path.join(artifacts_subdir, "best")
        os.makedirs(best_dir, exist_ok=True)
        save_config_snapshot(best_overall_cfg, best_dir)
        # best.ckpt
        torch.save(
            {
                "model_state_dict": best_overall_state,
                "config": best_overall_cfg,
                "best_val_acc": float(best_overall_acc),
                "best_run": best_overall_run,
            },
            os.path.join(best_dir, "best.ckpt"),
        )
        print(f"\n[{tag}] MEILLEUR CHOIX: {best_overall_run}")
        print(f"[{tag}] best_val_acc={best_overall_acc:.4f}")
        print(f"[{tag}] best snapshot: {best_dir}/config_snapshot.yaml")
        print(f"[{tag}] best model:   {best_dir}/best.ckpt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--refined", action="store_true", help="Lance la mini grid search M8 (LR/WD seulement).")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        base_cfg = yaml.safe_load(f)

    runs_dir = base_cfg.get("paths", {}).get("runs_dir", "./runs")
    artifacts_dir = base_cfg.get("paths", {}).get("artifacts_dir", "./artifacts")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    if args.refined:
        grid_search_refined(base_cfg, runs_dir, artifacts_dir)
    else:
        grid_search_base(base_cfg, runs_dir, artifacts_dir)


if __name__ == "__main__":
    main()
