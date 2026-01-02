import yaml
import torch
import torch.nn as nn

from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import set_seed, get_device

def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))

    # Seed stable (si tu as train.seed dans config)
    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed)

    device = get_device("auto")
    train_loader, _, _, meta = get_dataloaders(cfg)
    model = build_model(cfg).to(device)
    model.train()
    x, y = next(iter(train_loader))
    x = x.to(device)
    y = y.to(device)
    logits = model(x)  # (B, 200)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    # Gradient
    grad_sum = 0.0
    nb = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_sum += p.grad.data.norm(2).item()
            nb += 1
    print("initial loss:", float(loss.item()))
    print("L2 norms:", grad_sum, " (#tensors:", nb, ")")

if __name__ == "__main__":
    main()
