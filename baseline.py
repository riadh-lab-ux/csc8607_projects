import yaml
import torch
from src.data_loading import get_dataloaders

def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))
    _, val_loader, _, meta = get_dataloaders(cfg)

    num_classes = int(meta["num_classes"])

    # -------- Majority baseline (sur VAL) --------
    total = 0
    counts = torch.zeros(num_classes, dtype=torch.long)

    for _, labels in val_loader:
        labels = labels.cpu()
        counts += torch.bincount(labels, minlength=num_classes)
        total += labels.numel()

    maj_class = int(torch.argmax(counts))
    maj_count = int(counts[maj_class])
    acc_majority = maj_count / total

    # -------- Random uniform baseline (sur VAL, seed=42) --------
    torch.manual_seed(42)
    correct = 0
    total2 = 0

    for _, labels in val_loader:
        labels = labels.cpu()
        preds = torch.randint(low=0, high=num_classes, size=labels.shape)
        correct += int((preds == labels).sum())
        total2 += labels.numel()
    acc_random = correct / total2

 
    print(f"accuracy Classe majoritaire ={acc_majority:.4f}")
    print(f"accuracy Prédiction aléatoire uniforme (seed=42) = {acc_random:.4f}")

if __name__ == "__main__":
    main()
