import os
import yaml
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from data_loading import get_dataloaders

def denorm(x, mean, std):
    
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return x * std + mean

cfg = yaml.safe_load(open("configs/config.yaml"))
train_loader, val_loader, test_loader, meta = get_dataloaders(cfg)

images, labels = next(iter(train_loader))  # train => aug + preprocess
images = images.cpu()
labels = labels.cpu()

mean = cfg["preprocess"]["normalize"]["mean"]
std  = cfg["preprocess"]["normalize"]["std"]

plt.figure(figsize=(12, 4))

for i in range(3):
    img = denorm(images[i], mean, std).clamp(0, 1)
    plt.subplot(1, 3, i + 1)
    plt.imshow(F.to_pil_image(img))
    plt.title(f"Label: {labels[i].item()}")
    plt.axis("off")

plt.tight_layout()

os.makedirs("artifacts", exist_ok=True)
out_path = "examples.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Image sauvegardée dans {out_path}")
