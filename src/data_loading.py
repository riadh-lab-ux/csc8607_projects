"""
Chargement des données.

Signature imposée :
get_dataloaders(config: dict) -> (train_loader, val_loader, test_loader, meta: dict)

Le dictionnaire meta doit contenir au minimum :
- "num_classes": int
- "input_shape": tuple (ex: (3, 32, 32) pour des images)
"""
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from src.preporcessing import get_preprocess_transforms
from src.augmentation import get_augmentation_transforms


class TinyDataset(Dataset):
    def __init__(self, hf_ds, transform):
        self.hf_ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        item = self.hf_ds[int(idx)]
        img = item["image"]
        label = int(item["label"])
        if self.transform is not None:
            img = self.transform(img)
        return img, label




def get_dataloaders(config: dict):
    """
    Crée et retourne les DataLoaders d'entraînement/validation/test et des métadonnées.
    """

    dataset_config = config["dataset"]

    hf = load_dataset(
        dataset_config["name"],
        cache_dir=dataset_config.get("root", "./data")
    )

    split_cfg = dataset_config["split"]
    hf_train = hf[split_cfg["train"]]
    hf_valid = hf["valid"]

    val_size = int(split_cfg["val"])
    test_size = int(split_cfg["test"])

    num_classes = 200
    assert val_size % num_classes == 0, "val_size doit être multiple de 200"
    assert test_size % num_classes == 0, "test_size doit être multiple de 200"

    val_per_class = val_size // num_classes      # 40
    test_per_class = test_size // num_classes    # 10

    # --- split stratifié valid -> val/test
    labels = hf_valid["label"]  

    by_class = [[] for _ in range(num_classes)]
    for i, lab in enumerate(labels):
        by_class[int(lab)].append(i)

    g = torch.Generator().manual_seed(42)

    val_idx = []
    test_idx = []
    for c in range(num_classes):
        idx_c = torch.tensor(by_class[c], dtype=torch.long)
        perm_c = idx_c[torch.randperm(len(idx_c), generator=g)].tolist()

        test_idx.extend(perm_c[:test_per_class])
        val_idx.extend(perm_c[test_per_class:test_per_class + val_per_class])

    hf_val = hf_valid.select(val_idx)
    hf_test = hf_valid.select(test_idx)

    # preprocess 
    preprocess_tfm = get_preprocess_transforms(config)

    # augmentation pour train uniquement
    aug_tfm = get_augmentation_transforms(config)

    # train = augmentation + preprocess ; val/test = preprocess
    train_tfm = transforms.Compose([aug_tfm, preprocess_tfm])
    eval_tfm = preprocess_tfm

    train_dataset = TinyDataset(hf_train, train_tfm)
    val_dataset = TinyDataset(hf_val, eval_tfm)
    test_dataset = TinyDataset(hf_test, eval_tfm)

    batch_size = int(config["train"]["batch_size"])
    num_workers = int(dataset_config["num_workers"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    meta = {"num_classes": 200, "input_shape": (3, 64, 64)}
    return train_loader, val_loader, test_loader, meta
