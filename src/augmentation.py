"""
Data augmentation

Signature imposée :
get_augmentation_transforms(config: dict) -> objet/transform callable (ou None)
"""
from torchvision import transforms


def get_augmentation_transforms(config: dict):
    """Retourne les transformations d'augmentation. À implémenter."""
    aug_cfg = config["augment"]
    tfms = []

    # random_flip
    if bool(aug_cfg.get("random_flip", False)):
        tfms.append(transforms.RandomHorizontalFlip(p=0.5))

    # random_crop: dict ou null
    crop_cfg = aug_cfg.get("random_crop")
    if crop_cfg is not None:
        tfms.append(
            transforms.RandomCrop(
                size=int(crop_cfg["size"]),
                padding=int(crop_cfg["padding"])
            )
        )

    # color_jitter
    cj_cfg = aug_cfg.get("color_jitter")
    if cj_cfg is not None:
        tfms.append(
            transforms.ColorJitter(
                brightness=float(cj_cfg["brightness"]),
                contrast=float(cj_cfg["contrast"]),
                saturation=float(cj_cfg["saturation"]),
                hue=float(cj_cfg["hue"]),
            )
        )

    return transforms.Compose(tfms)

