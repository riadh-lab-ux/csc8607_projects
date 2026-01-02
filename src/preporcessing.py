"""
Pré-traitements.

Signature imposée :
get_preprocess_transforms(config: dict) -> objet/transform callable
"""
from torchvision import transforms

def get_preprocess_transforms(config: dict):
    resize = config["preprocess"]["resize"]
    mean = config["preprocess"]["normalize"]["mean"]
    std = config["preprocess"]["normalize"]["std"]
    tfm = transforms.Compose([
        transforms.Resize(resize),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return tfm
