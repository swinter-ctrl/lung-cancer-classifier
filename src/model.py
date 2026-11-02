"""Model definition and image transforms shared by training and the app."""
import torch.nn as nn
from torchvision import models, transforms

from config import IMG_SIZE

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0 pretrained on ImageNet, with a new classification head."""
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.efficientnet_b0(weights=weights)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m


def eval_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),   # CT slices are grayscale
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def train_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
