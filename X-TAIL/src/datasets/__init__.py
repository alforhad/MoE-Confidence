"""
Portable bundle: only datasets needed for the MoE adapter train / eval scripts.

Avoids importing optional stacks (``wilds``, extra ImageNet variants, etc.) so
``pip install`` on a fresh machine matches ``requirements.txt`` more closely.
"""
from .collections import (
    Aircraft,
    Caltech101,
    CIFAR10,
    CIFAR100,
    DTD,
    EuroSAT,
    Flowers,
    Food,
    MNIST,
    OxfordPet,
    StanfordCars,
    SUN397,
    TinyImagenet,
)

__all__ = [
    "Aircraft",
    "Caltech101",
    "CIFAR10",
    "CIFAR100",
    "DTD",
    "EuroSAT",
    "Flowers",
    "Food",
    "MNIST",
    "OxfordPet",
    "StanfordCars",
    "SUN397",
    "TinyImagenet",
]
