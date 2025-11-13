import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as dsets


def get_transforms(resolution=64):
    return T.Compose([
        T.Resize((resolution, resolution)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)  # output in [-1,1]
    ])


def load_dataset(name, batch_size, resolution=64, root="data"):
    tfm = get_transforms(resolution)

    if name.lower() == "cifar10":
        train = dsets.CIFAR10(root, train=True, download=True, transform=tfm)
        test = dsets.CIFAR10(root, train=False, download=True, transform=tfm)
    elif name.lower() == "celeba":
        train = dsets.CelebA(root, split="train", download=True, transform=tfm)
        test = dsets.CelebA(root, split="test", download=True, transform=tfm)
    elif name.lower() == "ffhq":
        # FFHQ not in torchvision, use custom or external
        # For now, raise error; in practice, implement FFHQ loader
        raise NotImplementedError("FFHQ dataset not implemented yet. Use external library like dataset-ffhq.")
    else:
        raise ValueError(f"Unknown dataset {name}")

    train_loader = DataLoader(train, batch_size, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test, batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader