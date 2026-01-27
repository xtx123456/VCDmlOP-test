# data.py — unified CIFAR-10/100 loaders + deterministic AUX split
from typing import Tuple
import numpy as np
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

# ---------- norms ----------
_C10_MEAN, _C10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
_C100_MEAN, _C100_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

def _norm(dataset: str):
    ds = dataset.lower()
    if ds == "cifar10":  return T.Normalize(_C10_MEAN,  _C10_STD)
    if ds == "cifar100": return T.Normalize(_C100_MEAN, _C100_STD)
    raise ValueError(f"unknown dataset: {dataset!r} (expected 'cifar10' or 'cifar100')")

def _transforms(dataset: str):
    nrm = _norm(dataset)
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), nrm])
    test_tf  = T.Compose([T.ToTensor(), nrm])
    return train_tf, test_tf

def num_classes_of(dataset: str) -> int:
    ds = dataset.lower()
    if ds == "cifar10":  return 10
    if ds == "cifar100": return 100
    raise ValueError(f"unknown dataset: {dataset!r}")

# ---------- loaders ----------
def get_dataloaders(dataset: str, root: str, batch_size: int, workers: int = 4,
                    download: bool = True, pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for CIFAR-10/100."""
    ds = dataset.lower()
    train_tf, test_tf = _transforms(ds)
    if ds == "cifar10":
        train_set = torchvision.datasets.CIFAR10(root=root, train=True,  download=download, transform=train_tf)
        test_set  = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=test_tf)
    elif ds == "cifar100":
        train_set = torchvision.datasets.CIFAR100(root=root, train=True,  download=download, transform=train_tf)
        test_set  = torchvision.datasets.CIFAR100(root=root, train=False, download=download, transform=test_tf)
    else:
        raise ValueError(f"unknown dataset: {dataset!r}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_set,  batch_size=256,      shuffle=False,
                              num_workers=workers, pin_memory=pin_memory)
    return train_loader, test_loader

# ---------- deterministic split: owner / aux ----------
def _split_indices(n: int, aux_frac: float, seed: int):
    assert 0.0 < aux_frac < 1.0, "aux_frac must be in (0,1)"
    rng = np.random.RandomState(seed)
    idx = np.arange(n); rng.shuffle(idx)
    m_aux = int(round(n * aux_frac))
    aux_idx = idx[:m_aux].tolist()
    owner_idx = idx[m_aux:].tolist()
    return owner_idx, aux_idx

def make_owner_and_aux_loaders(dataset: str, root: str, owner_frac: float,
                               batch_size: int, workers: int = 4, seed: int = 0,
                               download: bool = True, pin_memory: bool = True):
    """Create owner_loader / owner_val_loader / aux_loader with same-distribution split."""
    owner_ds, val_set, aux_ds = make_owner_and_aux_sets(
        dataset,
        root=root,
        owner_frac=owner_frac,
        seed=seed,
        download=download,
    )

    owner_loader = DataLoader(owner_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=pin_memory)
    aux_loader   = DataLoader(aux_ds,   batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=pin_memory)
    owner_val_loader = DataLoader(val_set, batch_size=256, shuffle=False,
                                  num_workers=workers, pin_memory=pin_memory)
    return owner_loader, owner_val_loader, aux_loader


def make_owner_and_aux_sets(dataset: str, root: str, owner_frac: float, *, seed: int = 0,
                            download: bool = True):
    """Return (owner_ds, val_set, aux_ds) for CIFAR-10/100 with a deterministic split.

    This is the dataset-level counterpart of :func:`make_owner_and_aux_loaders`.
    It is useful when downstream code needs to further subdivide the AUX split
    (e.g., to model partially-labeled adversary data).
    """
    ds = dataset.lower()
    train_tf, test_tf = _transforms(ds)

    if ds == "cifar10":
        full_train = torchvision.datasets.CIFAR10(root=root, train=True,  download=download, transform=train_tf)
        val_set    = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=test_tf)
    elif ds == "cifar100":
        full_train = torchvision.datasets.CIFAR100(root=root, train=True,  download=download, transform=train_tf)
        val_set    = torchvision.datasets.CIFAR100(root=root, train=False, download=download, transform=test_tf)
    else:
        raise ValueError(f"unknown dataset: {dataset!r}")

    n = len(full_train)
    aux_frac = 1.0 - float(owner_frac)
    owner_idx, aux_idx = _split_indices(n, aux_frac=aux_frac, seed=seed)
    owner_ds = Subset(full_train, owner_idx)
    aux_ds   = Subset(full_train, aux_idx)
    return owner_ds, val_set, aux_ds
