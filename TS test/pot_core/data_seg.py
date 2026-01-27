# pot_core/data_seg.py
import os, math, random
from typing import Tuple
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import VOCSegmentation
from torchvision import transforms as T

def _voc_tfms(image_size: int = 256):
    train_img = T.Compose([
        T.Resize((image_size, image_size)),                 # ← 强制 H×W 一致
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    train_mask = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),  # ← 同步到同尺寸
        T.PILToTensor(),
    ])
    val_img = T.Compose([
        T.Resize((image_size, image_size)),                 # ← 同上
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_mask = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),  # ← 同上
        T.PILToTensor(),
    ])
    return train_img, train_mask, val_img, val_mask


class _VOCWrap(torch.utils.data.Dataset):
    def __init__(self, ds, img_tf, mask_tf):
        self.ds, self.img_tf, self.mask_tf = ds, img_tf, mask_tf
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        img, mask = self.ds[idx]
        img  = self.img_tf(img)
        mask = self.mask_tf(mask).long().squeeze(0)  # [H,W], int64
        return img, mask

def get_seg_loaders(root: str, batch_size: int = 8, workers: int = 4, image_size: int = 256, download: bool = False):
    """
    Pascal VOC 2012 语义分割：返回 train_loader, val_loader
    """
    ti, tm, vi, vm = _voc_tfms(image_size)
    ds_train = VOCSegmentation(root=root, year="2012", image_set="train", download=download)
    ds_val   = VOCSegmentation(root=root, year="2012", image_set="val",   download=download)
    train = _VOCWrap(ds_train, ti, tm)
    val   = _VOCWrap(ds_val,   vi, vm)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val,   batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader

def make_owner_and_aux_loaders_seg(root: str, *, owner_frac: float = 0.7, batch_size: int = 8, workers: int = 4, image_size: int = 256, seed: int = 0, download: bool = False):
    """
    同分布切分：train → owner / aux
    返回：owner_train_loader, owner_val_loader(=val), aux_loader
    """
    ti, tm, vi, vm = _voc_tfms(image_size)
    base = VOCSegmentation(root=root, year="2012", image_set="train", download=download)
    val  = VOCSegmentation(root=root, year="2012", image_set="val",   download=download)

    n = len(base)
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_owner = int(round(owner_frac * n))
    owner_idx = idx[:n_owner]
    aux_idx   = idx[n_owner:]

    owner_ds = _VOCWrap(Subset(base, owner_idx), ti, tm)
    aux_ds   = _VOCWrap(Subset(base, aux_idx),   ti, tm)
    val_ds   = _VOCWrap(val, vi, vm)

    owner_loader = DataLoader(owner_ds, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True, drop_last=True)
    aux_loader   = DataLoader(aux_ds,   batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True, drop_last=True)
    owner_val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return owner_loader, owner_val_loader, aux_loader
