# pot_core/arch_utils.py
import json, os, torch
from .models import ResNet18CIFAR, AlexNetCIFAR, LeNetCIFAR, VGG16CIFAR
from .models_unet import UNetSeg

__all__ = ["get_model_cls_from_meta_or_arg", "infer_num_classes_from_meta_or_sd", "find_last_linear_name"]

def _norm(s: str) -> str:
    return ''.join(ch for ch in str(s).lower() if ch.isalnum())

_REGISTRY = {
    'resnet18cifar': ResNet18CIFAR,
    'alexnetcifar':  AlexNetCIFAR,
    'vgg16cifar':    VGG16CIFAR,
    'lenetcifar':    LeNetCIFAR,
    'unetseg': UNetSeg,
}
_ALIASES = {
    'resnet18': 'resnet18cifar',
    'alexnet':  'alexnetcifar',
    'vgg16':    'vgg16cifar',
    'lenet':    'lenetcifar',
    'unet': 'unetseg',
}

def _lookup_model_cls(name: str):
    key = _norm(name)
    if key in _REGISTRY: return _REGISTRY[key]
    if key in _ALIASES:  return _REGISTRY[_ALIASES[key]]
    return ResNet18CIFAR

def get_model_cls_from_meta_or_arg(chain_dir_or_meta: str|dict, arch_arg: str|None):
    """优先使用 --arch；否则从 chain_dir 的 metadata.json 读取 arch；均失败时回退 ResNet18CIFAR."""
    if arch_arg and arch_arg != 'auto':
        return _lookup_model_cls(arch_arg)
    meta = {}
    if isinstance(chain_dir_or_meta, dict):
        meta = chain_dir_or_meta
    else:
        meta_p = os.path.join(chain_dir_or_meta, 'metadata.json')
        try:
            with open(meta_p, 'r') as f: meta = json.load(f)
        except Exception:
            meta = {}
    return _lookup_model_cls(meta.get('arch', 'ResNet18CIFAR'))

def infer_num_classes_from_meta_or_sd(meta: dict, sd_final: dict, default=100) -> int:
    ds = str(meta.get('dataset', '')).strip().upper()
    if 'CIFAR10' in ds or ds in ('CIFAR-10','10'): return 10
    if 'CIFAR100' in ds or ds in ('CIFAR-100','100'): return 100
    # 扫描任意 Linear 的输出维度
    try:
        candidates = []
        for k, v in sd_final.items():
            if k.endswith('.weight') and v.ndim == 2:
                candidates.append(int(v.shape[0]))
        for target in (10, 100):
            if target in candidates: return target
    except Exception:
        pass
    return default

def find_last_linear_name(sd: dict) -> str|None:
    """找到 state_dict 中最后一个 Linear 的 weight 名（不假设叫 fc），兼容 AlexNet 的 classifier.*.weight。"""
    last_name = None
    for k, v in sd.items():
        if k.endswith('.weight') and v.ndim == 2:
            last_name = k
    return last_name
