# attacks/interp.py  (CPU-only, safe against device mismatch)
import os
import json
import time
import argparse
from typing import Dict, List

import numpy as np
import torch

from pot_core.init import apply_pot_init
from pot_core.checkpoints import load_chain
from pot_core.arch_utils import (
    get_model_cls_from_meta_or_arg,
    infer_num_classes_from_meta_or_sd,
)

def _to_cpu_sd(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Ensure all tensors in state_dict are on CPU (detach to avoid gradients)."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        out[k] = v.detach().cpu() if torch.is_tensor(v) else v
    return out

def _interpolate_sd(
    sd_a: Dict[str, torch.Tensor],
    sd_b: Dict[str, torch.Tensor],
    alpha: float
) -> Dict[str, torch.Tensor]:
    """
    对同形状的浮点参数执行线性插值：(1-a)*A + a*B
    非浮点（如 BN 运行统计）或形状不一致 → 直接用 sd_b（final）。
    遍历以 sd_b 的键为准，保证输出 state_dict 完整。
    这里默认 sd_a / sd_b 的张量均已在 CPU。
    """
    out: Dict[str, torch.Tensor] = {}
    for k, vb in sd_b.items():
        va = sd_a.get(k, None)
        if va is not None and torch.is_tensor(va) and torch.is_tensor(vb):
            if va.dtype.is_floating_point and vb.dtype.is_floating_point and va.shape == vb.shape:
                out[k] = (1.0 - alpha) * va + alpha * vb
            else:
                out[k] = vb.clone()
        else:
            out[k] = vb.clone() if torch.is_tensor(vb) else vb
    return out

def _alphas(alpha_start: float, alpha_end: float, alpha_step: float) -> List[float]:
    if alpha_step <= 0:
        raise ValueError("alpha_step must be positive.")
    if alpha_end < alpha_start:
        raise ValueError("alpha_end must be >= alpha_start.")
    return list(np.arange(alpha_start, alpha_end + 1e-9, alpha_step, dtype=float))

def main():
    ap = argparse.ArgumentParser(description="Interpolation attack: w(a)=(1-a)*w_rand + a*w_final")
    ap.add_argument('--victim', type=str, required=True, help='Victim chain directory (contains epoch_*.pt & metadata.json)')
    ap.add_argument('--out',    type=str, required=True, help='Output directory for forged chain')
    ap.add_argument('--arch',   type=str, default='auto', choices=['vgg16','resnet18','alexnet', 'lenet'],
                    help="Model arch to use for random init; 'auto' reads victim metadata")
    ap.add_argument('--alpha-start', type=float, default=0.0)
    ap.add_argument('--alpha-end',   type=float, default=1.0)
    ap.add_argument('--alpha-step',  type=float, default=0.01)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    # 1) 载入受害者链（最终权重 + 元信息），并统一到 CPU
    ckpts, v_meta = load_chain(args.victim)
    assert len(ckpts) >= 1, "victim chain is empty."
    sd_final_cpu = _to_cpu_sd(ckpts[-1]['model'])

    # 2) 选择模型类 & 推断类别数（兼容 AlexNet/ResNet）
    ModelCls  = get_model_cls_from_meta_or_arg(args.victim, args.arch)
    num_class = infer_num_classes_from_meta_or_sd(v_meta, sd_final_cpu)

    # 3) 构造同结构 PoT 随机初始化模型（随机起点）——保持在 CPU
    rand_model = ModelCls(num_classes=num_class)  # 不要 .to('cuda')
    rand_model.eval()
    with torch.no_grad():
        rand_model.apply(apply_pot_init)
    sd_rand_cpu = _to_cpu_sd(rand_model.state_dict())

    # 4) α 序列 & 输出目录
    alphas = _alphas(args.alpha_start, args.alpha_end, args.alpha_step)
    os.makedirs(args.out, exist_ok=True)

    # 5) 合成 meta：继承 victim 的 dataset/arch，注明攻击
    out_meta = {
        'epochs': int(len(alphas)),
        'batch_size': 0,
        'optimizer': 'N/A',
        'lr': 0.0,
        'weight_decay': 0.0,
        'train_acc': [],
        'val_acc': [],
        'val_loss': [],
        'timestamps': [],
        'arch': ModelCls.__name__,
        'dataset': str(v_meta.get('dataset', '') or ('CIFAR10' if num_class == 10 else 'CIFAR100')),
        'notes': 'interpolation attack: w = (1-a)*rand + a*final; val_acc is synthetic for P1',
        'attack': 'interp',
        'victim_from': os.path.abspath(args.victim),
    }

    # 6) 设定合成 val_acc 的上界（只写 meta，不参与任何度量）
    target_va = None
    try:
        if v_meta.get('val_acc'):
            target_va = float(v_meta['val_acc'][-1])
    except Exception:
        target_va = None
    if target_va is None:
        target_va = 0.85 if num_class == 10 else 0.65

    # 7) 生成 forged 链（全程 CPU）
    torch.set_grad_enabled(False)
    for i, a in enumerate(alphas):
        sd_i = _interpolate_sd(sd_rand_cpu, sd_final_cpu, float(a))
        torch.save({'epoch': i, 'model': sd_i}, os.path.join(args.out, f'epoch_{i:04d}.pt'))

        # 合成“看起来单调合理”的 val_acc（仅占位）
        va = (0.05 + 0.95 * (a ** 0.7)) * target_va
        out_meta['train_acc'].append(None)
        out_meta['val_acc'].append(float(va))
        out_meta['val_loss'].append(float(max(0.0, 1.5 - va)))
        out_meta['timestamps'].append(time.time())

    # 8) 写 meta
    with open(os.path.join(args.out, 'metadata.json'), 'w') as f:
        json.dump(out_meta, f, indent=2, allow_nan=False)

    if args.verbose:
        print(f"[attack/interp] forged {len(alphas)} checkpoints "
              f"(arch={ModelCls.__name__}, num_classes={num_class}, device=cpu) -> {args.out}")

if __name__ == '__main__':
    main()
