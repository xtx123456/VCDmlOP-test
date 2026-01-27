# pot_core/verify_v2.py
import os, json, random, time
from typing import Dict, List, Optional
import numpy as np
import torch

from .checkpoints import load_chain
from .init import apply_pot_init
from .metrics import (
    spearman_rank_correlation,
    l2_weight_distance,
    param_distribution_distance,
    property_p4_pca_ratio_on_init,
    sample_from_required_gmm,
    wasserstein_1d_exact,
)
from .arch_utils import (
    get_model_cls_from_meta_or_arg,
    infer_num_classes_from_meta_or_sd,
)

# 可选导入：支持 UNetSeg（分割）
try:
    from .models_unet import UNetSeg
    _HAS_UNET = True
except Exception:
    UNetSeg = None  # type: ignore
    _HAS_UNET = False


def _safe_list_like(x, n):
    """若 meta['val_acc'] 长度与 epochs 不一致：补齐/截断；并做前向填充以稳定 Spearman。"""
    if not isinstance(x, list):
        return [float("nan")] * n
    x = list(x)
    if len(x) == n:
        return x
    if len(x) < n:
        pad = [float("nan")] * (n - len(x))
        return x + pad
    return x[:n]


def _p1_from_meta_val_acc(meta: dict, epochs: int) -> float:
    """P1：基于 metadata['val_acc'] 的单调性（Spearman）。缺失/NaN 用最后有效值前向填充。"""
    vals_in = _safe_list_like(meta.get("val_acc", []), epochs)
    vals = []
    last = 0.0
    for v in vals_in:
        if isinstance(v, (int, float)) and np.isfinite(v):
            last = float(v)
        vals.append(last)
    return float(spearman_rank_correlation(vals))


def _infer_num_classes_seg_aware(meta: dict, final_sd: dict, default: int = 100) -> int:
    """
    分割友好的类数推断：
      1) 优先 meta['num_classes']（训练 seg 时你已在 metadata 填过 21）；
      2) 若存在 4D 权重（Conv2d）则取“最后一个卷积”的 out_channels 作为候选；
      3) 回退到 arch_utils.infer_num_classes_from_meta_or_sd（兼容分类链）。
    """
    # 1) 显式 meta
    try:
        if "num_classes" in meta:
            nc = int(meta["num_classes"])
            if 1 <= nc <= 1000:
                return nc
    except Exception:
        pass

    # 2) 扫描 Conv 层：最后出现的 4D 权重的 out_channels
    try:
        last_conv_out = None
        for k, v in final_sd.items():
            if isinstance(v, torch.Tensor) and v.ndim == 4:
                # weight shape: [out_channels, in_channels, kH, kW]
                last_conv_out = int(v.shape[0])
        if last_conv_out is not None and 1 <= last_conv_out <= 1000:
            return last_conv_out
    except Exception:
        pass

    # 3) 回退到原有推断（分类）
    return infer_num_classes_from_meta_or_sd(meta, final_sd, default=default)


def _pick_model_cls_for_rand(meta: dict, chain_dir: str):
    """
    用于 P6 的“随机对照模型类”选择：
      - 若 meta['arch'] 显示声称 UNet 且本地可用 → UNetSeg
      - 否则 → 走 arch_utils.get_model_cls_from_meta_or_arg（支持 resnet18/alexnet/vgg/lenet 等）
    """
    name = str(meta.get("arch", "")).strip().lower()
    if _HAS_UNET and ("unet" in name or "unetseg" in name):
        return UNetSeg
    # 默认：分类模型等
    return get_model_cls_from_meta_or_arg(chain_dir, arch_arg=None)


def verify_chain(
    chain_dir: str,
    emd_bins: int = 200,
    num_random: int = 20,
    seed: int = 0
) -> Dict[str, float]:

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    checkpoints, meta = load_chain(chain_dir)
    assert len(checkpoints) >= 2, "Need at least 2 checkpoints (initial + final)."
    sds = [c['model'] for c in checkpoints]
    epochs = len(sds)

    # 全部放 CPU，避免设备不一致
    sds = [{k: v.detach().cpu() for k, v in sd.items()} for sd in sds]
    final_sd = sds[-1]

    # ---- P1 ----
    p1 = _p1_from_meta_val_acc(meta, epochs)

    # ---- P2: 相邻权重分布 EMD 的最大值 ----
    p2 = 0.0
    for i in range(epochs - 1):
        try:
            d = param_distribution_distance(sds[i], sds[i + 1], bins=emd_bins)
        except TypeError:
            # 向后兼容：老实现无 bins 参数
            d = param_distribution_distance(sds[i], sds[i + 1])
        if d > p2:
            p2 = float(d)

    # ---- P3: init vs required GMM（权重分布的 Wasserstein-1D）----
    sd0 = sds[0]
    p3 = 0.0
    for k, w in sd0.items():
        if (not isinstance(w, torch.Tensor)) or (not w.dtype.is_floating_point):
            continue
        if not k.endswith(".weight"):
            continue
        W = w.float().view(-1).cpu().numpy()
        # fan-in 估计
        orig = w
        if orig.ndim == 4:
            fan_in = int(orig.shape[1] * orig.shape[2] * orig.shape[3])
        elif orig.ndim == 2:
            fan_in = int(orig.shape[1])
        else:
            continue
        ref = sample_from_required_gmm(W.size, fan_in)
        d = float(wasserstein_1d_exact(W, ref))
        if d > p3:
            p3 = d

    # ---- P4: 初始 PCA ratio ----
    p4 = float(property_p4_pca_ratio_on_init(sd0))

    # ---- P5: -||epoch_i - final|| 的 Spearman（越大越好）----
    dists = [l2_weight_distance(sd, final_sd) for sd in sds]
    p5 = float(spearman_rank_correlation([-d for d in dists]))

    # ---- P6: init-final vs rand-final ----
    init_final = float(dists[0])

    # 对类数与模型类的“分割友好”推断
    num_classes = _infer_num_classes_seg_aware(meta, final_sd, default=100)
    ModelCls = _pick_model_cls_for_rand(meta, chain_dir)

    # 随机对照：每次独立 PoT 初始化，并扰动随机种子
    rand_d = []
    base_seed = int(seed) if isinstance(seed, (int, float)) else 0
    for i in range(num_random):
        torch.manual_seed(base_seed + 12345 + i)
        np.random.seed(base_seed + 23456 + i)
        _py_random = __import__("random")
        _py_random.seed(base_seed + 34567 + i)

        m = ModelCls(num_classes=num_classes)
        with torch.no_grad():
            m.apply(apply_pot_init)
        rand_sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
        rand_d.append(l2_weight_distance(rand_sd, final_sd))

    rand_d = np.array(rand_d, dtype=float)
    mean_rand = float(rand_d.mean())
    std_rand = float(rand_d.std(ddof=1) if len(rand_d) > 1 else 1e-6)
    p6_small = bool(init_final < (mean_rand - 3.0 * std_rand))

    return {
        "P1_rho_val_acc": float(p1),
        "P2_max_EMD_consecutive": float(p2),
        "P3_max_EMD_init_vs_GMM": float(p3),
        "P4_max_PCA_ratio_init": float(p4),
        "P5_rho_neg_weight_distance": float(p5),
        "P6_init_final_distance": float(init_final),
        "P6_random_mean": float(mean_rand),
        "P6_random_std": float(std_rand),
        "P6_is_small_vs_random": bool(p6_small),
        "EPOCHS": float(epochs),
    }


def apply_strict(res: Dict[str, float], eps2=0.02, eps3=0.03, eps4=0.25, rho5=0.6, k6=3.0):
    """阈值判定（分类默认；分割可适当放宽 eps2/eps4）"""
    return {
        "PASS_P1": res["P1_rho_val_acc"] >= 0.6,
        "PASS_P2": res["P2_max_EMD_consecutive"] <= eps2,
        "PASS_P3": res["P3_max_EMD_init_vs_GMM"] <= eps3,
        "PASS_P4": res["P4_max_PCA_ratio_init"] <= eps4,
        "PASS_P5": res["P5_rho_neg_weight_distance"] >= rho5,
        "PASS_P6": res["P6_init_final_distance"] < (res["P6_random_mean"] - k6 * res["P6_random_std"]),
    }
