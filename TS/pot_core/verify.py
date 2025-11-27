import math, time, json, os, random, numpy as np, torch
from typing import Dict, List
# from .metrics import (spearman_rank_correlation, l2_weight_distance,
#                       param_distribution_distance, sample_from_required_gmm,
#                       property_p4_pca_ratio_on_init)
from .metrics import (spearman_rank_correlation, l2_weight_distance,
                      param_distribution_distance, sample_from_required_gmm,
                      property_p4_pca_ratio_on_init, wasserstein_1d_exact)
from .checkpoints import load_chain
from .models import ResNet18CIFAR, AlexNetCIFAR
from .init import apply_pot_init
from .arch_utils import (
    get_model_cls_from_meta_or_arg,
    infer_num_classes_from_meta_or_sd,
)



# def _infer_num_classes(meta: dict, sd_final: dict) -> int:
#     # 1) 优先从 final_sd 的 fc.weight 推断
#     try:
#         out_dim = int(next(v for k, v in sd_final.items() if k.endswith('fc.weight')).shape[0])
#         if out_dim in (10, 100):
#             return out_dim
#     except Exception:
#         pass
#     # 2) 再退回看 metadata['dataset']
#     ds = str(meta.get('dataset', '')).strip().upper()
#     if 'CIFAR10' in ds or ds in ('CIFAR-10', '10'):
#         return 10
#     if 'CIFAR100' in ds or ds in ('CIFAR-100', '100'):
#         return 100
#     # 3) 最后兜底
#     return 100

# def _infer_num_classes(meta: dict, sd_final: dict) -> int:
#     """
#     先信 metadata['dataset']，再从 Linear 层推断（兼容 AlexNet 没有 fc.weight 的情况），最后兜底 100。
#     """
#     # 1) metadata -> dataset
#     ds = str(meta.get('dataset', '')).strip().upper()
#     if 'CIFAR10' in ds or ds in ('CIFAR-10', '10'):
#         return 10
#     if 'CIFAR100' in ds or ds in ('CIFAR-100', '100'):
#         return 100

#     # 2) 扫描任意 Linear 的权重形状（2D），找 out_dim in {10,100}
#     try:
#         candidates = []
#         for k, v in sd_final.items():
#             if k.endswith('.weight') and v.ndim == 2:
#                 out_dim = int(v.shape[0])
#                 candidates.append(out_dim)
#         for target in (10, 100):
#             if target in candidates:
#                 return target
#     except Exception:
#         pass

#     # 3) 兜底
#     return 100


# def _get_model_cls_from_meta(meta: dict):
#     """
#     根据 metadata['arch'] 选模型类；支持常见别名；默认回退 ResNet18CIFAR。
#     """
#     raw = meta.get('arch', 'ResNet18CIFAR')
#     name = str(raw).strip()

#     registry = {
#         'ResNet18CIFAR': ResNet18CIFAR,
#         'AlexNetCIFAR':  AlexNetCIFAR,
#     }
#     aliases = {
#         'resnet18':      'ResNet18CIFAR',
#         'resnet18cifar': 'ResNet18CIFAR',
#         'alexnet':       'AlexNetCIFAR',
#         'alexnetcifar':  'AlexNetCIFAR',
#     }

#     # 1) 精确匹配
#     if name in registry:
#         return registry[name]

#     # 2) 别名（大小写不敏感）
#     lower = name.lower()
#     if lower in aliases:
#         return registry[aliases[lower]]

#     # 3) 宽松规范化（去掉非字母数字后比较）
#     def _norm(s: str) -> str:
#         return ''.join(ch for ch in s.lower() if ch.isalnum())

#     n = _norm(name)
#     for k in registry.keys():
#         if _norm(k) == n:
#             return registry[k]

#     # 4) 兜底
#     return ResNet18CIFAR



def verify_chain(chain_dir: str, emd_bins: int = 200, num_random: int = 20, seed: int = 0) -> Dict[str, float]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    checkpoints, meta = load_chain(chain_dir)
    assert len(checkpoints) >= 2, "Need at least 2 checkpoints (initial + final)."
    sds = [c['model'] for c in checkpoints]
    epochs = len(sds)
    val_acc = meta.get('val_acc', [])
    if len(val_acc) != epochs:
        val_acc = (val_acc + [float('nan')]*epochs)[:epochs]

    # accuracy monity
    vals, last = [], 0.0
    for v in val_acc:
        if not (isinstance(v, (int, float)) and np.isfinite(v)):
            vals.append(last)
        else:
            vals.append(v); last = v
    p1 = spearman_rank_correlation(vals)

    # distribution initial
    p2 = 0.0
    for i in range(epochs - 1):
        try:
            d = param_distribution_distance(sds[i], sds[i+1], bins=emd_bins)
        except TypeError:
            # 向后兼容：老版本没有 bins 参数
            d = param_distribution_distance(sds[i], sds[i+1])
        p2 = max(p2, d)

    # p2 = 0.0
    # for i in range(epochs - 1):
    #     d = param_distribution_distance(sds[i], sds[i+1])
    #     p2 = max(p2, d)

    # random inicial
    sd0 = sds[0]
    p3 = 0.0
    for k, w in sd0.items():
        if (not k.endswith('.weight')) or (not w.dtype.is_floating_point): continue
        W = w.detach().cpu().float().view(-1)
        orig = w.detach().cpu()
        if orig.ndim == 4:
            fan_in = orig.shape[1] * orig.shape[2] * orig.shape[3]
        elif orig.ndim == 2:
            fan_in = orig.shape[1]
        else:
            continue
        # ref = sample_from_required_gmm(W.numel(), int(fan_in))
        # d = np.mean(np.abs(np.sort(W.numpy()) - np.sort(ref)))
        ref = sample_from_required_gmm(W.numel(), int(fan_in))
        d = wasserstein_1d_exact(W.numpy(), ref)
        p3 = max(p3, float(d))

    # vs gmm
    p4 = property_p4_pca_ratio_on_init(sd0)

    # P5
    final_sd = sds[-1]
    dists = [l2_weight_distance(sd, final_sd) for sd in sds]
    p5 = spearman_rank_correlation([-d for d in dists])


    # # P6: init-final vs random-final
    # init_final = dists[0]

    # num_classes = infer_num_classes_from_meta_or_sd(meta, final_sd)
    # ModelCls = get_model_cls_from_meta_or_arg(chain_dir, arch_arg=None)  # 从 metadata 读取 arch
    # model = ModelCls(num_classes=num_classes)
    # model.apply(apply_pot_init)

    # rand_d = []
    # for _ in range(num_random):
    #     model.apply(apply_pot_init)
    #     rand_sd = model.state_dict()
    #     rand_d.append(l2_weight_distance(rand_sd, final_sd))
    # rand_d = np.array(rand_d)
    # mean_rand = float(rand_d.mean())
    # std_rand  = float(rand_d.std(ddof=1) if len(rand_d) > 1 else 1e-6)



    # P6: init-final vs random-final
    init_final = dists[0]

    # 统一把 final_sd 放到 CPU，避免设备不一致
    final_sd = {k: v.detach().cpu() for k, v in final_sd.items()}

    num_classes = infer_num_classes_from_meta_or_sd(meta, final_sd)
    ModelCls = get_model_cls_from_meta_or_arg(chain_dir, arch_arg=None)  # 从 metadata 读取 arch

    rand_d = []
    base_seed = int(seed) if isinstance(seed, (int, float)) else 0
    for i in range(num_random):
        # 关键：每次扰动随机种子，覆盖任何内部固定种子
        torch.manual_seed(base_seed + 12345 + i)
        np.random.seed(base_seed + 23456 + i)
        import random as _py_random
        _py_random.seed(base_seed + 34567 + i)

        # 新建同架构模型，并做 PoT 初始化
        m = ModelCls(num_classes=num_classes)
        with torch.no_grad():
            m.apply(apply_pot_init)

        # 取 state_dict（放 CPU）并计算距离
        rand_sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
        rand_d.append(l2_weight_distance(rand_sd, final_sd))

    rand_d = np.array(rand_d, dtype=float)
    mean_rand = float(rand_d.mean())
    std_rand  = float(rand_d.std(ddof=1) if len(rand_d) > 1 else 1e-6)

    # init_final = dists[0]
    
    # num_classes = _infer_num_classes(meta, final_sd)
    # ModelCls = _get_model_cls_from_meta(meta) 
    # #model = ResNet18CIFAR(num_classes=num_classes)
    # model = ModelCls(num_classes=num_classes)
    # model.apply(apply_pot_init)
    
    # rand_d = []
    # for _ in range(num_random):
    #     model.apply(apply_pot_init)
    #     rand_sd = model.state_dict()
    #     rand_d.append(l2_weight_distance(rand_sd, final_sd))   # 顶部已导入，别在循环里再 import
    # rand_d = np.array(rand_d)
    # mean_rand = float(rand_d.mean())
    # std_rand  = float(rand_d.std(ddof=1) if len(rand_d) > 1 else 1e-6)
    

    
    
    # rand_d = []
    # for _ in range(num_random):
    #     model.apply(apply_pot_init)
    #     rand_sd = model.state_dict()
    #     rand_d.append(l2_weight_distance(rand_sd, final_sd))
    # rand_d = np.array(rand_d)
    # mean_rand = float(rand_d.mean())
    # std_rand  = float(rand_d.std(ddof=1) if len(rand_d) > 1 else 1e-6)

    return {
        'rho_val_acc': float(p1),
        'max_EMD_consecutive': float(p2),
        'max_EMD_init_vs_GMM': float(p3),
        'max_PCA_ratio_init': float(p4),
        'rho_neg_weight_distance': float(p5),
        'init_final_distance': float(init_final),
        'random_mean': mean_rand,
        'random_std': std_rand,
        'init_final_distance_is_small_vs_random': bool(init_final < (mean_rand - 3.0 * std_rand)),
        'EPOCHS': float(epochs),
    }

def apply_strict(res: Dict[str, float], eps2=0.02, eps3=0.03, eps4=0.25, rho5=0.6, k6=3.0):
    return {
        'PASS_P1': res['rho_val_acc'] >= 0.6,
        'PASS_P2': res['max_EMD_consecutive'] <= eps2,
        'PASS_P3': res['max_EMD_init_vs_GMM'] <= eps3,
        'PASS_P4': res['max_PCA_ratio_init']  <= eps4,
        'PASS_P5': res['rho_neg_weight_distance'] >= rho5,
        'PASS_P6': res['init_final_distance'] < (res['random_mean'] - k6 * res['random_std']),
    }