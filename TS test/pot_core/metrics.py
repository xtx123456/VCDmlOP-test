from typing import Dict, List
import math
import numpy as np
import torch

def _is_weight_or_bias(key: str) -> bool:
    return key.endswith('.weight') or key.endswith('.bias')

def spearman_rank_correlation(values: List[float]) -> float:
    n = len(values)
    arr = np.array(values)
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(n)
    _, inv, counts = np.unique(arr, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    avg = sums / counts
    ranks = avg[inv]
    diffs = ranks - np.arange(n, dtype=float)
    den = n * (n**2 - 1)
    return 1.0 - 6.0 * np.sum(diffs**2) / den if den != 0 else 0.0

# def l2_weight_distance(sd_a: Dict[str, torch.Tensor], sd_b: Dict[str, torch.Tensor]) -> float:
#     s = 0.0; count = 0
#     for k, va in sd_a.items():
#         if k in sd_b and _is_weight_or_bias(k) and getattr(va, 'dtype', None) and va.dtype.is_floating_point:
#             vb = sd_b[k]
#             diff = (va - vb).float().view(-1)
#             s += torch.dot(diff, diff).item(); count += diff.numel()
#     return math.sqrt(s / max(1, count))

def l2_weight_distance(sd_a: Dict[str, torch.Tensor], sd_b: Dict[str, torch.Tensor]) -> float:
    # """RMS L2 distance over learnable params (.weight/.bias).
    # 形状不一致（如 10×512 vs 100×512）直接跳过，避免 RuntimeError。
    # """
    s = 0.0
    count = 0
    for k, va in sd_a.items():
        if k not in sd_b:
            continue
        if not _is_weight_or_bias(k):
            continue
        if getattr(va, 'dtype', None) is None or not va.dtype.is_floating_point:
            continue
        vb = sd_b[k]
        # 关键：保护不同形状（例如 CIFAR10 的 fc.weight 与 CIFAR100 不同）
        if va.shape != vb.shape:
            continue
        diff = (va - vb).float().view(-1)
        s += torch.dot(diff, diff).item()
        count += diff.numel()
    return math.sqrt(s / max(1, count))

def wasserstein_1d_exact(x: np.ndarray, y: np.ndarray) -> float:
    xs = np.sort(x.reshape(-1)); ys = np.sort(y.reshape(-1))
    n = min(xs.size, ys.size)
    if n == 0: return 0.0
    return float(np.mean(np.abs(xs[:n] - ys[:n])))

def param_distribution_distance(sd_a: Dict[str, torch.Tensor], sd_b: Dict[str, torch.Tensor]) -> float:
    maxdist = 0.0
    for k, va in sd_a.items():
        if k not in sd_b or not _is_weight_or_bias(k) or not va.dtype.is_floating_point:
            continue
        vb = sd_b[k]
        xa = va.detach().cpu().float().view(-1).numpy()
        xb = vb.detach().cpu().float().view(-1).numpy()
        d = wasserstein_1d_exact(xa, xb)
        maxdist = max(maxdist, d)
    return maxdist

def sample_from_required_gmm(numel: int, fan_in: int, device: str = 'cpu') -> np.ndarray:
    sigma = math.sqrt(2.0 / (5.0 * fan_in)); mu = 2.0 * sigma
    comp = torch.randint(0, 2, (numel,), device=device)
    means = torch.where(comp == 1, torch.full((numel,), mu, device=device), torch.full((numel,), -mu, device=device))
    samp = torch.randn(numel, device=device) * sigma + means
    return samp.cpu().numpy()

def property_p4_pca_ratio_on_init(sd0: Dict[str, torch.Tensor], max_groups: int = 2048) -> float:
    def pca_first_ratio(X: np.ndarray) -> float:
        if X.shape[0] < 2: return 1.0
        Xc = X - X.mean(axis=0, keepdims=True)
        _, S, _ = np.linalg.svd(Xc, full_matrices=False)
        var = (S ** 2); return float(var[0] / max(1e-12, var.sum()))
    max_ratio = 0.0
    for k, w in sd0.items():
        if not w.dtype.is_floating_point or (not k.endswith('.weight')):
            continue
        W = w.detach().cpu().float().numpy()
        if W.ndim == 4:
            out_c = W.shape[0]
            sel = np.arange(out_c)
            if out_c > max_groups:
                sel = np.random.choice(out_c, size=max_groups, replace=False)
            X = W[sel].reshape(len(sel), -1)
            max_ratio = max(max_ratio, pca_first_ratio(X))
        elif W.ndim == 2:
            out_f = W.shape[0]
            sel = np.arange(out_f)
            if out_f > max_groups:
                sel = np.random.choice(out_f, size=max_groups, replace=False)
            X = W[sel]
            max_ratio = max(max_ratio, pca_first_ratio(X))
    return max_ratio