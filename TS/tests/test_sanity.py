import numpy as np

# numpy-only sanity tests so they can pass even if torch is unavailable.
def wasserstein_1d_exact(x, y):
    xs = np.sort(np.ravel(x)); ys = np.sort(np.ravel(y))
    n = min(xs.size, ys.size)
    if n == 0: return 0.0
    return float(np.mean(np.abs(xs[:n] - ys[:n])))

def test_w1_zero_on_identical():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([0.0, 1.0, 2.0, 3.0])
    assert abs(wasserstein_1d_exact(a, b)) < 1e-12

def test_w1_positive_on_shift():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    c = np.array([1.0, 2.0, 3.0, 4.0])
    assert wasserstein_1d_exact(a, c) > 0.0

# Optional torch-based test (skip if torch not installed)
def test_l2_weight_distance_optional():
    try:
        import torch
    except Exception:
        return  # skip if torch missing
    from pot_core.metrics import l2_weight_distance
    t = torch
    sd1 = {'layer.weight': t.tensor([1.0, 2.0]), 'layer.bias': t.tensor([0.5, -0.5]), 'bn.running_mean': t.tensor([0.0, 0.0])}
    sd2 = {'layer.weight': t.tensor([1.1, 2.1]), 'layer.bias': t.tensor([0.4, -0.6]), 'bn.running_mean': t.tensor([10.0, 10.0])}
    d = l2_weight_distance(sd1, sd2)
    diffs = np.array([0.1, 0.1, -0.1, -0.1], dtype=float)
    expected = float(np.sqrt(np.dot(diffs, diffs) / diffs.size))  # 0.1
    assert abs(d - expected) < 1e-6