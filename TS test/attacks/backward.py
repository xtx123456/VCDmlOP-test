# attacks/backward.py
"""
Backward Construction Attack (PoT, WWW'23):

- Start from a stolen (victim) converged model M_teacher.
- Fine-tune it on an auxiliary labeled subset whose labels are increasingly poisoned.
- Add a regularizer to pull weights toward a random GMM-initialized model M_rand:
      L' = L + beta * d(M, M_rand)   (Eq.(9) in the paper)
- Reverse the produced trajectory and save as epoch_0000..epoch_N, so that
  accuracy tends to be non-decreasing along the saved chain (to try to pass P1/P5).
"""
import os, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from pot_core.init import apply_pot_init
from pot_core.data import make_owner_and_aux_sets, num_classes_of
from pot_core.checkpoints import load_chain
from pot_core.arch_utils import get_model_cls_from_meta_or_arg


# ----------------- helpers -----------------
def _load_victim(victim_path: str):
    """
    Accept:
    - a chain directory: use the last checkpoint as victim final
    - a single .pt file: interpret as state_dict or {'model': state_dict}
    Returns (state_dict, metadata_dict, chain_dir_or_None)
    """
    if os.path.isdir(victim_path):
        ckpts, meta = load_chain(victim_path)
        assert len(ckpts) >= 1, "victim chain is empty."
        return ckpts[-1]["model"], meta, victim_path

    payload = torch.load(victim_path, map_location="cpu")
    sd = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    return sd, {}, None


class _PoisonPlanSubset(torch.utils.data.Dataset):
    """
    Poison labels with a FIXED plan:
    - fixed permutation of indices
    - fixed wrong label per sample
    Poison set size increases monotonically with poison_rate.
    """
    def __init__(self, base_subset: Subset, num_classes: int, seed: int):
        self.base = base_subset
        self.num_classes = int(num_classes)

        rng = np.random.RandomState(int(seed))
        n = len(self.base)

        # fixed order: first k will be poisoned when poison_rate=k/n
        self.order = rng.permutation(n)

        # fixed wrong labels per sample
        self.wrong = rng.randint(0, self.num_classes, size=n, dtype=np.int64)

        self.k_poison = 0
        self.poison_mask = np.zeros(n, dtype=bool)

    def set_poison_rate(self, poison_rate: float):
        n = len(self.base)
        k = int(round(float(poison_rate) * n))
        k = max(0, min(n, k))
        self.k_poison = k

        # build a boolean mask (monotonic by construction)
        self.poison_mask[:] = False
        self.poison_mask[self.order[:k]] = True

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i: int):
        x, y = self.base[i]
        y = int(y)
        if self.poison_mask[i]:
            r = int(self.wrong[i])
            if r == y:
                r = (r + 1) % self.num_classes
            y = r
        return x, y


def _param_frob_sq_normalized(model: torch.nn.Module, ref: torch.nn.Module) -> torch.Tensor:
    """
    Normalized squared Frobenius distance over ALL parameters:
        d = ||theta - theta_ref||_F^2 / N
    where N is the total number of scalar parameters compared.

    This is a "global" normalization (sum over all tensors / total elements),
    which is typically more faithful than averaging per-tensor means.
    """
    device = next(model.parameters()).device
    sum_sq = torch.zeros((), device=device, dtype=torch.float32)
    count = 0

    for p, q in zip(model.parameters(), ref.parameters()):
        if p.shape != q.shape:
            continue
        diff = (p - q).float()
        sum_sq = sum_sq + diff.pow(2).sum()
        count += diff.numel()

    if count == 0:
        return torch.zeros((), device=device, dtype=torch.float32)

    return sum_sq / float(count)


@torch.no_grad()
def _acc_on_loader(model, loader, device) -> float:
    model.eval()
    tot = 0
    cor = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb).argmax(1)
        cor += (pred == yb).sum().item()
        tot += yb.numel()
    return float(cor / max(1, tot))


def main():
    ap = argparse.ArgumentParser(description="Backward Construction Attack (PoT-style)")
    ap.add_argument("--victim", type=str, required=True,
                    help="Victim chain directory (recommended) OR a single victim .pt")
    ap.add_argument("--data", type=str, required=True, help="Dataset root for CIFAR")
    ap.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100"])
    ap.add_argument("--out", type=str, required=True, help="Output chain directory")

    ap.add_argument("--arch", type=str, default="auto",
                    choices=["auto", "vgg16", "resnet18", "alexnet", "lenet"],
                    help="Model arch (auto reads from victim metadata if available)")

    # attacker split / labeled subset
    ap.add_argument("--aux-frac", type=float, default=0.10,
                    help="Attacker owns this fraction of training set (default 0.10)")
    ap.add_argument("--labeled-frac", type=float, default=1.0,
                    help="Fraction of AUX that has GT labels. For CV CIFAR, typically 1.0.")
    ap.add_argument("--split-seed", type=int, default=0)

    # backward attack schedule (poison rate increases)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--poison-start", type=float, default=0.40)
    ap.add_argument("--poison-inc", type=float, default=0.10)
    ap.add_argument("--poison-step", type=int, default=10,
                    help="Increase poison rate every K epochs")
    ap.add_argument("--poison-max", type=float, default=0.80)

    # optimization
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--beta", type=float, default=0.005,
                    help="Regularization coeff toward random GMM model (Eq.(9))")

    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = args.dataset.lower()
    num_classes = num_classes_of(dataset)

    # 1) Load victim model (stolen converged model)
    sd_victim, victim_meta, victim_chain_dir = _load_victim(args.victim)
    chain_ref = victim_chain_dir if victim_chain_dir is not None else victim_meta
    ModelCls = get_model_cls_from_meta_or_arg(chain_ref, args.arch)

    victim = ModelCls(num_classes=num_classes).to(device)
    victim.load_state_dict(sd_victim, strict=True)
    victim.eval()

    # 2) Build AUX split (same-distribution split from CIFAR train), and use official test as verifier/val
    owner_ds, val_set, aux_ds = make_owner_and_aux_sets(
        dataset,
        root=args.data,
        owner_frac=1.0 - float(args.aux_frac),
        seed=int(args.split_seed),
        download=args.download,
    )

    val_loader = DataLoader(
        val_set, batch_size=256, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # AUX labeled subset
    if not (0.0 < float(args.labeled_frac) <= 1.0):
        raise ValueError("--labeled-frac must be in (0,1].")
    n_aux = len(aux_ds)
    m_lab = int(round(n_aux * float(args.labeled_frac)))
    rng = np.random.RandomState(int(args.split_seed) + 1337)
    perm = np.arange(n_aux)
    rng.shuffle(perm)
    lab_idx = perm[:m_lab].tolist()
    aux_labeled = Subset(aux_ds, lab_idx)

    # 3) Random reference model (GMM init) for Eq.(9) regularizer
    ref = ModelCls(num_classes=num_classes).to(device)
    ref.apply(apply_pot_init)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    # 4) Attack trajectory: M0=victim, then fine-tune to get M1..ME (worse and worse)
    traj_state_dicts = []
    traj_metrics = []

    # M0: victim itself
    traj_state_dicts.append({k: v.detach().cpu() for k, v in victim.state_dict().items()})
    acc0 = _acc_on_loader(victim, val_loader, device)
    traj_metrics.append({"epoch_attack": 0, "poison_rate": 0.0, "val_acc": acc0})

    # attacker model = start from victim weights
    attacker = ModelCls(num_classes=num_classes).to(device)
    attacker.load_state_dict(sd_victim, strict=True)

    optim = torch.optim.SGD(
        attacker.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )

    os.makedirs(args.out, exist_ok=True)

    # fixed poison plan (monotonic increase)
    poisoned_ds = _PoisonPlanSubset(
        aux_labeled, num_classes=num_classes,
        seed=int(args.split_seed) * 100000 + 999
    )
    poisoned_ds.set_poison_rate(0.0)

    for ep in range(1, int(args.epochs) + 1):
        stage = (ep - 1) // max(1, int(args.poison_step))
        poison_rate = float(args.poison_start) + float(stage) * float(args.poison_inc)
        poison_rate = float(min(max(poison_rate, 0.0), float(args.poison_max)))

        poisoned_ds.set_poison_rate(poison_rate)
        train_loader = DataLoader(
            poisoned_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True
        )

        attacker.train()
        tot_loss = 0.0
        tot = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optim.zero_grad(set_to_none=True)
            logits = attacker(xb)
            loss_ce = F.cross_entropy(logits, yb)

            # Eq.(9) style reg toward random GMM model (normalized Frobenius^2)
            loss_reg = _param_frob_sq_normalized(attacker, ref)
            loss = loss_ce + float(args.beta) * loss_reg

            loss.backward()
            optim.step()

            bs = yb.size(0)
            tot_loss += float(loss.item()) * bs
            tot += bs

        traj_state_dicts.append({k: v.detach().cpu() for k, v in attacker.state_dict().items()})
        acc = _acc_on_loader(attacker, val_loader, device)
        traj_metrics.append({"epoch_attack": ep, "poison_rate": poison_rate, "val_acc": acc})

        if args.verbose:
            print(
                f"[backward] ep={ep:03d} poison={poison_rate:.2f} "
                f"loss={tot_loss/max(1,tot):.4f} val_acc={acc:.4f}",
                flush=True
            )

    # 5) Reverse trajectory: saved epoch_0000 = WORST, last epoch = victim
    traj_state_dicts_rev = list(reversed(traj_state_dicts))
    traj_metrics_rev = list(reversed(traj_metrics))

    # 6) Save chain files
    meta = {
        "attack": "backward_construction",
        "teacher_from": os.path.abspath(args.victim),
        "dataset": "CIFAR10" if num_classes == 10 else "CIFAR100",
        "arch": ModelCls.__name__,
        "epochs": int(args.epochs),
        "aux_frac": float(args.aux_frac),
        "labeled_frac": float(args.labeled_frac),
        "poison_start": float(args.poison_start),
        "poison_inc": float(args.poison_inc),
        "poison_step": int(args.poison_step),
        "poison_max": float(args.poison_max),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "beta": float(args.beta),
        "timestamps": [],
        "val_acc": [],
        "notes": "Trajectory generated from victim -> worse (increasing poison), then reversed and saved.",
        "trajectory_attack_order": traj_metrics,       # victim -> worse
        "trajectory_saved_order": traj_metrics_rev,    # worse -> victim
        "reg_distance": "normalized_frobenius_squared_over_all_params",
    }

    for i, sd in enumerate(traj_state_dicts_rev):
        torch.save({"epoch": i, "model": sd}, os.path.join(args.out, f"epoch_{i:04d}.pt"))
        meta["timestamps"].append(time.time())
        meta["val_acc"].append(float(traj_metrics_rev[i]["val_acc"]))

    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, allow_nan=False)

    print(f"[backward] wrote chain -> {args.out} (len={len(traj_state_dicts_rev)})", flush=True)


if __name__ == "__main__":
    main()
