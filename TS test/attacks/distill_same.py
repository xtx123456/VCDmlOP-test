# attacks/distill_same.py
import os, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.utils.data import DataLoader, Subset

from pot_core.init import apply_pot_init
from pot_core.data import make_owner_and_aux_sets, num_classes_of
from pot_core.checkpoints import load_chain
from pot_core.arch_utils import (
    get_model_cls_from_meta_or_arg,
    infer_num_classes_from_meta_or_sd,
)

def _kd_loss(student_logits, teacher_logits, T: float):
    log_p = F.log_softmax(student_logits / T, dim=1)
    q     = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction='batchmean') * (T * T)

def _load_teacher(victim_path: str):
    """
    支持两种输入：
    - 链目录：返回 (final_sd, meta, chain_dir)
    - 单个 .pt 文件：返回 (sd, {}, None)
    """
    if os.path.isdir(victim_path):
        ckpts, meta = load_chain(victim_path)
        assert len(ckpts) >= 1, "victim chain is empty."
        return ckpts[-1]['model'], meta, victim_path
    # 单文件
    payload = torch.load(victim_path, map_location='cpu')
    sd = payload['model'] if isinstance(payload, dict) and 'model' in payload else payload
    return sd, {}, None

def main():
    ap = argparse.ArgumentParser(description='Rule Distillation Attack (same-arch, owner/aux split)')
    ap.add_argument('--victim', type=str, required=True,
                    help='Victim chain directory (recommended) OR a single teacher checkpoint .pt')
    ap.add_argument('--data', type=str, required=True, help='Dataset root for CIFAR')
    ap.add_argument('--dataset', type=str, required=True, choices=['cifar10','cifar100'])
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--arch', type=str, default='auto', choices=['vgg16','resnet18','alexnet','lenet'],
                    help="Student/Teacher arch. 'auto' reads from victim metadata if available")
    # training / split
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--aux-frac', type=float, default=0.30)
    ap.add_argument('--split-seed', type=int, default=0)
    # partially-labeled AUX (per PoT w/o Training Data)
    ap.add_argument('--labeled-frac', type=float, default=0.10,
                    help='Fraction of attacker AUX split that has ground-truth labels (default: 0.10)')
    # KD hyper
    ap.add_argument('--tau', type=float, default=4.0)
    ap.add_argument('--tau-end', type=float, default=4.0)
    ap.add_argument('--lambda-kd', type=float, default=1.0)
    ap.add_argument('--lambda-end', type=float, default=1.0)
    ap.add_argument('--lambda-ce', type=float, default=1.0,
                    help='Weight for supervised CE on labeled AUX (start)')
    ap.add_argument('--lambda-ce-end', type=float, default=1.0,
                    help='Weight for supervised CE on labeled AUX (end)')

    #ap.add_argument('--gamma', type=float, default=0.0, help='Regularization weight for parameter-distance term (regulated distillation)')
    # optim
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--weight-decay', type=float, default=5e-4)
    # validation sampling
    ap.add_argument('--val-holdout', type=int, default=2000)
    # misc
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--download', action='store_true', help='Download CIFAR if not present')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    dataset = args.dataset.lower()
    num_classes_cli = num_classes_of(dataset)  # 10 or 100 by dataset arg

    # 1) Teacher
    sd_teacher, victim_meta, victim_chain_dir = _load_teacher(args.victim)

    # 选择架构类（优先 --arch，其次 victim metadata；都无则回退 ResNet18CIFAR）
    # 注意：当 --victim 是单个 .pt 时，victim_chain_dir 为 None，此时 arch_utils 会根据 --arch 决定
    chain_ref = victim_chain_dir if victim_chain_dir is not None else victim_meta
    ModelCls = get_model_cls_from_meta_or_arg(chain_ref, args.arch)

    # 类别数：优先根据 --dataset；如果未来你要“自动推断”，也可用 infer_*：
    # num_classes = infer_num_classes_from_meta_or_sd(victim_meta, sd_teacher, default=num_classes_cli)
    num_classes = num_classes_cli

    teacher = ModelCls(num_classes=num_classes).to(device)
    teacher.load_state_dict(sd_teacher, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # 2) Student (PoT init)
    student = ModelCls(num_classes=num_classes).to(device)
    student.apply(apply_pot_init)

    # 3) Data split (owner / aux) with partially-labeled attacker AUX
    owner_ds, owner_val_set, aux_ds = make_owner_and_aux_sets(
        dataset,
        root=args.data,
        owner_frac=1.0 - args.aux_frac,
        seed=args.split_seed,
        download=args.download,
    )

    owner_loader = DataLoader(owner_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    owner_val_loader = DataLoader(owner_val_set, batch_size=256, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)

    # A convenience loader over the full AUX split (used only for metrics/holdout sampling).
    aux_loader = DataLoader(aux_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=True)

    # AUX -> (labeled subset, unlabeled subset). Deterministic.
    if not (0.0 < float(args.labeled_frac) <= 1.0):
        raise ValueError("--labeled-frac must be in (0, 1].")
    n_aux = len(aux_ds)
    m_lab = int(round(n_aux * float(args.labeled_frac)))
    rng = np.random.RandomState(int(args.split_seed) + 1337)
    perm = np.arange(n_aux)
    rng.shuffle(perm)
    lab_idx = perm[:m_lab].tolist()
    unlab_idx = perm[m_lab:].tolist()
    aux_labeled_ds = Subset(aux_ds, lab_idx)
    aux_unlabeled_ds = Subset(aux_ds, unlab_idx)

    aux_labeled_loader = DataLoader(aux_labeled_ds, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.workers, pin_memory=True)
    aux_unlabeled_loader = DataLoader(aux_unlabeled_ds, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.workers, pin_memory=True)

    optimzr = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optimzr, T_max=args.epochs)

    # 4) Meta
    os.makedirs(args.out, exist_ok=True)
    meta = {
        'epochs': int(args.epochs),
        'batch_size': int(args.batch_size),
        'optimizer': 'SGD',
        'lr': float(args.lr),
        'weight_decay': float(args.weight_decay),
        'arch': ModelCls.__name__,                               # AlexNetCIFAR / ResNet18CIFAR
        'dataset': 'CIFAR10' if num_classes == 10 else 'CIFAR100',
        'train_acc': [],
        'val_acc': [],
        'val_loss': [],
        'timestamps': [],
        'attack': 'distill_same',
        'teacher_from': os.path.abspath(args.victim),
        'aux_frac': float(args.aux_frac),
        'labeled_frac': float(args.labeled_frac),
        'lambda_ce': float(args.lambda_ce),
        'lambda_ce_end': float(args.lambda_ce_end),
        'notes': 'val_acc = student-teacher agreement on held-out AUX slice; training uses KD on unlabeled AUX + (KD+CE) on labeled AUX',
    }

    # 5) Build held-out AUX for agreement metric
    heldout_imgs, heldout_teacher = [], []
    with torch.no_grad():
        collected = 0
        for xb in aux_loader:
            x = xb[0] if isinstance(xb, (list, tuple)) else xb
            x = x.to(device)
            tlog = teacher(x)
            heldout_imgs.append(x.cpu())
            heldout_teacher.append(tlog.cpu())
            collected += x.size(0)
            if collected >= args.val_holdout:
                break
    if len(heldout_imgs) > 0:
        heldout_imgs    = torch.cat(heldout_imgs, dim=0)
        heldout_teacher = torch.cat(heldout_teacher, dim=0)

    def agreement_on_heldout() -> float:
        if isinstance(heldout_imgs, list):  # 空
            return 0.0
        with torch.no_grad():
            slog = student(heldout_imgs.to(device))
            pred_s = slog.argmax(1)
            pred_t = heldout_teacher.argmax(1).to(device)
            return float((pred_s == pred_t).float().mean().item())

    # 6) Save epoch_0000 (PoT init) for verification baselines
    student.eval()
    val_agree0 = agreement_on_heldout()
    meta['train_acc'].append(None)
    meta['val_acc'].append(float(val_agree0))
    meta['val_loss'].append(float(1.0 - val_agree0))
    meta['timestamps'].append(time.time())
    torch.save({'epoch': 0, 'model': student.state_dict()}, os.path.join(args.out, f'epoch_{0:04d}.pt'))
    with open(os.path.join(args.out, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2, allow_nan=False)


    # 8) KD training
    for epoch in range(1, args.epochs + 1):
        student.train()
        # 线性调度 tau / lambda（如你原逻辑）
        progress = (epoch - 1) / max(1, args.epochs - 1)
        T_now    = args.tau + (args.tau_end - args.tau) * progress
        lam_now  = args.lambda_kd + (args.lambda_end - args.lambda_kd) * progress

        # unlabeled AUX: KD only
        for xb in aux_unlabeled_loader:
            # xb is (x, y) from CIFAR dataset, but we intentionally ignore y
            x = xb[0] if isinstance(xb, (list, tuple)) else xb
            x = x.to(device)
            with torch.no_grad():
                tlog = teacher(x)
            optimzr.zero_grad(set_to_none=True)
            slog = student(x)
            loss = lam_now * _kd_loss(slog, tlog, T=T_now)
            loss.backward()
            optimzr.step()

        # labeled AUX: supervised CE + KD (per partially-labeled adversary setting)
        lam_ce_now = args.lambda_ce + (args.lambda_ce_end - args.lambda_ce) * progress
        for xb in aux_labeled_loader:
            x, y = xb if isinstance(xb, (list, tuple)) and len(xb) >= 2 else (xb, None)
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                tlog = teacher(x)
            optimzr.zero_grad(set_to_none=True)
            slog = student(x)
            loss_kd = _kd_loss(slog, tlog, T=T_now)
            loss_ce = F.cross_entropy(slog, y)
            loss = lam_now * loss_kd + lam_ce_now * loss_ce
            loss.backward()
            optimzr.step()

        # log / save
        student.eval()
        val_agree = agreement_on_heldout()
        meta['train_acc'].append(None)
        meta['val_acc'].append(float(val_agree))
        meta['val_loss'].append(float(1.0 - val_agree))
        meta['timestamps'].append(time.time())
        torch.save({'epoch': epoch, 'model': student.state_dict()},
                   os.path.join(args.out, f'epoch_{epoch:04d}.pt'))
        with open(os.path.join(args.out, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2, allow_nan=False)

        sched.step()
        if args.verbose:
            print(f"[attack/distill] epoch {epoch}/{args.epochs} "
                  f"| agree={val_agree:.3f} | T={T_now:.2f} | lam_kd={lam_now:.3f} | lam_ce={lam_ce_now:.3f}")

if __name__ == '__main__':
    main()
