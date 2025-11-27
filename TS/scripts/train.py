# scripts/train.py
import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.optim as optim

from pot_core.init import apply_pot_init
from pot_core.data import get_dataloaders, num_classes_of
from pot_core.checkpoints import save_checkpoint
from pot_core.arch_utils import get_model_cls_from_meta_or_arg  # 统一注册表入口


def json_sanitize_meta(m):
    """把可能出现的 NaN/Inf 清到 0.0，避免 json.dump(allow_nan=False) 报错。"""
    def _clean(x):
        if isinstance(x, float) and not np.isfinite(x):
            return 0.0
        return x
    for k in ['train_acc', 'val_acc', 'val_loss', 'timestamps']:
        if k in m and isinstance(m[k], list):
            m[k] = [_clean(v) for v in m[k]]
    return m


def eval_model(model, loader, device):
    """只做前向与统计的验证函数。"""
    ce = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = ce(logits, labels)

            # 若出现非有限输出/损失，跳过本 batch
            if (not torch.isfinite(loss)) or (not torch.isfinite(logits).all()):
                continue

            bs = labels.size(0)
            loss_sum += float(loss.item()) * bs
            correct += int((logits.argmax(1) == labels).sum().item())
            total += bs

    if total == 0:
        return 0.0, 0.0
    acc = correct / total
    avg_loss = loss_sum / total
    if not np.isfinite(acc): acc = 0.0
    if not np.isfinite(avg_loss): avg_loss = 0.0
    return float(acc), float(avg_loss)


def main():
    ap = argparse.ArgumentParser(description='Train clean chain with PoT init (CIFAR-10/100)')
    ap.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'cifar10'])
    ap.add_argument('--data', type=str, required=True)
    # 统一注册表：支持 'resnet18' / 'alexnet' / 'auto'（auto 对训练等价于默认 ResNet18CIFAR）
    # ap.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'alexnet', 'auto'])
    ap.add_argument('--arch', type=str, default='resnet18',
                choices=['resnet18', 'alexnet', 'vgg16', 'lenet'])
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.01)  # 对 AlexNet-1024 更稳的默认
    ap.add_argument('--weight-decay', type=float, default=5e-4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--max-grad-norm', type=float, default=5.0)
    args = ap.parse_args()

    # 更快的 cuDNN 自动算法选择
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # 数据
    dataset = args.dataset.lower()
    train_loader, val_loader = get_dataloaders(dataset, args.data, args.batch_size, args.workers, download=True)
    num_classes = num_classes_of(dataset)

    if args.verbose:
        try:
            print(f"train_batches={len(train_loader)} val_batches={len(val_loader)}")
        except Exception:
            pass

    # 模型（统一从注册表获取类；训练阶段没有现成 meta，所以把 --arch 直接传入）
    # - 当 --arch 为 'resnet18' 或 'alexnet'：按注册表返回对应类
    # - 当 --arch 为 'auto'：对训练阶段等价于回退默认（ResNet18CIFAR）
    ModelCls = get_model_cls_from_meta_or_arg(chain_dir_or_meta={'arch': args.arch}, arch_arg=args.arch)
    model = ModelCls(num_classes=num_classes).to(device)
    arch_name = model.__class__.__name__  # 写进 metadata 供 verify/攻击脚本读取

    # PoT 初始化（在目标设备上进行）
    model.apply(apply_pot_init)

    # 损失与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 元信息与输出目录
    os.makedirs(args.out, exist_ok=True)
    meta = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'optimizer': 'SGD',
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'train_acc': [],
        'val_acc': [],
        'val_loss': [],
        'timestamps': [],
        'arch': arch_name,                 # e.g., 'AlexNetCIFAR' / 'ResNet18CIFAR'
        'dataset': args.dataset.upper(),   # 'CIFAR10' / 'CIFAR100'
    }

    # === 保存初始化（epoch_0000） ===
    init_val_acc, init_val_loss = eval_model(model, val_loader, device)
    meta['train_acc'].append(None)  # init 没有 train_acc
    meta['val_acc'].append(float(init_val_acc))
    meta['val_loss'].append(float(init_val_loss))
    meta['timestamps'].append(time.time())
    save_checkpoint(args.out, 0, model)
    with open(os.path.join(args.out, 'metadata.json'), 'w') as f:
        json.dump(json_sanitize_meta(meta), f, indent=2, allow_nan=False)

    if args.verbose:
        print(f"[epoch 0] init val_acc={init_val_acc:.3f} val_loss={init_val_loss:.3f} (saved epoch_0000.pt)")

    # === 训练循环（从 1 开始） ===
    best_acc = float(init_val_acc)
    for epoch in range(1, args.epochs + 1):
        model.train()
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)

            # 非有限数检查：跳过该 batch
            if (not torch.isfinite(loss)) or (not torch.isfinite(logits).all()):
                if args.verbose:
                    print("[warn] non-finite detected; skipping batch")
                continue

            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                correct += int((logits.argmax(1) == labels).sum().item())
                total += int(labels.size(0))

        train_acc = (correct / total) if total > 0 else 0.0
        va, vl = eval_model(model, val_loader, device)

        meta['train_acc'].append(float(train_acc))
        meta['val_acc'].append(float(va))
        meta['val_loss'].append(float(vl))
        meta['timestamps'].append(time.time())

        save_checkpoint(args.out, epoch, model)
        scheduler.step()

        with open(os.path.join(args.out, 'metadata.json'), 'w') as f:
            json.dump(json_sanitize_meta(meta), f, indent=2, allow_nan=False)

        if args.verbose:
            print(f"Epoch {epoch}/{args.epochs} | train_acc={train_acc:.3f} val_acc={va:.3f} val_loss={vl:.3f}")

        if va > best_acc:
            best_acc = va

    if args.verbose:
        print(f"Training finished. Best val acc: {best_acc:.4f}")


if __name__ == '__main__':
    main()
