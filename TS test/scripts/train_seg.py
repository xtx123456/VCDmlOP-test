# scripts/train_seg.py
import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from pot_core.init import apply_pot_init
from pot_core.checkpoints import save_checkpoint
from pot_core.data_seg import get_seg_loaders
from pot_core.models_unet import UNetSeg

IGNORE_INDEX = 255  # VOC ignore label

def compute_pixacc_miou(logits, mask, num_classes=21, ignore_index=IGNORE_INDEX):
    with torch.no_grad():
        pred = logits.argmax(1)
        valid = (mask != ignore_index)
        correct = (pred[valid] == mask[valid]).float().sum().item()
        total   = valid.float().sum().item()
        pixacc = (correct / max(1.0, total))
        pred = pred[valid]; gt = mask[valid]
        hist = torch.bincount(gt * num_classes + pred, minlength=num_classes*num_classes
               ).reshape(num_classes, num_classes).float()
        iou = torch.diag(hist) / (hist.sum(1) + hist.sum(0) - torch.diag(hist) + 1e-6)
        miou = float(iou.mean().item())
        return float(pixacc), float(miou)

def eval_model(model, loader, device, num_classes=21):
    model.eval()
    pix_sum = 0.0; iou_sum = 0.0; n = 0
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    loss_sum = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(imgs)
            loss = ce(logits, masks)
            pix, miou = compute_pixacc_miou(logits, masks, num_classes=num_classes)
            loss_sum += float(loss.item()); pix_sum += pix; iou_sum += miou; n += 1
    if n == 0: return 0.0, 0.0, 0.0
    return float(pix_sum/n), float(iou_sum/n), float(loss_sum/n)

def json_sanitize_meta(m):
    def _clean(x): return 0.0 if isinstance(x, float) and not np.isfinite(x) else x
    for k in ['train_acc','val_acc','val_loss','timestamps']:
        if k in m and isinstance(m[k], list): m[k] = [_clean(v) for v in m[k]]
    return m

def main():
    ap = argparse.ArgumentParser(description="UNet PoT training on VOC2012 (seg)")
    ap.add_argument('--data', type=str, required=True, help='VOC root (contains VOCdevkit/VOC2012)')
    ap.add_argument('--out',  type=str, required=True)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--img-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    # 关键：可选下载开关（默认不下载）
    ap.add_argument('--download', action='store_true',
                    help='set this flag only if you want torchvision to download VOC2012')
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # 不要写死 True！
    train_loader, val_loader = get_seg_loaders(
        root=args.data,
        batch_size=args.batch_size,
        workers=args.workers,
        image_size=args.img_size,
        download=args.download)

    num_classes = 21
    model = UNetSeg(num_classes=num_classes).to(device)
    model.apply(apply_pot_init)

    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.out, exist_ok=True)
    meta = {
        'epochs': args.epochs, 'batch_size': args.batch_size, 'optimizer': 'Adam',
        'lr': args.lr, 'weight_decay': args.weight_decay,
        'train_acc': [], 'val_acc': [], 'val_loss': [], 'timestamps': [],
        'arch': 'UNetSeg', 'dataset': 'VOC2012', 'num_classes': num_classes, 'metric_name': 'pixel-acc',
    }

    init_pix, init_miou, init_loss = eval_model(model, val_loader, device, num_classes)
    meta['train_acc'].append(None); meta['val_acc'].append(float(init_pix))
    meta['val_loss'].append(float(init_loss)); meta['timestamps'].append(time.time())
    save_checkpoint(args.out, 0, model)
    with open(os.path.join(args.out, 'metadata.json'), 'w') as f:
        json.dump(json_sanitize_meta(meta), f, indent=2, allow_nan=False)
    if args.verbose:
        print(f"[epoch 0] val_pix={init_pix:.3f} mIoU={init_miou:.3f} loss={init_loss:.3f} (saved epoch_0000.pt)")

    best = init_pix
    for ep in range(1, args.epochs + 1):
        model.train()
        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True); masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs); loss = ce(logits, masks)
            loss.backward(); clip_grad_norm_(model.parameters(), max_norm=5.0); optimizer.step()

        val_pix, val_miou, val_loss = eval_model(model, val_loader, device, num_classes)
        meta['train_acc'].append(None); meta['val_acc'].append(float(val_pix))
        meta['val_loss'].append(float(val_loss)); meta['timestamps'].append(time.time())
        save_checkpoint(args.out, ep, model); scheduler.step()
        with open(os.path.join(args.out, 'metadata.json'), 'w') as f:
            json.dump(json_sanitize_meta(meta), f, indent=2, allow_nan=False)
        if args.verbose:
            print(f"Epoch {ep}/{args.epochs} | val_pix={val_pix:.3f} mIoU={val_miou:.3f} val_loss={val_loss:.3f}")
        if val_pix > best: best = val_pix
    if args.verbose:
        print(f"Training finished. Best val pixel-acc: {best:.4f}")

if __name__ == '__main__':
    main()
