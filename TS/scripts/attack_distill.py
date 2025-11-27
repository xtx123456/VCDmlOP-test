# scripts/attack_distill.py
import os
import sys
import argparse
from attacks import distill_same as distill_mod  # 调用 attacks/distill_same.py 的 main()

def main():
    ap = argparse.ArgumentParser(
        description='Rule Distillation attack (wrapper -> attacks.distill_same)'
    )
    # 推荐新接口：传链目录（含 epoch_*.pt 与 metadata.json）
    ap.add_argument('--victim', type=str,
                    help='Victim chain directory (recommended) or a single teacher .pt')

    # 兼容旧接口：--teacher（等价于 --victim）
    ap.add_argument('--teacher', type=str,
                    help='[DEPRECATED] same as --victim; kept for backward compatibility')

    ap.add_argument('--data',    type=str, required=True, help='Dataset root (CIFAR)')
    ap.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'cifar100'])
    ap.add_argument('--out',     type=str, required=True, help='Output chain directory')

    # 架构选择：走 arch_utils（auto=从 victim 元数据读取）
    ap.add_argument('--arch', type=str, default='auto', choices=['vgg16', 'resnet18', 'alexnet','lenet'])

    # 训练/切分/KD/优化器参数（与 attacks/distill_same 保持一致）
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--aux-frac', type=float, default=0.30)
    ap.add_argument('--split-seed', type=int, default=0)

    ap.add_argument('--tau', type=float, default=4.0)
    ap.add_argument('--tau-end', type=float, default=4.0)
    ap.add_argument('--lambda-kd', type=float, default=1.0)
    ap.add_argument('--lambda-end', type=float, default=1.0)

    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--weight-decay', type=float, default=5e-4)

    ap.add_argument('--val-holdout', type=int, default=2000)

    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--download', action='store_true', help='Download dataset if missing')

    args = ap.parse_args()

    victim_path = args.victim or args.teacher
    if not victim_path:
        ap.error("You must provide --victim (recommended) or --teacher (deprecated).")

    # 组装 attacks.distill_same 的参数列表并转发
    argv = [
        '--victim', victim_path,
        '--data', args.data,
        '--dataset', args.dataset,
        '--out', args.out,
        '--arch', args.arch,
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--workers', str(args.workers),
        '--aux-frac', str(args.aux_frac),
        '--split-seed', str(args.split_seed),
        '--tau', str(args.tau),
        '--tau-end', str(args.tau_end),
        '--lambda-kd', str(args.lambda_kd),
        '--lambda-end', str(args.lambda_end),
        '--lr', str(args.lr),
        '--weight-decay', str(args.weight_decay),
        '--val-holdout', str(args.val_holdout),
    ]
    if args.cpu:      argv.append('--cpu')
    if args.verbose:  argv.append('--verbose')
    if args.download: argv.append('--download')

    # 把 argv 交给 attacks.distill_same.main()
    sys.argv = ['attacks.distill_same'] + argv
    distill_mod.main()

if __name__ == '__main__':
    main()
