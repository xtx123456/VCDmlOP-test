# scripts/attack_interp.py
import os
import sys
import argparse
from attacks import interp as interp_mod  # 使用 attacks/interp.py 的 main()

def main():
    ap = argparse.ArgumentParser(description='Interpolation attack (wrapper -> attacks.interp)')
    # 新接口：推荐传链目录
    ap.add_argument('--victim', type=str,
                    help='Victim chain directory (contains epoch_*.pt & metadata.json)')
    # 兼容旧接口：仅有最终权重文件（会取其所在目录作为链目录）
    ap.add_argument('--final', type=str,
                    help='[DEPRECATED] final checkpoint file; prefer --victim chain dir instead')
    ap.add_argument('--out', type=str, required=True, help='Output directory for forged chain')
    ap.add_argument('--arch', type=str, default='auto', choices=['vgg16', 'resnet18', 'alexnet', 'lenet'],
                    help="Model arch for random init; 'auto' reads from victim metadata")
    ap.add_argument('--alpha-start', type=float, default=0.0)
    ap.add_argument('--alpha-end',   type=float, default=1.0)
    ap.add_argument('--alpha-step',  type=float, default=0.01)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    # 参数检查与向 attacks.interp 转发
    if not args.victim and not args.final:
        ap.error("You must provide --victim (chain dir). For legacy usage, --final is accepted but deprecated.")

    victim_dir = args.victim or os.path.dirname(os.path.abspath(args.final))

    # 组装成 attacks.interp 的 argv 并调用其 main()
    argv = [
        '--victim', victim_dir,
        '--out', args.out,
        '--arch', args.arch,
        '--alpha-start', str(args.alpha_start),
        '--alpha-end', str(args.alpha_end),
        '--alpha-step', str(args.alpha_step),
    ]
    if args.verbose:
        argv.append('--verbose')

    # 将参数转交给 attacks.interp.main()
    sys.argv = ['attacks.interp'] + argv
    interp_mod.main()

if __name__ == '__main__':
    main()
