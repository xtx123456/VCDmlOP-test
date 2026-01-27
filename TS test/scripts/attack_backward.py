# scripts/attack_backward.py
import sys
import argparse
from attacks import backward as backward_mod

def main():
    ap = argparse.ArgumentParser(description="Backward Construction attack (wrapper -> attacks.backward)")

    ap.add_argument("--victim", type=str, required=True,
                    help="Victim chain directory (recommended) OR a single victim .pt")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100"])
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--arch", type=str, default="auto", choices=["vgg16", "resnet18", "alexnet", "lenet"])

    ap.add_argument("--aux-frac", type=float, default=0.10)
    ap.add_argument("--labeled-frac", type=float, default=1.0)
    ap.add_argument("--split-seed", type=int, default=0)

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--poison-start", type=float, default=0.40)
    ap.add_argument("--poison-inc", type=float, default=0.10)
    ap.add_argument("--poison-step", type=int, default=10)
    ap.add_argument("--poison-max", type=float, default=0.80)

    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--beta", type=float, default=0.005)

    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--download", action="store_true")

    args = ap.parse_args()

    argv = [
        "--victim", args.victim,
        "--data", args.data,
        "--dataset", args.dataset,
        "--out", args.out,
        "--arch", args.arch,
        "--aux-frac", str(args.aux_frac),
        "--labeled-frac", str(args.labeled_frac),
        "--split-seed", str(args.split_seed),
        "--epochs", str(args.epochs),
        "--poison-start", str(args.poison_start),
        "--poison-inc", str(args.poison_inc),
        "--poison-step", str(args.poison_step),
        "--poison-max", str(args.poison_max),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--beta", str(args.beta),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
    ]
    if args.cpu: argv.append("--cpu")
    if args.verbose: argv.append("--verbose")
    if args.download: argv.append("--download")

    sys.argv = ["attacks.backward"] + argv
    backward_mod.main()

if __name__ == "__main__":
    main()
