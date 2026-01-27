# scripts/sensitivity_ckpt_density.py
import argparse
import os
import json
import re
import shutil
import tempfile
import time
from typing import List, Tuple, Dict

from pot_core.verify import verify_chain, apply_strict

EPOCH_RE = re.compile(r"^epoch_(\d+)\.pt$")


def list_epochs(chain_dir: str) -> List[int]:
    eps = []
    for fn in os.listdir(chain_dir):
        m = EPOCH_RE.match(fn)
        if m:
            eps.append(int(m.group(1)))
    eps.sort()
    return eps


def read_meta(chain_dir: str) -> Dict:
    p = os.path.join(chain_dir, "metadata.json")
    if not os.path.isfile(p):
        return {}
    with open(p, "r") as f:
        return json.load(f)


def write_meta(meta: Dict, out_dir: str):
    p = os.path.join(out_dir, "metadata.json")
    with open(p, "w") as f:
        json.dump(meta, f, indent=2, allow_nan=False)


def make_thinned_chain(src_dir: str, dst_dir: str, every: int, max_epoch: int = None) -> Dict:
    """
    Create a thinned chain that mimics "save every K epochs":
      keep epoch_0000.pt and then keep epoch_%04d.pt where epoch % every == 0
      up to max_epoch (default: last epoch in src).
    Also trims metadata arrays (val_acc/train_acc/val_loss/timestamps) accordingly.
    """
    os.makedirs(dst_dir, exist_ok=True)

    epochs = list_epochs(src_dir)
    if not epochs:
        raise RuntimeError(f"No epoch_*.pt found in {src_dir}")
    last = epochs[-1]
    if max_epoch is None:
        max_epoch = last
    max_epoch = min(max_epoch, last)

    keep = []
    for e in epochs:
        if e == 0:
            keep.append(e)
        elif e <= max_epoch and (e % every == 0):
            keep.append(e)

    # copy checkpoints
    for e in keep:
        src = os.path.join(src_dir, f"epoch_{e:04d}.pt")
        dst = os.path.join(dst_dir, f"epoch_{e:04d}.pt")
        shutil.copy2(src, dst)

    # copy & trim metadata
    meta = read_meta(src_dir)
    if meta:
        def _trim_list(key: str):
            arr = meta.get(key, None)
            if not isinstance(arr, list):
                return
            trimmed = []
            for e in keep:
                if e < len(arr):
                    trimmed.append(arr[e])
                else:
                    trimmed.append(None)
            meta[key] = trimmed

        for k in ["train_acc", "val_acc", "val_loss", "timestamps"]:
            _trim_list(k)

        # epochs 字段不强制改（verify 用的是 checkpoint 数量），但改了更直观
        meta["thinned_from"] = os.path.abspath(src_dir)
        meta["thinned_every"] = every
        meta["thinned_keep_epochs"] = keep
        write_meta(meta, dst_dir)

    return {"keep_epochs": keep, "max_epoch": max_epoch, "last_epoch_in_src": last}


def strict_pass(chain_dir: str, emd_bins: int, num_rand: int, seed: int,
                eps2: float, eps3: float, eps4: float, rho5: float, k6: float) -> bool:
    res = verify_chain(chain_dir, emd_bins=emd_bins, num_random=num_rand, seed=seed)
    passes = apply_strict(res, eps2=eps2, eps3=eps3, eps4=eps4, rho5=rho5, k6=k6)
    return bool(passes["PASS_P1"] and passes["PASS_P2"] and passes["PASS_P3"] and
                passes["PASS_P4"] and passes["PASS_P5"] and passes["PASS_P6"])


def load_pairs_from_csv(path: str) -> List[Tuple[str, str]]:
    import csv
    pairs = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        # required columns: clean, attack
        for row in r:
            c = (row.get("clean") or "").strip()
            a = (row.get("attack") or "").strip()
            if c and a:
                pairs.append((c, a))
    return pairs


def main():
    ap = argparse.ArgumentParser(description="A2) Checkpoint density sensitivity (TPR/FPR on thinned chains)")
    ap.add_argument("--pair", action="append", nargs=2, metavar=("CLEAN_DIR", "ATTACK_DIR"),
                    help="Add a (clean, attack) pair. Can be repeated.")
    ap.add_argument("--pairs-csv", type=str, help="CSV with columns: clean, attack")

    ap.add_argument("--every", type=int, nargs="+", default=[1, 2, 5, 10],
                    help="Save frequency to simulate: keep every K epochs (default: 1 2 5 10)")
    ap.add_argument("--max-epoch", type=int, default=None,
                    help="Optional: truncate chain to max epoch before thinning (default: use full chain last epoch)")

    ap.add_argument("--emd-bins", type=int, default=200)
    ap.add_argument("--num-rand", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)

    # strict thresholds (same as compare/verify defaults)
    ap.add_argument("--eps2", type=float, default=0.02)
    ap.add_argument("--eps3", type=float, default=0.03)
    ap.add_argument("--eps4", type=float, default=0.25)
    ap.add_argument("--rho5", type=float, default=0.6)
    ap.add_argument("--k6", type=float, default=3.0)

    ap.add_argument("--workdir", type=str, default=None,
                    help="Where to write thinned chains (default: a temp directory)")
    ap.add_argument("--save", type=str, default=None,
                    help="Save summary JSON to this path")

    args = ap.parse_args()

    pairs: List[Tuple[str, str]] = []
    if args.pair:
        pairs.extend([(c, a) for c, a in args.pair])
    if args.pairs_csv:
        pairs.extend(load_pairs_from_csv(args.pairs_csv))

    if not pairs:
        raise SystemExit("No pairs provided. Use --pair CLEAN ATTACK (repeatable) or --pairs-csv ...")

    workdir = args.workdir or tempfile.mkdtemp(prefix="pot_thin_")
    os.makedirs(workdir, exist_ok=True)

    print(f"[A2] workdir = {workdir}")
    print(f"[A2] pairs   = {len(pairs)}")
    print(f"[A2] every   = {args.every}")
    print(f"[A2] strict thresholds: eps2={args.eps2}, eps3={args.eps3}, eps4={args.eps4}, rho5={args.rho5}, k6={args.k6}")
    print("")

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "workdir": workdir,
        "settings": {
            "every": args.every,
            "max_epoch": args.max_epoch,
            "emd_bins": args.emd_bins,
            "num_rand": args.num_rand,
            "seed": args.seed,
            "thresholds": {
                "eps2": args.eps2, "eps3": args.eps3, "eps4": args.eps4, "rho5": args.rho5, "k6": args.k6
            }
        },
        "results": {}
    }

    # For each K, compute TPR/FPR over all pairs
    print("A2) Checkpoint Density Sensitivity (TPR/FPR)")
    print("-" * 72)
    print(f"{'every(K)':>8s} | {'TPR(clean pass rate)':>22s} | {'FPR(attack pass rate)':>24s} | details")
    print("-" * 72)

    for K in args.every:
        clean_pass = 0
        attack_pass = 0

        per_pair = []
        for idx, (clean_dir, attack_dir) in enumerate(pairs):
            tag = f"pair{idx:02d}_K{K}"

            thin_clean = os.path.join(workdir, f"{tag}_clean")
            thin_attack = os.path.join(workdir, f"{tag}_attack")

            make_thinned_chain(clean_dir, thin_clean, every=K, max_epoch=args.max_epoch)
            make_thinned_chain(attack_dir, thin_attack, every=K, max_epoch=args.max_epoch)

            pc = strict_pass(thin_clean, args.emd_bins, args.num_rand, args.seed,
                             args.eps2, args.eps3, args.eps4, args.rho5, args.k6)
            pa = strict_pass(thin_attack, args.emd_bins, args.num_rand, args.seed,
                             args.eps2, args.eps3, args.eps4, args.rho5, args.k6)

            clean_pass += int(pc)
            attack_pass += int(pa)

            per_pair.append({
                "clean": clean_dir,
                "attack": attack_dir,
                "thin_clean": thin_clean,
                "thin_attack": thin_attack,
                "clean_pass": bool(pc),
                "attack_pass": bool(pa),
            })

        tpr = clean_pass / len(pairs)
        fpr = attack_pass / len(pairs)

        summary["results"][str(K)] = {
            "TPR": tpr,
            "FPR": fpr,
            "clean_pass": clean_pass,
            "attack_pass": attack_pass,
            "n_pairs": len(pairs),
            "per_pair": per_pair,
        }

        detail = f"{clean_pass}/{len(pairs)} clean, {attack_pass}/{len(pairs)} attack"
        print(f"{K:8d} | {tpr:22.4f} | {fpr:24.4f} | {detail}")

    print("-" * 72)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[A2] saved -> {args.save}")


if __name__ == "__main__":
    main()
