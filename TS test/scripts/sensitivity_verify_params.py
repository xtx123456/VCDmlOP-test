# scripts/sensitivity_verify_params.py
import argparse
import os
import json
import glob
import time
import csv
from typing import List, Dict, Any, Tuple

from pot_core.verify import verify_chain, apply_strict


CORE_PASS_KEYS = ["PASS_P1", "PASS_P2", "PASS_P3", "PASS_P4", "PASS_P5", "PASS_P6"]


def _read_meta(chain_dir: str) -> Dict[str, Any]:
    p = os.path.join(chain_dir, "metadata.json")
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _expand_chains(items: List[str]) -> List[str]:
    """
    items can be:
      - exact directories
      - glob patterns
      - a directory containing many chain subdirs (then we take subdirs that have checkpoints)
    """
    out = []
    for it in items:
        # glob
        matches = glob.glob(it)
        if matches:
            for m in matches:
                if os.path.isdir(m):
                    out.append(m)
            continue

        # plain dir
        if os.path.isdir(it):
            out.append(it)
            # if it is a "root" dir, also add its child dirs that look like chains
            # (heuristic: contains checkpoints/)
            for sub in sorted(glob.glob(os.path.join(it, "*"))):
                if os.path.isdir(sub) and os.path.isdir(os.path.join(sub, "checkpoints")):
                    out.append(sub)
            continue

    # unique, keep order
    seen = set()
    uniq = []
    for d in out:
        d = os.path.normpath(d)
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    # filter: must have checkpoints dir
    # filter: must look like a PoT chain
    def _looks_like_chain(d: str) -> bool:
        # case 1: checkpoints/epoch_*.pt
        ckpt_dir = os.path.join(d, "checkpoints")
        if os.path.isdir(ckpt_dir):
            for fn in os.listdir(ckpt_dir):
                if fn.startswith("epoch_") and fn.endswith(".pt"):
                    return True
        # case 2: epoch_*.pt directly under d
        for fn in os.listdir(d):
            if fn.startswith("epoch_") and fn.endswith(".pt"):
                return True
        return False

    uniq = [d for d in uniq if _looks_like_chain(d)]
    return uniq


# def _looks_like_chain(d: str) -> bool:
#     # case 1: checkpoints/epoch_*.pt
#     ckpt_dir = os.path.join(d, "checkpoints")
#     if os.path.isdir(ckpt_dir):
#         for fn in os.listdir(ckpt_dir):
#             if fn.startswith("epoch_") and fn.endswith(".pt"):
#                 return True
#     # case 2: epoch_*.pt directly under d
#     for fn in os.listdir(d):
#         if fn.startswith("epoch_") and fn.endswith(".pt"):
#             return True
#     return False

# uniq = [d for d in uniq if _looks_like_chain(d)]
# return uniq

def _decision_from_passes(passes: Dict[str, bool], enabled: List[str]) -> bool:
    """enabled is list of PASS_* keys that must all be True."""
    for k in enabled:
        if not bool(passes.get(k, False)):
            return False
    return True


def _mode_enabled(mode: str) -> List[str]:
    """
    Mode choices:
      - ALL: P1..P6
      - NO_P1
      - NO_P23 (drop P2 and P3 together)
      - NO_P4
      - NO_P6
      - ONLY_P1
      - ONLY_P23
      - ONLY_P4
      - ONLY_P6
    """
    mode = mode.upper()

    all_keys = CORE_PASS_KEYS[:]  # PASS_P1..PASS_P6

    if mode == "ALL":
        return all_keys
    if mode == "NO_P1":
        return [k for k in all_keys if k != "PASS_P1"]
    if mode == "NO_P23":
        return [k for k in all_keys if k not in ("PASS_P2", "PASS_P3")]
    if mode == "NO_P4":
        return [k for k in all_keys if k != "PASS_P4"]
    if mode == "NO_P6":
        return [k for k in all_keys if k != "PASS_P6"]

    if mode == "ONLY_P1":
        return ["PASS_P1"]
    if mode == "ONLY_P23":
        return ["PASS_P2", "PASS_P3"]
    if mode == "ONLY_P4":
        return ["PASS_P4"]
    if mode == "ONLY_P6":
        return ["PASS_P6"]

    raise ValueError(f"Unknown mode: {mode}")


def _grid_from_arg(s: str, cast=float) -> List[Any]:
    """
    Parse comma-separated list. Example: "100,200,400"
    """
    xs = []
    for t in (s or "").split(","):
        t = t.strip()
        if not t:
            continue
        xs.append(cast(t))
    return xs


def _verify_one(chain: str, emd_bins: int, num_rand: int, seed: int,
                eps2: float, eps3: float, eps4: float, rho5: float, k6: float) -> Dict[str, Any]:
    res = verify_chain(chain, emd_bins=emd_bins, num_random=num_rand, seed=seed)
    passes = apply_strict(res, eps2, eps3, eps4, rho5, k6)
    # normalize to bool
    for k, v in passes.items():
        res[k] = bool(v)
    return res


def _rate(bools: List[bool]) -> float:
    if not bools:
        return float("nan")
    return sum(1 for b in bools if b) / len(bools)


def main():
    ap = argparse.ArgumentParser(description="Verify-parameter sensitivity: emd_bins / num_rand / seed (+ val_holdout stratify if metadata exists)")
    ap.add_argument("--clean", nargs="+", required=True,
                    help="Clean chain dirs, or glob patterns, or root dir(s) containing chains. Example: results/victim_c10_*")
    ap.add_argument("--attack", nargs="+", required=True,
                    help="Attack chain dirs, or glob patterns, or root dir(s) containing chains. Example: results/attack_*")

    # verify params grid
    ap.add_argument("--emd-bins", type=str, default="200",
                    help="Comma-separated list. Example: 50,100,200,400")
    ap.add_argument("--num-rand", type=str, default="20",
                    help="Comma-separated list. Example: 5,10,20,50")
    ap.add_argument("--seed", type=str, default="0",
                    help="Comma-separated list. Example: 0,1,2,3")

    # thresholds (strict)
    ap.add_argument("--eps2", type=float, default=0.02)
    ap.add_argument("--eps3", type=float, default=0.03)
    ap.add_argument("--eps4", type=float, default=0.25)
    ap.add_argument("--rho5", type=float, default=0.6)
    ap.add_argument("--k6", type=float, default=3.0)

    # decision mode (predicate grouping)
    ap.add_argument("--mode", type=str, default="ALL",
                    choices=["ALL", "NO_P1", "NO_P23", "NO_P4", "NO_P6", "ONLY_P1", "ONLY_P23", "ONLY_P4", "ONLY_P6"],
                    help="Which predicate group to enable when computing pass/fail -> TPR/FPR")

    # stratify by val_holdout if present in metadata
    ap.add_argument("--stratify-val-holdout", action="store_true",
                    help="If set, additionally output TPR/FPR grouped by metadata['val_holdout'] (if exists).")

    # output
    ap.add_argument("--csv", type=str, default="", help="Append rows to CSV")
    ap.add_argument("--save", type=str, default="", help="Save JSON to path (or directory)")

    args = ap.parse_args()

    clean_chains = _expand_chains(args.clean)
    attack_chains = _expand_chains(args.attack)
    if not clean_chains:
        raise SystemExit("No clean chains found. Check --clean paths/globs.")
    if not attack_chains:
        raise SystemExit("No attack chains found. Check --attack paths/globs.")

    emd_bins_list = _grid_from_arg(args.emd_bins, int) or [200]
    num_rand_list = _grid_from_arg(args.num_rand, int) or [20]
    seed_list = _grid_from_arg(args.seed, int) or [0]

    enabled = _mode_enabled(args.mode)

    # pre-read metadata (for stratify)
    clean_meta = {c: _read_meta(c) for c in clean_chains}
    attack_meta = {c: _read_meta(c) for c in attack_chains}

    def _holdout_of(meta: Dict[str, Any]) -> str:
        if "val_holdout" in meta:
            return str(meta["val_holdout"])
        if "val_holdout_frac" in meta:
            return str(meta["val_holdout_frac"])
        return "NA"

    # output containers
    rows = []
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

    print(f"[sens] clean_chains={len(clean_chains)} attack_chains={len(attack_chains)} mode={args.mode} enabled={enabled}", flush=True)
    print(f"[sens] grid: emd_bins={emd_bins_list} num_rand={num_rand_list} seed={seed_list}", flush=True)

    for emd_bins in emd_bins_list:
        for num_rand in num_rand_list:
            for seed in seed_list:
                # verify all chains under this verify-param setting
                clean_pass = []
                attack_pass = []

                # optional stratification bucket
                clean_bucket: Dict[str, List[bool]] = {}
                attack_bucket: Dict[str, List[bool]] = {}

                for c in clean_chains:
                    res = _verify_one(c, emd_bins, num_rand, seed, args.eps2, args.eps3, args.eps4, args.rho5, args.k6)
                    ok = _decision_from_passes(res, enabled)
                    clean_pass.append(ok)
                    if args.stratify_val_holdout:
                        h = _holdout_of(clean_meta.get(c, {}))
                        clean_bucket.setdefault(h, []).append(ok)

                for a in attack_chains:
                    res = _verify_one(a, emd_bins, num_rand, seed, args.eps2, args.eps3, args.eps4, args.rho5, args.k6)
                    ok = _decision_from_passes(res, enabled)
                    attack_pass.append(ok)
                    if args.stratify_val_holdout:
                        h = _holdout_of(attack_meta.get(a, {}))
                        attack_bucket.setdefault(h, []).append(ok)

                tpr = _rate(clean_pass)   # clean pass rate
                fpr = _rate(attack_pass)  # attack pass rate (lower better)

                row = {
                    "timestamp": ts,
                    "mode": args.mode,
                    "enabled": ",".join(enabled),
                    "emd_bins": emd_bins,
                    "num_rand": num_rand,
                    "seed": seed,
                    "eps2": args.eps2, "eps3": args.eps3, "eps4": args.eps4, "rho5": args.rho5, "k6": args.k6,
                    "n_clean": len(clean_chains),
                    "n_attack": len(attack_chains),
                    "TPR": tpr,
                    "FPR": fpr,
                }
                rows.append(row)

                print(f"[sens] emd_bins={emd_bins:4d} num_rand={num_rand:3d} seed={seed:3d}  ->  TPR={tpr:.3f}  FPR={fpr:.3f}", flush=True)

                if args.stratify_val_holdout:
                    # print grouped summary
                    keys = sorted(set(clean_bucket.keys()) | set(attack_bucket.keys()))
                    for h in keys:
                        tpr_h = _rate(clean_bucket.get(h, []))
                        fpr_h = _rate(attack_bucket.get(h, []))
                        print(f"        [holdout={h}] TPR={tpr_h:.3f} FPR={fpr_h:.3f} (n_clean={len(clean_bucket.get(h, []))}, n_attack={len(attack_bucket.get(h, []))})", flush=True)

    # CSV append
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        write_header = not os.path.exists(args.csv)
        fieldnames = [
            "timestamp","mode","enabled","emd_bins","num_rand","seed",
            "eps2","eps3","eps4","rho5","k6",
            "n_clean","n_attack","TPR","FPR"
        ]
        with open(args.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"[sens] appended CSV -> {args.csv}", flush=True)

    # JSON save
    if args.save:
        save_path = args.save
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "sensitivity_verify_params.json")
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        payload = {
            "meta": {
                "timestamp": ts,
                "mode": args.mode,
                "enabled": enabled,
                "verify_grid": {
                    "emd_bins": emd_bins_list,
                    "num_rand": num_rand_list,
                    "seed": seed_list,
                },
                "thresholds": {
                    "eps2": args.eps2, "eps3": args.eps3, "eps4": args.eps4, "rho5": args.rho5, "k6": args.k6
                },
                "n_clean": len(clean_chains),
                "n_attack": len(attack_chains),
                "clean_chains": clean_chains,
                "attack_chains": attack_chains,
                "stratify_val_holdout": bool(args.stratify_val_holdout),
            },
            "rows": rows,
        }
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[sens] saved JSON -> {save_path}", flush=True)


if __name__ == "__main__":
    main()
