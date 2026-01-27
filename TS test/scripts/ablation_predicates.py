# scripts/ablation_predicates.py
import argparse
import os
import json
from typing import Dict, List

from pot_core.verify import verify_chain, apply_strict


P_PROPS = ["PASS_P1", "PASS_P2", "PASS_P3", "PASS_P4", "PASS_P5", "PASS_P6"]
P_METRICS = [
    "P1_rho_val_acc",
    "P2_max_EMD_consecutive",
    "P3_max_EMD_init_vs_GMM",
    "P4_max_PCA_ratio_init",
    "P5_rho_neg_weight_distance",
    "P6_init_final_distance",
]

def _read_meta(chain_dir: str) -> Dict[str, str]:
    meta_p = os.path.join(chain_dir, "metadata.json")
    out = {"dataset": "", "arch": "", "metric_name": "val_acc"}
    if os.path.isfile(meta_p):
        try:
            with open(meta_p, "r") as f:
                m = json.load(f)
            out["dataset"] = str(m.get("dataset", "")).upper()
            out["arch"] = str(m.get("arch", "")).strip()
            out["metric_name"] = str(m.get("metric_name", "val_acc"))
        except Exception:
            pass
    return out

def _overall_pass(passes: Dict[str, bool], enabled_props: List[str]) -> bool:
    # “启用的谓词全部通过” => overall PASS
    return all(bool(passes.get(k, False)) for k in enabled_props)

def _run_one_mode(strict_passes: Dict[str, bool], mode: str):
    """
    mode:
      - "ALL"
      - "DROP:PASS_Pi"
      - "ONLY:PASS_Pi"
    """
    enabled = list(P_PROPS)
    label = ""
    if mode == "ALL":
        label = "ALL (P1..P6)"
    elif mode.startswith("DROP:"):
        drop = mode.split(":", 1)[1]
        enabled = [p for p in enabled if p != drop]
        label = f"DROP {drop.replace('PASS_', '')}"
    elif mode.startswith("ONLY:"):
        only = mode.split(":", 1)[1]
        enabled = [only]
        label = f"ONLY {only.replace('PASS_', '')}"
    else:
        raise ValueError(f"Unknown mode {mode}")

    return label, enabled, _overall_pass(strict_passes, enabled)

def main():
    ap = argparse.ArgumentParser(description="A1) Ablation on predicates P1~P6 (ALL / DROP Pi / ONLY Pi)")
    ap.add_argument("--chain", type=str, required=True, help="checkpoint chain directory")
    ap.add_argument("--emd-bins", type=int, default=200)
    ap.add_argument("--num-rand", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)

    # strict thresholds
    ap.add_argument("--eps2", type=float, default=0.02)
    ap.add_argument("--eps3", type=float, default=0.03)
    ap.add_argument("--eps4", type=float, default=0.25)
    ap.add_argument("--rho5", type=float, default=0.6)
    ap.add_argument("--k6", type=float, default=3.0)

    ap.add_argument("--save", type=str, help="Save JSON results to this path")
    args = ap.parse_args()

    print("[ablation] verifying chain once ...", flush=True)
    res = verify_chain(args.chain, emd_bins=args.emd_bins, num_random=args.num_rand, seed=args.seed)

    strict = apply_strict(res, args.eps2, args.eps3, args.eps4, args.rho5, args.k6)
    # strict contains PASS_P1..PASS_P6 (plus maybe other keys)
    for k in P_PROPS:
        if k not in strict:
            raise RuntimeError(f"apply_strict did not return {k}. Got keys={list(strict.keys())}")

    meta = _read_meta(args.chain)
    title = f"A1) Predicate Ablation ({meta['dataset'] or '?'} | {meta['arch'] or '?'})"
    print("\n" + title)
    print("-" * len(title))

    # print metric values
    print("\nMetrics (computed once):")
    for k in P_METRICS:
        print(f"  {k}: {res[k]}")

    # print strict pass for each predicate
    print("\nStrict per-predicate PASS:")
    for k in P_PROPS:
        print(f"  {k}: {bool(strict[k])}")

    modes = ["ALL"]
    for p in P_PROPS:
        modes.append(f"DROP:{p}")
    for p in P_PROPS:
        modes.append(f"ONLY:{p}")

    print("\nAblation Overall Decision (enabled predicates must all PASS):")
    print(f"{'Mode':18s} | {'Enabled':28s} | {'OVERALL_PASS':12s}")
    print("-" * 70)
    rows = []
    for mode in modes:
        label, enabled, overall = _run_one_mode(strict, mode)
        enabled_str = ",".join([e.replace("PASS_", "") for e in enabled])
        print(f"{label:18s} | {enabled_str:28s} | {str(overall):>12s}")
        rows.append({"mode": label, "enabled": enabled_str, "overall_pass": bool(overall)})

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        payload = {
            "meta": {
                "chain": args.chain,
                "dataset": meta["dataset"],
                "arch": meta["arch"],
                "emd_bins": args.emd_bins,
                "num_rand": args.num_rand,
                "seed": args.seed,
                "thresholds": {
                    "eps2": args.eps2,
                    "eps3": args.eps3,
                    "eps4": args.eps4,
                    "rho5": args.rho5,
                    "k6": args.k6,
                },
            },
            "metrics": {k: res[k] for k in P_METRICS},
            "strict": {k: bool(strict[k]) for k in P_PROPS},
            "ablation": rows,
        }
        with open(args.save, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[ablation] saved -> {args.save}", flush=True)


if __name__ == "__main__":
    main()
