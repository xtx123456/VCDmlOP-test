# scripts/ablation_groups_tpr_fpr.py
import argparse, os, json, time, csv, glob
from typing import List, Dict, Tuple

from pot_core.verify import verify_chain, apply_strict


# 你规定的分组（P5不参与）
GROUPS = {
    "G1_P1": ["P1"],
    "G23_P2P3": ["P2", "P3"],
    "G4_P4": ["P4"],
    "G6_P6": ["P6"],
}

# 你要的模式：ALL / DROP / ONLY
MODES = [
    ("ALL",       ["G1_P1", "G23_P2P3", "G4_P4", "G6_P6"]),
    ("DROP_P1",   ["G23_P2P3", "G4_P4", "G6_P6"]),
    ("DROP_P23",  ["G1_P1", "G4_P4", "G6_P6"]),
    ("DROP_P4",   ["G1_P1", "G23_P2P3", "G6_P6"]),
    ("DROP_P6",   ["G1_P1", "G23_P2P3", "G4_P4"]),
    ("ONLY_P1",   ["G1_P1"]),
    ("ONLY_P23",  ["G23_P2P3"]),
    ("ONLY_P4",   ["G4_P4"]),
    ("ONLY_P6",   ["G6_P6"]),
]


def _expand_paths(items: List[str]) -> List[str]:
    """
    支持：
      - 直接给目录
      - glob：results/*_c10_* 这种
      - 文本文件：每行一个目录（以 .txt 结尾）
    """
    out = []
    for it in items:
        if it.endswith(".txt") and os.path.isfile(it):
            with open(it, "r") as f:
                for line in f:
                    p = line.strip()
                    if p:
                        out.append(p)
            continue

        # glob
        if any(ch in it for ch in ["*", "?", "["]):
            out.extend(glob.glob(it))
        else:
            out.append(it)

    # 去重 + 只保留存在的目录
    uniq = []
    seen = set()
    for p in out:
        rp = os.path.normpath(p)
        if rp in seen:
            continue
        seen.add(rp)
        if os.path.isdir(rp):
            uniq.append(rp)
    return uniq


def _read_meta(chain_dir: str) -> Dict[str, str]:
    meta_p = os.path.join(chain_dir, "metadata.json")
    out = {"dataset": "", "arch": ""}
    if os.path.isfile(meta_p):
        try:
            with open(meta_p, "r") as f:
                m = json.load(f)
            out["dataset"] = str(m.get("dataset", "")).upper()
            out["arch"] = str(m.get("arch", "")).strip()
        except Exception:
            pass
    return out


def _mode_enabled_preds(mode_groups: List[str]) -> List[str]:
    """把启用的 group 展开成谓词列表（P1/P2/P3/P4/P6）。"""
    preds = []
    for g in mode_groups:
        preds.extend(GROUPS[g])
    return preds


def _overall_pass(pass_flags: Dict[str, bool], enabled_preds: List[str]) -> bool:
    """enabled_preds 中每个谓词都必须 PASS 才算该模式通过。"""
    for p in enabled_preds:
        if not pass_flags.get(f"PASS_{p}", False):
            return False
    return True


def eval_chains(
    chains: List[str],
    emd_bins: int,
    num_rand: int,
    seed: int,
    eps2: float, eps3: float, eps4: float, rho5: float, k6: float,
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    返回：
      - per_chain：每条链的 PASS_Pi + 元信息
      - summary：暂时不在这里算（留给主流程按 mode 算）
    """
    per_chain = []
    for c in chains:
        res = verify_chain(c, emd_bins=emd_bins, num_random=num_rand, seed=seed)
        pf = apply_strict(res, eps2, eps3, eps4, rho5, k6)  # PASS_P1..PASS_P6
        meta = _read_meta(c)
        per_chain.append({
            "chain": c,
            "dataset": meta["dataset"],
            "arch": meta["arch"],
            **{k: bool(v) for k, v in pf.items()},  # PASS_P1..PASS_P6
            # 可选：也把原始指标放进去（方便你后面写论文/画图）
            "metrics": {k: res.get(k) for k in [
                "P1_rho_val_acc","P2_max_EMD_consecutive","P3_max_EMD_init_vs_GMM",
                "P4_max_PCA_ratio_init","P5_rho_neg_weight_distance","P6_init_final_distance",
                "P6_random_mean","P6_random_std","P6_is_small_vs_random","EPOCHS"
            ]},
        })
    return per_chain, {}


def rate_pass(per_chain: List[Dict], enabled_preds: List[str]) -> float:
    if not per_chain:
        return float("nan")
    ok = 0
    for item in per_chain:
        if _overall_pass(item, enabled_preds):
            ok += 1
    return ok / len(per_chain)


def main():
    ap = argparse.ArgumentParser(
        description="Grouped predicate ablation: report TPR (clean pass rate) & FPR (attack pass rate)"
    )
    ap.add_argument("--clean", nargs="+", required=True,
                    help="Clean chain dirs (support glob) or a .txt list file")
    ap.add_argument("--attack", nargs="+", required=True,
                    help="Attack chain dirs (support glob) or a .txt list file")

    ap.add_argument("--emd-bins", type=int, default=200)
    ap.add_argument("--num-rand", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)

    # strict thresholds（沿用 compare/verify 的默认值）
    ap.add_argument("--eps2", type=float, default=0.02)
    ap.add_argument("--eps3", type=float, default=0.03)
    ap.add_argument("--eps4", type=float, default=0.25)
    ap.add_argument("--rho5", type=float, default=0.6)
    ap.add_argument("--k6", type=float, default=3.0)

    ap.add_argument("--save", type=str, help="Save JSON to this path (or directory)")
    ap.add_argument("--csv", type=str, help="Append rows to CSV (one row per mode)")
    args = ap.parse_args()

    clean_chains = _expand_paths(args.clean)
    attack_chains = _expand_paths(args.attack)

    if not clean_chains:
        raise SystemExit("[error] no valid clean chains found")
    if not attack_chains:
        raise SystemExit("[error] no valid attack chains found")

    print(f"[ablation-groups] clean chains:  {len(clean_chains)}")
    print(f"[ablation-groups] attack chains: {len(attack_chains)}")
    print("[ablation-groups] verifying clean ...", flush=True)
    clean_items, _ = eval_chains(
        clean_chains, args.emd_bins, args.num_rand, args.seed,
        args.eps2, args.eps3, args.eps4, args.rho5, args.k6,
    )
    print("[ablation-groups] verifying attack ...", flush=True)
    attack_items, _ = eval_chains(
        attack_chains, args.emd_bins, args.num_rand, args.seed,
        args.eps2, args.eps3, args.eps4, args.rho5, args.k6,
    )

    # 标题（如果 clean/attack 里元信息一致，会更好看；不一致也没关系）
    meta0 = _read_meta(clean_items[0]["chain"])
    title = "A1) Grouped Predicate Ablation: TPR/FPR"
    if meta0["dataset"] or meta0["arch"]:
        title += f" ({meta0['dataset'] or '?'} | {meta0['arch'] or '?'})"

    print("\n" + title)
    print("-" * len(title))
    print(f"{'Mode':12s} | {'Enabled groups':30s} | {'Enabled preds':16s} | {'TPR(clean)':>10s} | {'FPR(attack)':>10s}")
    print("-" * 92)

    rows = []
    for mode_name, enabled_groups in MODES:
        enabled_preds = _mode_enabled_preds(enabled_groups)
        tpr = rate_pass(clean_items, enabled_preds)
        fpr = rate_pass(attack_items, enabled_preds)
        rows.append({
            "mode": mode_name,
            "enabled_groups": ",".join(enabled_groups),
            "enabled_preds": ",".join(enabled_preds),
            "TPR_clean": tpr,
            "FPR_attack": fpr,
        })
        print(f"{mode_name:12s} | {','.join(enabled_groups):30s} | {','.join(enabled_preds):16s} | {tpr:10.4f} | {fpr:10.4f}")

    print("-" * 92)
    print(f"Params: emd_bins={args.emd_bins}, num_rand={args.num_rand}, seed={args.seed} | "
          f"thr(eps2={args.eps2}, eps3={args.eps3}, eps4={args.eps4}, rho5={args.rho5}, k6={args.k6})")

    # 保存 JSON
    save_path = args.save
    if save_path:
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "ablation_groups_tpr_fpr.json")
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        payload = {
            "meta": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                "clean_chains": clean_chains,
                "attack_chains": attack_chains,
                "emd_bins": args.emd_bins,
                "num_rand": args.num_rand,
                "seed": args.seed,
                "thresholds": {
                    "eps2": args.eps2, "eps3": args.eps3, "eps4": args.eps4, "rho5": args.rho5, "k6": args.k6
                },
                "groups": GROUPS,
                "modes": MODES,
            },
            "rows": rows,
            "per_chain": {
                "clean": clean_items,
                "attack": attack_items
            }
        }
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[ablation-groups] saved -> {save_path}")

    # 追加 CSV（每种 mode 一行）
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        write_header = not os.path.exists(args.csv)
        fieldnames = [
            "timestamp", "mode", "enabled_groups", "enabled_preds",
            "TPR_clean", "FPR_attack",
            "emd_bins", "num_rand", "seed",
            "eps2", "eps3", "eps4", "rho5", "k6",
            "n_clean", "n_attack",
        ]
        with open(args.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
            for r in rows:
                w.writerow({
                    "timestamp": ts,
                    **r,
                    "emd_bins": args.emd_bins,
                    "num_rand": args.num_rand,
                    "seed": args.seed,
                    "eps2": args.eps2, "eps3": args.eps3, "eps4": args.eps4, "rho5": args.rho5, "k6": args.k6,
                    "n_clean": len(clean_items),
                    "n_attack": len(attack_items),
                })
        print(f"[ablation-groups] appended CSV -> {args.csv}")


if __name__ == "__main__":
    main()
