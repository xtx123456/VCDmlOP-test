# scripts/compare.py
import argparse
import os
import json
import csv
import time
from typing import Any, Dict, List, Optional, Tuple

from pot_core.verify import verify_chain, apply_strict


def _safe_json_load(p: str) -> Optional[Any]:
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _read_meta(chain_dir: str) -> Dict[str, str]:
    meta_p = os.path.join(chain_dir, "metadata.json")
    out = {"dataset": "", "arch": "", "metric_name": ""}
    m = _safe_json_load(meta_p)
    if isinstance(m, dict):
        out["dataset"] = str(m.get("dataset", "")).upper()
        out["arch"] = str(m.get("arch", "")).strip()
        out["metric_name"] = str(m.get("metric_name", "")).strip()
    return out


def _extract_series_from_obj(obj: Any) -> Optional[List[float]]:
    """
    尝试从任意 JSON 对象中抽取一个“看起来像指标序列”的 list[float]。
    支持：
      - {"val_accs":[...]} / {"val_acc":[...]} / {"metric_series":[...]} / {"history":{"val_acc":[...]}} 等
      - 直接就是 [0.1, 0.2, ...]
    """
    if obj is None:
        return None

    # 直接 list
    if isinstance(obj, list) and obj and all(isinstance(x, (int, float)) for x in obj):
        return [float(x) for x in obj]

    if not isinstance(obj, dict):
        return None

    # 常见字段候选（按优先级）
    key_candidates = [
        "val_accs",
        "val_acc",
        "val_accuracy",
        "val_acc_list",
        "metric_series",
        "metrics",
        "val_metric",
        "val_metrics",
        "history",
        "log",
        "records",
    ]

    # 1) 直接在顶层找 list[float]
    for k in key_candidates:
        v = obj.get(k, None)
        if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            return [float(x) for x in v]

    # 2) history / metrics 这类可能是 dict 里面再套一层
    for k in ["history", "metrics", "val_metrics", "log"]:
        v = obj.get(k, None)
        if isinstance(v, dict):
            for kk in ["val_acc", "val_accs", "val_accuracy", "val_metric", "val", "acc", "accuracy"]:
                vv = v.get(kk, None)
                if isinstance(vv, list) and vv and all(isinstance(x, (int, float)) for x in vv):
                    return [float(x) for x in vv]

    # 3) records 可能是 list[dict]：每个 epoch 一个
    rec = obj.get("records", None)
    if isinstance(rec, list) and rec and all(isinstance(x, dict) for x in rec):
        # 从每条记录里取一个字段
        for kk in ["val_acc", "val_accuracy", "val_metric", "val", "acc", "accuracy"]:
            xs = []
            ok = True
            for r in rec:
                if kk not in r or not isinstance(r[kk], (int, float)):
                    ok = False
                    break
                xs.append(float(r[kk]))
            if ok and xs:
                return xs

    return None


def _read_series_from_json_files(chain_dir: str) -> Optional[Tuple[str, List[float]]]:
    """
    尝试从 chain_dir 下的一些“可能存在”的 json 文件读取序列。
    返回 (source, series) 其中 source 表示来源文件名。
    """
    candidates = [
        "val_acc.json",
        "val_accs.json",
        "val_metrics.json",
        "metric_series.json",
        "metrics.json",
        "history.json",
        "train_history.json",
        "eval_history.json",
        "log.json",
    ]
    for fn in candidates:
        p = os.path.join(chain_dir, fn)
        if os.path.isfile(p):
            obj = _safe_json_load(p)
            s = _extract_series_from_obj(obj)
            if s is not None and len(s) > 0:
                return (fn, s)
    return None


def _read_series_from_jsonl(chain_dir: str) -> Optional[Tuple[str, List[float]]]:
    """
    读取 log.jsonl / train.jsonl 这类：每行一个 json dict，抽取 val_acc / val_accuracy 等字段形成序列。
    """
    candidates = ["log.jsonl", "train.jsonl", "events.jsonl", "metrics.jsonl"]
    for fn in candidates:
        p = os.path.join(chain_dir, fn)
        if not os.path.isfile(p):
            continue
        xs: List[float] = []
        try:
            with open(p, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    for kk in ["val_acc", "val_accuracy", "val_metric", "val", "acc", "accuracy"]:
                        if kk in obj and isinstance(obj[kk], (int, float)):
                            xs.append(float(obj[kk]))
                            break
            if xs:
                return (fn, xs)
        except Exception:
            pass
    return None


def _read_metric_series(chain_dir: str) -> Optional[Dict[str, Any]]:
    """
    尝试从 metadata.json 或其它文件中读取“验证指标序列”（val acc/val metric）。
    返回:
      {
        "name":  "val_acc" or "val_metric",
        "source": "metadata.json:val_accs" / "metrics.json" / "log.jsonl" ...
        "series": [float, ...]
      }
    """
    meta_p = os.path.join(chain_dir, "metadata.json")
    meta = _safe_json_load(meta_p)
    series = _extract_series_from_obj(meta)
    if series is not None and len(series) > 0:
        # 尽量猜一下字段名
        name = "val_acc"
        if isinstance(meta, dict):
            if "metric_name" in meta and isinstance(meta["metric_name"], str) and meta["metric_name"].strip():
                name = meta["metric_name"].strip()
            else:
                # 若 meta 里明确出现 val_metric 更合理
                if "val_metric" in meta or "val_metrics" in meta:
                    name = "val_metric"
        return {"name": name, "source": "metadata.json", "series": series}

    # json 文件
    r = _read_series_from_json_files(chain_dir)
    if r is not None:
        fn, s = r
        return {"name": "val_acc", "source": fn, "series": s}

    # jsonl 文件
    r2 = _read_series_from_jsonl(chain_dir)
    if r2 is not None:
        fn, s = r2
        return {"name": "val_acc", "source": fn, "series": s}

    return None


def _fmt_float(x: Any) -> str:
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return f"{float(x):.6f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser(description="Table-3 style comparison (clean vs attack)")
    ap.add_argument("--clean", type=str, required=True, help="Directory of clean chain")
    ap.add_argument("--attack", type=str, required=True, help="Directory of attack chain")
    ap.add_argument("--emd-bins", type=int, default=200)
    ap.add_argument("--num-rand", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", type=str, help="Append comparison rows to this CSV")
    ap.add_argument("--save", type=str, help="Save JSON with meta + results (+ optional series) to this path/dir")
    ap.add_argument("--strict", action="store_true", help="Also compute PASS/FAIL with thresholds")
    ap.add_argument("--eps2", type=float, default=0.02)
    ap.add_argument("--eps3", type=float, default=0.03)
    ap.add_argument("--eps4", type=float, default=0.25)
    ap.add_argument("--rho5", type=float, default=0.6)
    ap.add_argument("--k6", type=float, default=3.0)
    args = ap.parse_args()

    print("[compare] start", flush=True)

    # 如果 --save 是目录，自动补成 <dir>/compare.json
    save_path = args.save
    if save_path:
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "compare.json")
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Run verification for both chains
    print("[compare] verifying clean...", flush=True)
    res_clean = verify_chain(args.clean, emd_bins=args.emd_bins, num_random=args.num_rand, seed=args.seed)
    print("[compare] verifying attack...", flush=True)
    res_attack = verify_chain(args.attack, emd_bins=args.emd_bins, num_random=args.num_rand, seed=args.seed)

    # Title with dataset & arch (if available)
    m_clean = _read_meta(args.clean)
    m_attack = _read_meta(args.attack)
    title = "Table-3 Style Comparison (Clean vs Attack)"
    suffix = []
    if m_clean["dataset"] or m_clean["arch"]:
        suffix.append(f"clean={m_clean['dataset'] or '?'}|{m_clean['arch'] or '?'}")
    if m_attack["dataset"] or m_attack["arch"]:
        suffix.append(f"attack={m_attack['dataset'] or '?'}|{m_attack['arch'] or '?'}")
    if suffix:
        title = f"{title} ({', '.join(suffix)})"

    core_keys = [
        "P1_rho_val_acc",
        "P2_max_EMD_consecutive",
        "P3_max_EMD_init_vs_GMM",
        "P4_max_PCA_ratio_init",
        "P5_rho_neg_weight_distance",
        "P6_init_final_distance",
    ]
    extra_keys = ["P6_random_mean", "P6_random_std", "P6_is_small_vs_random", "EPOCHS"]

    print("\n" + title, flush=True)
    print("-" * max(41, len(title)), flush=True)
    print(f"{'Metric':35s} | {'Clean':>12s} | {'Attack':>12s}", flush=True)
    print("-" * 68, flush=True)
    for k in core_keys:
        print(f"{k:35s} | {_fmt_float(res_clean.get(k)):>12s} | {_fmt_float(res_attack.get(k)):>12s}", flush=True)
    print("-" * 68, flush=True)
    for k in extra_keys:
        print(f"{k:35s} | {_fmt_float(res_clean.get(k)):>12s} | {_fmt_float(res_attack.get(k)):>12s}", flush=True)

    # 尝试读取 val acc / metric 序列（如果存在）
    s_clean = _read_metric_series(args.clean)
    s_attack = _read_metric_series(args.attack)

    print("", flush=True)
    print("Metric series (if available)", flush=True)
    print("-" * 28, flush=True)

    def _print_series(tag: str, s: Optional[Dict[str, Any]]):
        if not s:
            print(f"{tag:6s}: (not found)", flush=True)
            return
        series = s["series"]
        name = s.get("name", "val_acc")
        src = s.get("source", "?")
        n = len(series)
        first = series[0]
        last = series[-1]
        print(f"{tag:6s}: name={name}, n={n}, first={first:.6f}, last={last:.6f}, source={src}", flush=True)

    _print_series("clean", s_clean)
    _print_series("attack", s_attack)

    # Strict PASS/FAIL
    if args.strict:
        pc = apply_strict(res_clean, args.eps2, args.eps3, args.eps4, args.rho5, args.k6)
        pa = apply_strict(res_attack, args.eps2, args.eps3, args.eps4, args.rho5, args.k6)
        print("\nStrict PASS/FAIL (thresholded)", flush=True)
        print("-" * 29, flush=True)
        print(f"{'Property':35s} | {'Clean':>6s} | {'Attack':>6s}", flush=True)
        print("-" * 54, flush=True)
        for prop in ["PASS_P1", "PASS_P2", "PASS_P3", "PASS_P4", "PASS_P5", "PASS_P6"]:
            print(f"{prop:35s} | {str(pc.get(prop)):>6s} | {str(pa.get(prop)):>6s}", flush=True)
        res_clean.update(pc)
        res_attack.update(pa)
        print(
            f"THRESHOLDS: eps2={args.eps2}, eps3={args.eps3}, eps4={args.eps4}, rho5={args.rho5}, k6={args.k6}",
            flush=True,
        )

    # CSV
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        write_header = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            fieldnames = [
                "timestamp",
                "clean_chain",
                "attack_chain",
                "dataset_clean",
                "arch_clean",
                "dataset_attack",
                "arch_attack",
                "emd_bins",
                "num_rand",
                "seed",
                "strict",
                "eps2",
                "eps3",
                "eps4",
                "rho5",
                "k6",
                "metric",
                "clean",
                "attack",
            ]
            # 额外：序列信息（只有摘要，不写全序列到 CSV）
            fieldnames += [
                "series_clean_found",
                "series_clean_n",
                "series_clean_first",
                "series_clean_last",
                "series_clean_source",
                "series_attack_found",
                "series_attack_n",
                "series_attack_first",
                "series_attack_last",
                "series_attack_source",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()

            def _series_summary(s: Optional[Dict[str, Any]]) -> Dict[str, Any]:
                if not s:
                    return dict(found=False, n="", first="", last="", source="")
                xs = s["series"]
                return dict(found=True, n=len(xs), first=xs[0], last=xs[-1], source=s.get("source", ""))

            sc = _series_summary(s_clean)
            sa = _series_summary(s_attack)

            for k in core_keys:
                w.writerow(
                    {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                        "clean_chain": args.clean,
                        "attack_chain": args.attack,
                        "dataset_clean": m_clean["dataset"],
                        "arch_clean": m_clean["arch"],
                        "dataset_attack": m_attack["dataset"],
                        "arch_attack": m_attack["arch"],
                        "emd_bins": args.emd_bins,
                        "num_rand": args.num_rand,
                        "seed": args.seed,
                        "strict": bool(args.strict),
                        "eps2": args.eps2,
                        "eps3": args.eps3,
                        "eps4": args.eps4,
                        "rho5": args.rho5,
                        "k6": args.k6,
                        "metric": k,
                        "clean": res_clean.get(k),
                        "attack": res_attack.get(k),
                        "series_clean_found": sc["found"],
                        "series_clean_n": sc["n"],
                        "series_clean_first": sc["first"],
                        "series_clean_last": sc["last"],
                        "series_clean_source": sc["source"],
                        "series_attack_found": sa["found"],
                        "series_attack_n": sa["n"],
                        "series_attack_first": sa["first"],
                        "series_attack_last": sa["last"],
                        "series_attack_source": sa["source"],
                    }
                )
        print(f"[info] Appended CSV -> {args.csv}", flush=True)

    # JSON
    if save_path:
        payload = {
            "meta": {
                "clean_chain": args.clean,
                "attack_chain": args.attack,
                "dataset_clean": m_clean["dataset"],
                "arch_clean": m_clean["arch"],
                "dataset_attack": m_attack["dataset"],
                "arch_attack": m_attack["arch"],
                "emd_bins": args.emd_bins,
                "num_rand": args.num_rand,
                "seed": args.seed,
                "strict": bool(args.strict),
                "thresholds": {
                    "eps2": args.eps2,
                    "eps3": args.eps3,
                    "eps4": args.eps4,
                    "rho5": args.rho5,
                    "k6": args.k6,
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            },
            "results": {"clean": res_clean, "attack": res_attack},
            "series": {
                "clean": s_clean,   # 可能为 None；若存在则包含 name/source/series
                "attack": s_attack,
            },
        }
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[info] Wrote JSON -> {save_path}", flush=True)

    print("[compare] done", flush=True)


if __name__ == "__main__":
    main()
