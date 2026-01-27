# scripts/verify_v2.py
import argparse, json, os, time
from pot_core.verify_v2 import verify_chain, apply_strict  # ← 使用 v2（支持 UNet/VOC）

def main():
    ap = argparse.ArgumentParser(description='PoT verify v2 (no data; supports UNet/VOC2012)')
    ap.add_argument('--chain', type=str, required=True, help='checkpoint chain directory')
    ap.add_argument('--emd-bins', type=int, default=200)
    ap.add_argument('--num-rand', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save', type=str, help='write results+meta as JSON to this path')
    ap.add_argument('--csv', type=str, help='append a results row into this CSV file')
    ap.add_argument('--strict', action='store_true')
    ap.add_argument('--eps2', type=float, default=0.02)
    ap.add_argument('--eps3', type=float, default=0.03)
    ap.add_argument('--eps4', type=float, default=0.25)
    ap.add_argument('--rho5', type=float, default=0.6)
    ap.add_argument('--k6', type=float, default=3.0)
    args = ap.parse_args()

    # 计算 P1–P6
    res = verify_chain(args.chain, emd_bins=args.emd_bins, num_random=args.num_rand, seed=args.seed)

    # 读取 meta 以美化标题（数据集/架构/指标名等）
    title = "PoT Verification Results"
    meta_path = os.path.join(args.chain, 'metadata.json')
    metric_name = 'val_acc'
    extra = []
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, 'r') as f:
                chain_meta = json.load(f)
            ds = chain_meta.get('dataset') or ''
            arch = chain_meta.get('arch') or ''
            metric_name = chain_meta.get('metric_name', metric_name)
            if ds:   extra.append(ds)
            if arch: extra.append(arch)
        except Exception:
            pass
    if extra:
        title = f"{title} ({', '.join(extra)})"

    # 打印
    print("\n" + title)
    print("-" * len(title))
    order = [
        'P1_rho_val_acc',
        'P2_max_EMD_consecutive',
        'P3_max_EMD_init_vs_GMM',
        'P4_max_PCA_ratio_init',
        'P5_rho_neg_weight_distance',
        'P6_init_final_distance',
        'P6_random_mean',
        'P6_random_std',
        'P6_is_small_vs_random',
        'EPOCHS',
    ]
    for k in order:
        print(f"{k}: {res[k]}")

    # 严格阈值
    if args.strict:
        passes = apply_strict(res, args.eps2, args.eps3, args.eps4, args.rho5, args.k6)
        for k, v in passes.items():
            res[k] = bool(v)
            print(f"{k}: {v}")
        print(f"THRESHOLDS: eps2={args.eps2}, eps3={args.eps3}, eps4={args.eps4}, rho5={args.rho5}, k6={args.k6}")

    # 保存 JSON
    if args.save:
        payload = {
            'meta': {
                'chain': args.chain,
                'emd_bins': args.emd_bins,
                'num_rand': args.num_rand,
                'seed': args.seed,
                'strict': bool(args.strict),
                'thresholds': {
                    'eps2': args.eps2, 'eps3': args.eps3, 'eps4': args.eps4, 'rho5': args.rho5, 'k6': args.k6
                },
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()),
            },
            'results': res,
        }
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        with open(args.save, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"[info] saved JSON -> {args.save}")

    # 追加 CSV
    if args.csv:
        import csv
        os.makedirs(os.path.dirname(args.csv) or '.', exist_ok=True)
        write_header = not os.path.exists(args.csv)
        head = ['timestamp', 'chain', 'emd_bins', 'num_rand', 'seed', 'strict',
                'eps2', 'eps3', 'eps4', 'rho5', 'k6'] + order
        with open(args.csv, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=head)
            if write_header:
                w.writeheader()
            row = {
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()),
                'chain': args.chain,
                'emd_bins': args.emd_bins,
                'num_rand': args.num_rand,
                'seed': args.seed,
                'strict': bool(args.strict),
                'eps2': args.eps2, 'eps3': args.eps3, 'eps4': args.eps4, 'rho5': args.rho5, 'k6': args.k6,
            }
            for k in order:
                row[k] = res[k]
            w.writerow(row)
        print(f"[info] appended CSV row -> {args.csv}")

if __name__ == '__main__':
    main()
