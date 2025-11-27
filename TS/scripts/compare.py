# scripts/compare.py
import argparse
import os
import json
import csv
import time

from pot_core.verify import verify_chain, apply_strict

def _read_meta(chain_dir: str):
    meta_p = os.path.join(chain_dir, 'metadata.json')
    out = {'dataset': '', 'arch': ''}
    try:
        with open(meta_p, 'r') as f:
            m = json.load(f)
        out['dataset'] = str(m.get('dataset', '')).upper()
        out['arch']    = str(m.get('arch',    '')).strip()
    except Exception:
        pass
    return out

def main():
    ap = argparse.ArgumentParser(description='Table-3 style comparison (clean vs attack)')
    ap.add_argument('--clean', type=str, required=True, help='Directory of clean chain')
    ap.add_argument('--attack', type=str, required=True, help='Directory of attack chain')
    ap.add_argument('--emd-bins', type=int, default=200)
    ap.add_argument('--num-rand', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--csv', type=str, help='Append comparison rows to this CSV')
    ap.add_argument('--save', type=str, help='Save JSON with meta + results to this path')
    ap.add_argument('--strict', action='store_true', help='Also compute PASS/FAIL with thresholds')
    ap.add_argument('--eps2', type=float, default=0.02)
    ap.add_argument('--eps3', type=float, default=0.03)
    ap.add_argument('--eps4', type=float, default=0.25)
    ap.add_argument('--rho5', type=float, default=0.6)
    ap.add_argument('--k6', type=float, default=3.0)
    args = ap.parse_args()

    print("[compare] start", flush=True)

    # 如果 --save 是目录，自动补成 <dir>/compare.json，避免写不进去
    save_path = args.save
    if save_path:
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "compare.json")
        # 若父目录不存在则创建
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    # Run verification for both chains
    print("[compare] verifying clean...", flush=True)
    res_clean  = verify_chain(args.clean,  emd_bins=args.emd_bins, num_random=args.num_rand, seed=args.seed)
    print("[compare] verifying attack...", flush=True)
    res_attack = verify_chain(args.attack, emd_bins=args.emd_bins, num_random=args.num_rand, seed=args.seed)

    # Title with dataset & arch (if available)
    m_clean  = _read_meta(args.clean)
    m_attack = _read_meta(args.attack)
    title = "Table-3 Style Comparison (Clean vs Attack)"
    suffix = []
    if m_clean['dataset'] or m_clean['arch']:
        suffix.append(f"clean={m_clean['dataset'] or '?'}|{m_clean['arch'] or '?'}")
    if m_attack['dataset'] or m_attack['arch']:
        suffix.append(f"attack={m_attack['dataset'] or '?'}|{m_attack['arch'] or '?'}")
    if suffix:
        title = f"{title} ({', '.join(suffix)})"

    core_keys = [
        'rho_val_acc',
        'max_EMD_consecutive',
        'max_EMD_init_vs_GMM',
        'max_PCA_ratio_init',
        'rho_neg_weight_distance',
        'init_final_distance',
    ]
    extra_keys = ['random_mean','random_std','init_final_distance_is_small_vs_random','EPOCHS']

    print("\n" + title, flush=True)
    print("-" * max(41, len(title)), flush=True)
    print(f"{'Metric':35s} | {'Clean':>12s} | {'Attack':>12s}", flush=True)
    print('-' * 68, flush=True)
    for k in core_keys:
        print(f"{k:35s} | {res_clean[k]:12.6f} | {res_attack[k]:12.6f}", flush=True)
    print('-' * 68, flush=True)
    for k in extra_keys:
        cv, av = res_clean[k], res_attack[k]
        if isinstance(cv, bool) or isinstance(av, bool):
            print(f"{k:35s} | {str(cv):>12s} | {str(av):>12s}", flush=True)
        else:
            print(f"{k:35s} | {cv:12.6f} | {av:12.6f}", flush=True)

    # Strict PASS/FAIL
    if args.strict:
        pc = apply_strict(res_clean,  args.eps2, args.eps3, args.eps4, args.rho5, args.k6)
        pa = apply_strict(res_attack, args.eps2, args.eps3, args.eps4, args.rho5, args.k6)
        print("\nStrict PASS/FAIL (thresholded)", flush=True)
        print("-" * 29, flush=True)
        print(f"{'Property':35s} | {'Clean':>6s} | {'Attack':>6s}", flush=True)
        print('-' * 54, flush=True)
        for prop in ['PASS_P1','PASS_P2','PASS_P3','PASS_P4','PASS_P5','PASS_P6']:
            print(f"{prop:35s} | {str(pc[prop]):>6s} | {str(pa[prop]):>6s}", flush=True)
        res_clean.update(pc); res_attack.update(pa)
        print(f"THRESHOLDS: eps2={args.eps2}, eps3={args.eps3}, eps4={args.eps4}, rho5={args.rho5}, k6={args.k6}", flush=True)

    # CSV
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or '.', exist_ok=True)
        write_header = not os.path.exists(args.csv)
        with open(args.csv, 'a', newline='') as f:
            fieldnames = [
                'timestamp','clean_chain','attack_chain','dataset_clean','arch_clean','dataset_attack','arch_attack',
                'emd_bins','num_rand','seed','strict','eps2','eps3','eps4','rho5','k6','metric','clean','attack'
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header: w.writeheader()
            for k in core_keys:
                w.writerow({
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()),
                    'clean_chain': args.clean,
                    'attack_chain': args.attack,
                    'dataset_clean': m_clean['dataset'],
                    'arch_clean': m_clean['arch'],
                    'dataset_attack': m_attack['dataset'],
                    'arch_attack': m_attack['arch'],
                    'emd_bins': args.emd_bins,
                    'num_rand': args.num_rand,
                    'seed': args.seed,
                    'strict': bool(args.strict),
                    'eps2': args.eps2, 'eps3': args.eps3, 'eps4': args.eps4, 'rho5': args.rho5, 'k6': args.k6,
                    'metric': k,
                    'clean': res_clean[k],
                    'attack': res_attack[k],
                })
        print(f"[info] Appended CSV -> {args.csv}", flush=True)

    # JSON
    if save_path:
        payload = {
            'meta': {
                'clean_chain': args.clean,
                'attack_chain': args.attack,
                'dataset_clean': m_clean['dataset'],
                'arch_clean': m_clean['arch'],
                'dataset_attack': m_attack['dataset'],
                'arch_attack': m_attack['arch'],
                'emd_bins': args.emd_bins,
                'num_rand': args.num_rand,
                'seed': args.seed,
                'strict': bool(args.strict),
                'thresholds': {
                    'eps2': args.eps2, 'eps3': args.eps3, 'eps4': args.eps4, 'rho5': args.rho5, 'k6': args.k6
                },
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()),
            },
            'results': {'clean': res_clean, 'attack': res_attack}
        }
        with open(save_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"[info] Wrote JSON -> {save_path}", flush=True)

    print("[compare] done", flush=True)

if __name__ == '__main__':
    main()
