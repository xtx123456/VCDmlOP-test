import os
import json
import glob
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats  # <--- 新增：用于计算置信区间

def parse_args():
    parser = argparse.ArgumentParser(description="Plot analysis results.")
    parser.add_argument("--dataset", type=str, default="c10", help="Dataset name (e.g., c10, c100, svhn)")
    parser.add_argument("--arch", type=str, default="resnet18", help="Architecture name (e.g., lenet, resnet18)")
    parser.add_argument("--base_dir", type=str, default="/root/autodl-tmp/attack-test/analysis_results", help="Base directory for results")
    return parser.parse_args()

def plot_results():
    args = parse_args()
    
    base_dir = args.base_dir
    dataset = args.dataset
    arch = args.arch

    # 定义映射
    metrics_map = {
        'p1': {'key': 'P1_rho_val_acc',           'title': 'P1: Rho Val Acc'},
        'p2': {'key': 'P2_max_EMD_consecutive',   'title': 'P2: Max EMD (Consecutive)'},
        'p3': {'key': 'P3_max_EMD_init_vs_GMM',   'title': 'P3: Max EMD (Init vs GMM)'},
        'p4': {'key': 'P4_max_PCA_ratio_init',    'title': 'P4: Max PCA Ratio (Init)'},
        'p5': {'key': 'P5_rho_neg_weight_distance','title': 'P5: Rho Neg Weight Dist'},
        'p6': {'key': 'P6_init_final_distance',   'title': 'P6: Init-Final Distance'}
    }
    
    clean_filename = f"metrics_{dataset}_{arch}_clean.json"
    clean_file_path = os.path.join(base_dir, clean_filename)
    attack_pattern = f"metrics_{dataset}_{arch}_beta_*.json"
    attack_file_pattern = os.path.join(base_dir, attack_pattern)
    output_dir = os.path.join(base_dir, "plots_output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"========================================")
    print(f"Working Directory: {base_dir}")
    print(f"Target Dataset   : {dataset}")
    print(f"Target Arch      : {arch}")
    print(f"========================================")

    # 1. 读取 Clean 数据
    clean_values = {}
    if os.path.exists(clean_file_path):
        with open(clean_file_path, 'r') as f:
            data = json.load(f)
            if 'metrics' in data:
                metrics_data = data['metrics']
                for m_code, m_info in metrics_map.items():
                    real_key = m_info['key']
                    if real_key in metrics_data:
                        clean_values[m_code] = metrics_data[real_key]

    # 2. 读取 Attack 数据
    data_by_beta = {}
    attack_files = glob.glob(attack_file_pattern)
    
    if not attack_files:
        print(f"[ERROR] No files found for {dataset}-{arch}")
        return

    for filepath in attack_files:
        match = re.search(r'beta_([0-9\.]+)\.json', filepath)
        if not match: continue
        beta_val = float(match.group(1))
        
        with open(filepath, 'r') as f:
            content = json.load(f)
            
        if beta_val not in data_by_beta:
            data_by_beta[beta_val] = {k: [] for k in metrics_map.keys()}
            
        for attack_key, attack_content in content.items():
            if isinstance(attack_content, dict) and 'metrics' in attack_content:
                m_data = attack_content['metrics']
                for m_code, m_info in metrics_map.items():
                    real_key = m_info['key']
                    if real_key in m_data:
                        data_by_beta[beta_val][m_code].append(m_data[real_key])

    sorted_betas = sorted(data_by_beta.keys())

    # 3. 绘图配置
    plt.rcParams.update({'font.size': 12})

    for m_code, m_info in metrics_map.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        
        means = []
        cis = [] # 存放置信区间的一半宽度 (Confidence Interval Error)
        valid_betas = []
        
        for beta in sorted_betas:
            vals = data_by_beta[beta][m_code]
            n = len(vals)
            if n > 0:
                mean_val = np.mean(vals)
                
                # === 计算 95% 置信区间 ===
                if n > 1:
                    # 计算标准误 (Standard Error)
                    sem = stats.sem(vals)
                    # 计算 95% CI 的半宽 (margin of error)
                    # stats.t.ppf((1 + confidence) / 2., n-1)
                    confidence = 0.95
                    h = sem * stats.t.ppf((1 + confidence) / 2., n-1)
                else:
                    h = 0.0 # 只有一个数据点，没有误差范围
                
                means.append(mean_val)
                cis.append(h)
                valid_betas.append(beta)
        
        if not valid_betas:
            plt.close()
            continue

        # 画 Clean 线
        if m_code in clean_values:
            clean_val = clean_values[m_code]
            ax.axhline(y=clean_val, color='#D35400', linestyle='--', linewidth=2, label='Clean Checkpoints')
        
        # 画 Attack 线 (使用置信区间作为 error bar)
        ax.errorbar(
            valid_betas, means, yerr=cis, 
            label='Launching CFP (Mean ± 95% CI)', # 更新图例说明
            color='#16A085',
            linestyle='-', linewidth=2, marker='o', markersize=8, capsize=5
        )

        ax.set_xscale('log')
        ax.set_xlabel('Beta (log scale)', fontweight='bold')
        ax.set_ylabel(m_info['title'], fontweight='bold')
        ax.set_title(f"{dataset}-{arch}: {m_code.upper()}", fontsize=14)
        
        ax.set_xticks(valid_betas)
        ax.set_xticklabels([str(b) for b in valid_betas])
        
        # === 关键修改：加密网格 ===
        ax.minorticks_on() # 开启次刻度
        # 主网格 (Major Grid) - 颜色稍深
        ax.grid(which='major', linestyle='--', linewidth='0.7', color='gray', alpha=0.6)
        # 次网格 (Minor Grid) - 颜色更浅，更密
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.3)
        
        ax.legend()
        
        save_filename = f"plot_{dataset}_{arch}_{m_code}.png"
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {save_filename}")

if __name__ == "__main__":
    plot_results()