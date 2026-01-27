# scripts/batch_compare.py
import os
import json
import time
import sys
import argparse

# 尝试导入 compare.py 中的核心验证函数
try:
    from compare import verify_chain
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from compare import verify_chain

def run_batch_analysis(dataset_name, arch_name):
    # ================= 配置区域 =================
    
    # 1. 数据集名称映射 (处理文件夹命名习惯)
    # 逻辑：输入 cifar10 -> 简称 c10 (用于 victim 文件名)
    #       输入 cifar10 -> 全称 cifar10 (用于 backward 路径)
    dataset_map = {
        "cifar10": "c10",
        "cifar100": "c100",
        "mnist": "mnist" # 预留
    }
    
    if dataset_name not in dataset_map:
        print(f"[Error] Unsupported dataset: {dataset_name}")
        return

    short_dataset = dataset_map[dataset_name] # e.g., c10
    full_dataset = dataset_name               # e.g., cifar10
    
    # 2. 生成文件前缀
    file_prefix = f"{short_dataset}_{arch_name}"  # 例如: c10_alexnet, c100_resnet18

    # 3. 动态构建路径
    # Clean Path 示例: /root/autodl-tmp/attack-test/results/victim_c10_alexnet
    clean_chain_path = f"/root/autodl-tmp/attack-test/results/victim_{short_dataset}_{arch_name}"
    
    # Attack Base Path 示例: /root/autodl-tmp/attack-test/results/backward/cifar10
    attack_base_path = f"/root/autodl-tmp/attack-test/results/backward/{full_dataset}"
    
    # 4. 定义变量 (如果不同模型 beta 不同，这里可能需要写个字典来映射)
    betas = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    attack_indices = [1, 2, 3, 4, 5]
    
    # 5. 输出文件保存目录
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 6. 验证参数
    emd_bins = 200
    num_rand = 20
    seed = 0
    # ===========================================

    print(f"[{time.strftime('%H:%M:%S')}] Start Analysis | Dataset: {full_dataset} ({short_dataset}) | Arch: {arch_name}")
    print(f"Prefix: {file_prefix}")

    # --- 1. 计算 Clean Chain 指标 ---
    print(f"Processing Clean Chain: {clean_chain_path}")
    if os.path.exists(clean_chain_path):
        clean_metrics = verify_chain(
            clean_chain_path, 
            emd_bins=emd_bins, 
            num_random=num_rand, 
            seed=seed
        )
        # 保存 Clean 结果
        clean_filename = f"metrics_{file_prefix}_clean.json"
        clean_json_path = os.path.join(output_dir, clean_filename)
        
        with open(clean_json_path, "w") as f:
            json.dump({
                "path": clean_chain_path,
                "dataset": short_dataset,
                "arch": arch_name,
                "metrics": clean_metrics
            }, f, indent=2)
        print(f"  -> Saved to {clean_json_path}")
    else:
        print(f"  [Error] Clean path not found: {clean_chain_path}")

    # --- 2. 批量计算 Attack Chains 指标 ---
    all_attack_results = {}

    for beta in betas:
        # 构造 beta 键名，例如 beta_0.001
        beta_key = f"beta_{beta}" 
        all_attack_results[beta_key] = {}
        
        for idx in attack_indices:
            # 构造路径: .../backward/cifar10/beta_0.001/alexnet/attack1
            # 注意这里使用了 arch_name 变量
            attack_path = os.path.join(
                attack_base_path, 
                beta_key, 
                arch_name, 
                f"attack{idx}"
            )
            
            print(f"Processing Attack Chain ({beta_key}, {arch_name}, id={idx}): {attack_path}")
            
            if os.path.exists(attack_path):
                try:
                    metrics = verify_chain(
                        attack_path, 
                        emd_bins=emd_bins, 
                        num_random=num_rand, 
                        seed=seed
                    )
                    
                    all_attack_results[beta_key][f"attack{idx}"] = {
                        "path": attack_path,
                        "metrics": metrics
                    }
                except Exception as e:
                    print(f"  [Error] Failed to verify {attack_path}: {e}")
                    all_attack_results[beta_key][f"attack{idx}"] = {"error": str(e)}
            else:
                # 很多时候某些 beta 下可能没有跑特定模型，报 Warning 即可
                print(f"  [Warning] Path not found: {attack_path}")
                all_attack_results[beta_key][f"attack{idx}"] = {"error": "Path not found"}

    # --- 3. 保存汇总结果 ---
    all_filename = f"metrics_{file_prefix}_attacks_all.json"
    all_json_path = os.path.join(output_dir, all_filename)
    with open(all_json_path, "w") as f:
        json.dump(all_attack_results, f, indent=2)
    print(f"  -> Saved all attack metrics to {all_json_path}")

    # --- 4. 分别保存每个 Beta 的结果 ---
    for beta_key, data in all_attack_results.items():
        # 过滤掉完全为空的结果（如果某个beta文件夹完全不存在）
        if not data: 
            continue
            
        beta_filename = f"metrics_{file_prefix}_{beta_key}.json"
        p = os.path.join(output_dir, beta_filename)
        with open(p, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  -> Saved {beta_key} metrics to {p}")

    print(f"[{time.strftime('%H:%M:%S')}] Batch Analysis Done for {file_prefix}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Analysis for Model Checkpoints")
    
    # 定义可选参数
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='cifar10', 
        choices=['cifar10', 'cifar100'],
        help='Dataset name (default: cifar10)'
    )
    
    parser.add_argument(
        '--arch', 
        type=str, 
        default='alexnet', 
        choices=['alexnet', 'lenet', 'resnet18', 'vgg16'],
        help='Model architecture (default: alexnet)'
    )

    args = parser.parse_args()
    
    run_batch_analysis(args.dataset, args.arch)