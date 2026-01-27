# MAMMON TS for CIFAR-10/100 — ResNet18 / AlexNet

## 快速开始（Quickstart）

> 下面命令默认项目根目录为当前工作目录。若使用 `python -m ...` 方式，请确保当前目录在 `PYTHONPATH` 的最前（或在 `scripts/` 下存在 `__init__.py`）。

### 0) 环境准备（可选）
```bash
pip install -r requirements.txt
```

### 1) 训练一条“干净链”
```bash
# ResNet18 + CIFAR-10
python -m scripts.train \
  --dataset cifar10 \
  --data /path/to/datasets/cifar10 \
  --arch resnet18 \
  --out runs/c10_resnet18_clean \
  --epochs 200 --batch-size 128 --lr 0.1 --verbose

# AlexNet + CIFAR-10（建议默认 lr=0.01）
python -m scripts.train \
  --dataset cifar10 \
  --data /path/to/datasets/cifar10 \
  --arch alexnet \
  --out runs/c10_alexnet_clean \
  --epochs 200 --batch-size 128 --lr 0.01 --verbose
```

> 说明：训练脚本会把 `arch` 与 `dataset` 写入 `metadata.json`，供 verify/攻击自动识别；`epoch_0000.pt` 为 **PoT 初始化**快照。

### 2) 插值攻击（无数据）
支持直接传**链目录**（推荐），从中读取最终 checkpoint 与元数据：
```bash
python -m scripts.attack_interp \
  --victim runs/c10_alexnet_clean \
  --out    runs/c10_alexnet_interp_attack \
  --arch   auto \
  --alpha-start 0.0 --alpha-end 1.0 --alpha-step 0.02 \
  --verbose
```
> `--arch auto`：从受害者链的 `metadata.json['arch']` 自动选择（AlexNet/ResNet）。插值仅对**同形状浮点参数**进行，不匹配的键直接采用 final。

### 3) 规则蒸馏攻击（同分布 AUX 切分）
```bash
python -m scripts.attack_distill \
  --victim  runs/c10_alexnet_clean \
  --data    /path/to/datasets/cifar10 \
  --dataset cifar10 \
  --out     runs/c10_alexnet_rd_attack \
  --arch    auto \
  --epochs  100 --batch-size 128 --lr 0.01 --tau 2.0 --lambda-kd 1.0 \
  --labeled-frac 0.10 --lambda-ce 1.0 \
  --verbose
```
- 该脚本会：
  1. 从 victim 链载入 **teacher**（最终权重）；
  2. 构建同架构 **student** 并进行 **PoT 初始化**；
  3. **保存 `epoch_0000.pt`**（真实 PoT init，用于 P3/P4/P6 基线）；
  4. **重新初始化 student** 后开始蒸馏训练（避免“训练起点=已保存的 init”）；
  5. 每个 epoch 保存 `epoch_XXXX.pt` 并更新元信息。

### 4) 验证一条链（P1–P6，严格判定可选）
```bash
python -m scripts.verify \
  --chain runs/c10_alexnet_rd_attack \
  --emd-bins 200 --num-rand 50 --seed 0 \
  --strict
```

### 5) 对比 干净链 vs 攻击链（表 3 风格 + 严格判定）
```bash
python -m scripts.compare \
  --clean  runs/c10_alexnet_clean \
  --attack runs/c10_alexnet_rd_attack \
  --num-rand 20 --strict \
  --csv  runs/compare_alexnet_rd.csv \
  --save runs/compare_alexnet_rd.json
```
> 若终端没有输出，请直接运行 `python scripts/compare.py ...` 或在脚本打印处加 `flush=True`；确保当前目录为项目根，`python -m scripts.compare -h` 能打印帮助。

---


