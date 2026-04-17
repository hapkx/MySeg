# 📚 训练和采样脚本使用指南

## 🎯 脚本概览

| 脚本 | 用途 | 关键特性 |
|------|------|--------|
| `segmentation_train_gate_v4.py` | 训练 | 灵活的配置，支持各种模块组合 |
| `segmentation_sample_gate_v4.py` | 采样+评估 | 全面的指标计算，自动反投影 |

---

## 📖 训练脚本详解

### 基础使用

**最简单的命令**（仅升级数据预处理v4）：
```bash
cd /path/to/MySeg
python scripts_new/segmentation_train_gate_v4.py \
    --data_dir ./data/mydata \
    --out_dir ./results/train/v4_test
```

### 完整参数说明

#### 必要参数
```bash
--data_dir PATH          # 数据目录（包含train_data/, train_mask/）
--out_dir PATH           # 输出目录（保存日志、checkpoints、结果）
```

#### 数据参数
```bash
--batch_size 8           # batch大小（根据GPU内存调整）
```

#### 优化参数
```bash
--lr 3e-5                # 学习率
--weight_decay 0.0       # 权重衰减
--lr_anneal_steps 0      # 学习率退火步数
--log_interval 100       # 日志输出间隔
--save_interval 500      # 模型保存间隔
```

#### 硬件参数
```bash
--gpu_dev "0,1"          # 使用的GPU设备
--use_fp16               # 是否使用FP16混合精度
```

#### 模型选择
```bash
--use_adapter            # 启用PVT Adapter
--adapter_lr 1e-4        # Adapter专用学习率
--freeze_pvt             # 冻结PVT原始参数
--use_edge_branch        # 添加边缘分支
--fusion_version v1      # 融合版本：None/v1/v2
```

#### Loss配置
```bash
--use_edge_loss          # 启用边缘监督Loss
--use_sdf_loss           # 启用SDF Loss
--use_hd_loss            # 启用Hausdorff Loss
--edge_loss_weight 0.2   # 边缘Loss权重
--sdf_loss_weight 0.2    # SDF Loss权重
--hd_loss_weight 0.1     # Hausdorff Loss权重
```

---

## 🔬 典型实验命令

### 实验1：基础baseline（数据v4）
```bash
python scripts_new/segmentation_train_gate_v4.py \
    --data_dir ./data/mydata \
    --out_dir ./results/train/exp1_baseline \
    --batch_size 8 \
    --lr 3e-5 \
    --log_interval 50 \
    --save_interval 200
```

### 实验2：+Adapter冻结
```bash
python scripts_new/segmentation_train_gate_v4.py \
    --data_dir ./data/mydata \
    --out_dir ./results/train/exp2_adapter \
    --use_adapter \
    --adapter_lr 1e-4 \
    --freeze_pvt \
    --batch_size 8 \
    --lr 1e-4
```

### 实验3：+边缘分支
```bash
python scripts_new/segmentation_train_gate_v4.py \
    --data_dir ./data/mydata \
    --out_dir ./results/train/exp3_edge_branch \
    --use_edge_branch \
    --use_edge_loss \
    --edge_loss_weight 0.2 \
    --batch_size 8 \
    --lr 3e-5
```

### 实验4：+SDF和Hausdorff Loss
```bash
python scripts_new/segmentation_train_gate_v4.py \
    --data_dir ./data/mydata \
    --out_dir ./results/train/exp4_advanced_loss \
    --use_sdf_loss \
    --use_hd_loss \
    --sdf_loss_weight 0.2 \
    --hd_loss_weight 0.1 \
    --batch_size 8 \
    --lr 3e-5
```

### 实验5：+融合v1（自适应权重）
```bash
python scripts_new/segmentation_train_gate_v4.py \
    --data_dir ./data/mydata \
    --out_dir ./results/train/exp5_fusion_v1 \
    --fusion_version v1 \
    --batch_size 8 \
    --lr 3e-5
```

### 实验6：+融合v2（CBA）
```bash
python scripts_new/segmentation_train_gate_v4.py \
    --data_dir ./data/mydata \
    --out_dir ./results/train/exp6_fusion_v2 \
    --fusion_version v2 \
    --batch_size 8 \
    --lr 3e-5
```

### 实验7：完整配置（所有优化）
```bash
python scripts_new/segmentation_train_gate_v4.py \
    --data_dir ./data/mydata \
    --out_dir ./results/train/exp7_complete \
    --use_adapter \
    --adapter_lr 1e-4 \
    --freeze_pvt \
    --use_edge_branch \
    --use_edge_loss \
    --use_sdf_loss \
    --use_hd_loss \
    --fusion_version v1 \
    --edge_loss_weight 0.2 \
    --sdf_loss_weight 0.2 \
    --hd_loss_weight 0.1 \
    --batch_size 8 \
    --lr 1e-4 \
    --log_interval 50
```

---

## 📊 采样脚本详解

### 基础使用

```bash
python scripts_new/segmentation_sample_gate_v4.py \
    --data_dir ./data/mydata \
    --model_path ./results/train/exp1_baseline/model_best.pt \
    --output_dir ./results/sample/exp1
```

### 完整参数说明

#### 必要参数
```bash
--data_dir PATH          # 数据目录
--model_path PATH        # 模型checkpoint路径
--output_dir PATH        # 输出目录
```

#### 采样参数
```bash
--batch_size 4           # batch大小（推荐4）
--num_samples 0          # 样本数量（0=全部）
```

#### 硬件参数
```bash
--gpu_dev "0"            # 使用的GPU设备
```

#### 模型匹配参数
```bash
--use_edge_branch        # 模型是否有边缘分支（要与训练匹配）
--use_adapter            # 模型是否有Adapter
--fusion_version v1      # 融合版本（None/v1/v2）
```

#### 输出参数
```bash
--save_vis               # 是否保存可视化结果
```

### 典型采样命令

**对应实验1的采样**：
```bash
python scripts_new/segmentation_sample_gate_v4.py \
    --data_dir ./data/mydata \
    --model_path ./results/train/exp1_baseline/model_best.pt \
    --output_dir ./results/sample/exp1 \
    --batch_size 4 \
    --save_vis
```

**对应实验3（有边缘分支）的采样**：
```bash
python scripts_new/segmentation_sample_gate_v4.py \
    --data_dir ./data/mydata \
    --model_path ./results/train/exp3_edge_branch/model_best.pt \
    --output_dir ./results/sample/exp3 \
    --use_edge_branch \
    --batch_size 4 \
    --save_vis
```

**对应实验7（完整配置）的采样**：
```bash
python scripts_new/segmentation_sample_gate_v4.py \
    --data_dir ./data/mydata \
    --model_path ./results/train/exp7_complete/model_best.pt \
    --output_dir ./results/sample/exp7 \
    --use_adapter \
    --use_edge_branch \
    --fusion_version v1 \
    --batch_size 4 \
    --save_vis
```

---

## 📁 输出目录结构

### 训练输出
```
./results/train/exp1_baseline/
├── log.txt                      # 训练日志
├── checkpoint.pt                # 最后一个checkpoint
├── model_best.pt                # 最好的checkpoint
└── config.json                  # 训练配置
```

### 采样输出
```
./results/sample/exp1/
├── metrics.json                 # 评估指标（JSON格式）
├── log.txt                      # 采样日志
└── visualizations/
    ├── sample_000.png           # 可视化结果（GT左，预测右）
    ├── sample_001.png
    └── ...
```

---

## 📈 指标解读

采样脚本会输出以下指标（保存在 `metrics.json`）：

```json
{
  "average_metrics": {
    "dice_mean": 0.85,              # Dice系数平均
    "iou_mean": 0.76,               # IoU平均
    "hd95_mean": 15.3,              # Hausdorff Distance 95
    "asd_mean": 8.2,                # Average Surface Distance
    "sensitivity_mean": 0.88,       # 灵敏度
    "specificity_mean": 0.92,       # 特异性
    "precision_mean": 0.83,         # 精确度
    "recall_mean": 0.88,            # 召回率
    "dice_class_1": 0.80,           # 前景1的Dice
    "dice_class_2": 0.90,           # 前景2的Dice
    ...
  }
}
```

**指标含义**：
- **Dice/IoU**：越高越好（0-1）
- **HD95/ASD**：越低越好（0+，单位：像素）
- **Sensitivity**：真阳率，捕捉目标能力
- **Specificity**：真阴率，避免误判能力
- **Precision**：预测准确性
- **Recall**：等同于Sensitivity

---

## 🎓 分析结果的方法

### 1. 快速比较各实验

```bash
# 提取所有实验的平均Dice
for dir in results/sample/exp*; do
    echo "$(basename $dir):"
    cat "$dir/metrics.json" | grep '"dice_mean"'
done
```

### 2. 查看详细指标

```bash
# 查看exp1的完整指标
cat results/sample/exp1/metrics.json | python -m json.tool
```

### 3. 对比指标差异

```bash
# 比较exp1和exp7的各类别Dice
echo "Exp1:"
cat results/sample/exp1/metrics.json | grep "dice_class"
echo "\nExp7:"
cat results/sample/exp7/metrics.json | grep "dice_class"
```

---

## ⚡ 性能优化建议

### 减少训练时间
```bash
# 降低日志频率，增加保存间隔
--log_interval 200 --save_interval 1000
```

### 提高采样速度
```bash
# 减少batch大小以避免内存溢出
--batch_size 2
```

### 节省GPU内存
```bash
# 使用FP16混合精度
--use_fp16 --fp16_scale_height 128
```

---

## 🐛 常见问题排查

### Q：Out of Memory错误
**A**：减小batch_size，或启用FP16：
```bash
--batch_size 4 --use_fp16
```

### Q：模型加载失败
**A**：确保--model_path正确，且与--use_adapter等参数匹配

### Q：数据加载失败
**A**：检查--data_dir结构：
```
data_dir/
├── train_data/     # 训练图像
├── train_mask/     # 训练mask
├── val_data/       # 验证图像
└── val_mask/       # 验证mask
```

### Q：输出目录权限问题
**A**：确保对输出目录有写权限：
```bash
chmod -R 755 ./results/
```

---

## 📚 关键论文参考

实现基于以下论文：
- DDPM：Ho et al. (2020) - Denoising Diffusion Probabilistic Models
- PVT Adapter：Houlsby et al. (2019) - Parameter-Efficient Transfer Learning for NLP
- SDF Loss：Kervadec et al. (2019) - Boundary Loss for Remote Sensing Image Segmentation
- Hausdorff Loss：Karimi et al. (2019) - Hausdorff Distance Loss for Semantic Segmentation

---

## 💾 保存和恢复训练

### 继续训练
```bash
python scripts_new/segmentation_train_gate_v4.py \
    --data_dir ./data/mydata \
    --out_dir ./results/train/exp1_baseline \
    --resume_checkpoint ./results/train/exp1_baseline/checkpoint.pt \
    --lr 3e-5
```

### 最佳模型选择
脚本自动保存最好的checkpoint为 `model_best.pt`，这就是要用于采样的模型。

---

祝训练顺利！ 🚀
