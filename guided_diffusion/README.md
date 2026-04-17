# 🚀 Enhanced Segmentation with DDPM - Module Guide

## 📁 文件清单

### 核心模块（guided_diffusion_new/）

#### 1. **eval_metrics.py** - 医学图像分割评估指标
```python
from guided_diffusion_new.eval_metrics import MetricsEvaluator

evaluator = MetricsEvaluator(include_hd=True, include_asd=True)
metrics = evaluator.compute_all(pred, target)
```
**支持指标**：
- Dice / IoU（完整性）
- HD95 / ASD（边界准确度）
- Sensitivity / Specificity（临床相关）
- Precision / Recall（预测质量）

---

#### 2. **fusion_modules.py** - 两种特征融合方案
```python
from guided_diffusion_new.fusion_modules import (
    AdaptiveFusionModule,  # v1: 自适应权重融合
    CrossBranchAttentionModule,  # v2: CBA融合
)
```
**方案对比**：
- v1：轻量级，参数少，稳定性好
- v2：表现力强，多头注意力，计算量大

---

#### 3. **advanced_losses.py** - 高级损失函数
```python
from guided_diffusion_new.advanced_losses import (
    SDFLoss,  # SDF基础的边界对齐
    HausdorffLoss,  # 可微分Hausdorff距离
    EdgeSupervisionLoss,  # 边缘二分类
)
```
**用途**：
- SDF Loss：直接优化边界位置
- Hausdorff Loss：惩罚离群错误
- Edge Loss：显式边缘监督

---

#### 4. **adapter.py** - PVT参数高效微调
```python
from guided_diffusion_new.adapter import PVTWithAdapters

pvt_adapter = PVTWithAdapters(pvt_model)
pvt_adapter.freeze_backbone()  # 冻结原始参数
```
**特点**：
- 降维→GELU→升维的轻量结构
- 零初始化保证恒等映射
- 仅训练Adapter，冻结PVT主体

---

#### 5. **mydataloader_v4.py** - 数据加载v4（方案4）
```python
from guided_diffusion_new.mydataloader_v4 import MyDatasetV4, unpad_and_rescale

dataset = MyDatasetV4(data_dir="./data", split='train')
sample = dataset[0]
# 返回：img, mask, edge, padding_info
```
**方案4特点**：
- 长边缩放到256，短边填充至256×256
- 保留宽高比信息，无锯齿
- Canny自动生成边缘伪标签
- 记录padding_info便于反投影

---

#### 6. **unet_gate_adapter.py** - UNet + Adapter
```python
from guided_diffusion_new.unet_gate_adapter import UNetModelWithAdapter

model = UNetModelWithAdapter(...)
model.freeze_pvt_backbone()
param_groups = model.get_parameter_groups(adapter_lr=1e-4, other_lr=1e-4)
```
**功能**：
- 集成PVT Adapter
- 分层学习率支持
- 参数统计信息

---

#### 7. **unet_gate_edge_branch.py** - UNet + 边缘分支
```python
from guided_diffusion_new.unet_gate_edge_branch import UNetModelWithEdgeBranch

model = UNetModelWithEdgeBranch(...)
output = model(x, t)  # (B, 4, H, W) = 3分割 + 1边缘

seg, edge = model.get_seg_and_edge(output)
```
**改进**：
- 输出扩展至4通道（3分割+1边缘）
- 显式边缘监督
- 便利的分离方法

---

#### 8. **unet_gate_fusion_v1.py** - 自适应融合融合
```python
from guided_diffusion_new.unet_gate_fusion_v1 import UNetModelWithFusionV1

model = UNetModelWithFusionV1(...)
```
**改进**：
- 替换GatedFusionModule为AdaptiveFusionModule
- Per-scale自适应权重
- 更稳定的融合

---

#### 9. **unet_gate_fusion_v2.py** - CBA融合
```python
from guided_diffusion_new.unet_gate_fusion_v2 import UNetModelWithFusionV2

model = UNetModelWithFusionV2(..., fusion_num_heads=8)
```
**改进**：
- 替换为CrossBranchAttentionModule
- 多头自注意力
- 更强的表现力

---

#### 10. **gaussian_diffusion_advanced.py** - 高级训练
```python
from guided_diffusion_new.gaussian_diffusion_advanced import GaussianDiffusionAdvanced

diffusion = GaussianDiffusionAdvanced(
    betas=...,
    use_edge_loss=True,
    use_sdf_loss=True,
    use_hd_loss=True,
)
```
**特点**：
- 支持边缘分支（4通道输出）
- 可启用/禁用各种Loss
- 便于消融实验

---

### 脚本文件（scripts_new/）

#### 1. **segmentation_sample_gate_v4.py** - 采样和评估
```bash
python scripts_new/segmentation_sample_gate_v4.py \
    --data_dir ./data/mydata \
    --model_path ./results/checkpoint.pt \
    --output_dir ./results/sample_v4 \
    --use_edge_branch
```
**功能**：
- 全面的评估指标计算
- 自动反投影回原图
- 支持边缘分支
- 可视化保存

---

#### 2. **segmentation_train_gate_v4.py** - 训练脚本
```bash
# 基础版本
python scripts_new/segmentation_train_gate_v4.py \
    --out_dir ./results/train/baseline

# 完整配置
python scripts_new/segmentation_train_gate_v4.py \
    --use_adapter \
    --use_edge_branch \
    --use_edge_loss \
    --use_sdf_loss \
    --use_hd_loss \
    --fusion_version v1 \
    --out_dir ./results/train/complete
```
**特点**：
- 灵活的模型选择
- Loss配置的启用/禁用
- 分层学习率设置
- 详细的日志输出

---

## 🧪 消融实验方案

### 实验序列

```
【实验1】基础配置 - 数据预处理v4
├─ 模型：unet_base（标准U-Net基础）
├─ 数据：mydataloader_v4
├─ 脚本：segmentation_train_gate_v4.py --baseline
└─ 期望改进：0%（baseline）

【实验2】+ Adapter冻结
├─ 模型：unet_gate_adapter
├─ 配置：--use_adapter --freeze_pvt
└─ 期望改进：+0.5~1% mDice

【实验3】+ 边缘监督分支
├─ 模型：unet_gate_edge_branch
├─ 配置：--use_edge_branch --use_edge_loss
└─ 期望改进：+1~3% mDice +边界清晰

【实验4】+ SDF+Hausdorff Loss
├─ 模型：原有（或任意）
├─ 配置：--use_sdf_loss --use_hd_loss
└─ 期望改进：+1~2% mDice

【实验5】+ 融合v1（自适应权重）
├─ 模型：unet_gate_fusion_v1
├─ 配置：--fusion_version v1
└─ 期望改进：+0.5~1.5% mDice

【实验6】+ 融合v2（CBA）
├─ 模型：unet_gate_fusion_v2
├─ 配置：--fusion_version v2
└─ 期望改进：+0.3~1% mDice

【实验7】完整配置
├─ 模型：各模块都启用
├─ 配置：所有flag都为True
└─ 期望改进：+3~6% mDice
```

---

## 🚀 快速开始

### 步骤1：基础训练（v4数据预处理）
```bash
python scripts_new/segmentation_train_gate_v4.py \
    --data_dir ./data/mydata \
    --out_dir ./results/train/v4_baseline \
    --batch_size 8 \
    --lr 3e-5 \
    --log_interval 100
```

### 步骤2：采样和评估
```bash
python scripts_new/segmentation_sample_gate_v4.py \
    --data_dir ./data/mydata \
    --model_path ./results/train/v4_baseline/model_best.pt \
    --output_dir ./results/sample/v4_baseline \
    --batch_size 4
```

### 步骤3：开始消融实验
```bash
# 实验2：+Adapter
python scripts_new/segmentation_train_gate_v4.py \
    --use_adapter \
    --out_dir ./results/train/v4_adapter

# 实验3：+边缘分支
python scripts_new/segmentation_train_gate_v4.py \
    --use_edge_branch \
    --use_edge_loss \
    --out_dir ./results/train/v4_edge
```

---

## 📊 关键参数说明

### 模型参数
```
--use_adapter          # 启用PVT Adapter冻结
--adapter_lr 1e-4      # Adapter学习率
--freeze_pvt           # 冻结PVT原始参数
--use_edge_branch      # 添加边缘分支
--fusion_version v1/v2 # 融合模块版本（None为原有GatedFusion）
```

### Loss参数
```
--use_edge_loss        # 启用边缘监督Loss
--use_sdf_loss         # 启用SDF Loss
--use_hd_loss          # 启用Hausdorff Loss
--edge_loss_weight 0.2 # 边缘Loss权重
--sdf_loss_weight 0.2  # SDF Loss权重
--hd_loss_weight 0.1   # Hausdorff Loss权重
```

---

## 📈 预期效果

| 改进 | 关键指标 | 预期提升 | 优先级 |
|------|--------|---------|--------|
| 数据v4 | Dice/IoU | +0~1% | 基础 |
| Adapter | Dice/IoU | +0.5~1% | 中 |
| 边缘分支 | Dice/IoU, 边界清晰 | +1~3% | 高 |
| SDF+HD Loss | Dice/IoU | +1~2% | 高 |
| 融合v1 | Dice/IoU | +0.5~1.5% | 中 |
| 融合v2 | Dice/IoU | +0.3~1% | 中 |
| **完整配置** | **综合效果** | **+3~6%** | **最高** |

---

## 💡 最佳实践

1. **从简到复**：先跑baseline，再逐个添加模块
2. **独立验证**：每个模块单独测试，记录贡献
3. **学习率调整**：添加新模块时可能需要调整lr
4. **定期评估**：每训练N个epoch保存checkpoint并评估
5. **保留原有脚本**：guided_diffusion下的原文件保持不变，便于对比

---

## 🔗 依赖项

```python
# 核心依赖
torch
numpy
scipy  # 用于距离变换
cv2    # 用于Canny边缘检测
timm   # 用于PVT
```

---

## 📝 常见问题

**Q: 如何同时启用Adapter和边缘分支？**
A: 目前实现是分开的，需要手动修改脚本组合。可以参考方案中的组合创建新的UNet变体。

**Q: 边缘分支的边缘标签从哪来？**
A: 从GT mask自动通过Canny边缘检测生成，无需手工标注。

**Q: 如何提高GPU利用率？**
A: 增加batch_size，或减少log_interval以减少I/O。

**Q: 反投影后的结果尺寸是什么？**
A: 返回原图的原始尺寸（1172×852或其他），同时保留填充信息用于可视化。

---

## 📞 支持

对于问题或建议，请查看：
- 原始实现：guided_diffusion/
- 原始脚本：scripts/
- 新增模块文档注释详尽

祝实验顺利！🎯
