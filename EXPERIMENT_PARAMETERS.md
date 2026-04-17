# 📊 实验参数对比表

基于 `launch.json` 中的实验配置，整理了所有实验的关键参数。

---

## 📋 实验分类

### 🟢 训练实验 (Training)

#### 1. train_no (无PVT基线)
```bash
python scripts/segmentation_train.py \
    --data_name mydata \
    --data_dir ./data/mydata \
    --lr 5e-5 \
    --batch_size 8 \
    --image_size 256 \
    --num_channels 128 \
    --num_res_blocks 2 \
    --num_heads 1 \
    --attention_resolutions 16 \
    --channel_mult 1,2,4,8 \
    --use_scale_shift_norm False \
    --learn_sigma False \
    --use_pvt_fusion False \
    --use_feb False \
    --use_cross_attn False \
    --freeze_pvt True \
    --out_dir ./results/train/no_0321_learnSigmaFalse \
    --gpu_dev 0
```

**特点**: 无PVT融合，标准UNet baseline

---

#### 2. train_pvt (PVT融合)
```bash
python scripts/segmentation_train_gate.py \
    --data_name mydata \
    --data_dir ./data/mydata \
    --lr 5e-5 \
    --batch_size 8 \
    --image_size 256 \
    --num_channels 32 \
    --num_res_blocks 2 \
    --num_heads 1 \
    --attention_resolutions 16,8 \
    --channel_mult 1,2,5,8 \
    --use_scale_shift_norm True \
    --learn_sigma False \
    --use_pvt_fusion True \
    --use_feb False \
    --use_cross_attn False \
    --freeze_pvt True \
    --out_dir ./results/train/pvt_gate_0331 \
    --gpu_dev 3 \
    --resume_checkpoint /home/nas2/biod/piankexin/AAAablation_model/results/train/pvt_gate_0331/savedmodel040000.pt
```

**特点**: 启用PVT融合，冻结PVT

**关键变化**:
- `--num_channels`: 128 → 32 (减小模型大小)
- `--attention_resolutions`: 16 → 16,8 (多层注意力)
- `--channel_mult`: 1,2,4,8 → 1,2,5,8
- `--use_scale_shift_norm`: False → True
- `--use_pvt_fusion`: False → True

---

#### 3. E1_train_patchmix_pvt_feb_frozen (PVT+FEB, Frozen)
```bash
python scripts/segmentation_train_gate_patchmix.py \
    --data_name mydata \
    --data_dir ./data/mydata \
    --lr 5e-5 \
    --weight_decay 1e-4 \
    --batch_size 8 \
    --image_size 256 \
    --num_channels 32 \
    --num_res_blocks 2 \
    --num_heads 1 \
    --attention_resolutions 16,8 \
    --channel_mult 1,2,5,8 \
    --use_scale_shift_norm True \
    --use_pvt_fusion True \
    --use_feb True \
    --use_cross_attn False \
    --freeze_pvt True \
    --pvt_train_mode frozen \
    --train_long_side 640 \
    --patches_per_image 6 \
    --fg_sample_ratio 0.6 \
    --min_fg_pixels 32 \
    --full_image_ratio 0.6 \
    --intensity_aug_p 0.4 \
    --out_dir ./results/train/pvt_feb_patchmix_frozen \
    --gpu_dev 0
```

**特点**: 启用PVT + FEB，patch-mix数据增强，PVT冻结

**新增参数**:
- `--weight_decay`: 1e-4
- `--use_feb`: True
- `--pvt_train_mode`: frozen (PVT不训练)
- `--train_long_side`: 640 (数据增强长边)
- `--patches_per_image`: 6
- `--fg_sample_ratio`: 0.6
- `--min_fg_pixels`: 32
- `--full_image_ratio`: 0.6
- `--intensity_aug_p`: 0.4

---

#### 4. E1.5_train_patchmix_pvt_feb_frozen_crossattn (PVT+FEB+CrossAttn, Frozen)
```bash
python scripts/segmentation_train_gate_patchmix.py \
    --data_name mydata \
    --data_dir ./data/mydata \
    --lr 5e-5 \
    --weight_decay 1e-4 \
    --batch_size 8 \
    --image_size 256 \
    --num_channels 32 \
    --num_res_blocks 2 \
    --num_heads 1 \
    --attention_resolutions 16,8 \
    --channel_mult 1,2,5,8 \
    --use_scale_shift_norm True \
    --use_pvt_fusion True \
    --use_feb True \
    --use_cross_attn True \
    --freeze_pvt True \
    --pvt_train_mode frozen \
    --train_long_side 640 \
    --patches_per_image 6 \
    --fg_sample_ratio 0.6 \
    --min_fg_pixels 32 \
    --full_image_ratio 0.6 \
    --intensity_aug_p 0.4 \
    --out_dir ./results/train/pvt_feb_patchmix_frozen_crossattn \
    --gpu_dev 0
```

**特点**: E1基础上 + Cross-Attention

**关键差异**:
- `--use_cross_attn`: False → True

---

#### 5. E5_train_patchmix_pvt_feb_last_stage (PVT+FEB, Last Stage)
```bash
python scripts/segmentation_train_gate_patchmix.py \
    --data_name mydata \
    --data_dir ./data/mydata \
    --lr 2e-5 \
    --weight_decay 1e-4 \
    --batch_size 8 \
    --image_size 256 \
    --num_channels 32 \
    --num_res_blocks 2 \
    --num_heads 1 \
    --attention_resolutions 16,8 \
    --channel_mult 1,2,5,8 \
    --use_scale_shift_norm True \
    --use_pvt_fusion True \
    --use_feb True \
    --use_cross_attn True \
    --freeze_pvt False \
    --pvt_train_mode last_stage \
    --train_long_side 640 \
    --patches_per_image 6 \
    --fg_sample_ratio 0.6 \
    --min_fg_pixels 32 \
    --full_image_ratio 0.6 \
    --intensity_aug_p 0.4 \
    --out_dir ./results/train/pvt_feb_patchmix_last_stage \
    --gpu_dev 3
```

**特点**: PVT训练模式改为last_stage，学习率降低

**关键差异**:
- `--lr`: 5e-5 → 2e-5 (学习率降低)
- `--freeze_pvt`: True → False
- `--pvt_train_mode`: frozen → last_stage (训练PVT最后一阶段)
- `--use_cross_attn`: False → True

---

#### 6. E6_train_patchmix_pvt_feb_last_two_stages (PVT+FEB, Last Two Stages)
```bash
python scripts/segmentation_train_gate_patchmix.py \
    --data_name mydata \
    --data_dir ./data/mydata \
    --lr 2e-5 \
    --weight_decay 1e-4 \
    --batch_size 8 \
    --image_size 256 \
    --num_channels 32 \
    --num_res_blocks 2 \
    --num_heads 1 \
    --attention_resolutions 16,8 \
    --channel_mult 1,2,5,8 \
    --use_scale_shift_norm True \
    --use_pvt_fusion True \
    --use_feb True \
    --use_cross_attn True \
    --freeze_pvt False \
    --pvt_train_mode last_two_stages \
    --train_long_side 640 \
    --patches_per_image 6 \
    --fg_sample_ratio 0.6 \
    --min_fg_pixels 32 \
    --full_image_ratio 0.6 \
    --intensity_aug_p 0.4 \
    --out_dir ./results/train/pvt_feb_patchmix_last_two_stages \
    --gpu_dev 5
```

**特点**: PVT训练最后两个阶段

**关键差异**:
- `--pvt_train_mode`: last_stage → last_two_stages

---

### 🔵 采样实验 (Sampling)

#### 7. E2_sample_patchmix_global_only_iou (Global Only)
```bash
python scripts/segmentation_sample_gate_window_iou.py \
    --data_name mydata \
    --data_dir ./data/mydata \
    --model_path ./results/train/pvt_feb_patchmix_frozen/emasavedmodel_0.9999_025000.pt \
    --image_size 256 \
    --num_channels 32 \
    --num_res_blocks 2 \
    --num_heads 1 \
    --attention_resolutions 16,8 \
    --channel_mult 1,2,5,8 \
    --use_scale_shift_norm True \
    --use_pvt_fusion True \
    --use_feb True \
    --use_cross_attn False \
    --freeze_pvt True \
    --pvt_train_mode frozen \
    --clip_denoised True \
    --use_global_branch True \
    --use_local_branch False \
    --num_ensemble_global 4 \
    --out_dir ./results/sample/pvt_feb_patchmix_25000 \
    --gpu_dev 0
```

**特点**: 仅使用全局分支，ensemble=4

---

#### 8. E2.5_sample_patchmix_global_only_iou (Global + CrossAttn)
```bash
python scripts/segmentation_sample_gate_window_iou.py \
    --data_name mydata \
    --data_dir ./data/mydata \
    --model_path ./results/train/pvt_feb_patchmix_frozen_crossattn/emasavedmodel_0.9999_030000.pt \
    --image_size 256 \
    --num_channels 32 \
    --num_res_blocks 2 \
    --num_heads 1 \
    --attention_resolutions 16,8 \
    --channel_mult 1,2,5,8 \
    --use_scale_shift_norm True \
    --use_pvt_fusion True \
    --use_feb True \
    --use_cross_attn True \
    --freeze_pvt True \
    --pvt_train_mode frozen \
    --clip_denoised True \
    --use_global_branch True \
    --use_local_branch False \
    --num_ensemble_global 4 \
    --out_dir ./results/sample/pvt_feb_patchmix_crossattn_30000 \
    --gpu_dev 0
```

**特点**: E2基础上启用cross-attention

---

#### 9. E3_sample_patchmix_local_window_iou (Local Only)
```bash
python scripts/segmentation_sample_gate_window_iou.py \
    --data_name mydata \
    --data_dir ./data/mydata \
    --model_path ./results/train/pvt_feb_patchmix_frozen/emasavedmodel_0.9999_025000.pt \
    --image_size 256 \
    --num_channels 32 \
    --use_global_branch False \
    --use_local_branch True \
    --num_ensemble_local 1 \
    --test_long_side 640 \
    --test_stride 128 \
    --out_dir ./results/sample/pvt_feb_patchmix_local_window_25000 \
    --gpu_dev 0
```

**特点**: 仅使用局部（window）分支，ensemble=1

**采样特定参数**:
- `--use_global_branch`: False
- `--use_local_branch`: True
- `--num_ensemble_local`: 1
- `--test_long_side`: 640
- `--test_stride`: 128

---

#### 10. E4_sample_patchmix_global_local_fusion_iou (Global + Local Fusion)
```bash
python scripts/segmentation_sample_gate_window_iou.py \
    --data_name mydata \
    --data_dir ./data/mydata \
    --model_path ./results/train/pvt_feb_patchmix_frozen/emasavedmodel_0.9999_025000.pt \
    --image_size 256 \
    --num_channels 32 \
    --use_global_branch True \
    --use_local_branch True \
    --num_ensemble_global 4 \
    --num_ensemble_local 1 \
    --fusion_alpha_global 0.4 \
    --test_long_side 640 \
    --test_stride 128 \
    --out_dir ./results/sample/pvt_feb_patchmix_global_local_fusion_25000 \
    --gpu_dev 0
```

**特点**: 全局和局部分支融合，融合权重alpha=0.4

**融合参数**:
- `--fusion_alpha_global`: 0.4 (全局权重)

---

#### 11. sample_pvt (PVT采样)
```bash
python scripts/segmentation_sample_gate.py \
    --data_name mydata \
    --data_dir ./data/mydata \
    --model_path ./results/train/pvt_0331/emasavedmodel_0.9999_040000.pt \
    --num_ensemble 4 \
    --image_size 256 \
    --num_channels 32 \
    --num_res_blocks 2 \
    --num_heads 1 \
    --attention_resolutions 16,8 \
    --channel_mult 1,2,5,8 \
    --use_scale_shift_norm True \
    --use_pvt_fusion True \
    --use_feb False \
    --use_cross_attn False \
    --freeze_pvt True \
    --out_dir ./results/sample/pvt_0331_040000 \
    --gpu_dev (未指定，默认0)
```

**特点**: 简单采样，仅PVT，ensemble=4

---

#### 12. sample_feb (FEB采样)
```bash
python scripts/segmentation_sample_gate.py \
    --data_name mydata \
    --data_dir ./data/mydata \
    --model_path ./results/train/feb_0331/emasavedmodel_0.9999_040000.pt \
    --num_ensemble 4 \
    --image_size 256 \
    --num_channels 32 \
    --use_pvt_fusion False \
    --use_feb True \
    --use_cross_attn False \
    --freeze_pvt True \
    --out_dir ./results/sample/feb_0331_040000 \
    --gpu_dev (未指定)
```

**特点**: 简单采样，仅FEB，无PVT，ensemble=4

---

#### 13. sample_pvt_feb (PVT+FEB采样)
```bash
python scripts/segmentation_sample_gate.py \
    --data_name mydata \
    --data_dir ./data/mydata \
    --model_path ./results/train/pvt_feb_0331/emasavedmodel_0.9999_030000.pt \
    --num_ensemble 1 \
    --image_size 256 \
    --num_channels 32 \
    --use_pvt_fusion True \
    --use_feb True \
    --use_cross_attn False \
    --freeze_pvt True \
    --out_dir ./results/sample/pvt_feb_0331_030000 ensemble=1 \
    --gpu_dev (未指定)
```

**特点**: 简单采样，PVT+FEB，ensemble=1

---

## 📊 实验对比总表

### 训练实验参数对比

| 参数 | train_no | train_pvt | E1 | E1.5 | E5 | E6 |
|------|----------|-----------|-----|------|-----|-----|
| **脚本** | segmentation_train.py | segmentation_train_gate.py | patchmix | patchmix | patchmix | patchmix |
| **lr** | 5e-5 | 5e-5 | 5e-5 | 5e-5 | **2e-5** | **2e-5** |
| **batch_size** | 8 | 8 | 8 | 8 | 8 | 8 |
| **num_channels** | 128 | 32 | 32 | 32 | 32 | 32 |
| **use_pvt_fusion** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **use_feb** | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **use_cross_attn** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| **freeze_pvt** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **pvt_train_mode** | - | - | frozen | frozen | last_stage | last_two_stages |
| **use_patchmix** | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **train_long_side** | - | - | 640 | 640 | 640 | 640 |
| **patches_per_image** | - | - | 6 | 6 | 6 | 6 |
| **gpu_dev** | 0 | 3 | 0 | 0 | 3 | 5 |

### 采样实验参数对比

| 参数 | E2 | E2.5 | E3 | E4 | sample_pvt | sample_feb | sample_pvt_feb |
|------|-----|-------|-----|-----|------------|------------|--------|
| **脚本** | window_iou | window_iou | window_iou | window_iou | simple | simple | simple |
| **模型** | E1@25k | E1.5@30k | E1@25k | E1@25k | pvt@40k | feb@40k | pvt+feb@30k |
| **use_global** | ✅ | ✅ | ❌ | ✅ | - | - | - |
| **use_local** | ❌ | ❌ | ✅ | ✅ | - | - | - |
| **num_ensemble_g** | 4 | 4 | - | 4 | - | - | - |
| **num_ensemble_l** | - | - | 1 | 1 | - | - | - |
| **fusion_alpha** | - | - | - | 0.4 | - | - | - |
| **test_long_side** | - | - | 640 | 640 | - | - | - |
| **test_stride** | - | - | 128 | 128 | - | - | - |
| **num_ensemble** | - | - | - | - | 4 | 4 | 1 |

---

## 🎯 实验说明

### 按复杂度分类

**基础** (Baseline):
- `train_no`: 纯UNet，无任何融合
- `sample_pvt`, `sample_feb`: 简单采样

**进阶** (Intermediate):
- `train_pvt`: 加入PVT融合
- `E1`: PVT + FEB，简单patch-mix

**高级** (Advanced):
- `E1.5`: + Cross-Attention
- `E5`: PVT partial finetune (last_stage)
- `E6`: PVT partial finetune (last_two_stages)
- `E2-E4`: 复杂采样策略（global/local融合）

### 关键改进点

1. **模型架构**:
   - baseline → PVT融合 → PVT+FEB → +Cross-Attention

2. **数据增强**:
   - 无 → Patch-Mix (640长边, 6个patch)

3. **PVT训练策略**:
   - Frozen → Last Stage → Last Two Stages

4. **采样策略**:
   - 简单全局采样 → Window-based local采样 → Global+Local融合

---

## 🔧 推荐使用

### 快速开始（推荐初学者）
```bash
# 使用E1配置，修改为v4脚本
python scripts_new/segmentation_train_gate_v4.py \
    --data_dir ./data/mydata \
    --out_dir ./results/train/e1_baseline \
    --lr 5e-5 \
    --batch_size 8 \
    --use_adapter \
    --use_edge_branch
```

### 完整实验（对标E1）
```bash
# E1 完整配置
python scripts_new/segmentation_train_gate_v4.py \
    --data_dir ./data/mydata \
    --out_dir ./results/train/e1_full \
    --lr 5e-5 \
    --batch_size 8 \
    --num_channels 32 \
    --attention_resolutions 16,8 \
    --channel_mult 1,2,5,8 \
    --use_scale_shift_norm \
    --use_adapter \
    --use_edge_branch \
    --use_edge_loss \
    --weight_decay 1e-4
```

### 高级实验（对标E5）
```bash
# E5 配置（PVT部分微调）
python scripts_new/segmentation_train_gate_v4.py \
    --data_dir ./data/mydata \
    --out_dir ./results/train/e5_advanced \
    --lr 2e-5 \
    --batch_size 8 \
    --use_adapter \
    --use_edge_branch \
    --use_edge_loss \
    --use_sdf_loss \
    --freeze_pvt False \
    --weight_decay 1e-4
```

