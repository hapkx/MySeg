# 📚 Script Utility 和 Train Utility 使用指南

## 📋 概述

新增两个工具文件，用于简化训练脚本的开发和维护：

| 文件 | 用途 | 包含的函数 |
|------|------|----------|
| `script_util_v4.py` | 参数管理和模型创建 | `model_and_diffusion_defaults()`, `create_model_and_diffusion()`, `add_dict_to_argparser()` |
| `train_util_v4.py` | 训练循环的扩展 | `TrainLoopV4`, `create_optimizer_and_schedule()` |

---

## 🔧 script_util_v4.py 详解

### 核心函数

#### 1. `model_and_diffusion_defaults()`
```python
from guided_diffusion_new.script_util_v4 import model_and_diffusion_defaults

defaults = model_and_diffusion_defaults()
# 返回包含所有参数默认值的字典
```

**返回的参数**：
- **原有参数**：image_size, num_channels, num_res_blocks等
- **新增参数**：
  - `use_adapter`: 是否使用Adapter
  - `use_edge_branch`: 是否有边缘分支
  - `fusion_version`: 融合模块版本
  - `use_edge_loss`, `use_sdf_loss`, `use_hd_loss`: Loss配置
  - 等等

**优点**：参数定义集中，易于维护

---

#### 2. `create_model_and_diffusion(...)`
```python
from guided_diffusion_new.script_util_v4 import create_model_and_diffusion

model, diffusion = create_model_and_diffusion(
    image_size=256,
    num_channels=128,
    use_adapter=True,
    use_edge_branch=True,
    use_edge_loss=True,
    # ... 其他参数
)
```

**特点**：
- ✅ 自动选择正确的模型类（adapter/edge_branch/fusion等）
- ✅ 自动选择正确的diffusion（普通或Advanced）
- ✅ 处理所有参数验证和初始化

**模型选择逻辑**：
```
if use_edge_branch:
    → UNetModelWithEdgeBranch
elif use_adapter:
    → UNetModelWithAdapter
elif fusion_version == 'v1':
    → UNetModelWithFusionV1
elif fusion_version == 'v2':
    → UNetModelWithFusionV2
else:
    → UNetModel（原有）
```

**Diffusion选择逻辑**：
```
if use_edge_loss or use_sdf_loss or use_hd_loss:
    → GaussianDiffusionAdvanced（支持新Loss）
else:
    → GaussianDiffusion（原有）
```

---

#### 3. `add_dict_to_argparser(parser, defaults)`
```python
from guided_diffusion_new.script_util_v4 import add_dict_to_argparser

parser = argparse.ArgumentParser()
add_dict_to_argparser(parser, defaults)
# 自动将defaults中的所有键值对添加为命令行参数
```

**自动处理**：
- 布尔值 → `--flag`（action="store_true"）
- 整数值 → `--param 8`（type=int）
- 浮点值 → `--lr 1e-4`（type=float）
- 字符串值 → `--fusion_version v1`（type=str）

---

### 使用示例

#### 完整训练脚本框架
```python
from guided_diffusion_new.script_util_v4 import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
)

# 1. 获取默认参数
defaults = model_and_diffusion_defaults()

# 2. 创建参数解析器
parser = argparse.ArgumentParser()
add_dict_to_argparser(parser, defaults)
args = parser.parse_args()

# 3. 创建模型和diffusion
model, diffusion = create_model_and_diffusion(
    image_size=args.image_size,
    num_channels=args.num_channels,
    num_res_blocks=args.num_res_blocks,
    # ... 传入args的所有参数
    use_adapter=args.use_adapter,
    use_edge_branch=args.use_edge_branch,
    # ... 其他参数
)

# 4. 开始训练
# ...
```

---

## ⚙️ train_util_v4.py 详解

### 核心类

#### `TrainLoopV4`
扩展的训练循环，继承自原有的`TrainLoop`

**新增功能**：
```python
from guided_diffusion_new.train_util_v4 import TrainLoopV4

train_loop = TrainLoopV4(
    model=model,
    diffusion=diffusion,
    data=data_loader,
    batch_size=8,
    lr=3e-5,
    # V4新增参数
    use_adapter=True,
    adapter_lr=1e-4,
    use_edge_branch=True,
    use_edge_loss=True,
    use_sdf_loss=True,
    use_hd_loss=True,
    log_loss_details=True,  # 详细记录Loss
)

train_loop.run_loop()
```

**改进点**：

1. **数据处理**（`_prepare_batch`）
   - 自动处理v4数据格式（dict with img, mask, edge）
   - 自动提取edge_target并传递给diffusion

2. **Loss详细记录**（`log_loss_details`）
   ```python
   if log_loss_details:
       logger.logkv("loss/seg_ce", ...)
       logger.logkv("loss/seg_dice", ...)
       logger.logkv("loss/seg_edge", ...)
       logger.logkv("loss/seg_sdf", ...)
       logger.logkv("loss/seg_hd", ...)
   ```

3. **分层学习率记录**
   ```python
   logger.logkv("lr_adapter", ...)
   logger.logkv("lr_other", ...)
   ```

4. **配置打印**（`_print_config`）
   - 启动时自动打印所有v4相关配置

---

### 核心函数

#### `create_optimizer_and_schedule(...)`
```python
from guided_diffusion_new.train_util_v4 import create_optimizer_and_schedule

optimizer = create_optimizer_and_schedule(
    model,
    lr=1e-4,
    weight_decay=0.0,
    use_adapter=True,
    adapter_lr=1e-4,
)
```

**特点**：
- ✅ 自动处理分层学习率
- ✅ 调用model的`get_parameter_groups()`方法（如果存在）
- ✅ 回退到普通学习率（如果模型不支持分层）

**参数组结构**：
```python
# 当use_adapter=True时
optimizer.param_groups = [
    {'params': adapter_params, 'lr': adapter_lr},
    {'params': other_params, 'lr': lr},
]
```

---

## 📖 完整使用示例

### 简单版本（无utility）
```python
# 传统方式：参数硬编码在脚本中
model = UNetModelWithAdapter(
    image_size=256,
    in_channels=4,
    model_channels=128,
    # ... 很多参数
)
diffusion = GaussianDiffusionAdvanced(
    betas=...,
    use_edge_loss=True,
    # ... 很多参数
)
```

### 现代版本（使用utility）
```python
# 使用utility：参数集中管理
from guided_diffusion_new.script_util_v4 import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
)

defaults = model_and_diffusion_defaults()
parser = argparse.ArgumentParser()
add_dict_to_argparser(parser, defaults)
args = parser.parse_args()

model, diffusion = create_model_and_diffusion(
    image_size=args.image_size,
    use_adapter=args.use_adapter,
    use_edge_branch=args.use_edge_branch,
    # ... 所有参数
)
```

**对比优势**：
- ✅ 参数定义只在一处（script_util_v4.py）
- ✅ 易于添加新参数（只需在defaults()中添加）
- ✅ 命令行参数自动生成
- ✅ 模型创建逻辑自动处理

---

## 🔗 与现有脚本的关系

### 文件关系图
```
scripts_new/
├── segmentation_train_gate_v4.py (原有，参数硬编码)
│   └── 直接导入各模块，手动配置
│
└── segmentation_train_gate_v4_enhanced.py (新增，使用utility)
    ├── 导入 script_util_v4
    ├── 导入 train_util_v4
    └── 简洁的参数管理和训练循环
```

---

## 🎯 何时使用Utility

| 场景 | 推荐 | 原因 |
|------|------|------|
| 简单实验，参数不会改变 | script_util不需要 | 硬编码足够 |
| 频繁调整参数 | ✅ 使用script_util | 参数集中管理 |
| 支持多种模型配置 | ✅ 使用script_util | 自动模型选择 |
| 需要分层学习率 | ✅ 使用train_util | 自动优化器创建 |
| 添加新Loss函数 | ✅ 使用script_util | 自动diffusion选择 |
| 团队协作 | ✅ 强烈推荐 | 减少错误和重复代码 |

---

## 📝 扩展指南

### 添加新参数的步骤

假设要添加新参数 `use_new_feature=False`：

1. **在`script_util_v4.py`中添加**：
```python
def model_and_diffusion_defaults():
    res = base_defaults()
    res.update(dict(
        # 原有参数...
        use_new_feature=False,  # 新参数
    ))
    return res
```

2. **在`create_model_and_diffusion`中处理**：
```python
def create_model_and_diffusion(
    # 原有参数...
    use_new_feature=False,  # 新参数
    **kwargs
):
    # 根据新参数做处理
    if use_new_feature:
        # ...
```

3. **在脚本中使用**：
```python
args = parser.parse_args()
model, diffusion = create_model_and_diffusion(
    # ...
    use_new_feature=args.use_new_feature,
)
```

**自动获得**：
- ✅ 命令行参数 `--use_new_feature`
- ✅ 默认值支持
- ✅ 类型检查

---

## 🐛 故障排查

| 问题 | 症状 | 解决方案 |
|------|------|--------|
| 参数未被识别 | `unrecognized arguments` | 检查参数是否在defaults()中 |
| 模型类选择错误 | 错误的模型被创建 | 检查create_model_and_diffusion()中的选择逻辑 |
| Loss函数未启用 | Loss=0或未记录 | 检查是否在create_model_and_diffusion中传递了use_*_loss |
| 分层学习率未工作 | 所有参数学习率相同 | 确保模型有get_parameter_groups()方法 |

---

## 💡 最佳实践

1. **始终使用defaults**
   ```python
   defaults = model_and_diffusion_defaults()
   # 这样新增参数时自动生效
   ```

2. **使用add_dict_to_argparser**
   ```python
   add_dict_to_argparser(parser, defaults)
   # 避免手动定义每个参数
   ```

3. **在create_model_and_diffusion中集中逻辑**
   ```python
   # ✅ 好：模型选择在create_model_and_diffusion中
   model, diffusion = create_model_and_diffusion(...)
   
   # ❌ 坏：模型选择分散在脚本中
   if use_adapter:
       model = UNetModelWithAdapter(...)
   ```

4. **利用TrainLoopV4的数据处理**
   ```python
   # 自动处理v4数据格式，无需手动处理
   train_loop = TrainLoopV4(...)
   ```

---

## 📚 参考

- 原有的script_util：`guided_diffusion/script_util.py`
- 原有的train_util：`guided_diffusion/train_util.py`
- V4增强脚本：`scripts_new/segmentation_train_gate_v4_enhanced.py`

---

## 🎓 学习路径

1. **理解参数管理**
   - 阅读`script_util_v4.py`的`model_and_diffusion_defaults()`
   - 理解参数传递流程

2. **理解模型创建**
   - 阅读`create_model_and_diffusion()`的逻辑
   - 跟踪各模块的导入和选择

3. **理解训练循环**
   - 阅读`TrainLoopV4`的改进点
   - 理解数据处理和Loss记录

4. **实践**
   - 运行`segmentation_train_gate_v4_enhanced.py`
   - 尝试修改defaults或添加新参数

祝使用愉快！ 🚀
