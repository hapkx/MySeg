"""
Parameter-Efficient Fine-Tuning for PVT

Adapter模块设计：
- 在PVT每个Block的Attention和FFN后各插入一个Adapter
- 冻结PVT原始参数，仅训练Adapter
- 通过零初始化up层实现恒等映射，不干扰预训练特征

参考论文：
  "Parameter-Efficient Transfer Learning for NLP" (Houlsby et al., 2019)
  应用到Vision Transformer进行医学图像适配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PVTAdapter(nn.Module):
    """
    轻量级Adapter模块

    结构：
      input → 降维 → GELU激活 → 升维 → 残差 → output

    特点：
    - 参数量少（降维因子r=16）
    - up层初始化为0，保证初始恒等映射
    - 与PVT的任何层兼容
    """

    def __init__(self, hidden_dim: int, reduction: int = 16):
        """
        Args:
            hidden_dim: 输入/输出特征维度
            reduction: 降维比例，默认16
                     即中间层维度 = hidden_dim // 16
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.reduction = reduction
        self.bottleneck_dim = max(hidden_dim // reduction, 1)

        # 降维
        self.down_proj = nn.Linear(hidden_dim, self.bottleneck_dim)
        self.activation = nn.GELU()

        # 升维（初始化为0，保证初始输出为0）
        self.up_proj = nn.Linear(self.bottleneck_dim, hidden_dim)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (B, N, hidden_dim) 或 (B, H, W, hidden_dim)

        Returns:
            残差连接后的输出，形状同输入
        """
        # 保存原始形状
        original_shape = x.shape

        # 转为(B*..., hidden_dim)进行处理
        x_flat = x.view(-1, self.hidden_dim)

        # Adapter：down -> activate -> up
        residual = self.down_proj(x_flat)
        residual = self.activation(residual)
        residual = self.up_proj(residual)

        # 恢复形状
        residual = residual.view(original_shape)

        # 残差连接
        output = x + residual

        return output


class AdapterConfig:
    """Adapter的配置"""

    def __init__(self, num_adapters: int = 4, reduction: int = 16):
        """
        Args:
            num_adapters: Adapter总数（对应PVT的stage数）
            reduction: 降维比例
        """
        self.num_adapters = num_adapters
        self.reduction = reduction


class PVTWithAdapters(nn.Module):
    """
    带Adapter的PVT包装类

    使用方式：
      from timm.models import pvt_v2_b0
      pvt = pvt_v2_b0(pretrained=True)
      pvt_with_adapters = PVTWithAdapters(pvt)
      pvt_with_adapters.freeze_backbone()
    """

    def __init__(self, pvt_model: nn.Module, adapter_config: AdapterConfig = None):
        """
        Args:
            pvt_model: timm中的PVT模型
            adapter_config: Adapter配置
        """
        super().__init__()
        self.pvt = pvt_model
        self.adapter_config = adapter_config or AdapterConfig()

        # 获取PVT的各层特征维度
        # 对于pvt_v2_b0: [64, 128, 256, 256]
        self.pvt_channels = self._get_pvt_channels()

        # 创建Adapters
        self.adapters = nn.ModuleList([
            PVTAdapter(ch, self.adapter_config.reduction)
            for ch in self.pvt_channels
        ])

    def _get_pvt_channels(self) -> list:
        """
        获取PVT各个stage的输出通道数

        对于不同的PVT版本，可能需要调整
        """
        if hasattr(self.pvt, 'feature_info'):
            return [info['num_chs'] for info in self.pvt.feature_info]
        else:
            # 默认配置（pvt_v2_b0）
            return [64, 128, 256, 256]

    def freeze_backbone(self):
        """冻结PVT的所有原始参数"""
        for param in self.pvt.parameters():
            param.requires_grad = False

    def unfreeze_adapters(self):
        """确保Adapter参数可训练"""
        for adapter in self.adapters:
            for param in adapter.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: 输入图像 (B, 3, H, W)

        Returns:
            features: 4个特征列表，每个特征经过Adapter处理
        """
        # 获取PVT的原始特征
        features = self.pvt(x)  # List of 4 features

        # 应用Adapters
        adapted_features = []
        for i, feat in enumerate(features):
            # Adapter处理前先转换格式（如果需要）
            # 通常特征已经是(B, C, H, W)格式
            B, C, H, W = feat.shape
            feat_flat = feat.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)

            # 应用Adapter
            adapted = self.adapters[i](feat_flat)  # (B, H*W, C)

            # 转换回(B, C, H, W)
            adapted = adapted.transpose(1, 2).view(B, C, H, W)
            adapted_features.append(adapted)

        return adapted_features

    def set_learning_rate(self, optimizer: torch.optim.Optimizer,
                         adapter_lr: float = 1e-4,
                         other_lr: float = 1e-4):
        """
        设置不同层的学习率

        Args:
            optimizer: PyTorch优化器（需要通过param_groups配置）
            adapter_lr: Adapter层的学习率
            other_lr: 其他可训练层的学习率
        """
        # 这个函数通常在训练脚本中使用
        # 示例：
        # optimizer = torch.optim.Adam([
        #     {'params': model.adapters.parameters(), 'lr': adapter_lr},
        #     {'params': model.fusion.parameters(), 'lr': other_lr},
        # ])
        pass


# ============================================================================
# 辅助函数
# ============================================================================

def create_pvt_with_adapters(pretrained: bool = True,
                            freeze_backbone: bool = True,
                            reduction: int = 16) -> PVTWithAdapters:
    """
    创建带Adapter的PVT模型

    Args:
        pretrained: 是否加载预训练权重
        freeze_backbone: 是否冻结PVT骨干
        reduction: Adapter的降维比例

    Returns:
        PVTWithAdapters实例
    """
    try:
        import timm
        pvt = timm.create_model('pvt_v2_b0', pretrained=pretrained, features_only=True)
    except ImportError:
        raise ImportError("Please install timm: pip install timm")

    adapter_config = AdapterConfig(num_adapters=4, reduction=reduction)
    pvt_with_adapters = PVTWithAdapters(pvt, adapter_config)

    if freeze_backbone:
        pvt_with_adapters.freeze_backbone()
        pvt_with_adapters.unfreeze_adapters()

    return pvt_with_adapters


if __name__ == "__main__":
    # 测试代码
    import torch

    # 创建模型
    try:
        pvt_adapter = create_pvt_with_adapters(pretrained=False)

        # 测试forward
        x = torch.randn(2, 3, 256, 256)
        features = pvt_adapter(x)

        print("PVT with Adapters test:")
        for i, feat in enumerate(features):
            print(f"  Feature {i}: {feat.shape}")

        # 统计参数
        total_params = sum(p.numel() for p in pvt_adapter.parameters())
        trainable_params = sum(p.numel() for p in pvt_adapter.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    except ImportError as e:
        print(f"Error: {e}")
        print("Please ensure timm is installed")
