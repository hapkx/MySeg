"""
UNet with PVT Adapter

基于unet_base.py的改造版本
添加PVT Adapter进行参数高效微调

使用方式：
  from guided_diffusion_new.unet_gate_adapter import UNetModelWithAdapter

  model = UNetModelWithAdapter(
      image_size=256,
      in_channels=4,
      model_channels=128,
      out_channels=3,
      num_res_blocks=2,
      attention_resolutions=[16, 8],
      channel_mult=(1, 2, 4, 8),
      use_pvt_fusion=True,
      freeze_pvt=True,
      adapter_reduction=16,
  )

  model.freeze_pvt_backbone()  # 冻结PVT原始参数
"""

import torch
import torch.nn as nn
from .unet_base import UNetModel
from .adapter import PVTAdapter


class PVTv2FeatureExtractorWithAdapter(nn.Module):
    """
    PVT特征提取器 + Adapter

    在每个PVT输出层后添加Adapter进行参数高效微调
    """

    def __init__(self, base_extractor, adapter_reduction: int = 16):
        """
        Args:
            base_extractor: 原有的PVTv2FeatureExtractor
            adapter_reduction: Adapter的降维比例
        """
        super().__init__()

        if base_extractor is None:
            raise ValueError("base_extractor cannot be None")

        self.base_extractor = base_extractor

        # 获取PVT通道列表
        try:
            self.pvt_channel_list = base_extractor.get_pvt_channels()
        except (AttributeError, RuntimeError) as e:
            raise RuntimeError(
                f"Failed to get PVT channels from base_extractor. "
                f"Ensure base_extractor is properly initialized. Error: {e}"
            )

        # 为每个输出层创建Adapter
        self.adapters = nn.ModuleList([
            PVTAdapter(ch, reduction=adapter_reduction)
            for ch in self.pvt_channel_list
        ])

    def get_pvt_channels(self):
        """返回PVT各个stage的通道数"""
        return self.pvt_channel_list

    def forward(self, x):
        """
        Args:
            x: 输入图像 (B, 3, H, W)

        Returns:
            adapted_features: 4个经过Adapter处理的特征列表
        """
        # 获取原有的特征
        features = self.base_extractor(x)  # List of 4 features

        if not isinstance(features, (list, tuple)):
            raise RuntimeError(
                f"Expected base_extractor to return list/tuple of features, "
                f"got {type(features)}"
            )

        if len(features) != len(self.adapters):
            raise RuntimeError(
                f"Feature count mismatch: base_extractor returned {len(features)} "
                f"features, but we have {len(self.adapters)} adapters"
            )

        # 应用Adapter
        adapted_features = []
        for i, feat in enumerate(features):
            B, C, H, W = feat.shape

            # 转为(B, H*W, C)进行Adapter处理
            feat_flat = feat.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)

            # 应用Adapter（添加残差）
            adapted = self.adapters[i](feat_flat)  # (B, H*W, C)

            # 转换回(B, C, H, W)
            adapted = adapted.transpose(1, 2).view(B, C, H, W)
            adapted_features.append(adapted)

        return adapted_features


class UNetModelWithAdapter(UNetModel):
    """
    带Adapter的UNet模型

    对原有UNetModel的扩展：
    1. PVT特征提取器包含Adapter用于参数高效微调
    2. 提供冻结PVT骨干的接口
    3. 支持分层学习率设置
    4. 完整的参数追踪和验证

    属性：
    - self.pvt_extractor: PVTv2FeatureExtractorWithAdapter实例（如果启用）
    - self.use_pvt_fusion: 是否启用PVT融合
    - self.freeze_pvt: 是否冻结PVT骨干
    - self.adapter_reduction: Adapter降维比例
    - self.pvt_trainable: PVT是否可训练
    """

    def __init__(self,
                 image_size: int,
                 in_channels: int,
                 model_channels: int,
                 out_channels: int,
                 num_res_blocks: int,
                 attention_resolutions,
                 dropout: float = 0.0,
                 channel_mult=(1, 2, 4, 8),
                 conv_resample: bool = True,
                 dims: int = 2,
                 use_checkpoint: bool = False,
                 num_heads: int = 8,
                 num_head_channels: int = -1,
                 num_heads_upsample: int = -1,
                 use_scale_shift_norm: bool = False,
                 resblock_updown: bool = False,
                 use_new_attention_order: bool = False,
                 use_pvt_fusion: bool = True,
                 freeze_pvt: bool = True,
                 use_feb: bool = False,
                 use_cross_attn: bool = False,
                 adapter_reduction: int = 16,
                 pvt_ckpt_path: str = None,
                 **kwargs):
        """
        Args:
            image_size: 输入图像大小 (256, 512等)
            in_channels: 输入通道数 (通常为4: image+mask)
            model_channels: 基础模型通道数
            out_channels: 输出通道数 (分割通常为3)
            num_res_blocks: 每个分辨率级别的ResBlock数量
            attention_resolutions: 使用注意力的分辨率级别
            dropout: Dropout概率
            channel_mult: 各分辨率的通道倍增系数
            conv_resample: 是否使用卷积下采样
            dims: 2D或3D卷积 (通常为2)
            use_checkpoint: 是否使用梯度检查点
            num_heads: 多头注意力的头数
            num_head_channels: 每个头的通道数
            num_heads_upsample: 上采样时的头数
            use_scale_shift_norm: 是否使用缩放移位归一化
            resblock_updown: ResBlock是否处理上下采样
            use_new_attention_order: 是否使用新的注意力顺序
            use_pvt_fusion: 是否启用PVT融合 ✅ 关键参数
            freeze_pvt: 是否冻结PVT原始参数（仅训练Adapter） ✅ 关键参数
            use_feb: 是否使用特征增强块
            use_cross_attn: 是否使用交叉注意力
            adapter_reduction: Adapter的降维比例（默认16倍） ✅ 关键参数
            pvt_ckpt_path: PVT预训练权重路径
        """
        # ✅ 第一步：存储所有参数为实例属性（在调用父类前）
        self.use_pvt_fusion = use_pvt_fusion
        self.freeze_pvt = freeze_pvt
        self.adapter_reduction = adapter_reduction
        self.pvt_ckpt_path = pvt_ckpt_path

        # ✅ 第二步：调用父类初始化
        super().__init__(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            use_pvt_fusion=use_pvt_fusion,
            freeze_pvt=freeze_pvt,
            use_feb=use_feb,
            use_cross_attn=use_cross_attn,
            **kwargs
        )

        # ✅ 第三步：如果启用PVT融合，用带Adapter的版本替换
        if use_pvt_fusion:
            self._wrap_pvt_with_adapter()

    def _wrap_pvt_with_adapter(self):
        """
        将父类创建的pvt_extractor用Adapter版本包装

        关键点：
        - 父类创建的是 self.pvt_extractor，而不是 self.pvt
        - 我们在它上面包装一层Adapter
        """
        # ✅ 检查父类是否创建了pvt_extractor
        if not hasattr(self, 'pvt_extractor'):
            raise RuntimeError(
                "Parent class UNetModel did not create self.pvt_extractor. "
                "This suggests use_pvt_fusion was not properly handled in parent __init__. "
                "Check unet_base.py for issues."
            )

        if self.pvt_extractor is None:
            raise RuntimeError(
                "self.pvt_extractor is None despite use_pvt_fusion=True. "
                "Check parent class initialization."
            )

        # ✅ 保存原有的PVT提取器
        original_pvt = self.pvt_extractor

        try:
            # ✅ 用Adapter版本替换
            self.pvt_extractor = PVTv2FeatureExtractorWithAdapter(
                original_pvt,
                adapter_reduction=self.adapter_reduction
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to wrap PVT extractor with Adapter. Error: {e}"
            )

        # ✅ 冻结PVT骨干（仅保留Adapter可训练）
        if self.freeze_pvt:
            self.freeze_pvt_backbone()

    def freeze_pvt_backbone(self):
        """
        冻结PVT的所有原始参数，仅保留Adapter可训练

        这实现了参数高效微调：
        - PVT原始参数：requires_grad = False（冻结）
        - Adapter参数：requires_grad = True（可训练）
        """
        # ✅ 检查pvt_extractor是否存在
        if not hasattr(self, 'pvt_extractor') or self.pvt_extractor is None:
            print("Warning: pvt_extractor not found, nothing to freeze")
            return

        # ✅ 检查是否是Adapter版本
        if not hasattr(self.pvt_extractor, 'base_extractor'):
            print("Warning: pvt_extractor is not wrapped with Adapter")
            return

        # ✅ 冻结base_extractor（原始PVT参数）
        try:
            for param in self.pvt_extractor.base_extractor.parameters():
                param.requires_grad = False
        except Exception as e:
            raise RuntimeError(f"Failed to freeze base_extractor: {e}")

        # ✅ 确保Adapter可训练
        if not hasattr(self.pvt_extractor, 'adapters'):
            raise RuntimeError(
                "pvt_extractor.adapters not found. "
                "Ensure PVTv2FeatureExtractorWithAdapter was properly initialized."
            )

        try:
            for param in self.pvt_extractor.adapters.parameters():
                param.requires_grad = True
        except Exception as e:
            raise RuntimeError(f"Failed to enable Adapter parameters: {e}")

    def get_adapter_parameters(self):
        """
        获取所有Adapter参数（用于设置学习率）

        Returns:
            list: Adapter中的所有可训练参数
        """
        # ✅ 检查pvt_extractor和adapters
        if (not hasattr(self, 'pvt_extractor') or
            self.pvt_extractor is None or
            not hasattr(self.pvt_extractor, 'adapters')):
            return []

        try:
            return list(self.pvt_extractor.adapters.parameters())
        except Exception as e:
            print(f"Warning: Failed to get adapter parameters: {e}")
            return []

    def get_backbone_parameters(self):
        """
        获取PVT骨干参数（通常应该冻结）

        Returns:
            list: PVT原始参数
        """
        # ✅ 检查pvt_extractor和base_extractor
        if (not hasattr(self, 'pvt_extractor') or
            self.pvt_extractor is None or
            not hasattr(self.pvt_extractor, 'base_extractor')):
            return []

        try:
            return list(self.pvt_extractor.base_extractor.parameters())
        except Exception as e:
            print(f"Warning: Failed to get backbone parameters: {e}")
            return []

    def get_other_parameters(self):
        """
        获取除PVT外的其他模型参数

        Returns:
            list: U-Net主干和其他模块的参数
        """
        all_params = set(self.parameters())
        adapter_params = set(self.get_adapter_parameters())
        backbone_params = set(self.get_backbone_parameters())

        other_params = all_params - adapter_params - backbone_params
        return list(other_params)

    def get_parameter_groups(self, adapter_lr: float = 1e-4,
                            other_lr: float = 1e-4):
        """
        获取用于分层学习率的参数组

        这支持不同部分使用不同的学习率：
        - Adapter通常使用较小的学习率（如1e-4）
        - 其他部分使用主学习率

        使用方式：
          optimizer = torch.optim.Adam(
              model.get_parameter_groups(adapter_lr=1e-4, other_lr=1e-4),
          )

        Args:
            adapter_lr: Adapter层的学习率
            other_lr: 其他可训练层的学习率

        Returns:
            list: 适合传给optimizer的参数组列表

        Example:
            >>> model = UNetModelWithAdapter(...)
            >>> param_groups = model.get_parameter_groups(
            ...     adapter_lr=5e-5,  # Adapter学习率较低
            ...     other_lr=1e-4     # U-Net主干学习率
            ... )
            >>> optimizer = torch.optim.Adam(param_groups)
        """
        adapter_params = self.get_adapter_parameters()
        other_params = self.get_other_parameters()

        param_groups = []

        # ✅ 只添加非空参数组
        if adapter_params:
            param_groups.append({
                'params': adapter_params,
                'lr': adapter_lr,
                'name': 'adapter'
            })

        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': other_lr,
                'name': 'other'
            })

        if not param_groups:
            raise RuntimeError("No trainable parameters found in model")

        return param_groups

    def print_parameter_info(self):
        """
        打印详细的参数统计信息

        输出：
        - 总参数量
        - 可训练参数量
        - Adapter参数量
        - 其他参数量
        - 可训练比例
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        adapter_params = sum(p.numel() for p in self.get_adapter_parameters())
        backbone_params = sum(p.numel() for p in self.get_backbone_parameters())
        other_params = trainable_params - adapter_params

        print("\n" + "=" * 70)
        print("Model Parameter Information (UNetModelWithAdapter)")
        print("=" * 70)
        print(f"Total parameters:           {total_params:>12,}")
        print(f"Trainable parameters:       {trainable_params:>12,}")
        print(f"  ├─ Adapter parameters:    {adapter_params:>12,}")
        print(f"  └─ Other trainable:       {other_params:>12,}")
        print(f"Frozen parameters:          {total_params - trainable_params:>12,}")
        print(f"  └─ PVT backbone:          {backbone_params:>12,}")
        print("-" * 70)
        print(f"Trainable ratio:            {trainable_params / total_params * 100:>12.2f}%")
        print(f"Adapter ratio:              {adapter_params / total_params * 100:>12.2f}%")
        print("=" * 70 + "\n")

        # ✅ 打印配置信息
        print("Configuration:")
        print(f"  use_pvt_fusion:   {self.use_pvt_fusion}")
        print(f"  freeze_pvt:       {self.freeze_pvt}")
        print(f"  adapter_reduction: {self.adapter_reduction}")
        print(f"  pvt_trainable:    {getattr(self, 'pvt_trainable', 'N/A')}")
        print()
