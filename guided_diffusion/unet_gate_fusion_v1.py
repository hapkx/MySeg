"""
UNet with Adaptive Fusion Module (v1)

使用自适应融合权重替换原有的门控融合

使用方式：
  from guided_diffusion_new.unet_gate_fusion_v1 import UNetModelWithFusionV1

  model = UNetModelWithFusionV1(
      image_size=256,
      in_channels=4,
      model_channels=128,
      out_channels=3,
      use_pvt_fusion=True,
      fusion_version='adaptive',  # or 'v1'
  )
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "guided_diffusion"))

try:
    from unet_base import UNetModel
except ImportError:
    raise ImportError("Please ensure unet_base.py is in guided_diffusion/")

from .fusion_modules import AdaptiveFusionModule
import torch
import torch.nn as nn


class UNetModelWithFusionV1(UNetModel):
    """
    使用自适应融合模块的UNet

    对原有UNetModel的改进：
    1. 将GatedFusionModule替换为AdaptiveFusionModule
    2. 支持per-scale的自适应权重
    3. 更稳定的融合策略
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
                 **kwargs):
        """
        初始化使用自适应融合的UNet

        Args:
            use_pvt_fusion: 是否使用PVT融合
            其他参数同原UNetModel
        """
        # 先调用父类初始化
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

        # 如果启用PVT融合，替换融合模块
        if use_pvt_fusion and hasattr(self, 'fusion_modules'):
            self._replace_fusion_modules()

    def _replace_fusion_modules(self):
        """
        将GatedFusionModule替换为AdaptiveFusionModule

        融合点：
        - [2, 4, 8, 16] 尺度
        """
        if not hasattr(self, 'fusion_modules'):
            return

        new_fusion_modules = nn.ModuleList()

        for i, fusion_module in enumerate(self.fusion_modules):
            # 获取原有的fusion_module的配置
            # 通常结构为 GatedFusionModule(pvt_ch, unet_ch, out_ch)

            # 创建新的AdaptiveFusionModule
            # 假设out_channels已知
            pvt_channels = self.pvt.get_pvt_channels()  # [64, 128, 256, 256]
            unet_channels = self._get_unet_channel_at_fusion(i)
            out_channels = unet_channels

            new_fusion = AdaptiveFusionModule(
                pvt_ch=pvt_channels[i],
                unet_ch=unet_channels,
                out_ch=out_channels,
                dims=self.dims
            )

            new_fusion_modules.append(new_fusion)

        # 替换
        self.fusion_modules = new_fusion_modules

    def _get_unet_channel_at_fusion(self, fusion_idx: int) -> int:
        """
        获取融合点处UNet的通道数

        Args:
            fusion_idx: 融合索引（对应PVT的stage索引）

        Returns:
            UNet在该融合点的通道数
        """
        # 融合映射：{2: [0], 4: [1], 8: [2], 16: [3]}
        # fusion_idx 0, 1, 2, 3 对应 2, 4, 8, 16

        # 根据channel_mult计算
        # 典型：channel_mult = (1, 2, 4, 8)
        # 对应通道数：base, 2*base, 4*base, 8*base

        if fusion_idx == 0:
            return self.model_channels  # 2倍尺度
        elif fusion_idx == 1:
            return self.model_channels * 2  # 4倍尺度
        elif fusion_idx == 2:
            return self.model_channels * 4  # 8倍尺度
        elif fusion_idx == 3:
            return self.model_channels * 8  # 16倍尺度
        else:
            return self.model_channels * 8

    def print_fusion_info(self):
        """打印融合模块信息"""
        print("=" * 60)
        print("Fusion Module Information (Adaptive v1)")
        print("=" * 60)

        if hasattr(self, 'fusion_modules'):
            for i, fusion_module in enumerate(self.fusion_modules):
                print(f"Fusion {i}: {fusion_module.__class__.__name__}")
                if hasattr(fusion_module, 'pvt_ch'):
                    print(f"  PVT channels:  {fusion_module.pvt_ch}")
                    print(f"  UNet channels: {fusion_module.unet_ch}")
                    print(f"  Out channels:  {fusion_module.out_ch}")
        else:
            print("No fusion modules found")

        print("=" * 60)
