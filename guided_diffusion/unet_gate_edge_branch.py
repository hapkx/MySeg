"""
UNet with Edge Supervision Branch

添加边缘预测分支用于显式边界监督

模型输出从 (B, 3, H, W) 扩展到 (B, 4, H, W)
最后一个通道用于边缘预测

使用方式：
  from guided_diffusion_new.unet_gate_edge_branch import UNetModelWithEdgeBranch

  model = UNetModelWithEdgeBranch(...)
  output = model(x, timesteps)  # (B, 4, H, W)

  # 分离分割和边缘输出
  seg_output = output[:, :3, :, :]
  edge_output = output[:, 3:, :, :]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "guided_diffusion"))

try:
    from unet_base import (
        UNetModel,
        TimestepEmbedSequential,
        ResBlock,
        AttentionBlock,
        conv_nd,
        normalization,
        nn,
        th,
    )
except ImportError:
    raise ImportError("Please ensure unet_base.py is in guided_diffusion/")

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeHead(nn.Module):
    """
    边缘预测头

    输入：UNet的中间特征
    输出：(B, 1, H, W) 边缘预测
    """

    def __init__(self, in_channels: int, out_channels: int = 1, dims: int = 2):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数（通常为1）
            dims: 维度
        """
        super().__init__()

        if dims == 2:
            Conv = nn.Conv2d
        else:
            Conv = nn.Conv3d

        self.net = nn.Sequential(
            Conv(in_channels, in_channels // 2, 3, padding=1),
            nn.SiLU(),
            Conv(in_channels // 2, out_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class UNetModelWithEdgeBranch(UNetModel):
    """
    带边缘分支的UNet模型

    对原有UNetModel的扩展：
    1. 添加边缘预测头
    2. 修改输出层以输出分割+边缘（4通道）
    3. 提供分离输出的便利方法
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
        初始化边缘分支UNet

        Args:
            out_channels: 分割的输出通道数（通常为3）
                         实际输出会是 out_channels + 1（加边缘通道）
            ... 其他参数同原UNetModel
        """
        # 保存参数
        self.edge_out_channels = 1
        self.original_out_channels = out_channels

        # 调用父类初始化
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

        # 添加边缘预测分支
        # 获取最后一个block的特征维度
        last_block_channels = model_channels * channel_mult[-1]

        # 添加边缘头
        self.edge_head = EdgeHead(
            in_channels=last_block_channels,
            out_channels=1,
            dims=dims
        )

        # 替换输出层投影以输出合并的结果
        self._replace_output_projection()

    def _replace_output_projection(self):
        """
        替换输出投影层

        原有：out_proj 输出 (B, out_channels, H, W)
        新增：
          - seg_proj 输出 (B, original_out_channels, H, W)
          - edge_head 输出 (B, 1, H, W)
          - 合并为 (B, original_out_channels + 1, H, W)
        """
        # 保存原有的out_proj
        self.seg_proj = self.out

        # 创建最终的合并投影
        def merged_output(x_seg, x_edge):
            return torch.cat([x_seg, x_edge], dim=1)

        self.merged_output = merged_output

    def forward(self, x, timesteps, y=None, **kwargs):
        """
        前向传播

        Args:
            x: (B, C, H, W) 输入
            timesteps: (B,) 时间步
            y: (B,) 类标签（可选）

        Returns:
            output: (B, original_out_channels + 1, H, W)
                   前original_out_channels个通道为分割
                   最后1个通道为边缘
        """
        # 调用父类的forward获取特征
        # 注意：这里需要override处理，因为父类直接输出结果

        # 使用父类的处理流程但截断最后的输出层
        # 我们需要在最后的解码阶段分岔

        # 重新实现forward以便在最后分岔

        # 获取时间步embedding
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # 如果有PVT融合，获取PVT特征
        if hasattr(self, 'pvt') and hasattr(kwargs, 'img_full'):
            img_for_pvt = kwargs.get('img_full')
            if img_for_pvt is None and hasattr(x, 'shape'):
                img_for_pvt = x[:, :1, :, :]
            if img_for_pvt is not None:
                pvt_feats = self.pvt(img_for_pvt)
        else:
            pvt_feats = None

        # 编码器过程
        h = x.type(self.dtype)
        for i, module in enumerate(self.input_blocks):
            if hasattr(self, 'pvt') and i == 0:
                # 在第一个block融合PVT特征
                h = module(h, emb)
                if pvt_feats is not None:
                    h = self.fusion_modules[0](pvt_feats[0], h)
            else:
                h = module(h, emb)
            hs.append(h)

        # 中间block（带cross-attention如果启用）
        if hasattr(self, 'middle_block'):
            if self.use_cross_attn and pvt_feats is not None:
                h = self.middle_block(h, emb)
            else:
                h = self.middle_block(h, emb)

        # 解码器过程
        for i, module in enumerate(self.output_blocks):
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)

        # 在最后输出层之前分岔
        # 获取分割输出
        seg_output = self.seg_proj(h.type(self.dtype))  # (B, original_out_channels, H, W)

        # 获取边缘输出（使用中间特征）
        edge_output = self.edge_head(h)  # (B, 1, H, W)

        # 合并输出
        output = torch.cat([seg_output, edge_output], dim=1)  # (B, original_out_channels + 1, H, W)

        return output

    def get_seg_and_edge(self, output: torch.Tensor):
        """
        分离分割和边缘输出

        Args:
            output: (B, original_out_channels + 1, H, W)

        Returns:
            seg: (B, original_out_channels, H, W)
            edge: (B, 1, H, W)
        """
        seg = output[:, :self.original_out_channels, :, :]
        edge = output[:, self.original_out_channels:, :, :]
        return seg, edge

    def print_edge_branch_info(self):
        """打印边缘分支信息"""
        print("=" * 60)
        print("Edge Branch Information")
        print("=" * 60)
        print(f"Original out_channels: {self.original_out_channels}")
        print(f"Edge out_channels:     {self.edge_out_channels}")
        print(f"Total output channels: {self.original_out_channels + self.edge_out_channels}")
        print(f"Output shape:          (B, {self.original_out_channels + self.edge_out_channels}, H, W)")
        print("=" * 60)


# 注意：上面的forward方法需要改进
# 更简单的实现方式是重写最后的输出层

class UNetModelWithEdgeBranchSimple(UNetModel):
    """
    简化版的边缘分支UNet

    直接替换输出层以输出4个通道
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
        初始化边缘分支UNet（简化版）

        Args:
            out_channels: 分割的输出通道数（e.g., 3）
                         实际输出会加1用于边缘
        """
        self.original_out_channels = out_channels

        # 调用父类初始化，但输出通道数+1
        super().__init__(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels + 1,  # 加1用于边缘
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

    def get_seg_and_edge(self, output: torch.Tensor):
        """
        分离分割和边缘输出

        Args:
            output: (B, original_out_channels + 1, H, W)

        Returns:
            seg: (B, original_out_channels, H, W)
            edge: (B, 1, H, W)
        """
        seg = output[:, :self.original_out_channels, :, :]
        edge = output[:, self.original_out_channels:, :, :]
        return seg, edge

    def get_seg_only(self, output: torch.Tensor):
        """
        仅获取分割部分（推理时可用）

        Args:
            output: (B, original_out_channels + 1, H, W)

        Returns:
            seg: (B, original_out_channels, H, W)
        """
        return output[:, :self.original_out_channels, :, :]

    def print_edge_branch_info(self):
        """打印边缘分支信息"""
        print("=" * 60)
        print("Edge Branch Information (Simple)")
        print("=" * 60)
        print(f"Segmentation channels: {self.original_out_channels}")
        print(f"Edge channel:          1")
        print(f"Total output channels: {self.original_out_channels + 1}")
        print(f"Output shape:          (B, {self.original_out_channels + 1}, H, W)")
        print("=" * 60)


# 使用简化版作为默认
UNetModelWithEdgeBranch = UNetModelWithEdgeBranchSimple
