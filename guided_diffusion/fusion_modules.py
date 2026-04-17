"""
特征融合模块的两种实现方案

方案v1：自适应融合权重 (Adaptive Fusion)
  F_fused = α·PVT_feat + β·UNet_feat
  其中α和β由注意力门控动态计算

方案v2：交叉分支注意力 (Cross-Branch Attention, CBA)
  使用UNet特征作Query，PVT特征作Key/Value
  通过多头注意力建模两个分支的交互
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 方案v1：自适应融合权重
# ============================================================================

class AdaptiveFusionModule(nn.Module):
    """
    自适应融合权重方案

    特点：
    1. 每个缩放层级有独立的α和β权重
    2. 权重由注意力门控动态计算
    3. 计算量小，参数少
    4. 易于理解和调试

    数学形式：
      F_fused_i = α_i · PVT_feat_i + β_i · UNet_feat_i
      其中 α_i, β_i = softmax([α_logit, β_logit])
    """

    def __init__(self, pvt_ch: int, unet_ch: int, out_ch: int, dims: int = 2):
        """
        Args:
            pvt_ch: PVT特征的通道数
            unet_ch: UNet特征的通道数
            out_ch: 输出通道数
            dims: 维度（2为2D图像，3为3D体积）
        """
        super().__init__()
        self.pvt_ch = pvt_ch
        self.unet_ch = unet_ch
        self.out_ch = out_ch
        self.dims = dims

        # 投影层：将两个特征投影到相同的通道维度
        if dims == 2:
            Conv = nn.Conv2d
            AvgPool = nn.AdaptiveAvgPool2d
        else:
            Conv = nn.Conv3d
            AvgPool = nn.AdaptiveAvgPool3d

        self.pvt_proj = nn.Sequential(
            Conv(pvt_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch) if dims == 2 else nn.BatchNorm3d(out_ch),
            nn.SiLU()
        )

        self.unet_proj = nn.Sequential(
            Conv(unet_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch) if dims == 2 else nn.BatchNorm3d(out_ch),
            nn.SiLU()
        )

        # 权重计算：基于两个特征的组合
        self.weight_gate = nn.Sequential(
            Conv(out_ch * 2, out_ch // 2, 1),
            nn.SiLU(),
            Conv(out_ch // 2, 2, 1),  # 输出2个权重值（alpha和beta）
        )

        # 输出投影
        self.out_proj = Conv(out_ch, out_ch, 3, padding=1)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.01)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

        # 融合强度控制
        self.fusion_strength = nn.Parameter(torch.tensor(0.0))

    def forward(self, f_pvt: torch.Tensor, x_unet: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_pvt: PVT特征 (B, pvt_ch, H, W)
            x_unet: UNet特征 (B, unet_ch, H, W)

        Returns:
            融合后的特征 (B, out_ch, H, W)
        """
        # 投影到相同通道
        f_pvt_proj = self.pvt_proj(f_pvt)  # (B, out_ch, H_pvt, W_pvt)
        x_unet_proj = self.unet_proj(x_unet)  # (B, out_ch, H, W)

        # 上采样PVT以匹配UNet的空间尺寸
        if f_pvt_proj.shape[2:] != x_unet_proj.shape[2:]:
            mode = 'bilinear' if self.dims == 2 else 'trilinear'
            f_pvt_proj = F.interpolate(
                f_pvt_proj,
                size=x_unet_proj.shape[2:],
                mode=mode,
                align_corners=False
            )

        # 组合特征用于权重计算
        combined = torch.cat([f_pvt_proj, x_unet_proj], dim=1)

        # 计算权重：[alpha, beta]
        weight_logits = self.weight_gate(combined)  # (B, 2, H, W)
        weights = F.softmax(weight_logits, dim=1)  # (B, 2, H, W) 求和为1
        alpha = weights[:, 0:1, :, :]  # (B, 1, H, W)
        beta = weights[:, 1:2, :, :]   # (B, 1, H, W)

        # 加权融合
        fused = alpha * f_pvt_proj + beta * x_unet_proj  # (B, out_ch, H, W)

        # 输出投影
        fused = self.out_proj(fused)

        # 融合强度控制
        fusion_strength = torch.tanh(self.fusion_strength)

        # 残差连接
        output = x_unet + fusion_strength * fused

        return output


# ============================================================================
# 方案v2：交叉分支注意力 (Cross-Branch Attention)
# ============================================================================

class CrossBranchAttentionModule(nn.Module):
    """
    交叉分支注意力方案 (CBA)

    特点：
    1. 基于多头自注意力机制
    2. UNet特征作为Query（查询），PVT特征作为Key/Value
    3. 建模UNet和PVT特征之间的交互关系
    4. 更复杂但表现力更强

    数学形式：
      Attention(Q, K, V) = softmax(Q·K^T / sqrt(d)) · V
      其中 Q来自UNet，K和V来自PVT
      最后加上残差连接保证信息流
    """

    def __init__(self, dim: int, num_heads: int = 8, dims: int = 2):
        """
        Args:
            dim: 特征维度（通道数）
            num_heads: 注意力头数
            dims: 维度（2为2D图像）
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dims = dims

        assert dim % num_heads == 0, f"dim({dim}) must be divisible by num_heads({num_heads})"

        # Q, K, V的投影层
        if dims == 2:
            Conv = nn.Conv2d
        else:
            Conv = nn.Conv3d

        self.query_proj = Conv(dim, dim, 1)
        self.key_proj = Conv(dim, dim, 1)
        self.value_proj = Conv(dim, dim, 1)

        # 输出投影
        self.out_proj = Conv(dim, dim, 1)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.01)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

        # 融合强度
        self.fusion_strength = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_unet: torch.Tensor, f_pvt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_unet: UNet特征 (B, C, H, W)
            f_pvt: PVT特征 (B, C, H_pvt, W_pvt)

        Returns:
            融合后的特征 (B, C, H, W)
        """
        B, C, H, W = x_unet.shape

        # 上采样PVT以匹配UNet的空间尺寸
        if f_pvt.shape[2:] != x_unet.shape[2:]:
            mode = 'bilinear' if self.dims == 2 else 'trilinear'
            f_pvt = F.interpolate(
                f_pvt,
                size=(H, W),
                mode=mode,
                align_corners=False
            )

        # 计算Q, K, V
        Q = self.query_proj(x_unet)  # (B, C, H, W)
        K = self.key_proj(f_pvt)      # (B, C, H, W)
        V = self.value_proj(f_pvt)    # (B, C, H, W)

        # 重形状为多头格式
        # (B, C, H, W) -> (B, num_heads, head_dim, H*W)
        Q = Q.view(B, self.num_heads, self.head_dim, H * W)
        K = K.view(B, self.num_heads, self.head_dim, H * W)
        V = V.view(B, self.num_heads, self.head_dim, H * W)

        # 计算注意力：(B, num_heads, H*W, H*W)
        attn = torch.matmul(Q.transpose(-2, -1), K)  # (B, num_heads, H*W, H*W)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)

        # 应用注意力到值
        out = torch.matmul(attn, V.transpose(-2, -1))  # (B, num_heads, H*W, head_dim)
        out = out.transpose(-2, -1)  # (B, num_heads, head_dim, H*W)

        # 重塑回原始格式
        out = out.view(B, C, H, W)

        # 输出投影
        out = self.out_proj(out)

        # 融合强度
        fusion_strength = torch.tanh(self.fusion_strength)

        # 残差连接：保证UNet特征的主要信息得以保留
        output = x_unet + fusion_strength * out

        return output


# ============================================================================
# 辅助函数：选择融合模块
# ============================================================================

def create_fusion_module(version: str, pvt_ch: int, unet_ch: int, out_ch: int,
                        num_heads: int = 8, dims: int = 2):
    """
    工厂函数：创建融合模块

    Args:
        version: 'v1' 或 'v2'
        pvt_ch: PVT特征通道数
        unet_ch: UNet特征通道数
        out_ch: 输出通道数
        num_heads: 仅用于v2
        dims: 维度

    Returns:
        融合模块实例
    """
    if version == 'v1':
        return AdaptiveFusionModule(pvt_ch, unet_ch, out_ch, dims)
    elif version == 'v2':
        # v2版本使用统一的out_ch
        return CrossBranchAttentionModule(out_ch, num_heads, dims)
    else:
        raise ValueError(f"Unknown fusion version: {version}")
