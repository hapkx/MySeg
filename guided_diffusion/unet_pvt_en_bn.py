"""
有encoder  有加载预训练pvt参数 有cross_attn

"""
from abc import abstractmethod
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
import timm
# from timm.models.pvt_v2 import checkpoint_filter_fn
import re

NUM_CLASSES = 3  # 多类别分割类别数

# pvt
def remap_pvtv2_official_to_timm(state_dict):
    """
    把官方键名映射到timm裸backbone键名
    """
    out_dict = {}
    if 'patch_embed.proj.weight' in state_dict or 'stages.0.blocks.0.attn.qkv.weight' in state_dict:
        return state_dict  # 已经是timm格式，无需转换
    
    for k, v in state_dict.items():
        if k.startswith('head.'):
            continue  # 跳过分类头权重)

        # patch embedding
        if k.startswith('patch_embed1'):
            k = k.replace('patch_embed1', 'patch_embed')
        elif k.startswith('patch_embed2'):
            k = k.replace('patch_embed2', 'stages_1.downsample')
        elif k.startswith('patch_embed3'):
            k = k.replace('patch_embed3', 'stages_2.downsample')
        elif k.startswith('patch_embed4'):
            k = k.replace('patch_embed4', 'stages_3.downsample')

        k = re.sub(
            r'^block(\d+)\.(\d+)\.(.*)',
            lambda x: f'stages_{int(x.group(1))-1}.blocks.{x.group(2)}.{x.group(3)}',
            k
        )
        k = re.sub(
            r'^norm(\d+)\.(.*)',
            lambda x: f'stages_{int(x.group(1))-1}.norm.{x.group(2)}',
            k
        )

        k = k.replace('dwconv.dwconv', 'dwconv') 

        out_dict[k] = v
    return out_dict


class PVTv2FeatureExtractor(nn.Module):
    """PVT v2 b0特征提取器，输出4层级特征F1~F4"""
    def __init__(self, pretrained=True, local_ckpt_path="pvt_v2_b0.pth"):
        super().__init__()
        # 加载预训练的PVT v2 b0，移除分类头，输出多层特征
        self.pvt = timm.create_model(
            'pvt_v2_b0',
            pretrained=False,  # 使用官方预训练权重
            features_only=True,  # 输出多层特征
            out_indices=(0, 1, 2, 3)  # 输出4层特征
        )
        # 2. 加载本地预训练权重（如果指定了pretrained）
        if pretrained:
            self._load_local_pretrained(local_ckpt_path)
        # 动态获取通道数
        self.pvt_channel_list = [info['num_chs'] for info in self.pvt.feature_info]

        # 验证PVT输出特征层数（必须为4层）
        assert len(self.pvt.feature_info) == 4, "PVT v2 b0必须输出4层特征"

    def _load_local_pretrained(self, ckpt_path):
        """加载本地.pth文件，并处理键名不匹配问题"""
        try:
            # 加载本地权重文件
            checkpoint = th.load(ckpt_path, map_location=th.device('cpu'))  # 先加载到CPU，避免GPU显存问题
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

            # 移除module.前缀（通用处理逻辑）
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    cleaned_state_dict[k[7:]] = v  # 去掉module.前缀
                else:
                    cleaned_state_dict[k] = v

            remapped_state_dict = remap_pvtv2_official_to_timm(cleaned_state_dict)  # 使用PVT提供的过滤函数处理键名            
            msg = self.pvt.load_state_dict(remapped_state_dict, strict=False)  # 加载权重，允许部分键名不匹配

            
            print(f"✅ 成功加载本地PVT权重：{ckpt_path}")
            print("missing_keys:", msg.missing_keys)  # 显示缺失的键（如果有）
            print("unexpected_keys:", msg.unexpected_keys)  # 显示多余的键（如果有）

            total_model_keys = len(self.pvt.state_dict().keys())
            loaded_keys = total_model_keys - len(msg.missing_keys)
            print(f"loaded_keys: {loaded_keys}/{total_model_keys}")
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ 本地权重文件不存在：{ckpt_path}，请检查文件路径")
        except Exception as e:
            raise RuntimeError(f"❌ 加载权重失败：{str(e)}，可能是权重文件格式不匹配")

    
    def get_pvt_channels(self):
        """返回PVT各层通道数列表"""
        return self.pvt_channel_list
    
    def forward(self, x):
        # x: [B, 3, H, W] → PVT标准输入为3通道RGB图像
        feats = self.pvt(x)  # 输出列表：[F1, F2, F3, F4]
        
        return feats
    

class FeatureFusionModule(nn.Module):
    """
    特征融合模块：
    1) 先把 PVT 特征用 1x1 conv 映射到目标通道
    2) 上采样到 UNet 特征大小
    3) 再用 3x3 conv 平滑
    4) 与 UNet 特征拼接融合
    """
    def __init__(self, pvt_channels, unet_channels, out_channels, dims=2):
        super().__init__()
        self.pvt_adapter = nn.Sequential(
            conv_nd(dims, pvt_channels, out_channels, 1),
            normalization(out_channels),
            nn.SiLU(),
        )
        self.pvt_refine = nn.Sequential(
            conv_nd(dims, out_channels, out_channels, 3, padding=1),
            normalization(out_channels),
            nn.SiLU(),
        )
        self.unet_adapter = nn.Sequential(
            conv_nd(dims, unet_channels, out_channels, 1),
            normalization(out_channels),
            nn.SiLU(),
        )
        self.fusion_conv = nn.Sequential(
            conv_nd(dims, out_channels * 2, out_channels, 3, padding=1),
            normalization(out_channels),
            nn.SiLU(),
        )

    def forward(self, f_pvt, x_unet):
        # PVT: 先通道映射，再上采样，再平滑
        f_pvt = self.pvt_adapter(f_pvt)
        if f_pvt.shape[2:] != x_unet.shape[2:]:
            f_pvt = F.interpolate(
                f_pvt,
                size=x_unet.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        f_pvt = self.pvt_refine(f_pvt)

        # UNet 特征通道对齐
        x_unet = self.unet_adapter(x_unet)

        fused = th.cat([f_pvt, x_unet], dim=1)
        fused = self.fusion_conv(fused)
        return fused
    
    
class CrossAttentionBlock(nn.Module):
    """
    真正的 cross-attention:
    query 来自当前 UNet bottleneck 特征
    key/value 来自 PVT 的 F4 特征
    """
    def __init__(self, channels, context_channels, num_heads=4, use_checkpoint=False):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.context_channels = context_channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm_x = normalization(channels)
        self.norm_ctx = normalization(context_channels)

        self.q_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv1d(context_channels, channels, kernel_size=1)
        self.v_proj = nn.Conv1d(context_channels, channels, kernel_size=1)

        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.proj_out = zero_module(nn.Conv1d(channels, channels, kernel_size=1))

    def forward(self, x, context):
        return checkpoint(self._forward, (x, context), self.parameters(), self.use_checkpoint)

    def _forward(self, x, context):
        b, c, h, w = x.shape
        _, cc, hc, wc = context.shape

        x_in = x

        x = self.norm_x(x).reshape(b, c, h * w)           # [B, C, HW]
        context = self.norm_ctx(context).reshape(b, cc, hc * wc)

        q = self.q_proj(x).permute(0, 2, 1)               # [B, HW, C]
        k = self.k_proj(context).permute(0, 2, 1)         # [B, HW_ctx, C]
        v = self.v_proj(context).permute(0, 2, 1)         # [B, HW_ctx, C]

        attn_out, _ = self.attn(q, k, v, need_weights=False)
        attn_out = self.proj_out(attn_out.permute(0, 2, 1)).reshape(b, c, h, w)

        return x_in + attn_out
# 
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction=16, dims=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) if dims == 2 else nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1) if dims == 2 else nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            conv_nd(dims, in_channels, in_channels // reduction, 1),
            nn.SiLU(),  # 使用与源代码一致的激活函数
            conv_nd(dims, in_channels // reduction, in_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7, dims=2):
        super(SpatialAttention, self).__init__()
        self.conv = conv_nd(dims, 2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = th.mean(x, dim=1, keepdim=True)
        max_out, _ = th.max(x, dim=1, keepdim=True)
        x_cat = th.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)
   
class DualLevelResidual(nn.Module):
    """
    双层次残差（DLR）组件
    """
    def __init__(
        self, 
        channels, 
        emb_channels, 
        out_channels=None, 
        dropout=0.0,
        dims=2,
        use_checkpoint=False,
        use_scale_shift_norm=False
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # 第一个并行卷积块 - 对应公式中的 X1 = ReLu(BN(conv3×3(Fi)))
        self.conv_block1 = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        
        # 第二个并行卷积块 - 对应公式中的 X2 = ReLu(BN(conv3×3(Xi)))
        self.conv_block2 = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        
        # 1x1卷积用于残差连接 - 对应公式中的 conv1×1(Fi)
        self.residual_conv = conv_nd(dims, channels, self.out_channels, 1)
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, self.out_channels),
        )
        
    def forward(self, x, emb):
        """
        Apply the DLR block to a Tensor, conditioned on a timestep embedding.
        
        :param x: an [N x C x ...] Tensor of features (Fi in your formula).
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs (X3 in your formula).
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )
    
    def _forward(self, x, emb):
        Fi = x
        Xi = x
        X1 = self.conv_block1(Fi)
        X2 = self.conv_block2(Xi)

        emb_out = self.emb_layers(emb).type(X1.dtype)
        while len(emb_out.shape) < len(X1.shape):
            emb_out = emb_out[..., None]

        X1 = X1 + emb_out
        X2 = X2 + emb_out
        h = X2 + X1 + self.residual_conv(Fi)
            
        return h

class AdaptiveFeatureSelection(nn.Module):
    """自适应特征选择（AFS）组件"""
    def __init__(self, channels, reduction=16, dims=2):
        super(AdaptiveFeatureSelection, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction, dims)
        self.spatial_attention = SpatialAttention(dims=dims)
        
    def forward(self, X3):
        # X4 = X3 ⊙ Channel(X3)
        channel_att = self.channel_attention(X3)
        X4 = X3 * channel_att
        
        # X5 = X4 ⊙ Spatial(X4)
        spatial_att = self.spatial_attention(X4)
        X5 = X4 * spatial_att
        
        return X5


class FeatureEnhancementBlock(TimestepBlock):
    """
    特征增强编码器块，用于替代UNet中的ResBlock
    结合了双层次残差(DLR)和自适应特征选择(AFS)
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout=0.0,
        out_channels=None,
        dims=2,
        use_checkpoint=False,
        use_scale_shift_norm=False,
        reduction=16,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        
        # 双层次残差组件
        self.dlr = DualLevelResidual(
            channels=channels,
            emb_channels=emb_channels,
            out_channels=self.out_channels,
            dropout=dropout,
            dims=dims,
            use_scale_shift_norm=use_scale_shift_norm
        )
        
        # 自适应特征选择组件
        self.afs = AdaptiveFeatureSelection(
            channels=self.out_channels,
            reduction=reduction,
            dims=dims
        )
        
        # 跳跃连接
        if self.out_channels != channels:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
            
    def forward(self, x, emb):
        """
        Apply the feature enhancement block to a Tensor, conditioned on timestep embedding.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        ) if self.use_checkpoint else self._forward(x, emb)
        
    def _forward(self, x, emb):
        # DLR组件处理 - 得到X3
        x3 = self.dlr(x, emb)
        
        # AFS组件处理 - 得到X5
        x5 = self.afs(x3)
        
        # 添加跳跃连接
        return self.skip_connection(x) + x5


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        # 记录在哪些 input_block 末尾做 PVT 融合
        # 当前浅层 U-Net: ds=1,2,4,8 分别对应 256,128,64,32
        self.fusion_block_ids = {}

        for level, mult in enumerate(channel_mult):
            for block_idx in range(num_res_blocks):
                layers = [
                    FeatureEnhancementBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels

                # 可选：在 32x32 这一层保留 self-attention
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

                # 每个 stage 最后一个 block 后做一次融合
                if block_idx == num_res_blocks - 1:
                    fusion_map = {1: 0, 2: 1, 4: 2, 8: 3}
                    if ds in fusion_map:
                        self.fusion_block_ids[len(self.input_blocks) - 1] = fusion_map[ds]

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
            
        self.middle_block1 = ResBlock(
            ch,
            time_embed_dim,
            dropout,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
        )

        self.middle_cross_attn = CrossAttentionBlock(
            channels=ch,
            context_channels=256,   # pvt_v2_b0 的 F4 通道数
            num_heads=num_heads,
            use_checkpoint=use_checkpoint,
        )

        self.middle_block2 = ResBlock(
            ch,
            time_embed_dim,
            dropout,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
        )

        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        # 初始化PVT特征提取器 
        self.pvt_extractor = PVTv2FeatureExtractor(
            pretrained=True, 
            local_ckpt_path="/home/nas2/biod/piankexin/model/guided_diffusion/pvt_v2_b0.pth"
        )
        # 可选：先冻结PVT预训练参数（训练后期可解冻微调）
        # for param in self.pvt_extractor.parameters():
        #     param.requires_grad = False
        # 初始化特征融合模块 
        pvt_channels = self.pvt_extractor.get_pvt_channels()
        # UNet各层通道数：channel_mult * model_channels
        self.fusion_stage_channels = [model_channels * mult for mult in channel_mult]
        # 初始化4个层级的融合模块
        self.fusion_modules = nn.ModuleList([
            FeatureFusionModule(
                pvt_channels=pvt_channels[i], 
                unet_channels=self.fusion_stage_channels[i], 
                out_channels=self.fusion_stage_channels[i],
                dims=dims,)
            for i in range(4)
        ])

    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block1.apply(convert_module_to_f16)
        self.middle_block2.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
        self.out.apply(convert_module_to_f16)
        self.fusion_modules.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block1.apply(convert_module_to_f32)
        self.middle_block2.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
        self.out.apply(convert_module_to_f32)
        self.fusion_modules.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)


        # if x.shape[1] != 3:
        #     x_gray = x[:, :-NUM_CLASSES, :, :]  
        #     x_pvt = th.cat([x_gray, x_gray, x_gray], dim=1)  # 单通道→3通道，多通道可根据需求调整
        # else:
        #     x_pvt = x
        if x.shape[1] != 4:
            raise ValueError(f"Expected input shape [b, 4, h, w], but got {x.shape[1]} channels.")

        x_img = x[:, :-NUM_CLASSES, :, :]
        x_pvt = x_img.repeat(1, 3, 1, 1)  # 从单通道复制到3通道，适配PVT输入要求
        x_pvt = (x_pvt + 1.0) / 2.0

        mean = th.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = th.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_pvt = (x_pvt - mean) / std

        # 提取PVT的4层特征 F1~F4
        pvt_feats = self.pvt_extractor(x_pvt)  # [F1, F2, F3, F4]

        h = x.type(self.dtype)
        hs = []

        for block_id, module in enumerate(self.input_blocks):
            h = module(h, emb)

            if block_id in self.fusion_block_ids:
                fusion_idx = self.fusion_block_ids[block_id]
                h = self.fusion_modules[fusion_idx](pvt_feats[fusion_idx], h)
                # print("fusion_idx:", fusion_idx, "pvt:", pvt_feats[fusion_idx].shape, "unet:", h.shape)

            hs.append(h)
        
        # 
        # print("PVT:", [f.shape for f in pvt_feats])
        # print("encoder:", [f.shape for f in hs])
        
        # middle
        h = self.middle_block1(h, emb)
        h = self.middle_cross_attn(h, pvt_feats[3])  # F4与middle层融合
        h = self.middle_block2(h, emb)

        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)
