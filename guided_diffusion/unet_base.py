"""
基础U-Net模型 - 标准实现

这是一个干净的、标准化的U-Net基础实现，作为所有U-Net变体的基础。
支持以下功能：
  - PVT v2 多尺度特征融合 (use_pvt_fusion)
  - 特征增强分支 (use_feb)
  - 交叉注意力机制 (use_cross_attn)

所有这些特性都可以通过参数进行开关，而不改变核心架构。

核心特性：
  - 标准的U-Net编码器-解码器架构
  - TimestepEmbedding用于扩散时间步
  - ResBlock用于跳跃连接
  - AttentionBlock用于自注意力
  - 轻量级可选模块用于PVT融合、FEB和CrossAttention

使用示例：
  from guided_diffusion.unet_base import UNetModel

  model = UNetModel(
      image_size=256,
      in_channels=4,
      model_channels=128,
      out_channels=3,
      num_res_blocks=2,
      attention_resolutions=[16, 8],
      channel_mult=(1, 2, 4, 8),
      use_pvt_fusion=True,
      freeze_pvt=True,
  )
"""

from abc import abstractmethod
import math
import re
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import timm

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


# =========================
# PVT v2 特征提取器
# =========================

def remap_pvtv2_official_to_timm(state_dict):
    """将官方PVT权重转换为timm格式"""
    out_dict = {}
    if 'patch_embed.proj.weight' in state_dict or 'stages.0.blocks.0.attn.qkv.weight' in state_dict:
        return state_dict

    for k, v in state_dict.items():
        if k.startswith('head.'):
            continue

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
            lambda x: f'stages_{int(x.group(1)) - 1}.blocks.{x.group(2)}.{x.group(3)}',
            k
        )
        k = re.sub(
            r'^norm(\d+)\.(.*)',
            lambda x: f'stages_{int(x.group(1)) - 1}.norm.{x.group(2)}',
            k
        )

        k = k.replace('dwconv.dwconv', 'dwconv')
        out_dict[k] = v
    return out_dict


class PVTv2FeatureExtractor(nn.Module):
    """PVT v2 b0 特征提取器，输出4层特征"""
    def __init__(self, pretrained=True, local_ckpt_path="pvt_v2_b0.pth"):
        super().__init__()
        self.pvt = timm.create_model(
            'pvt_v2_b0',
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        if pretrained:
            self._load_local_pretrained(local_ckpt_path)

        self.pvt_channel_list = [info['num_chs'] for info in self.pvt.feature_info]
        assert len(self.pvt.feature_info) == 4, "PVT v2 b0 must output 4 stages"

    def _load_local_pretrained(self, ckpt_path):
        checkpoint_data = th.load(ckpt_path, map_location=th.device('cpu'))
        state_dict = checkpoint_data['state_dict'] if 'state_dict' in checkpoint_data else checkpoint_data

        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                cleaned_state_dict[k[7:]] = v
            else:
                cleaned_state_dict[k] = v

        remapped_state_dict = remap_pvtv2_official_to_timm(cleaned_state_dict)
        msg = self.pvt.load_state_dict(remapped_state_dict, strict=False)

    def get_pvt_channels(self):
        return self.pvt_channel_list

    def forward(self, x):
        return self.pvt(x)


# =========================
# 融合模块
# =========================

class GatedFusionModule(nn.Module):
    """门控融合模块：将PVT特征融合到UNet中"""
    def __init__(self, pvt_channels, unet_channels, out_channels, dims=2):
        super().__init__()
        hidden = max(out_channels // 4, 16)

        self.pvt_proj = nn.Sequential(
            conv_nd(dims, pvt_channels, out_channels, 1),
            normalization(out_channels),
            nn.SiLU(),
        )
        self.unet_proj = nn.Sequential(
            conv_nd(dims, unet_channels, out_channels, 1),
            normalization(out_channels),
            nn.SiLU(),
        )

        self.gate = nn.Sequential(
            conv_nd(dims, out_channels * 2, hidden, 1),
            nn.SiLU(),
            conv_nd(dims, hidden, out_channels, 1),
            nn.Sigmoid(),
        )

        self.out_proj = zero_module(
            conv_nd(dims, out_channels, out_channels, 3, padding=1)
        )
        self.alpha = nn.Parameter(th.tensor(0.0))

    def forward(self, f_pvt, x_unet):
        f_pvt = self.pvt_proj(f_pvt)
        if f_pvt.shape[2:] != x_unet.shape[2:]:
            f_pvt = F.interpolate(
                f_pvt,
                size=x_unet.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        x_proj = self.unet_proj(x_unet)
        gate = self.gate(th.cat([x_proj, f_pvt], dim=1))
        injected = self.out_proj(gate * f_pvt)
        return x_unet + th.tanh(self.alpha) * injected


# =========================
# 核心块
# =========================

class TimestepBlock(nn.Module):
    """任何接收timestep embeddings的模块"""

    @abstractmethod
    def forward(self, x, emb):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """顺序模块，将timestep embeddings传递给支持的子模块"""

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
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
    """残差块"""
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
    """自注意力块"""
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
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels

        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)

        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x_ = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x_))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x_ + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class FeatureEnhancementBlock(TimestepBlock):
    """特征增强块 (FEB)"""
    def __init__(
        self,
        channels,
        emb_channels,
        dropout=0.0,
        out_channels=None,
        dims=2,
        use_checkpoint=False,
        use_scale_shift_norm=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint

        self.resblock = ResBlock(
            channels=channels,
            emb_channels=emb_channels,
            dropout=dropout,
            out_channels=self.out_channels,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
        )

        self.out_proj = zero_module(
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
        )

        if self.out_channels != channels:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x, emb):
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        ) if self.use_checkpoint else self._forward(x, emb)

    def _forward(self, x, emb):
        h = self.resblock(x, emb)
        h = self.out_proj(h)
        return self.skip_connection(x) + h


class CrossAttentionBlock(nn.Module):
    """交叉注意力块"""
    def __init__(self, channels, context_channels, num_heads=4, use_checkpoint=False):
        super().__init__()
        assert channels % num_heads == 0
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
        self.alpha = nn.Parameter(th.tensor(0.0))

    def forward(self, x, context):
        return checkpoint(self._forward, (x, context), self.parameters(), self.use_checkpoint)

    def _forward(self, x, context):
        b, c, h, w = x.shape
        _, cc, hc, wc = context.shape

        x_in = x

        x_ = self.norm_x(x).reshape(b, c, h * w)
        ctx_ = self.norm_ctx(context).reshape(b, cc, hc * wc)

        q = self.q_proj(x_).permute(0, 2, 1)
        k = self.k_proj(ctx_).permute(0, 2, 1)
        v = self.v_proj(ctx_).permute(0, 2, 1)

        attn_out, _ = self.attn(q, k, v, need_weights=False)
        attn_out = self.proj_out(attn_out.permute(0, 2, 1)).reshape(b, c, h, w)

        return x_in + th.tanh(self.alpha) * attn_out


# =========================
# UNet模型
# =========================

class UNetModel(nn.Module):
    """标准U-Net模型，支持可选的PVT融合、FEB和CrossAttention"""

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
        use_pvt_fusion=False,
        use_feb=False,
        use_cross_attn=False,
        pvt_ckpt_path="pvt_v2_b0.pth",
        freeze_pvt=True,
        image_channels=1,
        mask_channels=None,
        pvt_fusion_levels=(8,),
        feb_levels=(8,),
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

        self.use_pvt_fusion = use_pvt_fusion
        self.use_feb = use_feb
        self.use_cross_attn = use_cross_attn
        self.image_channels = image_channels
        self.mask_channels = out_channels if mask_channels is None else mask_channels
        self.pvt_fusion_levels = tuple(sorted(set(pvt_fusion_levels)))
        self.feb_levels = tuple(sorted(set(feb_levels)))

        assert self.image_channels > 0, "image_channels must be positive"
        assert self.image_channels + self.mask_channels == self.in_channels, (
            f"Expected image_channels + mask_channels == in_channels, but got "
            f"{self.image_channels} + {self.mask_channels} != {self.in_channels}"
        )

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        self.fusion_block_specs = {}
        pvt_stage_by_ds = {4: 0, 8: 1, 16: 2, 32: 3}

        for level, mult in enumerate(channel_mult):
            for block_idx in range(num_res_blocks):
                layers = [
                    ResBlock(
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

                if self.use_feb and ds in self.feb_levels and block_idx == num_res_blocks - 1:
                    layers.append(
                        FeatureEnhancementBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                block_id = len(self.input_blocks) - 1
                self._feature_size += ch
                input_block_chans.append(ch)

                if (
                    self.use_pvt_fusion
                    and block_idx == num_res_blocks - 1
                    and ds in self.pvt_fusion_levels
                    and ds in pvt_stage_by_ds
                ):
                    self.fusion_block_specs[block_id] = {
                        "pvt_idx": pvt_stage_by_ds[ds],
                        "channels": ch,
                    }

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
                        ) if resblock_updown else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.middle_cross_attn = None
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
                        ) if resblock_updown else Upsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        # 初始化PVT融合模块
        if self.use_pvt_fusion or self.use_cross_attn:
            self.pvt_extractor = PVTv2FeatureExtractor(
                pretrained=True,
                local_ckpt_path=pvt_ckpt_path,
            )

            for param in self.pvt_extractor.parameters():
                param.requires_grad = False

            if not freeze_pvt:
                for param in self.pvt_extractor.parameters():
                    param.requires_grad = True

            self.pvt_trainable = not freeze_pvt
            if not self.pvt_trainable:
                self.pvt_extractor.eval()

            pvt_channels = self.pvt_extractor.get_pvt_channels()
            self.fusion_modules = nn.ModuleDict()
            for block_id, spec in self.fusion_block_specs.items():
                self.fusion_modules[str(block_id)] = GatedFusionModule(
                    pvt_channels=pvt_channels[spec["pvt_idx"]],
                    unet_channels=spec["channels"],
                    out_channels=spec["channels"],
                    dims=dims,
                )

            if self.use_cross_attn:
                self.middle_cross_attn = CrossAttentionBlock(
                    channels=channel_mult[-1] * model_channels,
                    context_channels=pvt_channels[1],
                    num_heads=num_heads,
                    use_checkpoint=use_checkpoint,
                )
        else:
            self.pvt_extractor = None
            self.fusion_modules = None
            self.pvt_trainable = False

    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        if self.middle_cross_attn is not None:
            self.middle_cross_attn.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
        if self.fusion_modules is not None:
            self.fusion_modules.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        if self.middle_cross_attn is not None:
            self.middle_cross_attn.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
        if self.fusion_modules is not None:
            self.fusion_modules.apply(convert_module_to_f32)

    def _extract_pvt_feats(self, x):
        """
        从混合输入(B, 4, H, W)中提取PVT特征

        Args:
            x: (B, 4, H, W) 包含 [image(3ch) + mask(1ch)]
        """
        if self.pvt_extractor is None:
            return None

        x_img = x[:, :self.image_channels, :, :]
        if x_img.shape[1] == 1:
            x_pvt = x_img.repeat(1, 3, 1, 1)
        elif x_img.shape[1] == 3:
            x_pvt = x_img
        else:
            raise ValueError(f"Unsupported image channels for PVT: {x_img.shape[1]}")

        x_pvt = (x_pvt + 1.0) / 2.0
        mean = th.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = th.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x_pvt = (x_pvt - mean) / std

        if self.pvt_trainable:
            return self.pvt_extractor(x_pvt)
        with th.no_grad():
            return self.pvt_extractor(x_pvt)

    def _extract_pvt_feats_from_image(self, x_img):
        """
        直接从图像(B, 3, H, W)中提取PVT特征

        这用于从未噪声化的原始图像提取特征，
        而不是从噪声化的输入中提取

        Args:
            x_img: (B, 3, H, W) 原始图像
        """
        if self.pvt_extractor is None:
            return None

        if x_img.shape[1] == 1:
            x_pvt = x_img.repeat(1, 3, 1, 1)
        elif x_img.shape[1] == 3:
            x_pvt = x_img
        else:
            raise ValueError(f"Unsupported image channels for PVT: {x_img.shape[1]}")

        # ✅ 假设x_img已经归一化到[-1, 1]
        x_pvt = (x_pvt + 1.0) / 2.0
        mean = th.tensor([0.485, 0.456, 0.406], device=x_img.device, dtype=x_img.dtype).view(1, 3, 1, 1)
        std = th.tensor([0.229, 0.224, 0.225], device=x_img.device, dtype=x_img.dtype).view(1, 3, 1, 1)
        x_pvt = (x_pvt - mean) / std

        if self.pvt_trainable:
            return self.pvt_extractor(x_pvt)
        with th.no_grad():
            return self.pvt_extractor(x_pvt)

    def forward(self, x, timesteps, y=None, x_pvt=None):
        """
        前向传播

        Args:
            x: 输入张量 (B, 4, H, W)，包含 [image(3ch) + mask(1ch)]
            timesteps: 时间步
            y: 条件标签（如果使用类条件）
            x_pvt: 可选的原始图像用于PVT特征提取 (B, 3, H, W)
                  如果提供，则使用这个而不是从x中提取
        """
        assert (y is not None) == (self.num_classes is not None), \
            "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        # ✅ 优先使用提供的x_pvt，否则从x中提取
        if x_pvt is not None:
            pvt_feats = self._extract_pvt_feats_from_image(x_pvt)
        else:
            pvt_feats = self._extract_pvt_feats(x)

        # ✅ 如果启用PVT融合，确保特征被使用（防止DDP错误）
        if self.use_pvt_fusion and pvt_feats is None:
            # 这不应该发生，但如果发生了，需要立即报错
            raise RuntimeError(
                "use_pvt_fusion is True but pvt_feats is None. "
                "This indicates PVT extraction failed."
            )

        h = x.type(self.dtype)
        for block_id, module in enumerate(self.input_blocks):
            h = module(h, emb)
            # ✅ 确保融合模块被调用，防止DDP梯度问题
            if pvt_feats is not None and self.fusion_modules is not None:
                block_id_str = str(block_id)
                if block_id_str in self.fusion_modules:
                    spec = self.fusion_block_specs[block_id]
                    h = self.fusion_modules[block_id_str](pvt_feats[spec["pvt_idx"]], h)
            hs.append(h)

        h = self.middle_block(h, emb)
        # ✅ 确保交叉注意力被调用（如果启用）
        if self.middle_cross_attn is not None and pvt_feats is not None:
            h = self.middle_cross_attn(h, pvt_feats[1])
        elif self.middle_cross_attn is not None and pvt_feats is None and self.use_pvt_fusion:
            # 如果启用PVT但特征为None，这是错误的
            raise RuntimeError(
                "use_cross_attn is enabled but pvt_feats is None. "
                "This will cause gradient flow issues in DDP mode."
            )

        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        h = h.type(x.dtype)
        return self.out(h)
