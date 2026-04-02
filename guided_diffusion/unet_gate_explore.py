"""
多类别
baseline U-Net + PVTv2 + gated fusion
保留 FEB / cross-attn 开关，但当前推荐先关闭：
    use_pvt_fusion=True
    use_feb=False
    use_cross_attn=False
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
# PVT utils
# =========================

def remap_pvtv2_official_to_timm(state_dict):
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
    """
    PVT v2 b0 特征提取器，输出 4 层特征 [F1, F2, F3, F4]
    """
    def __init__(self, pretrained=True, local_ckpt_path="/home/nas2/biod/piankexin/AAAablation_model/guided_diffusion/pvt_v2_b0.pth"):
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

        print(f"Loaded local PVT weights: {ckpt_path}")
        print("missing_keys:", msg.missing_keys)
        print("unexpected_keys:", msg.unexpected_keys)
        total_model_keys = len(self.pvt.state_dict().keys())
        loaded_keys = total_model_keys - len(msg.missing_keys)
        print(f"loaded_keys: {loaded_keys}/{total_model_keys}")

    def get_pvt_channels(self):
        return self.pvt_channel_list

    def forward(self, x):
        return self.pvt(x)


class GatedFusionModule(nn.Module):
    """
    门控融合：
    1) PVT 特征投影到与 U-Net 特征相同通道
    2) 上采样到相同空间尺寸
    3) 用 [x_unet, f_pvt] 共同生成 gate
    4) gate 控制 PVT 注入强度
    5) 残差式加回原始 U-Net 特征
    """
    def __init__(self, pvt_channels, unet_channels, out_channels, dims=2):
        super().__init__()
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
            conv_nd(dims, out_channels * 2, out_channels, 1),
            nn.Sigmoid(),
        )

        # 零初始化，保证初始时接近恒等映射，更稳定
        self.out_proj = zero_module(
            conv_nd(dims, out_channels, out_channels, 3, padding=1)
        )

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

        injected = gate * f_pvt
        injected = self.out_proj(injected)

        return x_unet + injected


# =========================
# Core blocks
# =========================

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that support it.
    """

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


# =========================
# Optional modules (keep for future ablations)
# =========================

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, dims=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) if dims == 2 else nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1) if dims == 2 else nn.AdaptiveMaxPool3d(1)

        hidden = max(in_channels // reduction, 1)
        self.fc = nn.Sequential(
            conv_nd(dims, in_channels, hidden, 1),
            nn.SiLU(),
            conv_nd(dims, hidden, in_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, dims=2):
        super().__init__()
        self.conv = conv_nd(dims, 2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = th.mean(x, dim=1, keepdim=True)
        max_out, _ = th.max(x, dim=1, keepdim=True)
        x_cat = th.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class DualLevelResidual(nn.Module):
    def __init__(
        self,
        channels,
        emb_channels,
        out_channels=None,
        dropout=0.0,
        dims=2,
        use_checkpoint=False,
        use_scale_shift_norm=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint

        self.conv_block1 = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.conv_block2 = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, self.out_channels),
        )

    def forward(self, x, emb):
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        ) if self.use_checkpoint else self._forward(x, emb)

    def _forward(self, x, emb):
        x1 = self.conv_block1(x)

        emb_out = self.emb_layers(emb).type(x1.dtype)
        while len(emb_out.shape) < len(x1.shape):
            emb_out = emb_out[..., None]

        x1 = x1 + emb_out
        x2 = self.conv_block2(x1)
        x2 = x2 + emb_out

        return x1 + x2


class AdaptiveFeatureSelection(nn.Module):
    def __init__(self, channels, reduction=16, dims=2):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction, dims)
        self.spatial_attention = SpatialAttention(dims=dims)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class FeatureEnhancementBlock(TimestepBlock):
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

        self.dlr = DualLevelResidual(
            channels=channels,
            emb_channels=emb_channels,
            out_channels=self.out_channels,
            dropout=dropout,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
        )

        self.afs = AdaptiveFeatureSelection(
            channels=self.out_channels,
            reduction=reduction,
            dims=dims,
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
        h = self.dlr(x, emb)
        h = self.afs(h)
        h = self.out_proj(h)
        return self.skip_connection(x) + h


class CrossAttentionBlock(nn.Module):
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

        return x_in + attn_out


class AttentionPool2d(nn.Module):
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
        x = x.reshape(b, c, -1)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


# =========================
# UNet
# =========================

class UNetModel(nn.Module):
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

        use_pvt_fusion=True,
        use_feb=False,
        use_cross_attn=False,
        pvt_ckpt_path="pvt_v2_b0.pth",
        freeze_pvt=False,
        pvt_train_mode="",
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
        self.pvt_train_mode = pvt_train_mode

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

        # 记录 PVT 融合位置
        self.fusion_block_ids = {}

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

                # 可选 FEB，仅作为附加块，不替换整个主干
                if self.use_feb and ds in [4, 8] and block_idx == num_res_blocks - 1:
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
                self._feature_size += ch
                input_block_chans.append(ch)

                # 只在 ds=4 和 ds=8 进行 PVT 融合
                if self.use_pvt_fusion and block_idx == num_res_blocks - 1:
                    fusion_map = {4: 0, 8: 1}   # F1 -> ds4, F2 -> ds8
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
                        ) if resblock_updown else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # middle
        if self.use_cross_attn:
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
                context_channels=256,
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
        else:
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

        self._feature_size += ch

        # decoder
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

        # PVT branch
        if self.use_pvt_fusion or self.use_cross_attn:
            self.pvt_extractor = PVTv2FeatureExtractor(
                pretrained=True,
                local_ckpt_path=pvt_ckpt_path,
            )

            # PVT 微调策略：
            # frozen: 全冻结（当前最稳）
            # last_stage: 只解冻最后一层 stage_3
            # last_two_stages: 解冻 stage_2 和 stage_3
            # all: 全量解冻
            mode = (self.pvt_train_mode or ("frozen" if freeze_pvt else "shallow_two_stages")).lower()

            for param in self.pvt_extractor.parameters():
                param.requires_grad = False

            if mode == "all":
                for param in self.pvt_extractor.parameters():
                    param.requires_grad = True
            elif mode == "last_stage":
                for name, param in self.pvt_extractor.named_parameters():
                    if name.startswith("pvt.stages_3"):
                        param.requires_grad = True
            elif mode == "last_two_stages":
                for name, param in self.pvt_extractor.named_parameters():
                    if name.startswith("pvt.stages_2") or name.startswith("pvt.stages_3"):
                        param.requires_grad = True
            elif mode == "shallow_two_stages":
                for name, param in self.pvt_extractor.named_parameters():
                    if (
                        name.startswith("pvt.patch_embed")
                        or name.startswith("pvt.stages_0")
                        or name.startswith("pvt.stages_1")
                    ):
                        param.requires_grad = True
            elif mode == "frozen":
                pass
            else:
                raise ValueError(f"Unknown pvt_train_mode: {self.pvt_train_mode}")

            pvt_channels = self.pvt_extractor.get_pvt_channels()
            self.fusion_stage_channels = [model_channels * mult for mult in channel_mult]

            # F1 -> ds4 (160 channels), F2 -> ds8 (256 channels)
            self.fusion_modules = nn.ModuleList([
                GatedFusionModule(
                    pvt_channels=pvt_channels[0],
                    unet_channels=self.fusion_stage_channels[2],  # 160
                    out_channels=self.fusion_stage_channels[2],
                    dims=dims,
                ),
                GatedFusionModule(
                    pvt_channels=pvt_channels[1],
                    unet_channels=self.fusion_stage_channels[3],  # 256
                    out_channels=self.fusion_stage_channels[3],
                    dims=dims,
                ),
            ])
        else:
            self.pvt_extractor = None
            self.fusion_modules = None

    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        if self.use_cross_attn:
            self.middle_block1.apply(convert_module_to_f16)
            self.middle_block2.apply(convert_module_to_f16)
            self.middle_cross_attn.apply(convert_module_to_f16)
        else:
            self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
        if self.fusion_modules is not None:
            self.fusion_modules.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        if self.use_cross_attn:
            self.middle_block1.apply(convert_module_to_f32)
            self.middle_block2.apply(convert_module_to_f32)
            self.middle_cross_attn.apply(convert_module_to_f32)
        else:
            self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
        if self.fusion_modules is not None:
            self.fusion_modules.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None):
        assert (y is not None) == (self.num_classes is not None), \
            "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        # PVT branch: only use image channels (assume last 3 channels are mask/noisy mask)
        pvt_feats = None
        if self.pvt_extractor is not None:
            img_ch = self.in_channels - 3
            x_img = x[:, :img_ch, :, :]

            if x_img.shape[1] == 1:
                x_pvt = x_img.repeat(1, 3, 1, 1)
            elif x_img.shape[1] == 3:
                x_pvt = x_img
            else:
                raise ValueError(f"Unsupported image channels for PVT: {x_img.shape[1]}")

            # input image assumed in [-1, 1]
            x_pvt = (x_pvt + 1.0) / 2.0
            mean = th.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = th.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            x_pvt = (x_pvt - mean) / std

            if not any(p.requires_grad for p in self.pvt_extractor.parameters()):
                with th.no_grad():
                    pvt_feats = self.pvt_extractor(x_pvt)
            else:
                pvt_feats = self.pvt_extractor(x_pvt)

        # encoder
        h = x.type(self.dtype)
        for block_id, module in enumerate(self.input_blocks):
            h = module(h, emb)

            if pvt_feats is not None and block_id in self.fusion_block_ids:
                fusion_idx = self.fusion_block_ids[block_id]
                h = self.fusion_modules[fusion_idx](pvt_feats[fusion_idx], h)

            hs.append(h)

        # middle
        if self.use_cross_attn:
            h = self.middle_block1(h, emb)
            h = self.middle_cross_attn(h, pvt_feats[3])
            h = self.middle_block2(h, emb)
        else:
            h = self.middle_block(h, emb)

        # decoder
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        h = h.type(x.dtype)
        return self.out(h)


class SuperResModel(UNetModel):
    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


class EncoderUNetModel(nn.Module):
    """
    保留原始接口，基本不动
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
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
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

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
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
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

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
        self._feature_size += ch
        self.pool = pool
        self.gap = nn.AvgPool2d((8, 8))
        self.cam_feature_maps = None

        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d((image_size // ds), ch, num_head_channels, out_channels),
            )
        elif pool == "spatial":
            self.out = nn.Linear(256, self.out_channels)
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)

        if self.pool.startswith("spatial"):
            self.cam_feature_maps = h
            h = self.gap(h)
            n = h.shape[0]
            h = h.reshape(n, -1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            self.cam_feature_maps = h
            return self.out(h)