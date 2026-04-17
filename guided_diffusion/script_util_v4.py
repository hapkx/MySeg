"""
Script Utility Functions for V4 Configuration - Complete Self-Contained Implementation

提供参数解析、模型创建和diffusion创建的辅助函数
支持灵活的实验配置（V4独立版本，无原版依赖）

使用方式：
  from guided_diffusion_new.script_util_v4 import (
      model_and_diffusion_defaults,
      create_model_and_diffusion,
      add_dict_to_argparser,
      args_to_dict,
  )
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "guided_diffusion"))

from guided_diffusion_old import gaussian_diffusion as gd
from guided_diffusion_old.respace import SpacedDiffusion, space_timesteps
from guided_diffusion_old.unet import UNetModel

NUM_CLASSES = 3


# ============================================================================
# 默认参数定义
# ============================================================================

def diffusion_defaults():
    """
    Diffusion模型的默认参数

    Returns:
        dict: 包含所有diffusion参数的字典
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def model_and_diffusion_defaults():
    """
    模型和Diffusion的所有默认参数（原版+V4扩展）

    Returns:
        dict: 包含所有参数的字典
    """
    # 基础模型参数
    res = dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )

    # 原有PVT相关参数
    res.update(dict(
        use_pvt_fusion=False,
        use_feb=False,
        use_cross_attn=False,
        freeze_pvt=False,
        pvt_ckpt_path="pvt_v2_b0.pth",
    ))

    # V4新增参数
    res.update(dict(
        use_adapter=False,              # 是否使用PVT Adapter
        adapter_reduction=16,           # Adapter的降维比例
        adapter_lr=1e-4,                # Adapter的学习率
        use_edge_branch=False,          # 是否添加边缘分支
        fusion_version=None,            # 融合模块版本：None/v1/v2
        use_edge_loss=False,            # 是否使用边缘监督Loss
        use_sdf_loss=False,             # 是否使用SDF Loss
        use_hd_loss=False,              # 是否使用Hausdorff Loss
        edge_loss_weight=0.2,           # 边缘Loss权重
        sdf_loss_weight=0.2,            # SDF Loss权重
        hd_loss_weight=0.1,             # Hausdorff Loss权重
        data_version='v4',              # 数据加载版本
    ))

    # 合并diffusion默认参数
    res.update(diffusion_defaults())

    return res


# ============================================================================
# 模型创建函数
# ============================================================================

def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    # 原有PVT参数
    use_pvt_fusion=False,
    use_feb=False,
    use_cross_attn=False,
    freeze_pvt=False,
    pvt_ckpt_path="pvt_v2_b0.pth",
    # V4参数
    use_adapter=False,
    adapter_reduction=16,
    adapter_lr=1e-4,
    use_edge_branch=False,
    fusion_version=None,
    **kwargs
):
    """
    创建UNet模型

    自动选择模型变体：
    - use_edge_branch=True    → UNetModelWithEdgeBranch（4通道输出）
    - use_adapter=True        → UNetModelWithAdapter（PVT Adapter）
    - fusion_version='v1'     → UNetModelWithFusionV1（自适应融合）
    - fusion_version='v2'     → UNetModelWithFusionV2（交叉注意力融合）
    - else                    → UNetModel（基础模型）
    """

    # 解析channel_mult
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    # 解析attention_resolutions
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    # 确定输出通道数
    if use_edge_branch:
        # 边缘分支：3个分割通道 + 1个边缘通道
        out_channels = 4 if not learn_sigma else 8
    else:
        # 标准分割：3个通道
        out_channels = NUM_CLASSES if not learn_sigma else NUM_CLASSES * 2

    # 选择模型类
    if use_edge_branch:
        from .unet_gate_edge_branch import UNetModelWithEdgeBranch
        model_class = UNetModelWithEdgeBranch
    elif use_adapter:
        from .unet_gate_adapter import UNetModelWithAdapter
        model_class = UNetModelWithAdapter
    elif fusion_version == 'v1':
        from .unet_gate_fusion_v1 import UNetModelWithFusionV1
        model_class = UNetModelWithFusionV1
    elif fusion_version == 'v2':
        from .unet_gate_fusion_v2 import UNetModelWithFusionV2
        model_class = UNetModelWithFusionV2
    else:
        from guided_diffusion_old.unet_base import UNetModel as BaseUNetModel
        model_class = BaseUNetModel

    # 创建模型实例
    model = model_class(
        image_size=image_size,
        in_channels=4,
        model_channels=num_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        # 原有参数
        use_pvt_fusion=use_pvt_fusion,
        use_feb=use_feb,
        use_cross_attn=use_cross_attn,
        freeze_pvt=freeze_pvt,
        pvt_ckpt_path=pvt_ckpt_path,
    )

    # 如果使用Adapter，需要特殊处理
    if use_adapter and hasattr(model, 'freeze_pvt_backbone'):
        if freeze_pvt:
            model.freeze_pvt_backbone()

    return model


# ============================================================================
# Diffusion创建函数
# ============================================================================

def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    # V4参数
    use_edge_loss=False,
    use_sdf_loss=False,
    use_hd_loss=False,
    edge_loss_weight=0.2,
    sdf_loss_weight=0.2,
    hd_loss_weight=0.1,
    **kwargs
):
    """
    创建Diffusion模型

    如果启用了任何V4 loss函数，则返回GaussianDiffusionAdvanced
    否则返回标准的SpacedDiffusion
    """

    # 获取beta调度
    betas = gd.get_named_beta_schedule(noise_schedule, steps)

    # 确定loss类型
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE

    # 处理timestep_respacing
    if not timestep_respacing:
        timestep_respacing = [steps]

    # 如果启用了高级loss，使用GaussianDiffusionAdvanced
    if use_edge_loss or use_sdf_loss or use_hd_loss:
        from .gaussian_diffusion_advanced import GaussianDiffusionAdvanced
        return GaussianDiffusionAdvanced(
            use_timesteps=space_timesteps(steps, timestep_respacing),
            betas=betas,
            model_mean_type=(
                gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    gd.ModelVarType.FIXED_LARGE
                    if not sigma_small
                    else gd.ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
            # V4参数
            use_edge_loss=use_edge_loss,
            use_sdf_loss=use_sdf_loss,
            use_hd_loss=use_hd_loss,
            edge_loss_weight=edge_loss_weight,
            sdf_loss_weight=sdf_loss_weight,
            hd_loss_weight=hd_loss_weight,
        )
    else:
        # 标准diffusion
        return SpacedDiffusion(
            use_timesteps=space_timesteps(steps, timestep_respacing),
            betas=betas,
            model_mean_type=(
                gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    gd.ModelVarType.FIXED_LARGE
                    if not sigma_small
                    else gd.ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
        )


def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    # 原有PVT参数
    use_pvt_fusion=False,
    use_feb=False,
    use_cross_attn=False,
    freeze_pvt=False,
    pvt_ckpt_path="pvt_v2_b0.pth",
    # V4参数
    use_adapter=False,
    adapter_reduction=16,
    adapter_lr=1e-4,
    use_edge_branch=False,
    fusion_version=None,
    use_edge_loss=False,
    use_sdf_loss=False,
    use_hd_loss=False,
    edge_loss_weight=0.2,
    sdf_loss_weight=0.2,
    hd_loss_weight=0.1,
    data_version='v4',
    **kwargs
):
    """
    创建模型和diffusion

    Parameters:
        ... (所有原有参数)
        use_adapter: 是否使用PVT Adapter
        adapter_reduction: Adapter降维比例
        use_edge_branch: 是否添加边缘分支
        fusion_version: 融合模块版本（None/v1/v2）
        use_edge_loss: 是否使用边缘Loss
        use_sdf_loss: 是否使用SDF Loss
        use_hd_loss: 是否使用Hausdorff Loss

    Returns:
        tuple: (model, diffusion)
    """

    # 创建模型
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
        # 原有参数
        use_pvt_fusion=use_pvt_fusion,
        use_feb=use_feb,
        use_cross_attn=use_cross_attn,
        freeze_pvt=freeze_pvt,
        pvt_ckpt_path=pvt_ckpt_path,
        # V4参数
        use_adapter=use_adapter,
        adapter_reduction=adapter_reduction,
        adapter_lr=adapter_lr,
        use_edge_branch=use_edge_branch,
        fusion_version=fusion_version,
    )

    # 创建diffusion
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        # V4参数
        use_edge_loss=use_edge_loss,
        use_sdf_loss=use_sdf_loss,
        use_hd_loss=use_hd_loss,
        edge_loss_weight=edge_loss_weight,
        sdf_loss_weight=sdf_loss_weight,
        hd_loss_weight=hd_loss_weight,
    )

    return model, diffusion


# ============================================================================
# 参数解析函数
# ============================================================================

def str2bool(v):
    """
    将字符串转换为布尔值

    参考: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    """
    将字典中的参数添加到argparse parser

    Parameters:
        parser: argparse.ArgumentParser实例
        default_dict: 包含参数及默认值的字典
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    """
    从argparse Namespace中提取指定参数

    Parameters:
        args: argparse.Namespace对象
        keys: 要提取的参数名称列表

    Returns:
        dict: 提取的参数字典
    """
    return {k: getattr(args, k) for k in keys}
