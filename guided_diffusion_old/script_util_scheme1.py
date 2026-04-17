"""
【方案1专用】Script Utilities for Scheme 1

缩放原图到256×256版本的配置管理模块

核心特性：
- 针对方案1的专用参数配置
- 自动处理img_patch和img_full_small的输入
- 简化的参数管理
"""

import argparse
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet_gate_scheme1 import UNetModelScheme1 as UNetModel

NUM_CLASSES = 3


def diffusion_defaults():
    """
    扩散过程的默认参数
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
    模型和扩散的默认参数（方案1优化版）
    """
    res = dict(
        image_size=256,              # 固定256×256
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="1,2,4,8",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,

        # 【方案1特定参数】
        use_pvt_fusion=True,         # 方案1必须启用
        use_feb=False,               # 推荐不用
        use_cross_attn=False,        # 推荐不用
        freeze_pvt=False,            # 允许PVT微调
        pvt_ckpt_path="pvt_v2_b0.pth",
        pvt_train_mode="shallow_two_stages",  # 推荐配置
    )
    res.update(diffusion_defaults())
    return res


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
    use_pvt_fusion=True,
    use_feb=False,
    use_cross_attn=False,
    freeze_pvt=False,
    pvt_ckpt_path="pvt_v2_b0.pth",
    pvt_train_mode="shallow_two_stages",
):
    """创建模型和扩散对象"""

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
        use_pvt_fusion=use_pvt_fusion,
        use_feb=use_feb,
        use_cross_attn=use_cross_attn,
        freeze_pvt=freeze_pvt,
        pvt_ckpt_path=pvt_ckpt_path,
        pvt_train_mode=pvt_train_mode,
    )

    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )

    return model, diffusion


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    use_pvt_fusion,
    use_feb,
    use_cross_attn,
    freeze_pvt,
    pvt_ckpt_path,
    pvt_train_mode,
):
    """创建UNetModel"""

    if isinstance(channel_mult, str):
        if channel_mult == "":
            channel_mult = ()
        else:
            channel_mult = tuple(int(x) for x in channel_mult.split(","))

    attention_resolutions = tuple(int(x) for x in attention_resolutions.split(","))

    model = UNetModel(
        image_size=image_size,
        in_channels=4,  # 1 image + 3 mask
        model_channels=num_channels,
        out_channels=3,  # 3 class outputs
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
        channel_mult=channel_mult,
        use_checkpoint=use_checkpoint,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_new_attention_order=use_new_attention_order,
        use_pvt_fusion=use_pvt_fusion,
        use_feb=use_feb,
        use_cross_attn=use_cross_attn,
        freeze_pvt=freeze_pvt,
        pvt_ckpt_path=pvt_ckpt_path,
        pvt_train_mode=pvt_train_mode,
    )
    return model


def create_gaussian_diffusion(
    steps,
    learn_sigma,
    noise_schedule,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    timestep_respacing,
):
    """创建高斯扩散对象"""

    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.KL
    else:
        loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not rescale_learned_sigmas
                else gd.ModelVarType.FIXED_LARGE_LOG
            )
            if not learn_sigma
            else (
                gd.ModelVarType.LEARNED if not rescale_learned_sigmas else gd.ModelVarType.LEARNED_RANGE
            )
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def args_to_dict(args, keys):
    """将argparse对象转换为字典（仅包含指定的键）"""
    return {k: getattr(args, k, None) for k in keys}


def add_dict_to_argparser(parser, default_dict):
    """将字典项添加到argparse解析器"""
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")
