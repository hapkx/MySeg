"""
【方案2改进版】Training Script for Scheme 2

使用全新的script_util_scheme2和train_util_scheme

启动命令：
  python scripts/segmentation_train_scheme2_new.py \
      --data_dir ./data/mydata \
      --batch_size 8 \
      --lr 3e-5 \
      --out_dir ./results/train/mydata_scheme2_new
"""

import sys
import argparse
import os

sys.path.append("..")
sys.path.append(".")

from guided_diffusion_old import dist_util, logger
from guided_diffusion_old.resample import create_named_schedule_sampler
from guided_diffusion_old.mydataloader_scheme2 import MyDatasetScheme2
from guided_diffusion_old.script_util_scheme2 import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion_old.train_util_scheme import TrainLoopScheme
import torch as th


def main():
    """Main training function for Scheme 2"""
    args = create_argparser().parse_args()

    # 设置GPU和日志
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_dev
    dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)

    logger.log("=" * 80)
    logger.log("SCHEME 2 TRAINING - PVT和UNet都输入相同patch版本")
    logger.log("=" * 80)
    logger.log(f"Data directory: {args.data_dir}")
    logger.log(f"Output directory: {args.out_dir}")
    logger.log(f"Batch size: {args.batch_size}")
    logger.log(f"Learning rate: {args.lr}")

    logger.log("\nCreating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    logger.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion, maxt=1000
    )

    logger.log("\nLoading dataset...")
    ds = MyDatasetScheme2(args, args.data_dir, mode='training')
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    logger.log(f"Dataset size: {len(ds)}")
    logger.log(f"DataLoader batches: {len(datal)}")

    logger.log("\n" + "=" * 80)
    logger.log("Starting training loop...")
    logger.log("=" * 80)

    train_loop = TrainLoopScheme(
        model=model,
        diffusion=diffusion,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        scheme="scheme2",  # 关键：指定方案2
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
    )
    train_loop.run_loop()

    logger.log("\n" + "=" * 80)
    logger.log("Training completed!")
    logger.log("=" * 80)


def create_argparser():
    """Create argument parser for Scheme 2"""
    defaults = dict(
        data_name="mydata",
        data_dir="./data/mydata",
        schedule_sampler="uniform",
        lr=3e-5,
        weight_decay=1e-4,
        lr_anneal_steps=0,
        warmup_steps=1000,  # 前1000步线性预热
        total_steps=50000,  # 总训练步数，用于cosine衰减
        batch_size=8,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',
        use_fp16=False,
        fp16_scale_growth=1e-3,
        out_dir='./results/train/mydata_scheme2_new',
        gpu_dev="0",
        # Scheme2特定参数
        train_long_side=640,
        patches_per_image=8,
        fg_sample_ratio=0.6,
        min_fg_pixels=64,
        full_image_ratio=0.7,
        intensity_aug_p=0.5,
    )

    # 添加model和diffusion的默认参数
    defaults.update(model_and_diffusion_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
