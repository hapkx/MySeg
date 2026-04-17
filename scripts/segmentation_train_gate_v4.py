"""
【方案4增强版】训练脚本 - 参考原始脚本写法

支持灵活的实验配置：
  - 数据加载：v4（长边缩放+填充）
  - UNet变体：base, adapter, edge_branch, fusion_v1, fusion_v2
  - Loss配置：edge_loss, sdf_loss, hd_loss的启用/禁用
  - 分层学习率

启动命令示例：

基础：
  python scripts_new/segmentation_train_gate_v4.py \
      --data_dir ./data/mydata \
      --out_dir ./results/train/v4_baseline

完整：
  python scripts_new/segmentation_train_gate_v4.py \
      --data_dir ./data/mydata \
      --out_dir ./results/train/v4_complete \
      --use_adapter \
      --use_edge_branch \
      --use_edge_loss \
      --use_sdf_loss \
      --use_hd_loss
"""

import sys
import argparse
import os

sys.path.append("..")
sys.path.append(".")

import torch as th

from guided_diffusion_old import dist_util, logger
from guided_diffusion_old.resample import create_named_schedule_sampler
from guided_diffusion_new.mydataloader_v4 import MyDatasetV4
from guided_diffusion_new.script_util_v4 import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion_old.train_util import TrainLoop


def main():
    """Main training function"""
    args = create_argparser().parse_args()

    # 设置GPU和日志
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_dev
    dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)

    logger.log("=" * 80)
    logger.log("SEGMENTATION TRAINING - V4 CONFIGURATION")
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
    ds = MyDatasetV4(
        data_dir=args.data_dir,
        split='train',
        target_size=256,
    )
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

    train_loop = TrainLoop(
        model=model,
        diffusion=diffusion,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatches,
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
    )
    train_loop.run_loop()

    logger.log("\n" + "=" * 80)
    logger.log("Training complete!")
    logger.log("=" * 80)


def create_argparser():
    """Create argument parser following original script pattern"""
    defaults = dict(
        data_dir="./data/mydata",
        out_dir="./results/train/v4_test",
        schedule_sampler="uniform",
        lr=3e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=4,
        microbatches=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=500,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1.0,
        gpu_dev="0",
    )

    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
