"""
【终极版本】训练脚本 - 使用utility文件和TrainLoopV4

比segmentation_train_gate_v4.py更强大的版本，使用TrainLoopV4支持：
- 详细Loss记录
- 分层学习率
- 边缘分支处理
- 高级Loss函数

启动命令示例：

基础：
  python scripts_new/segmentation_train_gate_v4_enhanced.py \
      --data_dir ./data/mydata \
      --out_dir ./results/train/baseline

完整：
  python scripts_new/segmentation_train_gate_v4_enhanced.py \
      --data_dir ./data/mydata \
      --out_dir ./results/train/complete \
      --use_adapter \
      --use_edge_branch \
      --use_edge_loss \
      --use_sdf_loss \
      --use_hd_loss \
      --fusion_version v1 \
      --log_loss_details
"""

import sys
import os

sys.path.append("..")
sys.path.append(".")

import argparse
import torch as th

from guided_diffusion_old import dist_util, logger
from guided_diffusion_old.resample import create_named_schedule_sampler
from guided_diffusion_new.script_util_v4 import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion_new.train_util_v4 import TrainLoopV4, create_optimizer_and_schedule
from guided_diffusion_new.mydataloader_v4 import MyDatasetV4


def main():
    """Main training function"""
    args = create_argparser().parse_args()

    # 设置GPU和日志
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_dev
    dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)

    logger.log("=" * 80)
    logger.log("SEGMENTATION TRAINING - V4 ENHANCED (TrainLoopV4 + Detailed Logging)")
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

    # 创建优化器
    optimizer = create_optimizer_and_schedule(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_adapter=args.use_adapter,
        adapter_lr=args.adapter_lr,
    )

    logger.log(f"\nOptimizer: {len(optimizer.param_groups)} param groups")

    logger.log("\n" + "=" * 80)
    logger.log("Starting training with TrainLoopV4...")
    logger.log("=" * 80)

    train_loop = TrainLoopV4(
        model=model,
        diffusion=diffusion,
        data=datal,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        schedule_sampler=schedule_sampler,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_height=args.fp16_scale_height,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        checkpoint_path=args.out_dir,
        # V4参数
        use_adapter=args.use_adapter,
        adapter_lr=args.adapter_lr,
        use_edge_branch=args.use_edge_branch,
        use_edge_loss=args.use_edge_loss,
        use_sdf_loss=args.use_sdf_loss,
        use_hd_loss=args.use_hd_loss,
        log_loss_details=args.log_loss_details,
        optimizer=optimizer,
    )

    train_loop.run_loop()

    logger.log("\n" + "=" * 80)
    logger.log("Training complete!")
    logger.log("=" * 80)


def create_argparser():
    """Create argument parser with all model/diffusion/training defaults"""
    defaults = model_and_diffusion_defaults()

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    # 额外的训练参数
    parser.add_argument("--gpu_dev", default="0")
    parser.add_argument("--log_loss_details", action="store_true",
                       help="Log detailed loss terms for each component")

    return parser


if __name__ == "__main__":
    main()
