"""
Patch/full-image mixed training for current gate PVT+FEB model.
"""
import sys
import argparse
import os
sys.path.append("..")
sys.path.append(".")

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.mydataloader_gate_patchmix import MyDataset
from guided_diffusion.script_util_gate_explore import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_dev
    dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)
    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, maxt=1000)

    logger.log("creating mixed patch/full-image data loader...")
    ds = MyDataset(args, args.data_dir, mode='training')
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    data = iter(datal)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
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
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_name="mydata",
        data_dir="./data/mydata",
        schedule_sampler="uniform",
        lr=3e-5,
        weight_decay=1e-4,
        lr_anneal_steps=0,
        batch_size=8,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',
        use_fp16=False,
        fp16_scale_growth=1e-3,
        out_dir='./results/train/mydata_patchmix',
        gpu_dev="0",
        multi_gpu=None,
        # patch-mix settings
        train_long_side=640,
        patches_per_image=6,
        fg_sample_ratio=0.6,
        min_fg_pixels=32,
        full_image_ratio=0.6,
        intensity_aug_p=0.4,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
