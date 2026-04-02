"""
Train a diffusion model on images.
"""
import sys
import argparse
sys.path.append("..")
sys.path.append(".")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.mydataloader import MyDataset
from PIL import Image

import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

from guided_diffusion.script_util_gate import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop
from visdom import Visdom
viz = Visdom(port=8850)
import torchvision.transforms as transforms
import os

def main():
    args = create_argparser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_dev
    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)
    logger.log("creating model and diffusion...")

    # 使用解析的参数创建模型和扩散过程，这允许在模型架构和训练过程中灵活配置。
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # 创建一个文件对象，模式为写入 (w)
    with open('model_gate 1.txt', 'w') as f:
        # 将 print() 的输出重定向到文件
        print(model, file=f)
    model.to(dist_util.dev())   # 将模型移动到适当的设备（CPU 或 GPU）

    # 初始化调度采样器，该采样器决定训练过程中模型如何采样噪声水平。
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)
    
    logger.log("creating data loader...")
    # 数据加载器设置

    tran_list = [transforms.Resize((args.image_size,args.image_size), interpolation=Image.NEAREST), transforms.ToTensor(),]
    transform_train = transforms.Compose(tran_list)
    print("Your current directory : ",args.data_dir)
    ds = MyDataset(args, args.data_dir, transform=transform_train, mode='training')
    args.in_ch = 4

    # data_test = ds.__getitem__(0)
    print(f"使用的GPU: {args.gpu_dev}")

    datal= th.utils.data.DataLoader(
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
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',#'"./results/pretrainedmodel.pt",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        out_dir='./results/train/mydata/0625',
        gpu_dev = "2",
        multi_gpu = None, #"0,1,2"
    )
    # 使用 model_and_diffusion_defaults() 更新默认值，包含模型和扩散相关的参数。
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()