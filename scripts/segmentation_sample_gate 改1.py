"""
多类别 

Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import nibabel as nib
from visdom import Visdom
viz = Visdom(port=8850)
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.mydataloader import MyDataset
import torchvision.utils as vutils
from guided_diffusion.utils import staple
from guided_diffusion.script_util_gate import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F

seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

NUM_CLASSES = 3

# modify
def visualize_segmentation_mask_from_classes(class_indices):
    """
    将类别索引转换为RGB彩色图像
    Args:
        class_indices: (H, W) 的类别索引tensor，值为0,1,2
    Returns:
        rgb_image: (H, W, 3) 的RGB图像，数值范围0-255
    """
    if isinstance(class_indices, th.Tensor):
        class_indices = class_indices.cpu().numpy()
    
    H, W = class_indices.shape
    rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
    
    # 类别映射到颜色
    # 背景 (类别0): 黑色 (0, 0, 0) - 已经是0
    # 类别1: 红色 (255, 0, 0) 
    rgb_image[class_indices == 1] = [255, 0, 0]
    # 类别2: 绿色 (0, 255, 0)
    rgb_image[class_indices == 2] = [0, 255, 0]
    
    return rgb_image

def visualize_segmentation_mask_from_onehot(onehot_mask):
    """
    原来的函数，处理one-hot编码
    """
    if isinstance(onehot_mask, th.Tensor):
        onehot_mask = onehot_mask.cpu()
    
    # 获取每个像素的类别索引 (H, W)
    class_indices = th.argmax(onehot_mask, dim=0)
    
    return visualize_segmentation_mask_from_classes(class_indices)

def save_comparison_images_improved(predicted_classes, gt_mask, original_img, save_path, filename):
    """
    保存预测结果与groundtruth的对比图像
    Args:
        predicted_classes: (H, W) 预测的类别索引
        gt_mask: (3, H, W) groundtruth one-hot mask 或 (H, W) 类别索引
        original_img: (1, H, W) 原始图像
        save_path: 保存路径
        filename: 文件名
    """
    # 转换预测结果为可视化图像
    pred_rgb = visualize_segmentation_mask_from_classes(predicted_classes)
    
    # 处理ground truth
    if len(gt_mask.shape) == 3:  # one-hot编码
        gt_rgb = visualize_segmentation_mask_from_onehot(gt_mask)
    else:  # 类别索引
        gt_rgb = visualize_segmentation_mask_from_classes(gt_mask)
    
    # 处理原始图像
    if isinstance(original_img, th.Tensor):
        original_img = original_img.cpu().squeeze()
        if original_img.min() < 0:
            original_img = (original_img + 1.0) / 2.0 * 255
        else:
            original_img = original_img * 255
        original_img = original_img.numpy().astype(np.uint8)
    
    orig_rgb = np.stack([original_img, original_img, original_img], axis=-1)
    
    # 水平拼接：原图 | 预测结果 | groundtruth
    comparison = np.concatenate([pred_rgb, gt_rgb], axis=1)
    
    # 保存图像
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{filename}_comparison.png")
    Image.fromarray(comparison).save(save_file)
    
    # 单独保存预测结果
    pred_file = os.path.join(save_path, f"{filename}_predicted.png")
    Image.fromarray(pred_rgb).save(pred_file)
    
    return save_file, pred_file

def save_detailed_results(predicted_classes, sample_mask, gt_mask, original_img, save_path, filename):
    """
    保存详细的结果，包括概率图、类别图等
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 1. 保存预测的类别图
    pred_rgb = visualize_segmentation_mask_from_classes(predicted_classes)
    pred_file = os.path.join(save_path, f"{filename}_predicted_classes.png")
    Image.fromarray(pred_rgb).save(pred_file)
    
    # 2. 保存原始概率图（如果需要）
    if sample_mask is not None:
        # 转换为概率
        if sample_mask.abs().max() > 1.0:
            probs = F.softmax(sample_mask, dim=0)
        else:
            probs = sample_mask
        
    if gt_mask is not None:
        save_comparison_images_improved(predicted_classes, gt_mask, original_img, save_path, filename)
    
    print(f"Saved detailed results to {save_path}")
    return pred_file

# modify
def improved_classification(sample_mask, threshold=0.7, background_threshold=0.3):
    """
    改进的分类方法，基于阈值而不是简单的argmax
    
    Args:
        sample_mask: (3, H, W) 模型输出的logits或概率
        threshold: 前景类别的最小置信度阈值
        background_threshold: 所有类别都低于此值时归为背景
    
    Returns:
        class_indices: (H, W) 类别索引
    """
    probs = F.softmax(sample_mask, dim=0)
    
    # 获取最大概率和对应的类别
    max_probs, argmax_classes = th.max(probs, dim=0)
    
    # 创建结果张量，初始化为背景类(0)
    class_indices = th.zeros_like(argmax_classes)
    
    # 只有当最大概率超过阈值时，才分配给对应类别
    confident_mask = max_probs >= threshold
    class_indices[confident_mask] = argmax_classes[confident_mask]
    
    # 额外检查：如果所有前景类别概率都很低，强制归为背景
    all_low_confidence = th.all(probs[1:] < background_threshold, dim=0)  # 检查类别1和2
    class_indices[all_low_confidence] = 0
    
    return class_indices


def dice_score(pred, targs, eps=1e-6):
    pred = (pred > 0).float()
    return (2.0 * (pred * targs).sum() + eps) / ((pred + targs).sum() + eps)

def cls_to_onehot(pred_cls, num_classes=3):
    """
    pred_cls: [H, W] or [B, H, W]
    return: [C, H, W] or [B, C, H, W]
    """
    if pred_cls.dim() == 2:
        onehot = F.one_hot(pred_cls.long(), num_classes=num_classes)   # [H, W, C]
        onehot = onehot.permute(2, 0, 1).float()                       # [C, H, W]
    elif pred_cls.dim() == 3:
        onehot = F.one_hot(pred_cls.long(), num_classes=num_classes)   # [B, H, W, C]
        onehot = onehot.permute(0, 3, 1, 2).float()                    # [B, C, H, W]
    else:
        raise ValueError(f"Unsupported pred_cls shape: {pred_cls.shape}")
    return onehot


def dice_score_per_class(pred_classes, gt_classes, class_id, eps=1e-6):
    pred_binary = (pred_classes == class_id).float()
    gt_binary = (gt_classes == class_id).float()
    return (2.0 * (pred_binary * gt_binary).sum() + eps) / ((pred_binary + gt_binary).sum() + eps)


def compute_avg_foreground_dice(pred_classes, gt_classes, fg_class_ids=(1, 2)):
    """
    只计算前景类平均 Dice，不计背景
    pred_classes, gt_classes: [H, W]
    """
    dice_scores = []
    for class_id in fg_class_ids:
        gt_binary = (gt_classes == class_id).float()
        if gt_binary.sum() > 0:
            dice = dice_score_per_class(pred_classes, gt_classes, class_id)
            dice_scores.append(dice.item())

    if len(dice_scores) == 0:
        return 0.0, {}

    dice_dict = {f"class_{fg_class_ids[i]}": dice_scores[i] for i in range(len(dice_scores))}
    return float(np.mean(dice_scores)), dice_dict


def multi_sample_predict(
    diffusion,
    model,
    input_batch,
    batch_size,
    image_size,
    sample_fn,
    num_ensemble=5,
    clip_denoised=True,
    device=None,
):
    """
    对同一张图采样 num_ensemble 次，平均 softmax 概率，返回最终预测
    input_batch: [B, 4, H, W]
    return:
        pred_classes: [H, W]
        prob_mean: [C, H, W]
        pred_start_list: 长度为 num_ensemble 的原始 pred_start 列表（可选）
    """
    if device is None:
        device = next(model.parameters()).device

    input_batch = input_batch.to(device)
    prob_list = []
    pred_start_list = []

    for i in range(num_ensemble):
        logger.log(f"Ensemble {i+1}/{num_ensemble}")

        model_kwargs = {}
        sample, pred_start, x_noisy, img_used = sample_fn(
            model,
            (batch_size, 4, image_size, image_size),
            input_batch,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
        )

        # pred_start: [B, 3, H, W]
        pred_start_list.append(pred_start.detach().cpu())

        probs = F.softmax(pred_start, dim=1)   # [B, 3, H, W]
        prob_list.append(probs.detach())

    prob_mean = th.stack(prob_list, dim=0).mean(dim=0)   # [B, 3, H, W]
    pred_classes = th.argmax(prob_mean, dim=1)           # [B, H, W]

    return pred_classes, prob_mean, pred_start_list

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    ds = MyDataset(args, args.data_dir, mode = 'test')
    # ds = MyDataset(args, args.data_dir, mode = 'training')
    args.in_ch = 4

    data_test = ds.__getitem__(0)
    datal = th.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    data = iter(datal)
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    all_dice_scores = []
    save_detailed = True

    # 建议直接遍历 dataloader，而不是 iter + next
    for batch_idx, (b, m, path) in enumerate(datal):
        # b: [B,1,H,W], m: [B,3,H,W]
        slice_ID = path[0].split('.')[0].split('/')[-1]
        logger.log(f"sampling processing {slice_ID}...")

        # 这里直接用 [image + gt_mask] 作为输入模板
        # p_sample_loop_known 里真正会保留 image，并自动重新生成随机噪声 mask
        input_batch = th.cat((b, m), dim=1).to(dist_util.dev())

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)

        start.record()
        sample_fn = (
            diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
        )

        with th.no_grad():
            pred_classes_batch, prob_mean_batch, pred_start_list = multi_sample_predict(
                diffusion=diffusion,
                model=model,
                input_batch=input_batch,
                batch_size=args.batch_size,
                image_size=args.image_size,
                sample_fn=sample_fn,
                num_ensemble=args.num_ensemble,
                clip_denoised=args.clip_denoised,
                device=dist_util.dev(),
            )

        end.record()
        th.cuda.synchronize()
        print(f'time for {args.num_ensemble} ensemble samples: {start.elapsed_time(end):.4f} ms')

        # 当前 batch_size 默认是 1，这里按单张处理
        pred_classes = pred_classes_batch[0].cpu()    # [H, W]
        prob_mean = prob_mean_batch[0].cpu()          # [3, H, W]
        gt_mask_single = m[0].cpu()                   # [3, H, W]
        gt_classes = th.argmax(gt_mask_single, dim=0) # [H, W]
        original_img = b[0].cpu()                     # [1, H, W]

        # 计算最终 Dice（注意：这是 ensemble 平均后的最终结果）
        avg_dice, dice_dict = compute_avg_foreground_dice(pred_classes, gt_classes, fg_class_ids=(1, 2))
        all_dice_scores.append(avg_dice)

        logger.log(f"[{batch_idx+1}/{len(datal)}] {slice_ID}")
        logger.log(f"Final Ensemble Dice: {avg_dice:.4f}")
        if len(dice_dict) > 0:
            logger.log(f"Per-class Dice: {dice_dict}")
        logger.log(f"Running Mean Dice: {np.mean(all_dice_scores):.4f}")
        logger.log("--------------------------------------------------")

        # 保存最终平均概率得到的预测结果
        save_name = f"{slice_ID}_ensembleMean_{args.num_ensemble}"

        if save_detailed:
            save_detailed_results(
                predicted_classes=pred_classes,
                sample_mask=prob_mean,      # 这里传平均概率图
                gt_mask=gt_mask_single,
                original_img=original_img,
                save_path=args.out_dir,
                filename=save_name
            )
        else:
            pred_rgb = visualize_segmentation_mask_from_classes(pred_classes)
            pred_file = os.path.join(args.out_dir, f"{save_name}_predicted.png")
            os.makedirs(args.out_dir, exist_ok=True)
            Image.fromarray(pred_rgb).save(pred_file)
            print(f"Saved prediction: {pred_file}")

    # 最终总体统计
    if len(all_dice_scores) > 0:
        mean_dice = np.mean(all_dice_scores)
        std_dice = np.std(all_dice_scores)
        logger.log("\nOverall Results:")
        logger.log(f"Mean Dice Score: {mean_dice:.4f} ± {std_dice:.4f}")
        logger.log("====================================\n")




def create_argparser():
    defaults = dict(
        data_name = 'mydata',
        data_dir="./data/mydata",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=5,      #number of samples in the ensemble
        out_dir='./results/sample/mydata/0625/',  #output directory for the generated masks
        gpu_dev = "3",
        multi_gpu = None, #"0,1,2"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
