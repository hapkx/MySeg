"""
【方案4+评估指标增强版】采样和评估脚本 - 参考原始脚本写法

特点：
1. 支持数据集v4（长边缩放+填充）
2. 包含所有评估指标（Dice, IoU, HD95, ASD等）
3. 支持多样本集合（ensemble）
4. 支持多个UNet变体
5. 遵循原始脚本的初始化和参数加载方式

启动命令：
  python scripts_new/segmentation_sample_gate_v4.py \
      --model_path ./results/train/model_best.pt \
      --data_dir ./data/mydata \
      --out_dir ./results/sample/v4 \
      --num_ensemble 5
"""

import argparse
import os
import json
import sys
import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image

sys.path.append(".")

from guided_diffusion_old import dist_util, logger
from guided_diffusion_new.mydataloader_v4 import MyDatasetV4, unpad_and_rescale
from guided_diffusion_new.eval_metrics import MetricsEvaluator, print_metrics
from guided_diffusion_new.script_util_v4 import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


def visualize_segmentation_mask_from_classes(class_indices):
    """
    将类别索引转换为RGB彩色图像
    Args:
        class_indices: (H, W) 的类别索引
    Returns:
        rgb_image: (H, W, 3) 的RGB图像，数值范围0-255
    """
    if isinstance(class_indices, th.Tensor):
        class_indices = class_indices.cpu().numpy()

    H, W = class_indices.shape
    rgb_image = np.zeros((H, W, 3), dtype=np.uint8)

    rgb_image[class_indices == 1] = [255, 0, 0]    # 类别1: 红色
    rgb_image[class_indices == 2] = [0, 255, 0]    # 类别2: 绿色

    return rgb_image


def dice_score_per_class(pred_classes, gt_classes, class_id, eps=1e-6):
    """计算某一类别的Dice分数"""
    pred_binary = (pred_classes == class_id).float()
    gt_binary = (gt_classes == class_id).float()
    return (2.0 * (pred_binary * gt_binary).sum() + eps) / ((pred_binary + gt_binary).sum() + eps)


def compute_avg_foreground_dice(pred_classes, gt_classes, fg_class_ids=(1, 2)):
    """
    只计算前景类平均 Dice
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


def save_detailed_results(predicted_classes, prob_mean, gt_mask, vis_dir, filename):
    """
    保存详细的结果，包括预测和GT对比
    """
    # 预测类别图
    pred_rgb = visualize_segmentation_mask_from_classes(predicted_classes)

    # GT类别图
    gt_classes = th.argmax(gt_mask, dim=0) if isinstance(gt_mask, th.Tensor) else np.argmax(gt_mask, axis=0)
    gt_rgb = visualize_segmentation_mask_from_classes(gt_classes)

    # 并行显示
    comparison = np.hstack([gt_rgb, pred_rgb])

    os.makedirs(vis_dir, exist_ok=True)
    save_file = os.path.join(vis_dir, f"{filename}_comparison.png")
    Image.fromarray(comparison).save(save_file)

    return save_file


def multi_sample_predict(
    diffusion,
    model,
    input_batch,
    batch_size,
    sample_fn,
    num_ensemble=5,
    clip_denoised=False,
    device=None,
):
    """
    对同一张图采样 num_ensemble 次，平均概率，返回最终预测
    input_batch: dict or tuple
    return:
        pred_classes: [H, W]
        prob_mean: [C, H, W]
    """
    if device is None:
        device = next(model.parameters()).device

    prob_list = []

    for i in range(num_ensemble):
        logger.log(f"  Ensemble sample {i+1}/{num_ensemble}")

        model_kwargs = {}

        if isinstance(input_batch, dict):
            img = input_batch['img'].to(device)
            mask_gt = input_batch['mask'].to(device)
            sample_input = th.cat([img, th.randn_like(mask_gt)], dim=1)
        else:
            sample_input = input_batch.to(device)

        sample = sample_fn(
            model,
            sample_input.shape,
            sample_input,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
        )

        # sample 返回格式取决于diffusion实现
        if isinstance(sample, (tuple, list)):
            pred_output = sample[1] if len(sample) > 1 else sample[0]
        else:
            pred_output = sample

        # 取分割通道（前3个）
        if pred_output.shape[1] >= 3:
            pred_seg = pred_output[:, :3, :, :]
        else:
            pred_seg = pred_output

        probs = F.softmax(pred_seg, dim=1)
        prob_list.append(probs.detach())

    prob_mean = th.stack(prob_list, dim=0).mean(dim=0)   # [B, C, H, W]
    pred_classes = th.argmax(prob_mean, dim=1)           # [B, H, W]

    return pred_classes, prob_mean


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)

    logger.log("=" * 80)
    logger.log("SEGMENTATION SAMPLING - V4 CONFIGURATION")
    logger.log("=" * 80)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # 加载数据集
    ds = MyDatasetV4(args.data_dir, split='val', target_size=256)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    # 加载模型权重
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = os.path.join(args.out_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    all_dice_scores = []
    all_metrics = []
    evaluator = MetricsEvaluator(include_hd=True, include_asd=True)

    # 采样处理
    sample_fn = diffusion.p_sample_loop_known

    logger.log("\nStarting sampling and evaluation...")

    for batch_idx, batch in enumerate(datal):
        if args.num_samples > 0 and batch_idx >= args.num_samples:
            break

        logger.log(f"[{batch_idx+1}/{len(datal)}] Processing batch...")

        with th.no_grad():
            pred_classes_batch, prob_mean_batch = multi_sample_predict(
                diffusion=diffusion,
                model=model,
                input_batch=batch,
                batch_size=args.batch_size,
                sample_fn=sample_fn,
                num_ensemble=args.num_ensemble,
                clip_denoised=args.clip_denoised,
                device=dist_util.dev(),
            )

        # 处理batch中的每个样本
        B = pred_classes_batch.shape[0]
        for i in range(B):
            pred_classes = pred_classes_batch[i].cpu()
            prob_mean = prob_mean_batch[i].cpu()

            # 从batch中提取数据
            if isinstance(batch, dict):
                mask_gt = batch['mask'][i].cpu()
                padding_info = batch['padding_info'][i]
                img = batch['img'][i].cpu()
            else:
                # Fallback for tuple/list格式
                mask_gt = batch[1][i].cpu() if len(batch) > 1 else None
                padding_info = batch[2][i] if len(batch) > 2 else None
                img = batch[0][i].cpu() if len(batch) > 0 else None

            # 计算反投影后的指标（如果有padding_info）
            if padding_info is not None:
                pred_classes_orig = unpad_and_rescale(
                    prob_mean.numpy(),
                    padding_info
                )
                mask_gt_orig = unpad_and_rescale(
                    mask_gt.numpy(),
                    padding_info
                )

                # 计算指标
                metrics = evaluator.compute_all(
                    pred_classes_orig,
                    mask_gt_orig,
                    class_ids=[1, 2]
                )
                all_metrics.append(metrics)

                # 计算Dice
                pred_cls_orig = np.argmax(pred_classes_orig, axis=0)
                gt_cls_orig = np.argmax(mask_gt_orig, axis=0)
                avg_dice, dice_dict = compute_avg_foreground_dice(
                    th.from_numpy(pred_cls_orig).long(),
                    th.from_numpy(gt_cls_orig).long()
                )
            else:
                # 不进行反投影
                metrics = evaluator.compute_all(
                    prob_mean.numpy(),
                    mask_gt.numpy(),
                    class_ids=[1, 2]
                )
                all_metrics.append(metrics)

                avg_dice, dice_dict = compute_avg_foreground_dice(
                    pred_classes,
                    th.argmax(mask_gt, dim=0)
                )

            all_dice_scores.append(avg_dice)

            logger.log(f"  Sample {i+1}: Ensemble Dice = {avg_dice:.4f}")
            if dice_dict:
                logger.log(f"    Per-class: {dice_dict}")
            logger.log(f"    Running Mean Dice: {np.mean(all_dice_scores):.4f}")

            # 保存visualization
            if args.save_vis:
                save_detailed_results(
                    predicted_classes=pred_classes,
                    prob_mean=prob_mean,
                    gt_mask=mask_gt,
                    vis_dir=vis_dir,
                    filename=f"sample_{batch_idx * args.batch_size + i:04d}"
                )

    # 最终结果统计
    logger.log("\n" + "=" * 80)
    logger.log("EVALUATION RESULTS")
    logger.log("=" * 80)

    if len(all_dice_scores) > 0:
        mean_dice = np.mean(all_dice_scores)
        std_dice = np.std(all_dice_scores)
        logger.log(f"Mean Dice Score: {mean_dice:.4f} ± {std_dice:.4f}")

    if len(all_metrics) > 0:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        print_metrics(avg_metrics, title="Averaged Metrics")

        # 保存结果
        results_path = os.path.join(args.out_dir, "metrics.json")
        with open(results_path, 'w') as f:
            json.dump({
                'average_metrics': {k: float(v) for k, v in avg_metrics.items()},
                'per_sample_metrics': [{k: float(v) for k, v in m.items()} for m in all_metrics]
            }, f, indent=2)
        logger.log(f"Metrics saved to {results_path}")

    logger.log("=" * 80)
    logger.log("Sampling complete!")
    logger.log("=" * 80)


def create_argparser():
    defaults = dict(
        data_dir="./data/mydata",
        model_path="./results/train/model_best.pt",
        out_dir="./results/sample/v4",
        clip_denoised=True,
        batch_size=1,
        num_samples=0,
        num_ensemble=5,
        gpu_dev="0",
        save_vis=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
