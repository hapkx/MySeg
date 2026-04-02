"""
Current gate-model sampling with optional global branch, sliding-window local branch,
and Dice + IoU reporting.
"""
import argparse
import os
import sys
import random
import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image

sys.path.append(".")
from guided_diffusion import dist_util, logger
from guided_diffusion.mydataloader_gate_patchmix import MyDataset
from guided_diffusion.script_util_gate_explore import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

seed = 10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


def visualize_segmentation_mask_from_classes(class_indices):
    if isinstance(class_indices, th.Tensor):
        class_indices = class_indices.cpu().numpy()
    h, w = class_indices.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_image[class_indices == 1] = [255, 0, 0]
    rgb_image[class_indices == 2] = [0, 255, 0]
    return rgb_image


def overlay_mask_on_image(gray_tensor, class_indices, alpha=0.45):
    if isinstance(gray_tensor, th.Tensor):
        gray = gray_tensor.cpu().squeeze().numpy()
    else:
        gray = gray_tensor
    if gray.min() < 0:
        gray = (gray + 1.0) / 2.0
    gray = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    mask_rgb = visualize_segmentation_mask_from_classes(class_indices)
    out = gray_rgb.copy().astype(np.float32)
    fg = np.any(mask_rgb > 0, axis=-1)
    out[fg] = (1 - alpha) * out[fg] + alpha * mask_rgb[fg].astype(np.float32)
    return out.astype(np.uint8)


def save_results_at_original_size(pred_classes, gt_classes, original_img, save_path, filename):
    os.makedirs(save_path, exist_ok=True)
    pred_rgb = visualize_segmentation_mask_from_classes(pred_classes)
    gt_rgb = visualize_segmentation_mask_from_classes(gt_classes)
    pred_overlay = overlay_mask_on_image(original_img, pred_classes)
    gt_overlay = overlay_mask_on_image(original_img, gt_classes)
    if isinstance(original_img, th.Tensor):
        gray = original_img.cpu().squeeze().numpy()
    else:
        gray = original_img
    if gray.min() < 0:
        gray = (gray + 1.0) / 2.0
    gray = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    orig_rgb = np.stack([gray, gray, gray], axis=-1)
    comparison = np.concatenate([orig_rgb, pred_overlay, gt_overlay, pred_rgb, gt_rgb], axis=1)
    Image.fromarray(comparison).save(os.path.join(save_path, f"{filename}_comparison.png"))


def dice_score_per_class(pred_classes, gt_classes, class_id, eps=1e-6):
    pred_binary = (pred_classes == class_id).float()
    gt_binary = (gt_classes == class_id).float()
    return (2.0 * (pred_binary * gt_binary).sum() + eps) / ((pred_binary + gt_binary).sum() + eps)


def iou_score_per_class(pred_classes, gt_classes, class_id, eps=1e-6):
    pred_binary = (pred_classes == class_id).float()
    gt_binary = (gt_classes == class_id).float()
    inter = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum() - inter
    return (inter + eps) / (union + eps)


def compute_avg_foreground_metrics(pred_classes, gt_classes, fg_class_ids=(1, 2)):
    dice_scores, iou_scores = [], []
    dice_dict, iou_dict = {}, {}
    for class_id in fg_class_ids:
        gt_binary = (gt_classes == class_id).float()
        if gt_binary.sum() > 0:
            dice = dice_score_per_class(pred_classes, gt_classes, class_id)
            iou = iou_score_per_class(pred_classes, gt_classes, class_id)
            dice_scores.append(dice.item())
            iou_scores.append(iou.item())
            dice_dict[f"class_{class_id}"] = dice.item()
            iou_dict[f"class_{class_id}"] = iou.item()
    mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    return mean_dice, mean_iou, dice_dict, iou_dict


def resize_image_keep_ratio(img_tensor, long_side):
    _, _, orig_h, orig_w = img_tensor.shape
    scale = long_side / max(orig_h, orig_w)
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))
    img_resized = F.interpolate(img_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
    info = {"orig_h": orig_h, "orig_w": orig_w, "scaled_h": new_h, "scaled_w": new_w, "scale": scale}
    return img_resized, info


def resize_and_pad_to_square(img_tensor, target_size):
    _, _, orig_h, orig_w = img_tensor.shape
    scale = min(target_size / orig_h, target_size / orig_w)
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))
    img_resized = F.interpolate(img_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    img_pad = F.pad(img_resized, (left, right, top, bottom), mode="constant", value=-1.0)
    meta = {"orig_h": orig_h, "orig_w": orig_w, "new_h": new_h, "new_w": new_w, "top": top, "left": left}
    return img_pad, meta


def restore_square_prob_to_orig(prob_square, meta):
    prob_crop = prob_square[:, :, meta["top"]:meta["top"] + meta["new_h"], meta["left"]:meta["left"] + meta["new_w"]]
    return F.interpolate(prob_crop, size=(meta["orig_h"], meta["orig_w"]), mode="bilinear", align_corners=False)


def pad_for_sliding_window(img_tensor, patch_size, stride):
    _, _, h, w = img_tensor.shape
    if h <= patch_size:
        pad_h = patch_size - h
    else:
        rem_h = (h - patch_size) % stride
        pad_h = 0 if rem_h == 0 else (stride - rem_h)
    if w <= patch_size:
        pad_w = patch_size - w
    else:
        rem_w = (w - patch_size) % stride
        pad_w = 0 if rem_w == 0 else (stride - rem_w)
    img_pad = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode="constant", value=-1.0)
    return img_pad, {"pad_h": pad_h, "pad_w": pad_w}


def build_weight_map(patch_size, device):
    w1 = th.hann_window(patch_size, periodic=False, device=device)
    w2 = th.hann_window(patch_size, periodic=False, device=device)
    weight = th.outer(w1, w2)
    weight = weight / weight.max()
    return weight.clamp(min=1e-3).view(1, 1, patch_size, patch_size)


def multi_sample_predict(diffusion, model, input_batch, batch_size, image_size, sample_fn, num_ensemble=1, clip_denoised=True, device=None):
    if device is None:
        device = next(model.parameters()).device
    input_batch = input_batch.to(device)
    prob_list = []
    for i in range(num_ensemble):
        logger.log(f"Ensemble {i+1}/{num_ensemble}")
        sample, pred_start, _, _ = sample_fn(
            model, (batch_size, 4, image_size, image_size), input_batch,
            clip_denoised=clip_denoised, model_kwargs={}, device=device,
        )
        prob_list.append(F.softmax(pred_start, dim=1).detach())
    prob_mean = th.stack(prob_list, dim=0).mean(dim=0)
    pred_classes = th.argmax(prob_mean, dim=1)
    return pred_classes, prob_mean


def global_diffusion_inference(diffusion, model, img_full, image_size, num_ensemble, clip_denoised, device):
    img_square, meta = resize_and_pad_to_square(img_full, image_size)
    dummy_mask = th.zeros((1, NUM_CLASSES, image_size, image_size), device=device)
    input_batch = th.cat([img_square, dummy_mask], dim=1)
    _, prob_square = multi_sample_predict(diffusion, model, input_batch, 1, image_size, diffusion.p_sample_loop_known, num_ensemble, clip_denoised, device)
    return restore_square_prob_to_orig(prob_square, meta)


def sliding_window_diffusion_inference(diffusion, model, img_full, patch_size, stride, num_ensemble, clip_denoised, device):
    img_pad, pad_info = pad_for_sliding_window(img_full, patch_size, stride)
    _, _, H, W = img_pad.shape
    prob_acc = th.zeros((1, NUM_CLASSES, H, W), device=device)
    weight_acc = th.zeros((1, 1, H, W), device=device)
    weight_map = build_weight_map(patch_size, device)
    for top in range(0, H - patch_size + 1, stride):
        for left in range(0, W - patch_size + 1, stride):
            img_patch = img_pad[:, :, top:top + patch_size, left:left + patch_size]
            dummy_mask = th.zeros((1, NUM_CLASSES, patch_size, patch_size), device=device)
            input_patch = th.cat([img_patch, dummy_mask], dim=1)
            _, prob_patch = multi_sample_predict(diffusion, model, input_patch, 1, patch_size, diffusion.p_sample_loop_known, num_ensemble, clip_denoised, device)
            prob_acc[:, :, top:top + patch_size, left:left + patch_size] += prob_patch * weight_map
            weight_acc[:, :, top:top + patch_size, left:left + patch_size] += weight_map
    prob_full = prob_acc / weight_acc.clamp(min=1e-6)
    scaled_h = H - pad_info["pad_h"]
    scaled_w = W - pad_info["pad_w"]
    return prob_full[:, :, :scaled_h, :scaled_w]


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    ds = MyDataset(args, args.data_dir, mode='test')
    datal = th.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location='cpu'))
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    all_dice, all_iou = [], []
    for batch_idx, (b, m, path) in enumerate(datal):
        slice_ID = path[0].split('.')[0].split('/')[-1]
        logger.log(f"sampling processing {slice_ID}...")
        b = b.to(dist_util.dev())
        m = m.to(dist_util.dev())
        orig_h, orig_w = b.shape[-2], b.shape[-1]
        prob_branches = []
        branch_weights = []
        if args.use_global_branch:
            prob_global_orig = global_diffusion_inference(diffusion, model, b, args.image_size, args.num_ensemble_global, args.clip_denoised, dist_util.dev())
            prob_branches.append(prob_global_orig)
            branch_weights.append(args.fusion_alpha_global)
        if args.use_local_branch:
            img_scaled, _ = resize_image_keep_ratio(b, args.test_long_side)
            prob_scaled = sliding_window_diffusion_inference(diffusion, model, img_scaled, args.image_size, args.test_stride, args.num_ensemble_local, args.clip_denoised, dist_util.dev())
            prob_local_orig = F.interpolate(prob_scaled, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            prob_branches.append(prob_local_orig)
            branch_weights.append(1.0 - args.fusion_alpha_global if args.use_global_branch else 1.0)
        weight_sum = sum(branch_weights)
        prob_final = sum(w * p for w, p in zip(branch_weights, prob_branches)) / max(weight_sum, 1e-6)
        pred_classes = th.argmax(prob_final, dim=1)[0].cpu()
        gt_classes = th.argmax(m, dim=1)[0].cpu()
        mean_dice, mean_iou, dice_dict, iou_dict = compute_avg_foreground_metrics(pred_classes, gt_classes)
        all_dice.append(mean_dice)
        all_iou.append(mean_iou)
        logger.log(f"[{batch_idx+1}/{len(datal)}] {slice_ID}")
        logger.log(f"Final Dice (orig size): {mean_dice:.4f}")
        logger.log(f"Final IoU  (orig size): {mean_iou:.4f}")
        if dice_dict:
            logger.log(f"Per-class Dice: {dice_dict}")
            logger.log(f"Per-class IoU: {iou_dict}")
        logger.log(f"Running Mean Dice: {np.mean(all_dice):.4f}")
        logger.log(f"Running Mean IoU: {np.mean(all_iou):.4f}")
        logger.log('--------------------------------------------------')
        save_results_at_original_size(pred_classes, gt_classes, b[0].cpu(), args.out_dir, f"{slice_ID}_pred")
    logger.log("\nOverall Results:")
    logger.log(f"Mean Dice Score: {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    logger.log(f"Mean IoU Score: {np.mean(all_iou):.4f} ± {np.std(all_iou):.4f}")
    logger.log("====================================\n")


def create_argparser():

    defaults = dict(
        data_name='mydata',
        data_dir='./data/mydata',
        clip_denoised=True,
        batch_size=1,
        use_ddim=False,
        model_path='',
        out_dir='./results/sample/mydata/window_iou/',
        gpu_dev='0',
        multi_gpu=None,
        use_global_branch=True,
        use_local_branch=True,
        num_ensemble_global=1,
        num_ensemble_local=1,
        fusion_alpha_global=0.4,
        test_long_side=640,
        test_stride=128,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == '__main__':
    main()
