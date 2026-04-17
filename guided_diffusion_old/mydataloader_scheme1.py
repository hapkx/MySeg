"""
【方案1】缩放原图到256×256版本

核心思路：
  - UNet输入：256×256 patch（裁剪的局部）
  - PVT输入：1186×852原图缩放到256×256（全局视图）
  - 优点：完全对齐，不需要坐标映射
  - 缺点：PVT看到的是降分辨率的全局信息

训练模式：
  1) full-image分支：整图缩放到256×256，PVT/UNet都输入这个256×256
  2) patch分支：从原图裁patch，UNet输入256×256 patch，PVT输入同一张图的缩小版本(256×256)
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import torch.nn.functional as F


# =========================================================
# 基础工具函数
# =========================================================
def resize_long_side_pil(img, long_side, is_mask=False):
    """保持宽高比，把长边缩放到 long_side"""
    w, h = img.size
    scale = long_side / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    interp = Image.NEAREST if is_mask else Image.BILINEAR
    return img.resize((new_w, new_h), interp)


def resize_with_aspect_and_pad(img, target_size, is_mask=False, fill=0):
    """保持宽高比缩放，然后 pad 到 target_size"""
    target_h, target_w = target_size
    w, h = img.size

    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    interp = Image.NEAREST if is_mask else Image.BILINEAR
    img = img.resize((new_w, new_h), interp)

    pad_w = target_w - new_w
    pad_h = target_h - new_h

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    img = ImageOps.expand(img, border=(left, top, right, bottom), fill=fill)
    return img


def pad_if_smaller_np(img_np, mask_np, patch_size, fill_img=0, fill_mask=0):
    """如果缩放后图像仍小于 patch_size，则右下补零"""
    h, w = img_np.shape[:2]
    target_h = max(h, patch_size)
    target_w = max(w, patch_size)

    if target_h == h and target_w == w:
        return img_np, mask_np

    img_pad = np.full((target_h, target_w), fill_img, dtype=img_np.dtype)
    mask_pad = np.full((target_h, target_w), fill_mask, dtype=mask_np.dtype)

    img_pad[:h, :w] = img_np
    mask_pad[:h, :w] = mask_np
    return img_pad, mask_pad


def random_crop_with_fg_preference(
    img_np,
    mask_np,
    patch_size=256,
    fg_sample_ratio=0.6,
    min_fg_pixels=64,
    max_tries=20,
):
    """优先裁取包含前景的 patch"""
    img_np, mask_np = pad_if_smaller_np(img_np, mask_np, patch_size)
    h, w = img_np.shape[:2]

    def crop_at(top, left):
        img_patch = img_np[top:top + patch_size, left:left + patch_size]
        mask_patch = mask_np[top:top + patch_size, left:left + patch_size]
        return img_patch, mask_patch

    has_fg = np.any(mask_np > 0)
    use_fg = has_fg and (random.random() < fg_sample_ratio)

    if use_fg:
        ys, xs = np.where(mask_np > 0)
        for _ in range(max_tries):
            idx = random.randint(0, len(ys) - 1)
            cy, cx = ys[idx], xs[idx]

            top = np.clip(cy - random.randint(0, patch_size - 1), 0, h - patch_size)
            left = np.clip(cx - random.randint(0, patch_size - 1), 0, w - patch_size)

            img_patch, mask_patch = crop_at(top, left)

            if np.sum(mask_patch > 0) >= min_fg_pixels:
                return img_patch, mask_patch

    # 回退到随机裁剪
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    return crop_at(top, left)


def robust_normalize_gray(img_np, lower_q=1.0, upper_q=99.0):
    """对灰度图做分位数裁剪，再归一化到 [0,1]"""
    img_np = img_np.astype(np.float32)

    lo = np.percentile(img_np, lower_q)
    hi = np.percentile(img_np, upper_q)

    if hi <= lo:
        lo = img_np.min()
        hi = img_np.max() + 1e-6

    img_np = np.clip(img_np, lo, hi)
    img_np = (img_np - lo) / (hi - lo + 1e-6)
    return img_np


def random_intensity_augment(img_pil, p=0.5):
    """对灰度图做轻量亮度/对比度/gamma扰动"""
    if random.random() > p:
        return img_pil

    b = random.uniform(0.9, 1.1)
    img_pil = TF.adjust_brightness(img_pil, b)

    c = random.uniform(0.9, 1.1)
    img_pil = TF.adjust_contrast(img_pil, c)

    g = random.uniform(0.9, 1.1)
    img_pil = TF.adjust_gamma(img_pil, g)

    return img_pil


# =========================================================
# Dataset - 方案1：缩放原图到256×256
# =========================================================
class MyDatasetScheme1(Dataset):
    """
    【方案1】缩放原图版本

    返回值:
      img_patch: (1, 256, 256) - UNet的patch输入
      img_full_small: (1, 256, 256) - PVT的输入（原图缩小到256×256）
      mask_patch: (3, 256, 256) - ground truth
    """
    def __init__(self, args, data_path, transform=None, mode='training', plane=False, num_classes=3):
        csv_name = f"mydata_{mode}_groundtruth.csv"
        df = pd.read_csv(os.path.join(data_path, csv_name), encoding='gbk')

        self.name_list = df.iloc[:, 1].tolist()
        self.label_list = df.iloc[:, 2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.num_classes = num_classes

        # 训练相关参数
        self.patch_size = getattr(args, "image_size", 256)
        self.train_long_side = getattr(args, "train_long_side", 640)
        self.patches_per_image = getattr(args, "patches_per_image", 8)
        self.fg_sample_ratio = getattr(args, "fg_sample_ratio", 0.6)
        self.min_fg_pixels = getattr(args, "min_fg_pixels", 64)
        self.full_image_ratio = getattr(args, "full_image_ratio", 0.7)
        self.intensity_aug_p = getattr(args, "intensity_aug_p", 0.5)

        self.color_map = {
            (0, 0, 0): 0,        # 背景
            (128, 0, 0): 1,      # 类别1
            (0, 128, 0): 2,      # 类别2
        }

    def __len__(self):
        if self.mode == 'training':
            return len(self.name_list) * self.patches_per_image
        return len(self.name_list)

    def rgb_to_mask(self, rgb_mask):
        """RGB mask -> 类别索引mask"""
        rgb_mask = np.array(rgb_mask)
        h, w = rgb_mask.shape[:2]
        mask = np.zeros((h, w), dtype=np.int64)

        for color, class_id in self.color_map.items():
            diff = np.abs(rgb_mask.astype(np.int32) - np.array(color))
            distance = np.sum(diff, axis=-1)
            matches = distance < 10
            mask[matches] = class_id

        return mask

    def _load_full_sample(self, real_index):
        name = self.name_list[real_index]
        img_path = os.path.join(self.data_path, name)
        mask_name = self.label_list[real_index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('L')
        mask_rgb = Image.open(msk_path).convert('RGB')

        mask = self.rgb_to_mask(mask_rgb)
        mask = Image.fromarray(mask.astype(np.uint8))

        return img, mask, name

    def _apply_basic_aug(self, img_pil, mask_pil):
        """对 full-image 和 patch 两个分支共用的几何增强"""
        if torch.rand(1) < 0.5:
            img_pil = TF.hflip(img_pil)
            mask_pil = TF.hflip(mask_pil)

        angle = torch.randint(-10, 11, (1,)).item()
        img_pil = TF.rotate(img_pil, angle, interpolation=Image.BILINEAR, fill=0)
        mask_pil = TF.rotate(mask_pil, angle, interpolation=Image.NEAREST, fill=0)

        return img_pil, mask_pil

    def _img_to_tensor_with_robust_norm(self, img_pil):
        """PIL灰度图 -> robust normalize -> tensor -> [-1,1]"""
        img_np = np.array(img_pil, dtype=np.uint8)
        img_np = robust_normalize_gray(img_np, lower_q=1.0, upper_q=99.0)
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).float()
        img_tensor = img_tensor * 2.0 - 1.0
        return img_tensor

    def _img_to_tensor_robust_0to1(self, img_pil):
        """PIL灰度图 -> robust normalize -> tensor，保留在 [0,1]"""
        img_np = np.array(img_pil, dtype=np.uint8)
        img_np = robust_normalize_gray(img_np, lower_q=1.0, upper_q=99.0)
        return torch.from_numpy(img_np).unsqueeze(0).float()

    def _intensity_augment_tensor(self, img_tensor, p=0.5):
        """在归一化后的 [0,1] tensor 空间做轻量强度扰动"""
        if random.random() > p:
            return img_tensor

        b = random.uniform(0.9, 1.1)
        img_tensor = img_tensor * b

        c = random.uniform(0.9, 1.1)
        mean_val = img_tensor.mean()
        img_tensor = (img_tensor - mean_val) * c + mean_val

        g = random.uniform(0.9, 1.1)
        img_tensor = img_tensor.clamp(0.0, 1.0).pow(g)

        return img_tensor.clamp(0.0, 1.0)

    def _mask_to_onehot(self, mask_pil):
        mask_tensor = torch.from_numpy(np.array(mask_pil)).long()
        mask_onehot = F.one_hot(mask_tensor, num_classes=self.num_classes)
        mask_onehot = mask_onehot.permute(2, 0, 1).float()
        return mask_onehot

    def __getitem__(self, index):
        """
        返回:
          img_patch: UNet输入的256×256 patch
          img_full_small: PVT输入的256×256缩小版全图
          mask_patch: 对应的mask
        """
        if self.mode == 'training':
            real_index = index % len(self.name_list)
        else:
            real_index = index

        img, mask, name = self._load_full_sample(real_index)

        if self.mode == 'training':
            use_full_image = (random.random() < self.full_image_ratio)

            if use_full_image:
                # ===== Full-image分支 =====
                # 缩放原图到256×256
                img_full = resize_with_aspect_and_pad(
                    img,
                    (self.patch_size, self.patch_size),
                    is_mask=False,
                    fill=0
                )
                mask_full = resize_with_aspect_and_pad(
                    mask,
                    (self.patch_size, self.patch_size),
                    is_mask=True,
                    fill=0
                )

                img_full, mask_full = self._apply_basic_aug(img_full, mask_full)

                # 转换到tensor
                img_tensor = self._img_to_tensor_robust_0to1(img_full)
                img_tensor = self._intensity_augment_tensor(img_tensor, p=self.intensity_aug_p)
                img_tensor = img_tensor * 2.0 - 1.0  # [-1,1]

                mask_onehot = self._mask_to_onehot(mask_full)

                # 【方案1关键】返回两个相同的输入
                return img_tensor, img_tensor, mask_onehot, name

            else:
                # ===== Patch分支 =====
                img_patch_src = resize_long_side_pil(img, self.train_long_side, is_mask=False)
                mask_patch_src = resize_long_side_pil(mask, self.train_long_side, is_mask=True)

                img_np = np.array(img_patch_src, dtype=np.uint8)
                mask_np = np.array(mask_patch_src, dtype=np.int64)

                img_patch, mask_patch = random_crop_with_fg_preference(
                    img_np,
                    mask_np,
                    patch_size=self.patch_size,
                    fg_sample_ratio=self.fg_sample_ratio,
                    min_fg_pixels=self.min_fg_pixels,
                    max_tries=20,
                )

                img_patch = Image.fromarray(img_patch)
                mask_patch = Image.fromarray(mask_patch.astype(np.uint8))

                img_patch, mask_patch = self._apply_basic_aug(img_patch, mask_patch)

                # 转换到tensor
                img_patch_tensor = self._img_to_tensor_robust_0to1(img_patch)
                img_patch_tensor = self._intensity_augment_tensor(img_patch_tensor, p=self.intensity_aug_p)
                img_patch_tensor = img_patch_tensor * 2.0 - 1.0  # [-1,1]

                # 【方案1关键】原图也缩小到256×256，与patch共享上下文
                img_full = resize_with_aspect_and_pad(
                    img,
                    (self.patch_size, self.patch_size),
                    is_mask=False,
                    fill=0
                )
                img_full_tensor = self._img_to_tensor_robust_0to1(img_full)
                img_full_tensor = self._intensity_augment_tensor(img_full_tensor, p=self.intensity_aug_p)
                img_full_tensor = img_full_tensor * 2.0 - 1.0  # [-1,1]

                mask_onehot = self._mask_to_onehot(mask_patch)

                return img_patch_tensor, img_full_tensor, mask_onehot, name

        else:
            # 测试时返回原图大小
            img_tensor = self._img_to_tensor_with_robust_norm(img)

            # 为了推理兼容性，也返回缩小版本
            img_full = resize_with_aspect_and_pad(
                img,
                (self.patch_size, self.patch_size),
                is_mask=False,
                fill=0
            )
            img_full_tensor = self._img_to_tensor_with_robust_norm(img_full)

            mask_onehot = self._mask_to_onehot(mask)
            return img_tensor, img_full_tensor, mask_onehot, name
