"""
单通道图像 + 3通道mask

训练：
    混合训练模式：
    1) patch训练：等比例缩放到较大尺寸 + 前景优先随机patch
    2) full-image训练：整图等比例缩放 + pad到 image_size
    并加入：
    - 轻量灰度增强（brightness / contrast / gamma）
    - 稳健归一化（percentile clipping + normalization）

测试：
    返回原图大小，不做resize，供 sample.py 做滑窗推理并恢复原图尺寸
    测试时只做稳健归一化，不做随机灰度增强
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
    """
    保持宽高比，把长边缩放到 long_side
    """
    w, h = img.size
    scale = long_side / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    interp = Image.NEAREST if is_mask else Image.BILINEAR
    return img.resize((new_w, new_h), interp)


def resize_with_aspect_and_pad(img, target_size, is_mask=False, fill=0):
    """
    保持宽高比缩放，然后 pad 到 target_size
    img: PIL.Image
    target_size: (target_h, target_w)
    """
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
    """
    如果缩放后图像仍小于 patch_size，则右下补零
    img_np: [H, W] uint8
    mask_np: [H, W] int64
    """
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
    """
    优先裁取包含前景的 patch
    img_np: [H, W]
    mask_np: [H, W], 类别索引
    """
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


# =========================================================
# 新增：稳健归一化 + 轻量灰度增强
# =========================================================
def robust_normalize_gray(img_np, lower_q=1.0, upper_q=99.0):
    """
    对灰度图做分位数裁剪，再归一化到 [0,1]
    img_np: numpy array, [H, W], uint8 or float
    """
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
    """
    对灰度图做轻量亮度/对比度/gamma扰动
    只建议训练时使用
    """
    if random.random() > p:
        return img_pil

    # brightness
    b = random.uniform(0.9, 1.1)
    img_pil = TF.adjust_brightness(img_pil, b)

    # contrast
    c = random.uniform(0.9, 1.1)
    img_pil = TF.adjust_contrast(img_pil, c)

    # gamma
    g = random.uniform(0.9, 1.1)
    img_pil = TF.adjust_gamma(img_pil, g)

    return img_pil


# =========================================================
# Dataset
# =========================================================
class MyDataset(Dataset):
    def __init__(self, args, data_path, transform=None, mode='training', plane=False, num_classes=3):
        csv_name = f"mydata_{mode}_groundtruth.csv"
        df = pd.read_csv(os.path.join(data_path, csv_name), encoding='gbk')

        self.name_list = df.iloc[:, 1].tolist()
        self.label_list = df.iloc[:, 2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.num_classes = num_classes

        # ===== 训练相关参数 =====
        self.patch_size = getattr(args, "image_size", 256)

        # patch支路：先把长边缩放到更大尺寸，再裁patch
        self.train_long_side = getattr(args, "train_long_side", 640)

        # 每张图每个epoch随机采样多少次
        self.patches_per_image = getattr(args, "patches_per_image", 8)

        # patch裁剪参数
        self.fg_sample_ratio = getattr(args, "fg_sample_ratio", 0.6)
        self.min_fg_pixels = getattr(args, "min_fg_pixels", 64)

        # full image混合训练比例
        self.full_image_ratio = getattr(args, "full_image_ratio", 0.7)

        # 轻量灰度增强概率
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
        """
        RGB mask -> 类别索引mask
        """
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
        """
        对 full-image 和 patch 两个分支共用的几何增强
        """
        if torch.rand(1) < 0.5:
            img_pil = TF.hflip(img_pil)
            mask_pil = TF.hflip(mask_pil)

        angle = torch.randint(-10, 11, (1,)).item()
        img_pil = TF.rotate(img_pil, angle, interpolation=Image.BILINEAR, fill=0)
        mask_pil = TF.rotate(mask_pil, angle, interpolation=Image.NEAREST, fill=0)

        return img_pil, mask_pil

    def _img_to_tensor_with_robust_norm(self, img_pil):
        """
        PIL灰度图 -> robust normalize -> tensor -> [-1,1]
        """
        img_np = np.array(img_pil, dtype=np.uint8)
        img_np = robust_normalize_gray(img_np, lower_q=1.0, upper_q=99.0)   # [0,1]
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).float()           # [1,H,W]
        img_tensor = img_tensor * 2.0 - 1.0                                  # [-1,1]
        return img_tensor

    def _img_to_tensor_robust_0to1(self, img_pil):
        """
        PIL灰度图 -> robust normalize -> tensor，保留在 [0,1]（供后续增强使用）
        """
        img_np = np.array(img_pil, dtype=np.uint8)
        img_np = robust_normalize_gray(img_np, lower_q=1.0, upper_q=99.0)
        return torch.from_numpy(img_np).unsqueeze(0).float()                 # [1,H,W] in [0,1]

    def _intensity_augment_tensor(self, img_tensor, p=0.5):
        """
        在归一化后的 [0,1] tensor 空间做轻量强度扰动，保证增强不被归一化抵消。
        img_tensor: [1,H,W], float, in [0,1]
        """
        if random.random() > p:
            return img_tensor

        # brightness: 乘以系数
        b = random.uniform(0.9, 1.1)
        img_tensor = img_tensor * b

        # contrast: 绕均值缩放
        c = random.uniform(0.9, 1.1)
        mean_val = img_tensor.mean()
        img_tensor = (img_tensor - mean_val) * c + mean_val

        # gamma: 幂次变换（仅在 [0,1] 空间有意义）
        g = random.uniform(0.9, 1.1)
        img_tensor = img_tensor.clamp(0.0, 1.0).pow(g)

        return img_tensor.clamp(0.0, 1.0)

    def _mask_to_onehot(self, mask_pil):
        mask_tensor = torch.from_numpy(np.array(mask_pil)).long()            # [H,W]
        mask_onehot = F.one_hot(mask_tensor, num_classes=self.num_classes)
        mask_onehot = mask_onehot.permute(2, 0, 1).float()                   # [3,H,W]
        return mask_onehot

    def __getitem__(self, index):
        """
        training:
            混合两种训练方式：
            1) full-image分支：整图等比例缩放 + pad -> [256,256]
            2) patch分支：长边缩放到较大尺寸，再裁 patch -> [256,256]
            并加入轻量灰度增强 + 稳健归一化
        test:
            返回原图大小完整图，只做稳健归一化，不做随机灰度增强
        """
        if self.mode == 'training':
            real_index = index % len(self.name_list)
        else:
            real_index = index

        img, mask, name = self._load_full_sample(real_index)

        if self.mode == 'training':
            use_full_image = (random.random() < self.full_image_ratio)

            # =========================
            # 分支1：full image训练
            # =========================
            if use_full_image:
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

                # 先归一化到 [0,1]，再做强度增强，避免 percentile clip 抵消增强效果
                img_tensor = self._img_to_tensor_robust_0to1(img_full)
                img_tensor = self._intensity_augment_tensor(img_tensor, p=self.intensity_aug_p)
                img_tensor = img_tensor * 2.0 - 1.0                          # [-1,1]

                mask_onehot = self._mask_to_onehot(mask_full)
                return img_tensor, mask_onehot, name

            # =========================
            # 分支2：patch训练
            # =========================
            else:
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

                # 先归一化到 [0,1]，再做强度增强，避免 percentile clip 抵消增强效果
                img_tensor = self._img_to_tensor_robust_0to1(img_patch)
                img_tensor = self._intensity_augment_tensor(img_tensor, p=self.intensity_aug_p)
                img_tensor = img_tensor * 2.0 - 1.0                          # [-1,1]

                mask_onehot = self._mask_to_onehot(mask_patch)
                return img_tensor, mask_onehot, name

        else:
            # 测试时不resize，直接返回原图大小
            # 不做随机强度增强，只做稳健归一化
            img_tensor = self._img_to_tensor_with_robust_norm(img)
            mask_onehot = self._mask_to_onehot(mask)
            return img_tensor, mask_onehot, name