"""
数据加载器v4：方案4实现

特点：
1. 长边缩放到256，短边填充至256×256
2. 保留原始宽高比信息
3. 生成Canny边缘伪标签
4. 记录padding_info便于后续反投影

数据流：
  原图(H, W) → 等比缩放 → 对称填充 → (256, 256)
             → Canny边缘生成
             → 返回：img, mask, edge_label, padding_info
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Dict, Optional


class MyDatasetV4(Dataset):
    """
    医学图像分割数据集 - v4版本（方案4）

    特点：
    - 长边缩放到256，短边填充
    - 返回填充信息用于反投影
    - 包含Canny边缘伪标签
    """

    def __init__(self, data_dir: str, split: str = 'train',
                 target_size: int = 256,
                 canny_threshold1: int = 50,
                 canny_threshold2: int = 150,
                 normalize: bool = True):
        """
        Args:
            data_dir: 数据目录路径
            split: 数据集划分 ('train', 'val', 'test')
            target_size: 目标尺寸
            canny_threshold1: Canny算子阈值1
            canny_threshold2: Canny算子阈值2
            normalize: 是否归一化图像到[0, 1]
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.target_size = target_size
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        self.normalize = normalize

        # 查找数据文件
        self.img_dir = self.data_dir / f"{split}_data"
        self.mask_dir = self.data_dir / f"{split}_mask"  # 如果存在
        self.img_files = sorted(self.img_dir.glob("*.png"))

        if len(self.img_files) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

        print(f"[DataLoader V4] Found {len(self.img_files)} images in {split} split")

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Dict:
        """
        Args:
            idx: 数据索引

        Returns:
            dict包含：
              - 'img': 填充后的图像 (1, 256, 256)
              - 'mask': 填充后的mask (3, 256, 256) one-hot编码
              - 'edge': Canny边缘 (1, 256, 256)
              - 'padding_info': 填充信息用于反投影
        """
        img_path = self.img_files[idx]

        # 读取图像
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        # 读取mask（如果存在）
        mask_path = self.mask_dir / img_path.name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
            # 假设mask是RGB格式，其中R=bg, G=fg1, B=fg2
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        else:
            # 如果mask不存在，创建虚拟mask
            mask = np.zeros((*img.shape, 3), dtype=np.uint8)

        # 应用方案4：长边缩放+填充
        img_padded, mask_padded, padding_info = self._resize_and_pad(img, mask)

        # 生成Canny边缘
        edge_label = self._generate_canny_edge(mask_padded)

        # 归一化
        if self.normalize:
            img_padded = img_padded.astype(np.float32) / 255.0
            mask_padded = mask_padded.astype(np.float32) / 255.0
        else:
            img_padded = img_padded.astype(np.float32)
            mask_padded = mask_padded.astype(np.float32)

        edge_label = edge_label.astype(np.float32)

        # 转为torch张量
        img_tensor = torch.from_numpy(img_padded).unsqueeze(0)  # (1, 256, 256)
        mask_tensor = torch.from_numpy(mask_padded).permute(2, 0, 1)  # (3, 256, 256)
        edge_tensor = torch.from_numpy(edge_label).unsqueeze(0)  # (1, 256, 256)

        return {
            'img': img_tensor,
            'mask': mask_tensor,
            'edge': edge_tensor,
            'padding_info': padding_info,
        }

    def _resize_and_pad(self, img: np.ndarray, mask: np.ndarray,
                       pad_value: int = 0) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        长边缩放到target_size，短边填充至target_size×target_size

        Args:
            img: 原始图像 (H, W)
            mask: 原始mask (H, W, 3)
            pad_value: 填充值

        Returns:
            img_padded: (256, 256)
            mask_padded: (256, 256, 3)
            padding_info: dict包含缩放和填充信息
        """
        H, W = img.shape[:2]

        # 计算缩放因子（以长边为准）
        scale = self.target_size / max(H, W)
        new_h = int(H * scale)
        new_w = int(W * scale)

        # 缩放图像和mask
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # 计算填充量
        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # 填充（图像用pad_value，mask用0）
        if len(img_resized.shape) == 2:
            img_padded = np.pad(
                img_resized,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=pad_value
            )
        else:
            img_padded = np.pad(
                img_resized,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant',
                constant_values=pad_value
            )

        mask_padded = np.pad(
            mask_resized,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant',
            constant_values=0  # 背景类
        )

        # 记录padding信息
        padding_info = {
            'scale': float(scale),
            'original_size': (H, W),
            'resized_size': (new_h, new_w),
            'padded_size': (self.target_size, self.target_size),
            'pad_coords': (pad_top, pad_bottom, pad_left, pad_right),
        }

        return img_padded, mask_padded, padding_info

    def _generate_canny_edge(self, mask: np.ndarray,
                            threshold1: Optional[int] = None,
                            threshold2: Optional[int] = None) -> np.ndarray:
        """
        从mask生成Canny边缘伪标签

        Args:
            mask: (H, W, 3) 或 (256, 256, 3)
            threshold1: Canny阈值1
            threshold2: Canny阈值2

        Returns:
            edge: (256, 256) 二值边缘图
        """
        if threshold1 is None:
            threshold1 = self.canny_threshold1
        if threshold2 is None:
            threshold2 = self.canny_threshold2

        # 获取前景类（class 1 + class 2）
        if mask.shape[2] == 3:
            fg_mask = (mask[:, :, 1] > 127).astype(np.uint8) | (mask[:, :, 2] > 127).astype(np.uint8)
        else:
            fg_mask = (mask[:, :, 0] > 127).astype(np.uint8)

        # Canny边缘检测
        edges = cv2.Canny(fg_mask, threshold1, threshold2)

        # 二值化
        edge_label = (edges > 0).astype(np.uint8)

        return edge_label


# ============================================================================
# 辅助函数
# ============================================================================

def unpad_and_rescale(pred_seg: np.ndarray, padding_info: Dict) -> np.ndarray:
    """
    将256×256的预测结果反投影回原图尺寸

    Args:
        pred_seg: 预测分割 (256, 256) 或 (C, 256, 256) 或 (B, C, 256, 256)
        padding_info: 包含缩放和填充信息的字典

    Returns:
        seg_original: 反投影后的分割 (H_original, W_original) 或 (C, H_orig, W_orig)
    """
    # 处理不同的输入shape
    if isinstance(pred_seg, torch.Tensor):
        pred_seg = pred_seg.detach().cpu().numpy()

    original_shape = pred_seg.shape
    if pred_seg.ndim == 4:
        B, C, H, W = pred_seg.shape
        pred_seg = pred_seg[0]  # 取第一个batch
    elif pred_seg.ndim == 3:
        C, H, W = pred_seg.shape
    elif pred_seg.ndim == 2:
        H, W = pred_seg.shape
        pred_seg = np.expand_dims(pred_seg, axis=0)
    else:
        raise ValueError(f"Unsupported input shape: {original_shape}")

    pad_top, pad_bottom, pad_left, pad_right = padding_info['pad_coords']
    scale = padding_info['scale']
    H_orig, W_orig = padding_info['original_size']

    # 1. 去掉填充
    if pred_seg.ndim == 3:
        seg_unpadded = pred_seg[:, pad_top:256-pad_bottom, pad_left:256-pad_right]
    else:
        seg_unpadded = pred_seg[pad_top:256-pad_bottom, pad_left:256-pad_right]

    # 2. 缩放回原图尺寸
    if seg_unpadded.ndim == 3:
        # (C, H_resized, W_resized) -> (H_resized, W_resized, C) -> resize -> (H_orig, W_orig, C) -> (C, H_orig, W_orig)
        seg_unpadded = seg_unpadded.transpose(1, 2, 0)
        seg_original = cv2.resize(
            seg_unpadded,
            (W_orig, H_orig),
            interpolation=cv2.INTER_LINEAR
        )
        seg_original = seg_original.transpose(2, 0, 1)
    else:
        seg_original = cv2.resize(
            seg_unpadded,
            (W_orig, H_orig),
            interpolation=cv2.INTER_LINEAR
        )

    return seg_original


def rescale_to_target(img: np.ndarray, target_size: int = 256) -> Tuple[np.ndarray, Dict]:
    """
    快速版本：仅用于推理

    Args:
        img: 输入图像 (H, W)
        target_size: 目标尺寸

    Returns:
        img_resized: 调整后的图像 (256, 256)
        padding_info: 填充信息
    """
    H, W = img.shape[:2]

    # 缩放
    scale = target_size / max(H, W)
    new_h = int(H * scale)
    new_w = int(W * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 填充
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    img_padded = np.pad(
        img_resized,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=0
    )

    padding_info = {
        'scale': float(scale),
        'original_size': (H, W),
        'resized_size': (new_h, new_w),
        'pad_coords': (pad_top, pad_bottom, pad_left, pad_right),
    }

    return img_padded, padding_info


if __name__ == "__main__":
    # 测试代码
    import matplotlib.pyplot as plt

    # 创建数据集
    dataset = MyDatasetV4(
        data_dir="./data/mydata",
        split="train",
        target_size=256
    )

    # 获取一个样本
    sample = dataset[0]

    print("Sample keys:", sample.keys())
    print(f"  img shape: {sample['img'].shape}")
    print(f"  mask shape: {sample['mask'].shape}")
    print(f"  edge shape: {sample['edge'].shape}")
    print(f"  padding_info: {sample['padding_info']}")
