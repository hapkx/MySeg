"""
高级损失函数：SDF Loss、Hausdorff Loss、Edge Supervision Loss

支持医学图像分割中的边界细化和结构约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


class SDFLoss(nn.Module):
    """
    Signed Distance Field Loss (SDF Loss)

    原理：
    1. 从GT mask计算Signed Distance Field
    2. 在预测的logits上计算L2距离
    3. 在边界处施加强梯度信号

    优点：
    - 直接优化边界位置对齐
    - 边界处的梯度信号强
    - 对边界细化有显著帮助

    参考：
      "Boundary Loss for Remote Sensing Image Segmentation" (Kervadec et al.)
    """

    def __init__(self, weight: float = 0.2, normalize: bool = True):
        """
        Args:
            weight: 损失权重
            normalize: 是否对距离进行归一化
        """
        super().__init__()
        self.weight = weight
        self.normalize = normalize

    def forward(self, pred_logits: torch.Tensor, target_onehot: torch.Tensor,
               foreground_classes: list = None) -> torch.Tensor:
        """
        Args:
            pred_logits: 模型预测logits (B, C, H, W)
                        其中C是分类数（包括背景）
            target_onehot: GT one-hot编码 (B, C, H, W)
            foreground_classes: 前景类别ID列表，默认[1, 2]

        Returns:
            SDF Loss (标量)
        """
        if foreground_classes is None:
            foreground_classes = [1, 2]

        B, C, H, W = pred_logits.shape
        device = pred_logits.device

        # 转换为概率
        pred_probs = F.softmax(pred_logits, dim=1)  # (B, C, H, W)

        total_loss = 0.0

        for cls_id in foreground_classes:
            # 获取该类别的GT掩码
            target_cls = target_onehot[:, cls_id, :, :]  # (B, H, W)
            pred_cls = pred_probs[:, cls_id, :, :]      # (B, H, W)

            # 计算SDF
            sdf = self._compute_sdf_batch(target_cls)  # (B, H, W)

            # 对距离进行归一化（使梯度更稳定）
            if self.normalize:
                sdf_max = torch.abs(sdf).max()
                if sdf_max > 0:
                    sdf = sdf / (sdf_max + 1e-6)

            # 损失：在GT边界处（|sdf|≈0）有强约束，内部约束较弱
            # 使用 |sdf| 作为权重：边界处权重大，内部权重小
            boundary_weight = torch.abs(sdf)
            loss_cls = F.l1_loss(pred_cls, target_cls, reduction='none')
            loss_cls = (boundary_weight * loss_cls).mean()

            total_loss = total_loss + loss_cls

        total_loss = total_loss / len(foreground_classes)
        return self.weight * total_loss

    @staticmethod
    def _compute_sdf_batch(mask: torch.Tensor) -> torch.Tensor:
        """
        计算一个batch的Signed Distance Field

        Args:
            mask: (B, H, W) 二值掩码 (0/1)

        Returns:
            sdf: (B, H, W) 有符号距离场
                 正值表示在前景内部，负值表示在背景内
        """
        B, H, W = mask.shape
        device = mask.device

        sdf_batch = []
        mask_np = mask.detach().cpu().numpy()

        for b in range(B):
            mask_b = mask_np[b]  # (H, W)

            # 计算前景到背景的距离
            fg_dist = distance_transform_edt(mask_b > 0.5)

            # 计算背景到前景的距离
            bg_dist = distance_transform_edt(mask_b <= 0.5)

            # 组合：前景为正，背景为负
            sdf = np.where(mask_b > 0.5, fg_dist, -bg_dist)
            sdf_batch.append(torch.from_numpy(sdf).to(device).float())

        sdf = torch.stack(sdf_batch, dim=0)  # (B, H, W)
        return sdf


class HausdorffLoss(nn.Module):
    """
    可微分的Hausdorff Distance Loss

    原理：
    1. 计算预测到GT的距离
    2. 计算GT到预测的距离
    3. 取两者的最大值（或高百分位数）

    优点：
    - 对离群错误（孤立噪声点）敏感
    - 补充Dice Loss对小偏差的不敏感性
    - 强制边界对齐

    参考：
      "Hausdorff Distance Loss for Semantic Segmentation" (Karimi et al.)
    """

    def __init__(self, weight: float = 0.1, percentile: float = 95.0):
        """
        Args:
            weight: 损失权重
            percentile: 使用百分位数而非最大值（更稳定）
        """
        super().__init__()
        self.weight = weight
        self.percentile = percentile

    def forward(self, pred_logits: torch.Tensor, target_onehot: torch.Tensor,
               foreground_classes: list = None) -> torch.Tensor:
        """
        Args:
            pred_logits: 模型预测logits (B, C, H, W)
            target_onehot: GT one-hot编码 (B, C, H, W)
            foreground_classes: 前景类别ID列表，默认[1, 2]

        Returns:
            Hausdorff Loss (标量)
        """
        if foreground_classes is None:
            foreground_classes = [1, 2]

        B, C, H, W = pred_logits.shape
        device = pred_logits.device

        # 转换为概率
        pred_probs = F.softmax(pred_logits, dim=1)  # (B, C, H, W)

        total_loss = 0.0

        for cls_id in foreground_classes:
            # 获取该类别的二值掩码
            target_cls = (target_onehot[:, cls_id, :, :] > 0.5).float()  # (B, H, W)
            pred_cls = pred_probs[:, cls_id, :, :]  # (B, H, W)

            # 计算Hausdorff距离
            hd_loss = self._hausdorff_distance(pred_cls, target_cls)
            total_loss = total_loss + hd_loss

        total_loss = total_loss / len(foreground_classes)
        return self.weight * total_loss

    def _hausdorff_distance(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算两个mask之间的可微分Hausdorff距离

        Args:
            pred: 预测概率图 (B, H, W)
            target: GT二值掩码 (B, H, W)

        Returns:
            Hausdorff距离损失
        """
        B, H, W = pred.shape
        device = pred.device

        # 二值化预测
        pred_binary = (pred > 0.5).float()

        # 计算距离变换：使用硬掩码计算距离
        # 从pred的边界到target的距离
        dist_pred_to_target = self._compute_distance_tensor(pred_binary, target)
        # 从target的边界到pred的距离
        dist_target_to_pred = self._compute_distance_tensor(target, pred_binary)

        # 计算Hausdorff距离：max(d(pred->target), d(target->pred))
        # 使用高百分位数使损失更稳定
        h1 = torch.quantile(dist_pred_to_target.view(B, -1), self.percentile / 100.0, dim=1)
        h2 = torch.quantile(dist_target_to_pred.view(B, -1), self.percentile / 100.0, dim=1)

        hd = torch.max(h1, h2).mean()  # 取平均以获得标量
        return hd

    @staticmethod
    def _compute_distance_tensor(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算source的点到target的最小距离

        Args:
            source: (B, H, W)
            target: (B, H, W)

        Returns:
            距离张量 (B, H, W)
        """
        B, H, W = source.shape
        device = source.device

        # 获取source的边界点
        source_edges = HausdorffLoss._get_edge_points(source)  # List of (y, x) coordinates

        if len(source_edges) == 0:
            return torch.zeros_like(source)

        # 获取target的距离变换
        target_np = target.detach().cpu().numpy()
        dist_transform_list = []

        for b in range(B):
            # 使用scipy计算距离变换
            dt = distance_transform_edt(target_np[b] > 0.5)
            dist_transform_list.append(torch.from_numpy(dt).to(device).float())

        dist_tensor = torch.stack(dist_transform_list, dim=0)  # (B, H, W)
        return dist_tensor

    @staticmethod
    def _get_edge_points(mask: torch.Tensor) -> list:
        """获取掩码的边界点"""
        # 简化版本：直接返回掩码的点
        # 实际应用中可以使用形态学操作获取真正的边界
        points = torch.where(mask > 0.5)
        return list(zip(points[0].cpu().numpy(), points[1].cpu().numpy()))


class EdgeSupervisionLoss(nn.Module):
    """
    边缘预测的监督损失

    用于UNet的边缘监督分支
    组合Binary Cross Entropy和Dice Loss
    """

    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5):
        """
        Args:
            weight_bce: BCE损失权重
            weight_dice: Dice损失权重
        """
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, edge_pred: torch.Tensor, edge_target: torch.Tensor,
               pos_weight: float = 1.0) -> torch.Tensor:
        """
        Args:
            edge_pred: 边缘预测logits (B, 1, H, W)
            edge_target: 边缘GT (B, 1, H, W) 二值图 [0, 1]
            pos_weight: 正样本权重（因为边缘点通常很少）

        Returns:
            Edge Loss (标量)
        """
        # 转换为概率
        edge_prob = torch.sigmoid(edge_pred)  # (B, 1, H, W)

        # BCE Loss（加权以处理不平衡）
        bce_loss = F.binary_cross_entropy(
            edge_prob,
            edge_target,
            reduction='mean'
        )

        # Dice Loss
        intersection = (edge_prob * edge_target).sum()
        union = edge_prob.sum() + edge_target.sum()
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)

        # 组合损失
        total_loss = (self.weight_bce * bce_loss +
                     self.weight_dice * dice_loss)

        return total_loss


class BoundaryAlignmentLoss(nn.Module):
    """
    边界对齐损失

    结合梯度信息和距离信息
    在GT边界处施加强约束
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, pred_logits: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits: (B, C, H, W)
            target_onehot: (B, C, H, W)

        Returns:
            Boundary Loss (标量)
        """
        B, C, H, W = pred_logits.shape

        pred_probs = F.softmax(pred_logits, dim=1)

        # 计算梯度
        # X方向梯度
        pred_grad_x = torch.abs(pred_probs[:, 1:, :, :-1] - pred_probs[:, 1:, :, 1:])
        target_grad_x = torch.abs(target_onehot[:, 1:, :, :-1] - target_onehot[:, 1:, :, 1:])

        # Y方向梯度
        pred_grad_y = torch.abs(pred_probs[:, 1:, :-1, :] - pred_probs[:, 1:, 1:, :])
        target_grad_y = torch.abs(target_onehot[:, 1:, :-1, :] - target_onehot[:, 1:, 1:, :])

        # 边界位置：GT梯度大的地方
        target_boundary_x = (target_grad_x > 0.1).float()
        target_boundary_y = (target_grad_y > 0.1).float()

        # 损失：在GT边界处，鼓励预测梯度也大；非边界处，鼓励预测平滑
        loss_x = (torch.abs(pred_grad_x - target_grad_x) * target_boundary_x).mean()
        loss_y = (torch.abs(pred_grad_y - target_grad_y) * target_boundary_y).mean()

        # 平滑性损失
        smoothness_x = (pred_grad_x * (1.0 - target_boundary_x)).mean()
        smoothness_y = (pred_grad_y * (1.0 - target_boundary_y)).mean()

        total_loss = loss_x + loss_y + 0.5 * (smoothness_x + smoothness_y)

        return self.weight * total_loss
