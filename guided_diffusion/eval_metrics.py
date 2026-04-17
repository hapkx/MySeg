"""
医学图像分割评估指标计算

支持指标：
  - Dice: 基础分割指标
  - IoU: 基础分割指标
  - HD95: Hausdorff Distance 95%ile，捕捉离群错误
  - ASD: Average Surface Distance，平均边界偏差
  - Sensitivity: TP / (TP + FN)，捕捉真正目标能力
  - Specificity: TN / (TN + FP)，避免误判能力
  - Precision: TP / (TP + FP)，预测准确度
  - Recall: 同Sensitivity
"""

import numpy as np
import torch as th
from scipy.ndimage import distance_transform_edt
from typing import Dict, Tuple


class MetricsEvaluator:
    """医学图像分割的完整评估器"""

    def __init__(self, include_hd=True, include_asd=True):
        """
        Args:
            include_hd: 是否计算HD95（计算量较大）
            include_asd: 是否计算ASD（计算量较大）
        """
        self.include_hd = include_hd
        self.include_asd = include_asd

    def compute_all(self, pred: np.ndarray, target: np.ndarray,
                   class_ids: list = None, eps=1e-6) -> Dict[str, float]:
        """
        计算所有评估指标

        Args:
            pred: 预测分割 (H, W) 或 (B, H, W) or (B, C, H, W)
                  如果是(B, C, H, W)，会做argmax
            target: GT分割，格式同pred
            class_ids: 要评估的类别ID列表，默认为[1, 2]（前景类）
            eps: 数值稳定性

        Returns:
            metrics_dict: {
                'dice_class_1': float, 'dice_class_2': float, 'dice_mean': float,
                'iou_class_1': float, 'iou_class_2': float, 'iou_mean': float,
                'hd95_class_1': float, 'hd95_class_2': float, 'hd95_mean': float,
                'asd_class_1': float, 'asd_class_2': float, 'asd_mean': float,
                'sensitivity_class_1': float, 'sensitivity_class_2': float,
                'specificity_class_1': float, 'specificity_class_2': float,
                'precision_class_1': float, 'precision_class_2': float,
                'recall_class_1': float, 'recall_class_2': float,
            }
        """
        # 转换为numpy
        if isinstance(pred, th.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, th.Tensor):
            target = target.detach().cpu().numpy()

        # 处理不同的输入格式
        pred = self._prepare_input(pred)      # -> (B, H, W)
        target = self._prepare_input(target)  # -> (B, H, W)

        if class_ids is None:
            class_ids = [1, 2]

        metrics = {}

        # 计算各类别的指标
        dice_scores = []
        iou_scores = []
        hd95_scores = []
        asd_scores = []
        sen_scores = []
        spe_scores = []
        prec_scores = []
        rec_scores = []

        for class_id in class_ids:
            # 二值化：该类别 vs 其他
            pred_binary = (pred == class_id).astype(np.uint8)
            target_binary = (target == class_id).astype(np.uint8)

            # 基础指标
            dice = self.compute_dice(pred_binary, target_binary, eps)
            iou = self.compute_iou(pred_binary, target_binary, eps)

            metrics[f'dice_class_{class_id}'] = dice
            metrics[f'iou_class_{class_id}'] = iou

            dice_scores.append(dice)
            iou_scores.append(iou)

            # 距离指标（可选）
            if self.include_hd:
                hd95 = self.compute_hd95(pred_binary, target_binary)
                metrics[f'hd95_class_{class_id}'] = hd95
                hd95_scores.append(hd95)

            if self.include_asd:
                asd = self.compute_asd(pred_binary, target_binary)
                metrics[f'asd_class_{class_id}'] = asd
                asd_scores.append(asd)

            # 敏感性/特异性/精确度/召回率
            sen = self.compute_sensitivity(pred_binary, target_binary, eps)
            spe = self.compute_specificity(pred_binary, target_binary, eps)
            prec = self.compute_precision(pred_binary, target_binary, eps)
            rec = self.compute_recall(pred_binary, target_binary, eps)

            metrics[f'sensitivity_class_{class_id}'] = sen
            metrics[f'specificity_class_{class_id}'] = spe
            metrics[f'precision_class_{class_id}'] = prec
            metrics[f'recall_class_{class_id}'] = rec

            sen_scores.append(sen)
            spe_scores.append(spe)
            prec_scores.append(prec)
            rec_scores.append(rec)

        # 计算平均值
        metrics['dice_mean'] = np.mean(dice_scores)
        metrics['iou_mean'] = np.mean(iou_scores)
        metrics['sensitivity_mean'] = np.mean(sen_scores)
        metrics['specificity_mean'] = np.mean(spe_scores)
        metrics['precision_mean'] = np.mean(prec_scores)
        metrics['recall_mean'] = np.mean(rec_scores)

        if self.include_hd:
            metrics['hd95_mean'] = np.mean(hd95_scores)
        if self.include_asd:
            metrics['asd_mean'] = np.mean(asd_scores)

        return metrics

    @staticmethod
    def _prepare_input(x: np.ndarray) -> np.ndarray:
        """
        将输入标准化为 (B, H, W) 格式

        支持格式：
          (H, W) -> (1, H, W)
          (B, H, W) -> (B, H, W)
          (B, C, H, W) -> argmax -> (B, H, W)
        """
        if x.ndim == 2:
            return np.expand_dims(x, axis=0)
        elif x.ndim == 3:
            return x
        elif x.ndim == 4:
            # 假设是(B, C, H, W)，取argmax
            return np.argmax(x, axis=1)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

    @staticmethod
    def compute_dice(pred: np.ndarray, target: np.ndarray, eps=1e-6) -> float:
        """Dice系数，越高越好（0-1）"""
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target)
        dice = (2.0 * intersection + eps) / (union + eps)
        return float(dice)

    @staticmethod
    def compute_iou(pred: np.ndarray, target: np.ndarray, eps=1e-6) -> float:
        """IoU (Jaccard)，越高越好（0-1）"""
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target) - intersection
        iou = (intersection + eps) / (union + eps)
        return float(iou)

    @staticmethod
    def compute_hd95(pred: np.ndarray, target: np.ndarray) -> float:
        """
        Hausdorff Distance 95%ile

        定义：max(d(pred, target), d(target, pred))的95%ile
        越小越好，单位为像素
        """
        if np.sum(pred) == 0 or np.sum(target) == 0:
            return 373.0  # 返回最大可能距离（对于256×256的对角线）

        # 计算距离变换
        dist_pred_to_target = distance_transform_edt(~pred.astype(bool))
        dist_target_to_pred = distance_transform_edt(~target.astype(bool))

        # 计算pred内部到target的距离
        h1 = np.percentile(dist_pred_to_target[pred > 0], 95) if np.sum(pred) > 0 else 0

        # 计算target内部到pred的距离
        h2 = np.percentile(dist_target_to_pred[target > 0], 95) if np.sum(target) > 0 else 0

        hd95 = max(h1, h2)
        return float(hd95)

    @staticmethod
    def compute_asd(pred: np.ndarray, target: np.ndarray) -> float:
        """
        Average Surface Distance (ASD)

        定义：(d(pred, target) + d(target, pred)) / 2
        越小越好，单位为像素
        """
        if np.sum(pred) == 0 or np.sum(target) == 0:
            return 373.0

        # 计算距离变换（包括内部）
        dist_pred_to_target = distance_transform_edt(~pred.astype(bool))
        dist_target_to_pred = distance_transform_edt(~target.astype(bool))

        # 获取边界点（通过梯度）
        pred_boundary = (pred > 0) & (distance_transform_edt(pred > 0) == 1)
        target_boundary = (target > 0) & (distance_transform_edt(target > 0) == 1)

        # 计算平均距离
        if np.sum(pred_boundary) > 0:
            d1 = np.mean(dist_pred_to_target[pred_boundary])
        else:
            d1 = 0

        if np.sum(target_boundary) > 0:
            d2 = np.mean(dist_target_to_pred[target_boundary])
        else:
            d2 = 0

        asd = (d1 + d2) / 2.0
        return float(asd)

    @staticmethod
    def compute_sensitivity(pred: np.ndarray, target: np.ndarray, eps=1e-6) -> float:
        """
        Sensitivity (Recall, True Positive Rate)
        定义：TP / (TP + FN) = 能捕捉的目标 / 所有真实目标
        越高越好，范围0-1
        """
        tp = np.sum(pred * target)
        fn = np.sum((1 - pred) * target)
        sen = (tp + eps) / (tp + fn + eps)
        return float(sen)

    @staticmethod
    def compute_specificity(pred: np.ndarray, target: np.ndarray, eps=1e-6) -> float:
        """
        Specificity (True Negative Rate)
        定义：TN / (TN + FP) = 正确判断的负例 / 所有负例
        越高越好，范围0-1
        """
        tn = np.sum((1 - pred) * (1 - target))
        fp = np.sum(pred * (1 - target))
        spe = (tn + eps) / (tn + fp + eps)
        return float(spe)

    @staticmethod
    def compute_precision(pred: np.ndarray, target: np.ndarray, eps=1e-6) -> float:
        """
        Precision (Positive Predictive Value)
        定义：TP / (TP + FP) = 预测为正的准确度
        越高越好，范围0-1
        """
        tp = np.sum(pred * target)
        fp = np.sum(pred * (1 - target))
        prec = (tp + eps) / (tp + fp + eps)
        return float(prec)

    @staticmethod
    def compute_recall(pred: np.ndarray, target: np.ndarray, eps=1e-6) -> float:
        """
        Recall (同Sensitivity)
        定义：TP / (TP + FN) = 能捕捉的目标 / 所有真实目标
        越高越好，范围0-1
        """
        tp = np.sum(pred * target)
        fn = np.sum((1 - pred) * target)
        rec = (tp + eps) / (tp + fn + eps)
        return float(rec)


def print_metrics(metrics: Dict[str, float], title: str = "Segmentation Metrics"):
    """格式化打印评估指标"""
    print("=" * 80)
    print(f"{title:^80}")
    print("=" * 80)

    # 分类打印
    print("\n[Dice Coefficient]")
    for k, v in metrics.items():
        if 'dice' in k:
            print(f"  {k:20s}: {v:.4f}")

    print("\n[IoU / Jaccard]")
    for k, v in metrics.items():
        if 'iou' in k:
            print(f"  {k:20s}: {v:.4f}")

    if any('hd95' in k for k in metrics.keys()):
        print("\n[Hausdorff Distance 95%]")
        for k, v in metrics.items():
            if 'hd95' in k:
                print(f"  {k:20s}: {v:.4f}")

    if any('asd' in k for k in metrics.keys()):
        print("\n[Average Surface Distance]")
        for k, v in metrics.items():
            if 'asd' in k:
                print(f"  {k:20s}: {v:.4f}")

    print("\n[Sensitivity / Recall]")
    for k, v in metrics.items():
        if 'sensitivity' in k:
            print(f"  {k:20s}: {v:.4f}")

    print("\n[Specificity]")
    for k, v in metrics.items():
        if 'specificity' in k:
            print(f"  {k:20s}: {v:.4f}")

    print("\n[Precision]")
    for k, v in metrics.items():
        if 'precision' in k:
            print(f"  {k:20s}: {v:.4f}")

    print("=" * 80 + "\n")
