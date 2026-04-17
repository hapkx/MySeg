"""
多类别
Refined version:
    - segmentation supervision on pred_xstart is stabilized
    - segmentation time weight no longer collapses to ~0 at high t
    - optional logit clamp for CE/Dice branch
    加入 presence 和 focal loss，增强对小目标的监督
"""
"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py
Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
from torch.autograd import Variable
import enum
import torch.nn.functional as F
from torchvision.utils import save_image
import torch
import math
# from visdom import Visdom
# viz = Visdom(port=8850)
import numpy as np
import torch as th
from .train_util import visualize
from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from scipy import ndimage
from torchvision import transforms

from timm.models.pvt_v2 import checkpoint_filter_fn

NUM_CLASSES=3

import torch as th
import torch.nn.functional as F


def multiclass_focal_loss(logits, target_onehot, alpha=None, gamma=2.0, eps=1e-6):
    """
    logits: [B,C,H,W]
    target_onehot: [B,C,H,W]
    alpha: list/tuple, e.g. [0.2, 1.8, 1.0]
    """
    probs = th.softmax(logits, dim=1).clamp(min=eps, max=1.0 - eps)
    ce = -target_onehot * th.log(probs)              # [B,C,H,W]
    mod = (1.0 - probs) ** gamma
    loss = mod * ce

    if alpha is not None:
        alpha_t = th.tensor(alpha, device=logits.device, dtype=logits.dtype).view(1, -1, 1, 1)
        loss = loss * alpha_t

    # 先对类别求和，再对空间求均值，返回每个样本一个值
    loss = loss.sum(dim=1).mean(dim=(1, 2))               # [B]
    return loss


def presence_loss(logits, target_onehot, fg_ids=(1,), target_prob=0.35, eps=1e-6):
    # Penalize model for ignoring a fg class that exists in GT.
    # Uses mean predicted prob at GT fg pixels (not whole image).
    # Perfect prediction -> pred_at_fg ~1.0 -> loss ~0.
    probs = th.softmax(logits, dim=1)
    total_loss = 0.0
    count = 0

    for cid in fg_ids:
        gt_mask = target_onehot[:, cid]                                        # [B, H, W]
        gt_exist = (gt_mask.sum(dim=(1, 2)) > 0).float()                      # [B]
        # 只在 GT 前景像素位置计算平均预测概率
        pred_at_fg = (probs[:, cid] * gt_mask).sum(dim=(1, 2)) / (gt_mask.sum(dim=(1, 2)) + eps)  # [B]
        # 只有低于 target_prob 才处罚，避免过度推高 class_1
        cls_loss = gt_exist * F.relu(target_prob - pred_at_fg)
        total_loss += cls_loss.mean()
        count += 1

    return total_loss / max(count, 1)

# modify
def multiclass_dice_loss(pred_probs, target_onehot, eps=1e-6):
    """
    pred_probs: [B, C, H, W], softmax后的概率
    target_onehot: [B, C, H, W], one-hot标签
    """
    assert pred_probs.shape == target_onehot.shape

    # 只计算前景类1,2
    pred_probs = pred_probs[:, 1:, ...]  # [B, C-1, H, W]
    target_onehot = target_onehot[:, 1:, ...]  # [B, C-1, H, W]

    dims = (2, 3)  # 按 batch 和空间维度求和，保留通道维
    intersection = (pred_probs * target_onehot).sum(dim=dims)
    union = pred_probs.sum(dim=dims) + target_onehot.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)  # [B, 2]
    # class_1 (small target) gets 2x weight vs class_2 (large target)
    # equal weighting lets class_1 collapse unpenalized since its pixel count is tiny
    weights = pred_probs.new_tensor([1.2, 1.0])  # [class_1, class_2]
    dice_weighted = (dice * weights).sum(dim=1) / weights.sum()
    return 1.0 - dice_weighted  # [B]


def boundary_smooth_loss(pred_probs, target_onehot, weight=0.05, eps=1e-6):
    """
    边界平滑和清晰化损失
    鼓励在GT边界处的预测边界清晰且平滑

    Args:
        pred_probs: [B, C, H, W] 预测概率（softmax后）
        target_onehot: [B, C, H, W] 目标one-hot编码
        weight: 损失权重系数
        eps: 数值稳定性

    Returns:
        标量损失值 [B]
    """
    B, C, H, W = pred_probs.shape

    # ===== 计算预测的边界（梯度） =====
    # 使用Sobel算子计算梯度
    # X方向梯度
    pred_grad_x = th.abs(pred_probs[:, :, :, :-1] - pred_probs[:, :, :, 1:])  # [B, C, H, W-1]
    # Y方向梯度
    pred_grad_y = th.abs(pred_probs[:, :, :-1, :] - pred_probs[:, :, 1:, :])  # [B, C, H-1, W]

    # 只关注前景类（class_1和class_2）
    pred_grad_x = pred_grad_x[:, 1:, :, :]  # [B, 2, H, W-1]
    pred_grad_y = pred_grad_y[:, 1:, :, :]  # [B, 2, H-1, W]

    # ===== 计算GT的边界 =====
    target_grad_x = th.abs(target_onehot[:, 1:, :, :-1] - target_onehot[:, 1:, :, 1:])  # [B, 2, H, W-1]
    target_grad_y = th.abs(target_onehot[:, 1:, :-1, :] - target_onehot[:, 1:, 1:, :])  # [B, 2, H-1, W]

    # 二值化：边界为1，非边界为0
    target_boundary_x = (target_grad_x > 0.1).float()
    target_boundary_y = (target_grad_y > 0.1).float()

    # ===== 边界一致性损失 =====
    # 在GT有边界的地方，鼓励预测也有边界（梯度大）
    # 在GT没有边界的地方，鼓励预测边界小（平滑）

    # X方向
    boundary_loss_x = th.abs(pred_grad_x - target_grad_x) * target_boundary_x
    boundary_loss_x = boundary_loss_x.mean()

    # Y方向
    boundary_loss_y = th.abs(pred_grad_y - target_grad_y) * target_boundary_y
    boundary_loss_y = boundary_loss_y.mean()

    # ===== 平滑性损失 =====
    # 惩罚在非边界区域的高频波动（锯齿）
    smoothness_x = pred_grad_x * (1.0 - target_boundary_x)
    smoothness_y = pred_grad_y * (1.0 - target_boundary_y)

    smoothness_loss = smoothness_x.mean() + smoothness_y.mean()

    # 总损失
    total_loss = boundary_loss_x + boundary_loss_y + 0.5 * smoothness_loss

    return weight * total_loss

# 对输入图像进行标准化处理，使其均值为 0，标准差为 1。
def standardize(img):
    mean = th.mean(img)
    std = th.std(img)
    img = (img - mean) / std
    return img

# 根据给定的调度名称返回相应的 β 值调度。
def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

# 创建一个 β 值调度，离散化给定的 alpha_t_bar 函数。该函数定义了从t =[0,1]开始的（1-beta）随时间的累积积。
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.
    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)   # α的累乘
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])     # 前一个时间步的 α 值的累积乘积，首个值设为 1.0
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)      # 后一个时间步的 α 值的累积乘积，最后一个值设为 0.0
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)     # 累积的 α 值的平方根
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)     # ( \sqrt{1 - \alpha_{t}} )，即当前时间步的 1 减去累积 α 值的平方根。
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)   # ( \sqrt{1 - \alpha_{t}} )，即当前时间步的 1 减去累积 α 值的平方根。
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)     # 1/(a_t) 的平方根
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)     # 1/(a_t)-1 的平方根


        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])  # 保持t=0时方差为0
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        xt~N(sqrt_alphas_cumprod * x0, 1-alphas_cumprod)
        这个函数用来求xt的方差和期望
        计算前向过程中q(x_t | x_0)的均值、方差和对数方差。
        返回一个元组 (mean, variance, log_variance)，分别表示均值、方差和对数方差。
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        对给定数量的扩散步骤进行数据扩散。
        In other words, sample from q(x_t | x_0).
        从q(x_t | x_0)中采样，即对初始数据x_0添加t步噪声，生成x_t。
        从方差和期望求xt
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        计算扩散后验分布q(x_{t-1} | x_t, x_0)的均值和方差
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        应用模型计算 p(x_{t-1} | x_t) 和预测初始 x_0
        利用模型预测反向过程中p(x_{t-1} | x_t)的均值、方差和x_0的估计值。
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x.shape[:2]
        C = NUM_CLASSES
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        x=x[:,-NUM_CLASSES:,...]  # 只保留最后NUM_CLASSES个通道作为输入  
        
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)
            
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }



    def _predict_xstart_from_eps(self, x_t, t, eps):
        # 从扩散后的图像和噪声预测 x_0
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        # 从前一张图像 x_prev 预测 x_0
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        # 从 x_0 预测噪声。
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        # 将时间步缩放到（0 到 1000）范围。
        if self.rescale_timesteps:

            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample(
        self,
        model,
        x, 
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
    ):
        """
        从模型中生成 x_{t-1} 样本。
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model, x, t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x[:, -NUM_CLASSES:,...])
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop_known(
        self,
        model,
        shape,
        img,
        org=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        conditioner = None,
        classifier=None
    ):
        """
        从模型生成已知信息的样本。
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        img=img.to(device)
        original_image = img[:, :-NUM_CLASSES, ...]  # [B, C_img, H, W]，不参与去噪
        if noise is None:
            noise = th.randn_like(img[:, -NUM_CLASSES:, ...]).to(device)  # [B, NUM_CLASSES, H, W]
        x_noisy = th.cat([original_image, noise], dim=1)
        
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=x_noisy,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            org=org,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        
        return final["sample"], final["pred_xstart"], x_noisy, img


    def p_sample_loop_progressive(
        self,
        model,
        shape,
        time=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        org=None,
        model_kwargs=None,
        device=None,
        progress=False,
        ):
        """
        从模型生成样本并逐步返回每个时间步的样本。
        从模型中生成样本，并从扩散的每个时间步长产生中间样本。
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """

        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if time is None:
            time = self.num_timesteps

        indices = list(range(time))[::-1]
        org_img = img[:, :-NUM_CLASSES, ...]
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            # if i%100==0:
            #     print('sampling step', i)
                # viz.image(visualize(img.cpu()[0, -NUM_CLASSES,...]), opts=dict(caption="sample"+ str(i) ))

            with th.no_grad():
                if img.shape[1] != shape[1]:
                    img = th.cat((org_img,img[:, -NUM_CLASSES:, ...]), dim=1) 

                out = self.p_sample(
                    model,
                    img.float(),
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        标准的训练损失计算方法。
        调用 training_losses_segmentation（传入 classifier=None）
        返回：损失字典（仅terms，不包含model_output）
        """
        terms, model_output = self.training_losses_segmentation(
            model=model,
            classifier=None,
            x_start=x_start,
            t=t,
            model_kwargs=model_kwargs,
            noise=noise
        )
        return terms

    def training_losses_segmentation(self, model, classifier, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        Refinements compared with the original version:
        1) segmentation supervision is applied on a clamped pred_xstart branch,
           which makes CE/Dice less sensitive to early noisy predictions;
        2) segmentation time weighting keeps a non-zero floor so the auxiliary
           segmentation objective does not disappear at large timesteps.
        """
        if model_kwargs is None:
            model_kwargs = {}

        if noise is None:
            noise = th.randn_like(x_start[:, -NUM_CLASSES:, ...])

        mask = x_start[:, -NUM_CLASSES:, ...].clone()   # [B, 3, H, W]
        res_t = self.q_sample(mask, t, noise=noise)

        x_t = x_start.float().clone()
        x_t[:, -NUM_CLASSES:, ...] = res_t.float()

        terms = {}

        if self.loss_type not in [LossType.MSE, LossType.RESCALED_MSE]:
            raise NotImplementedError(
                f"training_losses_segmentation only supports MSE / RESCALED_MSE, but got {self.loss_type}"
            )

        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            B, C = x_t.shape[:2]
            assert model_output.shape == (B, NUM_CLASSES * 2, *x_t.shape[2:])
            model_output, model_var_values = th.split(model_output, NUM_CLASSES, dim=1)

            frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
            terms["vb"] = self._vb_terms_bpd(
                model=lambda *args, r=frozen_out: r,
                x_start=mask,
                x_t=res_t,
                t=t,
                clip_denoised=False,
            )["output"]

            if self.loss_type == LossType.RESCALED_MSE:
                terms["vb"] *= self.num_timesteps / 1000.0

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=mask, x_t=res_t, t=t
            )[0],
            ModelMeanType.START_X: mask,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]

        # diffusion loss
        terms["mse"] = mean_flat((target - model_output) ** 2)

        # reconstruct pred_xstart for segmentation supervision
        if self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(
                x_t=res_t, t=t, eps=model_output
            )
        elif self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        else:
            pred_xstart = self._predict_xstart_from_xprev(
                x_t=res_t, t=t, xprev=model_output
            )

        gt_classes = th.argmax(mask, dim=1)  # [B, H, W]

        # Stabilize the auxiliary segmentation branch. This branch is only used
        # for CE/Dice and does not change the diffusion target itself.
        pred_logits_for_seg = pred_xstart.clamp(min=-4.0, max=4.0)
        pred_probs = th.softmax(pred_logits_for_seg, dim=1)

        # 改进：更平衡的class权重 (减少class_1的过度强调)
        # 原: [0.3, 1.3, 1.0] 导致class_1过度强调，class_2collapse
        # 新: [0.3, 1.1, 1.0] 更温和的平衡
        class_weights = pred_xstart.new_tensor([0.3, 1.1, 1.0])

        seg_ce = F.cross_entropy(
            pred_logits_for_seg,
            gt_classes,
            weight=class_weights,
            reduction="none",
        ).mean(dim=(1, 2))
        seg_dice = multiclass_dice_loss(pred_probs, mask)

        # -------------------------
        # Focal
        # multiclass_focal_loss 当前返回标量
        # 为兼容现有 terms 结构，这里扩成 [B]
        # 改进：降低class_1的focal权重以平衡两个类
        # 原: [0.2, 1.5, 1.0] 导致class_1过度强调
        # 新: [0.2, 1.2, 1.0] 更温和的平衡
        # -------------------------
        seg_focal = multiclass_focal_loss(
            pred_logits_for_seg,
            mask,
            alpha=[0.2, 1.2, 1.0],
            gamma=2.0,
        )

        # -------------------------
        # Boundary smoothness loss
        # 边界平滑和清晰化损失，减少分割边界的锯齿和马赛克现象
        # 鼓励在GT边界处的预测边界清晰且平滑
        # -------------------------
        seg_boundary = boundary_smooth_loss(
            pred_probs,
            mask,
            weight=0.05  # 边界损失的权重系数
        )

        terms["seg_ce"] = seg_ce
        terms["seg_dice"] = seg_dice
        terms["seg_focal"] = seg_focal
        terms["seg_boundary"] = seg_boundary

        # 最终简化方案：保留Focal，删除Presence
        #
        # 4个损失函数的作用：
        # 1. seg_ce      - 基础分类准确率
        # 2. seg_dice    - 补充CE，强调完整性和IoU
        # 3. seg_focal   - Hard example mining，同时处理：
        #                  • 错误预测出不存在的类（false positive/hallucination）
        #                  • 遗漏存在的类（false negative/collapse）
        # 4. seg_boundary- 边界清晰和平滑（减少锯齿）
        #
        # 删除Presence的原因：
        # Focal Loss通过调制因子(1-p_t)^γ可以同时处理false+和false-问题，
        # 而Presence只处理false negative中的完全collapse问题。
        # Focal的功能更全面，保留它能更好地处理你观察到的两种情况。

        lambda_seg = 1.0
        seg_time_weight = 0.1 + 0.9 * (1.0 - t.float() / self.num_timesteps)
        seg_time_weight = seg_time_weight.clamp(min=0.05)

        seg_loss = seg_time_weight * (
            seg_ce              # 主损失：分类准确率
            + seg_dice          # 补充：完整性和IoU
            + 0.3 * seg_focal   # Hard examples：collapse和hallucination
            + seg_boundary      # 边界：清晰和平滑（已含权重0.05）
        )


        if "vb" in terms:
            terms["loss"] = terms["mse"] + terms["vb"] + lambda_seg * seg_loss
        else:
            terms["loss"] = terms["mse"] + lambda_seg * seg_loss

        return terms, model_output
        
    
# 从一个一维 NumPy 数组中提取特定的值，并将其转换为一个具有指定形状的张量
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    # # 调试打印
    # print(f"arr shape: {arr.shape}, timesteps: {timesteps}, timesteps device: {timesteps.device}")
    
    # 检查 timesteps 是否在有效范围内
    if not ((timesteps >= 0).all() and (timesteps < len(arr)).all()):
        raise ValueError(f"Invalid timesteps: {timesteps}, arr length: {len(arr)}")
    
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()

    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)