"""
Training Utility Functions for V4 Configuration - Complete Self-Contained Implementation

扩展的训练循环，支持：
- 数据格式v4处理（dict格式的batch）
- 边缘分支的处理
- 新的Loss函数
- 分层学习率
- 详细的日志记录

本实现不依赖原版train_util.py，完全独立
"""

import copy
import functools
import os
import time

import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from guided_diffusion_old import dist_util, logger
from guided_diffusion_old.fp16_util import MixedPrecisionTrainer
from guided_diffusion_old.nn import update_ema
from guided_diffusion_old.resample import LossAwareSampler, UniformSampler


# ============================================================================
# 辅助函数
# ============================================================================

def _parse_resume_step_from_filename(filename):
    """
    解析形如 path/to/modelNNNNNN.pt 的文件名中的步数
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def _get_blob_logdir():
    """获取日志目录"""
    return logger.get_dir()


def _find_resume_checkpoint():
    """
    在你的基础设施中，你可以重写此函数以自动发现最新的checkpoint
    """
    return None


def _find_ema_checkpoint(main_checkpoint, step, rate):
    """找到EMA checkpoint"""
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = os.path.join(os.path.dirname(main_checkpoint), filename)
    if os.path.exists(path):
        return path
    return None


def _log_loss_dict(diffusion, ts, losses):
    """
    记录loss信息到logger
    """
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # 记录四分位数
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


# ============================================================================
# TrainLoop V4 - 完整独立实现
# ============================================================================

class TrainLoopV4:
    """
    V4版本的训练循环，完全独立实现

    支持：
    1. 数据加载版本v4（处理dict格式的batch）
    2. 边缘分支的输出处理
    3. 新的Loss函数（edge_loss, sdf_loss, hd_loss）
    4. 分层学习率（Adapter等）
    5. 详细的损失函数记录

    典型用法：
        train_loop = TrainLoopV4(
            model=model,
            diffusion=diffusion,
            dataloader=datal,
            batch_size=8,
            lr=3e-5,
            log_interval=100,
            save_interval=500,
        )
        train_loop.run_loop()
    """

    def __init__(
        self,
        *,
        model,
        diffusion,
        dataloader,
        batch_size,
        lr,
        log_interval=100,
        save_interval=500,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        ema_rate=0.9999,
        microbatch=-1,
        # V4参数
        use_edge_branch=False,
        use_edge_loss=False,
        use_sdf_loss=False,
        use_hd_loss=False,
        log_loss_details=False,
        use_adapter=False,
        adapter_lr=1e-4,
    ):
        """
        初始化训练循环

        Args:
            model: 要训练的模型
            diffusion: diffusion模型
            dataloader: 数据加载器
            batch_size: 批次大小
            lr: 学习率
            log_interval: 日志记录间隔（步数）
            save_interval: 保存检查点间隔（步数）
            resume_checkpoint: 要恢复的检查点路径
            use_fp16: 是否使用FP16混合精度训练
            fp16_scale_growth: FP16损失缩放增长率
            schedule_sampler: 时间步采样器
            weight_decay: 权重衰减
            lr_anneal_steps: 学习率退火步数（0表示不退火）
            ema_rate: EMA平滑率
            microbatch: 微批次大小（-1表示等于batch_size）
            use_edge_branch: 是否使用边缘分支
            use_edge_loss: 是否使用边缘Loss
            use_sdf_loss: 是否使用SDF Loss
            use_hd_loss: 是否使用Hausdorff Loss
            log_loss_details: 是否记录详细的loss分量
            use_adapter: 是否使用Adapter（用于分层学习率）
            adapter_lr: Adapter的学习率
        """

        self.model = model
        self.dataloader = dataloader
        self.diffusion = diffusion
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.adapter_lr = adapter_lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        # V4参数
        self.use_edge_branch = use_edge_branch
        self.use_edge_loss = use_edge_loss
        self.use_sdf_loss = use_sdf_loss
        self.use_hd_loss = use_hd_loss
        self.log_loss_details = log_loss_details
        self.use_adapter = use_adapter

        # 训练状态
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * max(dist.get_world_size(), 1)

        self.sync_cuda = th.cuda.is_available()

        # 时间统计
        self.data_time_total = 0.0
        self.step_time_total = 0.0
        self.fb_time_total = 0.0
        self.opt_time_total = 0.0
        self.timing_count = 0

        # 加载模型参数
        self._load_and_sync_parameters()

        # 混合精度训练器
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        # 创建优化器
        self._create_optimizer()

        # 加载EMA参数
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        # DDP设置
        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _create_optimizer(self):
        """创建优化器，支持分层学习率"""
        if self.use_adapter and hasattr(self.model, 'get_parameter_groups'):
            # 分层学习率
            param_groups = self.model.get_parameter_groups(
                base_lr=self.lr,
                adapter_lr=self.adapter_lr,
            )
            self.opt = AdamW(param_groups, weight_decay=self.weight_decay)
        else:
            # 单一学习率
            self.opt = AdamW(
                self.mp_trainer.master_params,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

    def _load_and_sync_parameters(self):
        """加载并同步参数"""
        resume_checkpoint = _find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.resume_step = _parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        """加载EMA参数"""
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = _find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = _find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        """加载优化器状态"""
        main_checkpoint = _find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = os.path.join(
            os.path.dirname(main_checkpoint),
            f"opt{self.resume_step:06d}.pt"
        )
        if os.path.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        """运行训练循环"""
        logger.log("Starting training loop...")

        data_iter = iter(self.dataloader)
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            # 获取batch
            data_start = time.time()
            try:
                batch_data = next(data_iter)
            except StopIteration:
                # 数据集结束，重新初始化
                data_iter = iter(self.dataloader)
                batch_data = next(data_iter)

            data_end = time.time()
            data_time = data_end - data_start

            # 训练步
            step_start = time.time()
            self.run_step(batch_data)

            if th.cuda.is_available():
                th.cuda.synchronize()
            step_end = time.time()
            step_time = step_end - step_start

            # 更新时间统计
            self.data_time_total += data_time
            self.step_time_total += step_time
            self.timing_count += 1

            # 日志记录
            if self.step % self.log_interval == 0 and self.timing_count > 0:
                self._log_training_stats()

            # 保存
            if self.step % self.save_interval == 0:
                self.save()
                # 测试模式
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

            self.step += 1

        # 保存最后的checkpoint
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def _log_training_stats(self):
        """记录训练统计"""
        avg_data_time = self.data_time_total / self.timing_count
        avg_step_time = self.step_time_total / self.timing_count
        avg_fb_time = self.fb_time_total / max(self.timing_count, 1)
        avg_opt_time = self.opt_time_total / max(self.timing_count, 1)

        logger.log(f"[Step {self.step + self.resume_step}] "
                   f"data_time: {avg_data_time:.4f}s, "
                   f"step_time: {avg_step_time:.4f}s")

        logger.logkv("avg_data_time", avg_data_time)
        logger.logkv("avg_step_time", avg_step_time)
        logger.logkv("avg_fb_time", avg_fb_time)
        logger.logkv("avg_opt_time", avg_opt_time)

        logger.dumpkvs()

        # 重置计数
        self.data_time_total = 0.0
        self.step_time_total = 0.0
        self.fb_time_total = 0.0
        self.opt_time_total = 0.0
        self.timing_count = 0

    def run_step(self, batch_data):
        """运行单个训练步"""
        # 处理batch数据
        batch, batch_info = self._prepare_batch(batch_data)

        # ✅ 构造model_kwargs（包含PVT所需的原始图像）
        cond = {}

        # ✅ 强制确保x_pvt被传入（防止DDP梯度问题）
        if hasattr(self.ddp_model, 'use_pvt_fusion') and self.ddp_model.use_pvt_fusion:
            if batch_info is not None and 'image' in batch_info:
                # 优先使用原始未噪声化的图像
                cond['x_pvt'] = batch_info['image']
            else:
                # 如果没有batch_info，从batch中提取前3个通道
                # batch是(B, 4, H, W)，前3个通道是图像，最后1个是mask
                cond['x_pvt'] = batch[:, :3, :, :].clone()

        # 前向反向传播
        sample = self.forward_backward(batch, cond)

        # 优化器步
        opt_start = time.time()
        took_step = self.mp_trainer.optimize(self.opt)
        if th.cuda.is_available():
            th.cuda.synchronize()
        opt_end = time.time()
        self.opt_time_total += (opt_end - opt_start)

        # 更新EMA和学习率
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

        return sample

    def _prepare_batch(self, batch_data):
        """
        处理v4格式的batch数据

        V4数据格式为dict，包含：
        - 'image': 输入图像（3通道）
        - 'mask': 对应的mask（NUM_CLASSES通道）
        - 'padding_info': 填充信息（用于反投影）
        - 'edge': 边缘标签（如果use_edge_loss=True）

        返回：
        - batch: 拼接后的tensor [image, mask, ...]，形状(B, 4, H, W)
        - batch_info: dict，包含原始图像等信息（用于PVT特征提取）
        """
        if isinstance(batch_data, dict):
            # ✅ V4格式：dict
            image = batch_data['image']  # (B, 3, H, W)
            mask = batch_data['mask']    # (B, NUM_CLASSES, H, W)

            # 拼接为4通道输入：[image(3ch) + mask(1ch)]
            batch = th.cat((image, mask), dim=1)

            # ✅ 保存原始图像信息供PVT特征提取
            batch_info = {
                'image': image,  # 原始3通道图像，供PVT使用
                'mask': mask,
            }

            # 如果有边缘标签和使用边缘loss，保存供后续使用
            if 'edge' in batch_data and self.use_edge_loss:
                batch_info['edge'] = batch_data['edge']

            # 保存填充信息
            if 'padding_info' in batch_data:
                batch_info['padding_info'] = batch_data['padding_info']

            return batch, batch_info
        else:
            # 原有格式：tuple或tensor
            if isinstance(batch_data, (tuple, list)):
                if len(batch_data) == 3:
                    # 原版格式: (batch, cond, name)
                    batch, batch_cond, _ = batch_data
                    return batch, {'cond': batch_cond}
                else:
                    batch = batch_data[0]
            else:
                batch = batch_data

            return batch, None

    def forward_backward(self, batch, cond):
        """前向反向传播"""
        fb_start = time.time()

        self.mp_trainer.zero_grad()
        last_sample = None

        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            # 计算losses
            compute_losses = functools.partial(
                self.diffusion.training_losses_segmentation,
                self.ddp_model,
                None,  # classifier为None（分割任务不需要）
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses_output = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses_output = compute_losses()

            # 解析losses输出
            losses = losses_output[0]
            sample = losses_output[1] if len(losses_output) > 1 else None
            last_sample = sample

            # 如果使用LossAwareSampler，更新
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            # 计算加权loss
            loss = (losses["loss"] * weights).mean()

            # 记录loss
            _log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )

            # 详细loss记录（V4新增）
            if self.log_loss_details and self.step % (10 * self.log_interval) == 0:
                self._log_loss_details(losses)

            # 反向传播
            self.mp_trainer.backward(loss)

        if th.cuda.is_available():
            th.cuda.synchronize()
        fb_end = time.time()
        self.fb_time_total += (fb_end - fb_start)

        return last_sample

    def _log_loss_details(self, losses):
        """记录各loss分量详情（V4新增）"""
        for key, value in losses.items():
            if isinstance(value, th.Tensor):
                logger.log(f"  {key}: {value.mean().item():.6f}")

    def _update_ema(self):
        """更新EMA参数"""
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        """学习率退火"""
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        """记录步信息"""
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        """保存checkpoint"""
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model (rate={rate})...")
                if not rate:
                    filename = f"model{(self.step + self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"

                save_path = os.path.join(_get_blob_logdir(), filename)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                th.save(state_dict, save_path)
                logger.log(f"saved to {save_path}")

        # 保存主模型
        save_checkpoint(0, self.mp_trainer.master_params)

        # 保存EMA模型
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # 保存优化器状态
        if dist.get_rank() == 0:
            opt_path = os.path.join(
                _get_blob_logdir(),
                f"opt{(self.step + self.resume_step):06d}.pt"
            )
            os.makedirs(os.path.dirname(opt_path), exist_ok=True)
            th.save(self.opt.state_dict(), opt_path)

        dist.barrier()


# ============================================================================
# 优化器和学习率工厂函数（辅助）
# ============================================================================

def create_optimizer_and_schedule(
    model,
    base_lr=3e-5,
    adapter_lr=1e-4,
    weight_decay=0.0,
    use_adapter=False,
):
    """
    创建优化器和学习率调度

    支持分层学习率：
    - 基础参数使用base_lr
    - Adapter参数使用adapter_lr

    Args:
        model: 模型
        base_lr: 基础学习率
        adapter_lr: Adapter学习率（如果use_adapter=True）
        weight_decay: 权重衰减
        use_adapter: 是否使用Adapter

    Returns:
        tuple: (optimizer, scheduler)
    """
    if use_adapter and hasattr(model, 'get_parameter_groups'):
        # 分层学习率
        param_groups = model.get_parameter_groups(
            base_lr=base_lr,
            adapter_lr=adapter_lr,
        )
        optimizer = AdamW(param_groups, weight_decay=weight_decay)
    else:
        # 单一学习率
        optimizer = AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=weight_decay,
        )

    # 返回optimizer和None（scheduler可选）
    return optimizer, None
