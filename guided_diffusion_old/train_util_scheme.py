"""
【改进版】统一的Training Utilities

支持两个方案的训练循环：
- 方案1：处理(img_patch, img_full_small, mask)三元组
- 方案2：处理(img_patch, mask)二元组

核心改进：
- 灵活的批次处理
- 自动检测数据格式
- 清晰的日志输出
- 完整的checkpoint管理
"""

import copy
import functools
import os
import time

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoopScheme:
    """
    统一的训练循环，支持两个方案

    方案1数据格式：(img_patch, img_full_small, mask, filename)
    方案2数据格式：(img_patch, mask, filename)
    """

    def __init__(
        self,
        *,
        model,
        diffusion,
        dataloader,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        scheme="scheme1",  # "scheme1" 或 "scheme2"
        warmup_steps=0,  # Warmup步数，0表示不使用warmup
        total_steps=0,   # 总训练步数，用于学习率衰减（0表示不衰减）
    ):
        self.model = model
        self.dataloader = dataloader
        self.diffusion = diffusion
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
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
        self.scheme = scheme  # 记录方案
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps if total_steps > 0 else lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        # 统计信息
        self.data_time_total = 0.0
        self.step_time_total = 0.0
        self.timing_count = 0

        logger.log("=" * 80)
        logger.log(f"TRAINING CONFIG - {self.scheme.upper()}")
        logger.log("=" * 80)
        logger.log(f"  Scheme: {self.scheme}")
        logger.log(f"  Batch size: {self.batch_size}")
        logger.log(f"  Learning rate: {self.lr}")
        logger.log(f"  Warmup steps: {self.warmup_steps}")
        if self.total_steps > 0:
            logger.log(f"  Total steps: {self.total_steps}")
        logger.log(f"  EMA rates: {self.ema_rate}")
        logger.log(f"  Weight decay: {self.weight_decay}")
        logger.log(f"  Log interval: {self.log_interval} steps")
        logger.log(f"  Save interval: {self.save_interval} steps")
        logger.log(f"  Global batch size: {self.global_batch}")
        logger.log("=" * 80)

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

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

    def _get_lr(self, step):
        """
        计算当前的学习率
        支持：Linear Warmup + Cosine Annealing衰减
        """
        total_step = step + self.resume_step

        # Warmup阶段
        if self.warmup_steps > 0 and total_step < self.warmup_steps:
            return self.lr * (total_step / self.warmup_steps)

        # Cosine衰减阶段
        if self.total_steps > 0 and total_step >= self.warmup_steps:
            progress = (total_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            # 余弦衰减：从lr衰减到lr/100
            return self.lr * (0.5 * (1.0 + th.cos(th.tensor(3.14159 * progress))))

        # 无衰减：返回原学习率
        return self.lr

    def _set_lr(self, new_lr):
        """设置优化器学习率"""
        for param_group in self.opt.param_groups:
            param_group['lr'] = new_lr

    def _load_and_sync_parameters(self):
        """加载checkpoint或初始化模型"""
        resume_checkpoint = self.resume_checkpoint

        if resume_checkpoint:
            logger.log(f"Loading model from checkpoint: {resume_checkpoint}...")
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
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
        return ema_params

    def _load_optimizer_state(self):
        """加载优化器状态"""
        main_checkpoint = self.resume_checkpoint
        if main_checkpoint and os.path.exists(main_checkpoint):
            opt_checkpoint = os.path.join(
                os.path.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
            )
            if os.path.exists(opt_checkpoint):
                logger.log(f"Loading optimizer state from: {opt_checkpoint}")
                state_dict = dist_util.load_state_dict(
                    opt_checkpoint, map_location=dist_util.dev()
                )
                self.opt.load_state_dict(state_dict)

    def run_loop(self):
        """主训练循环"""
        data_iter = iter(self.dataloader)

        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            # 获取数据
            data_start = time.time()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            data_time = time.time() - data_start
            self.data_time_total += data_time

            # 处理数据格式（方案1 vs 方案2）
            if self.scheme == "scheme1":
                # 方案1：(img_patch, img_full_small, mask, filename)
                if len(batch) == 4:
                    img_patch, img_full_small, mask, filename = batch
                    img_patch = img_patch.to(dist_util.dev())
                    img_full_small = img_full_small.to(dist_util.dev())
                    mask = mask.to(dist_util.dev())

                    # 拼接成模型输入
                    x = th.cat([img_patch, mask], dim=1)  # (B, 4, 256, 256)

                    # 时间步采样
                    t = th.randint(
                        0, self.diffusion.num_timesteps, (x.shape[0],),
                        device=dist_util.dev()
                    )

                    # 前向传播（方案1特有：传img_full）
                    compute_losses = self.diffusion.training_losses(
                        self.ddp_model,
                        x,
                        t,
                        model_kwargs={"img_full": img_full_small},
                    )
                else:
                    raise ValueError(f"Scheme1 expects 4 elements, got {len(batch)}")

            elif self.scheme == "scheme2":
                # 方案2：(img_patch, mask, filename)
                if len(batch) == 3:
                    img_patch, mask, filename = batch
                    img_patch = img_patch.to(dist_util.dev())
                    mask = mask.to(dist_util.dev())

                    # 拼接成模型输入
                    x = th.cat([img_patch, mask], dim=1)  # (B, 4, 256, 256)

                    # 时间步采样
                    t = th.randint(
                        0, self.diffusion.num_timesteps, (x.shape[0],),
                        device=dist_util.dev()
                    )

                    # 前向传播（方案2：不传img_full）
                    compute_losses = self.diffusion.training_losses(
                        self.ddp_model,
                        x,
                        t,
                    )
                else:
                    raise ValueError(f"Scheme2 expects 3 elements, got {len(batch)}")
            else:
                raise ValueError(f"Unknown scheme: {self.scheme}")

            # 计算loss
            loss = compute_losses["loss"].mean()

            # 更新学习率（如果配置了warmup或total_steps）
            if self.warmup_steps > 0 or self.total_steps > 0:
                current_lr = self._get_lr(self.step)
                self._set_lr(current_lr)

            # 反向传播
            self.opt.zero_grad()
            self.mp_trainer.backward(loss)
            self.opt.step()

            # 更新EMA
            for rate, params in zip(self.ema_rate, self.ema_params):
                update_ema(params, self.mp_trainer.master_params, rate=rate)

            # 日志记录
            if self.step % self.log_interval == 0:
                logger.logkv("step", self.step + self.resume_step)
                logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
                logger.logkv("loss", loss.item())
                logger.logkv("data_time", data_time)

                # 详细损失项（遍历所有损失）
                for key, values in compute_losses.items():
                    if key != "loss":  # 跳过总loss，已单独记录
                        if isinstance(values, th.Tensor):
                            logger.logkv(key, values.mean().item())

                # 学习率信息
                for param_group in self.opt.param_groups:
                    logger.logkv("lr", param_group["lr"])
                    break

                logger.dumpkvs()

            # 保存checkpoint
            if self.step % self.save_interval == 0 and self.step > self.resume_step:
                self.save()
                logger.log(f"[CHECKPOINT] Saved at step {self.step + self.resume_step}")

            self.step += 1

    def save(self):
        """保存模型、EMA模型和优化器状态"""
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"savedmodel{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"emasavedmodel_{rate}_{(self.step+self.resume_step):06d}.pt"
                output_path = os.path.join(logger.get_dir(), filename)
                th.save(state_dict, output_path)

        # 保存主模型
        save_checkpoint(0, self.mp_trainer.master_params)

        # 保存每个EMA模型
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # 保存优化器状态
        if dist.get_rank() == 0:
            opt_output = os.path.join(
                logger.get_dir(),
                f"opt_{self.step + self.resume_step:06d}.pt"
            )
            th.save(self.opt.state_dict(), opt_output)


def parse_resume_step_from_filename(filename):
    """从文件名中解析resume step"""
    try:
        return int(filename.split("model_")[-1].split(".")[0])
    except ValueError:
        return 0


def find_resume_checkpoint():
    """查找最新的checkpoint"""
    checkpoints = []
    for fn in os.listdir(logger.get_dir()):
        if fn.startswith("model_"):
            checkpoints.append(fn)
    if not checkpoints:
        return None
    return os.path.join(logger.get_dir(), max(checkpoints))


def args_to_dict(args, keys):
    """将argparse转换为字典"""
    return {k: getattr(args, k, None) for k in keys}
