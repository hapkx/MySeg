"""
【方案1专用】UNet模型 - 灵活支持img_full参数

基于unet_gate_improved.py，扩展forward方法以支持img_full参数：
- 【训练时】接受img_full（缩小的完整图）给PVT，img_patch给UNet
- 【推理时】如果不传img_full，自动从x的image部分提取
- 两个模式完全兼容，无需修改代码

================================================================================

【使用场景1】训练时 - 方案1（有img_full）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

数据流：
  原图(1186×852)
    ├─ 缩小到256×256 → img_full_small  (B, 1, 256, 256)
    ├─ 裁patch到256×256 → img_patch    (B, 1, 256, 256)
    └─ mask                            (B, 3, 256, 256)

模型调用：
  x = torch.cat([img_patch, mask], dim=1)      # (B, 4, 256, 256)
  output = model(x, timesteps, img_full=img_full_small)

效果：
  ✅ PVT看到全局信息（缩小的完整图）
  ✅ UNet看到局部细节（patch）
  ✅ 在256×256尺度完全对齐，融合效果最优

================================================================================

【使用场景2】推理时 - 直接兼容（无需img_full）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

数据流：
  测试图像(任意尺寸)
    ↓ MyDataset缩放到256×256
    ├─ img: (B, 1, 256, 256)
    └─ mask: (B, 3, 256, 256)

模型调用（推理脚本自动处理）：
  x = torch.cat([img, mask], dim=1)  # (B, 4, 256, 256)
  output = model(x, timesteps)       # ← 无需传img_full

自动处理：
  model内部检测img_full=None，自动从x[:,:1]提取image部分

效果：
  ✅ 兼容现有推理脚本（无需修改sample.py）
  ✅ 自动降级到方案2/本地模式
  ✅ 推理流程完全透明

================================================================================

【兼容性矩阵】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  调用方式                  是否需要img_full?  兼容的场景
  ─────────────────────────────────────────────────────
  model(x, t)              NO（自动提取）    推理、方案2训练
  model(x, t, img_full=v)  YES（显式传入）   方案1训练

================================================================================
"""

# This file is a wrapper/adapter that imports from unet_gate_improved
# and modifies the forward method to support scheme1

from guided_diffusion_old.unet_gate_improved import *
import torch as th
import torch.nn.functional as F


class UNetModelScheme1(UNetModel):
    """
    增强型UNet模型：灵活支持img_full参数

    特点：
    ✅ 继承自UNetModel，完全兼容所有父类功能
    ✅ forward方法支持img_full参数（可选）
    ✅ 训练和推理使用同一个模型，无需修改
    ✅ 自动处理img_full的默认情况

    forward方法签名：
      forward(self, x, timesteps, img_full=None, y=None)

    参数说明：
      x: 模型主输入 (B, 4, 256, 256)
         = (B, 1, 256, 256) image + (B, 3, 256, 256) mask

      timesteps: 扩散时间步索引 (B,)

      img_full: 【可选】PVT的独立输入 (B, 1, 256, 256)
                ├─ 训练时（方案1）：传入缩小的完整图
                ├─ 推理或方案2：不传（默认从x提取）
                └─ 默认值：None

      y: 【可选】类标签，仅当model初始化时num_classes!=None时需要

    使用示例：
      # 方案1训练
      output = model(x_train, timesteps, img_full=img_full_small)

      # 推理或方案2训练（自动处理）
      output = model(x_test, timesteps)
    """

    def forward(self, x, timesteps, img_full=None, y=None):
        """
        改进的forward方法，支持img_full输入

        ===== 使用场景 =====

        【训练时（方案1）】：
        - x: 包含patch和mask的拼接 (B, 4, 256, 256) = (image + mask)
        - img_full: 缩小到256×256的完整图 (B, 1, 256, 256)
        - 示例调用: model(x, timesteps, img_full=img_full_small)

        【推理时】：
        - x: 包含图像和mask的拼接 (B, 4, 256, 256)
        - img_full: 可选，如果不传则从x中自动提取image部分
        - 示例调用: model(x, timesteps)  # 自动从x[:,:1]提取

        Args:
            x: 模型输入 (B, 4, 256, 256) = image(1ch) + mask(3ch)
            timesteps: 时间步索引
            img_full: 可选，缩小的完整图 (B, 1, 256, 256)
                     如果为None，则从x的image部分自动提取
            y: 类标签（如果模型是class-conditional）
        """
        assert (y is not None) == (self.num_classes is not None), \
            "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        # ===== PVT特征提取（关键改进）=====
        pvt_feats = None
        if self.pvt_extractor is not None:
            # 【改进点】智能选择PVT的输入
            if img_full is not None:
                # 【方案1】训练时：使用传入的img_full（缩小的完整图）
                # 这样PVT能看到全局信息，而UNet处理局部patch
                x_pvt = img_full
            else:
                # 【方案2/推理时降级】从x的image部分提取
                # x的格式：(B, 1, 256, 256) image + (B, 3, 256, 256) mask
                img_ch = self.in_channels - 3  # in_channels通常是4 = 1+3
                x_pvt = x[:, :img_ch, :, :]  # 提取image部分 (B, 1, 256, 256)

            # 转换为3通道输入给PVT（PVT期望3通道RGB输入）
            if x_pvt.shape[1] == 1:
                # 灰度图 → 复制到3通道
                x_pvt = x_pvt.repeat(1, 3, 1, 1)  # (B, 1, H, W) → (B, 3, H, W)
            elif x_pvt.shape[1] == 3:
                # 已经是3通道，保持不变
                pass
            else:
                raise ValueError(f"Unsupported image channels for PVT: {x_pvt.shape[1]}")

            # 【归一化】从[-1, 1]转到[0, 1]
            x_pvt = (x_pvt + 1.0) / 2.0

            # 【ImageNet统计量归一化】使用灰度等价的ImageNet统计
            # 参考：DDPM论文中对灰度图的处理
            mean = th.tensor([0.449, 0.449, 0.449], device=x.device, dtype=x_pvt.dtype).view(1, 3, 1, 1)
            std = th.tensor([0.226, 0.226, 0.226], device=x.device, dtype=x_pvt.dtype).view(1, 3, 1, 1)
            x_pvt = (x_pvt - mean) / std

            # 【PVT前向传播】
            if not any(p.requires_grad for p in self.pvt_extractor.parameters()):
                # PVT冻结时，不计算梯度
                with th.no_grad():
                    pvt_feats = self.pvt_extractor(x_pvt)
            else:
                # PVT微调时，计算梯度
                pvt_feats = self.pvt_extractor(x_pvt)

            # 【医学图像适配】从ImageNet特征适配到医学图像域
            if self.use_pvt_fusion and hasattr(self, 'medical_adapter') and self.medical_adapter is not None:
                pvt_feats = self.medical_adapter(pvt_feats)

        # ===== UNet编码器 - 与PVT特征融合 =====
        h = x.type(self.dtype)
        for block_id, module in enumerate(self.input_blocks):
            h = module(h, emb)

            # 【关键】PVT特征与UNet特征在此处进行融合
            # 两个方案的区别在这里完全消失：
            # - 方案1：PVT看到了缩小的完整图（全局信息）
            # - 方案2/推理：PVT看到的也是patch的image部分（局部信息）
            # 但都在256×256的尺度上对齐，所以融合逻辑相同
            if pvt_feats is not None and block_id in self.fusion_block_ids:
                for fusion_idx in self.fusion_block_ids[block_id]:
                    h = self.fusion_modules[fusion_idx](pvt_feats[fusion_idx], h)

            hs.append(h)

        # ===== 中间层 - 可选的交叉注意力 =====
        if self.use_cross_attn:
            h = self.middle_block1(h, emb)
            # 交叉注意力：用PVT的最深层特征F3作为全局上下文
            h = self.middle_cross_attn(h, pvt_feats[3])  # F3: 16×16, 256ch
            h = self.middle_block2(h, emb)
        else:
            # 标准UNet中间块
            h = self.middle_block(h, emb)

        # ===== UNet解码器 =====
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        h = h.type(x.dtype)
        return self.out(h)
