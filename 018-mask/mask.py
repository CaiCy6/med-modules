# -*- coding: utf-8 -*-
"""
【医学图像分割模块】Mask to Concept —— Mask to Concept: Auto-Promptable SAM3 via Efficient Test-Tim

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskToConcept(nn.Module):
    """
    Mask to Concept：把参考掩码压成概念嵌入。
    输入:
        feat:  [B, C, H, W]  图像/特征图（来自主干，如 U-Net bottleneck）
        mask:  [B, 1, H, W]  参考掩码，值域 {0,1}，与 feat 同空间尺寸
    输出:
        concept: [B, D]      概念嵌入，可直接作为提示向量使用
    """

    def __init__(self, in_channels, concept_dim=256, hidden_dim=512, num_refs=1):
        super().__init__()
        self.concept_dim = concept_dim
        self.num_refs = num_refs  # 支持多张参考掩码（少样本）

        # 1) 把主干特征投影到聚合空间，降维减少计算
        self.feat_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        # 2) 聚合后的向量 -> 概念嵌入的映射（MLP）
        #    输入维度是 hidden_dim * 2（均值池化 + 最大池化拼接）
        self.to_concept = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, concept_dim),
        )

        # 3) 可学习温度，用于后续与概念空间对齐时的相似度缩放
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def _masked_pool(self, feat, mask):
        """对 feat 在 mask 区域内做均值池化和最大池化。"""
        B, C, H, W = feat.shape
        # 把 mask 下采样/上采样到 feat 的空间尺寸，保证对齐
        if mask.shape[-2:] != (H, W):
            mask = F.interpolate(mask, size=(H, W), mode="nearest")
        mask = (mask > 0.5).float()  # 二值化，避免软掩码带来的歧义

        # 展平空间维，方便做掩码池化
        feat_flat = feat.view(B, C, -1)          # [B, C, H*W]
        mask_flat = mask.view(B, 1, -1)          # [B, 1, H*W]

        # 每个样本的有效像素数，clamp 防止除零
        denom = mask_flat.sum(dim=-1).clamp(min=1.0)  # [B, 1]

        # 均值池化：掩码内特征求和 / 有效像素数
        mean_pool = (feat_flat * mask_flat).sum(dim=-1) / denom  # [B, C]

        # 最大池化：把掩码外位置置为极小值再取 max
        neg_inf = torch.finfo(feat.dtype).min
        masked_for_max = feat_flat.masked_fill(mask_flat <= 0, neg_inf)
        max_pool = masked_for_max.max(dim=-1).values  # [B, C]

        return mean_pool, max_pool

    def forward(self, feat, mask):
        # 投影特征到聚合空间
        feat = self.feat_proj(feat)  # [B, hidden_dim, H, W]

        # 掩码引导的池化，得到两个全局描述子
        mean_pool, max_pool = self._masked_pool(feat, mask)  # 各 [B, hidden_dim]

        # 拼接均值与最大池化，信息互补
        pooled = torch.cat([mean_pool, max_pool], dim=-1)  # [B, hidden_dim*2]

        # 映射到概念嵌入空间
        concept = self.to_concept(pooled)  # [B, concept_dim]

        # L2 归一化，方便后续做余弦相似度对齐
        concept = F.normalize(concept, dim=-1)
        return concept
