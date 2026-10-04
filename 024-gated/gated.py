# -*- coding: utf-8 -*-
"""
【医学图像分割模块】T —— T-Gated Adapter: A Lightweight Temporal Adapter for Vision-L

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn


class TemporalGatedAdapter(nn.Module):
    """
    T-Gated Adapter：轻量时序门控适配块。
    输入: x (B, T, C, H, W)，T 为邻域帧数，中间帧为当前帧。
    输出: (B, C, H, W)，当前帧被邻帧门控修正后的特征。
    """

    def __init__(self, channels, reduction=4, neighbor=True):
        super().__init__()
        self.channels = channels
        self.neighbor = neighbor  # 是否使用邻帧；False 时退化为恒等

        # 邻帧增量投影：把邻帧特征压成与当前帧同维的增量 Δ
        # 用 1x1 卷积，逐像素、跨通道混合，参数量极小
        self.delta_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # 门控网络：先全局池化拿到通道描述，再经瓶颈 MLP 输出逐通道门控
        hidden = max(channels // reduction, 8)  # 瓶颈维度，防止通道太小时塌成 0
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # (B, C, 1, 1) 全局上下文
            nn.Conv2d(channels, hidden, 1),   # 降维
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),   # 升回通道数
            nn.Sigmoid(),                     # 门控值压到 0~1
        )

        # 零初始化增量投影，使模块初始近似恒等映射，插入预训练网络不破坏特征
        nn.init.zeros_(self.delta_proj.weight)

    def forward(self, x):
        # x: (B, T, C, H, W)
        if not self.neighbor or x.dim() != 5 or x.size(1) < 2:
            # 没有邻帧可用时，直接返回中间帧（或原张量），保证接口安全
            if x.dim() == 5:
                return x[:, x.size(1) // 2]
            return x

        t_mid = x.size(1) // 2          # 当前帧在时间维的索引
        cur = x[:, t_mid]               # (B, C, H, W) 当前帧特征

        # 邻帧取平均，作为上下文先验；避免对帧序敏感
        neigh = torch.cat([x[:, :t_mid], x[:, t_mid + 1:]], dim=1)  # (B, T-1, C, H, W)
        neigh = neigh.mean(dim=1)       # (B, C, H, W)

        delta = self.delta_proj(neigh)  # (B, C, H, W) 邻帧增量
        g = self.gate(cur)              # (B, C, 1, 1) 逐通道门控

        # 残差门控修正：门控趋 0 时输出≈当前帧
        return cur + g * delta
