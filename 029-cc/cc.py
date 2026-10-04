# -*- coding: utf-8 -*-
"""
【医学图像分割模块】CC —— CC-SAM: SAM with Cross-feature Attention and Context for Ult

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn


class CC(nn.Module):
    """
    CC: Context / Cross-feature Context 模块（ECCV 2024, CC-SAM）
    即插即用的通道上下文注意力，输入输出形状完全一致。
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        # 瓶颈的中间维度，至少为 1，避免通道太少时降成 0
        hidden = max(channels // reduction, 1)

        # 全局平均池化：把 H×W 压成 1×1，得到每个通道的全局描述子
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 瓶颈：先降维，学通道间的非线性关系，再升回原通道数
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),  # 降维
            nn.ReLU(inplace=True),                                   # 非线性
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),  # 升维
        )

        # sigmoid 把权重压到 (0,1)，作为逐通道的缩放系数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        # 1) 全局上下文：每个通道一个标量
        context = self.gap(x)              # (B, C, 1, 1)

        # 2) 瓶颈建模通道间关系，再映射回 C 维权重
        weight = self.fc(context)          # (B, C, 1, 1)

        # 3) 归一化到 (0,1)
        weight = self.sigmoid(weight)      # (B, C, 1, 1)

        # 4) 逐通道相乘，广播回 H×W；形状不变
        return x * weight                  # (B, C, H, W)
