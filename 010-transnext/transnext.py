# -*- coding: utf-8 -*-
"""
【医学图像分割模块】TransNeXt —— 聚合注意力（Aggregated Attention）—— 像人眼一样做中心凹+周边视觉

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedAggregatedAttention(nn.Module):
    """
    简化版聚合注意力：两路设计
    - 中心凹分支：深度可分离卷积，局部高分辨率聚合
    - 周边分支：全局平均池化 + 线性投影，低分辨率全局聚合
    两路输出相加后过投影，输出形状与输入一致，可直接替换标准注意力。
    """

    def __init__(self, dim, foveal_kernel=3):
        super().__init__()
        self.dim = dim

        # ---- 中心凹分支：深度可分离卷积，模拟局部高分辨率感知 ----
        # 用 depthwise 卷积控制参数量，padding 保持空间尺寸不变
        self.foveal_conv = nn.Conv2d(
            dim, dim,
            kernel_size=foveal_kernel,
            padding=foveal_kernel // 2,
            groups=dim,          # depthwise，每个通道独立卷积
            bias=False
        )
        self.foveal_bn = nn.BatchNorm2d(dim)

        # ---- 周边分支：全局平均池化到 1x1，再线性映射回 dim ----
        # 这一步把整张图压成一个全局描述子，复杂度与分辨率线性相关
        self.peripheral_fc = nn.Linear(dim, dim)

        # ---- 融合后的输出投影 ----
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.out_bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # 中心凹分支：局部卷积 + BN + 激活
        foveal = self.foveal_conv(x)          # (B, C, H, W)
        foveal = self.foveal_bn(foveal)
        foveal = F.gelu(foveal)

        # 周边分支：全局池化 -> 线性 -> 广播回原尺寸
        peripheral = F.adaptive_avg_pool2d(x, 1)   # (B, C, 1, 1)
        peripheral = peripheral.flatten(1)          # (B, C)
        peripheral = self.peripheral_fc(peripheral) # (B, C)
        peripheral = peripheral[:, :, None, None]   # (B, C, 1, 1)
        peripheral = peripheral.expand(-1, -1, H, W)  # 广播到 (B, C, H, W)

        # 两路融合：逐元素相加，让每个位置同时拥有局部和全局信息
        out = foveal + peripheral

        # 输出投影 + BN，保持与输入同分布
        out = self.out_proj(out)
        out = self.out_bn(out)

        # 残差连接，方便直接塞进现有网络
        return x + out
