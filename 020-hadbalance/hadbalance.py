# -*- coding: utf-8 -*-
"""
【医学图像分割模块】HadBalance —— HadBalance: A Plug-and-Play Unified Global Geometric Prior F

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HadBalance(nn.Module):
    """
    即插即用几何先验平衡块。
    输入:  (B, C, H, W)
    输出:  (B, C, H, W)  形状不变，可直接替换原特征
    """

    def __init__(self, channels, reduction=8, kernel_size=7):
        super().__init__()
        self.channels = channels

        # ---- 全局几何分支：估计近凸包络 ----
        # 用深度可分离卷积 + 大核平滑，近似"把局部凹陷填平"的凸化效果
        self.geo_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size,
                      padding=kernel_size // 2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            # 1x1 做通道混合，让几何分支能跨通道整合形状信息
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # ---- 局部证据分支：保留原始局部响应 ----
        # 一个轻量残差，避免直接恒等导致分支无参数可学
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        # ---- 自适应平衡门控 ----
        # 输入是两路特征的拼接，输出每个位置/通道的融合权重 g ∈ (0,1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, 1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),  # 权重压到 0~1
        )

        # 输出投影，融合后回到原特征空间
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def _soft_convex_envelope(self, x):
        """
        软凸包络近似：对空间维度做软最大，得到"全局主导响应"，
        再广播回原尺寸。直觉上相当于把局部小凹陷用全局形状填平。
        """
        # x: (B, C, H, W)
        # 在 H*W 上做 softmax 加权求和，得到一个全局描述子 (B, C, 1, 1)
        b, c, h, w = x.shape
        flat = x.view(b, c, h * w)
        # 温度系数控制"软"的程度，越小越接近硬最大
        attn = F.softmax(flat * 2.0, dim=-1)          # (B, C, H*W)
        global_desc = (flat * attn).sum(dim=-1)        # (B, C) 全局加权响应
        global_desc = global_desc.view(b, c, 1, 1)
        # 广播回原尺寸，作为"凸参考"的基础
        return global_desc.expand(b, c, h, w)

    def forward(self, x):
        # 1) 全局几何分支：先平滑，再叠加软凸包络
        geo = self.geo_conv(x)
        envelope = self._soft_convex_envelope(geo)
        # 几何表示 = 平滑特征 与 全局包络 的残差组合
        geo_feat = geo + envelope

        # 2) 局部证据分支
        local_feat = self.local_conv(x)

        # 3) 自适应平衡：门控看两路特征，决定每个位置信谁更多
        g = self.gate(torch.cat([geo_feat, local_feat], dim=1))  # (B, C, H, W)
        fused = g * geo_feat + (1.0 - g) * local_feat            # 凸组合

        # 4) 输出投影 + 残差回注，保证即插即用不退化
        out = self.bn(self.out_proj(fused))
        return x + out  # 残差连接：初始近似恒等，训练稳定
