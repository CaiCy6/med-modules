# -*- coding: utf-8 -*-
"""
【医学图像分割模块】MambaLiteUNet —— MambaLiteUNet: Cross-Gated Adaptive Feature Fusion for Robus

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossGatedAttention(nn.Module):
    """
    Cross-Gated Attention (CGA) 交叉门控注意力块。
    双输入单输出：接收两路特征 (x_skip, x_deep)，输出门控融合后的特征。
    通道数保持不变，可直接替换 U-Net 跳跃连接处的 concat。
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        # 门控分支用的中间通道数，先降维再升维，控制参数量
        hidden = max(channels // reduction, 8)

        # 对第一路输入做轻量变换，对齐到 hidden 维
        self.proj_a = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        # 对第二路输入做同样的变换
        self.proj_b = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        # 交叉门控：用 a 生成门控 b 的权重图，用 b 生成门控 a 的权重图
        self.gate_b = nn.Conv2d(hidden, hidden, kernel_size=1, bias=True)
        self.gate_a = nn.Conv2d(hidden, hidden, kernel_size=1, bias=True)

        # 融合后回到原通道数
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x_skip, x_deep):
        # 两路输入先各自投影到 hidden 维
        a = self.proj_a(x_skip)   # 浅层：细节强
        b = self.proj_b(x_deep)   # 深层：语义强

        # 交叉门控：a 门控 b，b 门控 a
        # sigmoid 输出逐像素 0~1 权重，实现空间自适应的软掩码
        g_b = torch.sigmoid(self.gate_b(a))   # 用 a 决定 b 保留多少
        g_a = torch.sigmoid(self.gate_a(b))   # 用 b 决定 a 保留多少

        # 加权后相加，得到双向筛选过的融合特征
        out = a * g_a + b * g_b

        # 投影回原通道数，方便直接接后续卷积
        out = self.fuse(out)

        # 残差：把原始两路相加作为恒等路径，稳定训练
        return out + x_skip + x_deep
