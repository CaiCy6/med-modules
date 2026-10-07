# -*- coding: utf-8 -*-
"""
【医学图像分割模块】GFNet —— 全局滤波器网络 —— 用 FFT 在频域做"全局混频"

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn


class GlobalFilter(nn.Module):
    """频域全局滤波：2D FFT -> 逐元素乘可学习滤波器 -> 2D IFFT"""

    def __init__(self, dim, h, w):
        super().__init__()
        # 可学习的全局滤波器，形状为 (dim, h, w//2+1)
        # 用复数存储：实部 + 虚部两个实数张量
        # rfft2 的输出频率维度是 w//2+1（实数输入的厄米对称性）
        self.complex_weight = nn.Parameter(
            torch.randn(dim, h, w // 2 + 1, 2) * 0.02
        )
        self.h = h
        self.w = w

    def forward(self, x):
        # x: (B, H, W, C) —— 注意这里用 channels-last，方便和 rfft2 对齐
        B, H, W, C = x.shape

        # 1) 2D 实数 FFT，输出复数张量 (B, H, W//2+1, C)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        # 2) 把可学习滤波器拼成复数：real + i*imag
        weight = torch.view_as_complex(self.complex_weight)  # (C, H, W//2+1)

        # 3) 频域逐元素相乘（广播到 batch 维）—— 这一步就是 token mixing
        x = x * weight

        # 4) 2D 逆 FFT，回到空间域 (B, H, W, C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        return x


class GFNetBlock(nn.Module):
    """一个最小的 GFNet 块：全局滤波 + 通道 MLP + 残差"""

    def __init__(self, dim, h, w, mlp_ratio=4.0):
        super().__init__()
        self.filter = GlobalFilter(dim, h, w)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # 通道 MLP，和 Transformer 里的 FFN 一个作用
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        # x: (B, C, H, W) —— 对外保持 NCHW，方便塞进 U-Net
        B, C, H, W = x.shape

        # 转成 channels-last 给频域滤波用
        x_ = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_ = self.norm1(x_)
        x_ = self.filter(x_)
        x = x + x_.permute(0, 3, 1, 2)  # 残差，转回 NCHW

        # 通道 MLP 分支（同样走残差）
        x_ = x.permute(0, 2, 3, 1)
        x_ = self.norm2(x_)
        x_ = self.mlp(x_)
        x = x + x_.permute(0, 3, 1, 2)
        return x
