# -*- coding: utf-8 -*-
"""
【医学图像分割模块】GET —— GET: Generative Embedding Translation for Medical Image Segm

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn


class MobileBottleneck(nn.Module):
    """轻量瓶颈卷积：先压通道 -> 深度卷积 -> 再升通道。"""
    def __init__(self, in_ch, out_ch, expand=2, stride=1):
        super().__init__()
        mid_ch = in_ch * expand  # 瓶颈中间通道数
        self.use_res = (stride == 1 and in_ch == out_ch)  # 形状一致才加残差

        self.block = nn.Sequential(
            # 1x1 升维（逐点卷积），把通道扩到 mid_ch
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
            # 深度卷积，逐通道做空间卷积，省参数
            nn.Conv2d(mid_ch, mid_ch, 3, stride, 1,
                      groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
            # 1x1 降维回 out_ch
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            out = out + x  # 残差连接，稳定训练
        return out


class GETBlock(nn.Module):
    """GET 可插拔翻译块：输入输出同形状，可堆叠。"""
    def __init__(self, channels, expand=2, num_layers=2):
        super().__init__()
        # 堆叠若干 MobileBottleneck 做嵌入翻译
        layers = []
        for _ in range(num_layers):
            layers.append(MobileBottleneck(channels, channels, expand))
        self.translate = nn.Sequential(*layers)
        # 轻量通道注意力，强调对翻译有用的通道
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.translate(x)          # 嵌入翻译
        feat = feat * self.attn(feat)     # 通道重加权
        return x + feat                   # 残差，保证即插即用不破坏原特征
