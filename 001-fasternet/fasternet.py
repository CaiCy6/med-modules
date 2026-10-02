# -*- coding: utf-8 -*-
"""
【医学图像分割模块】FasterNet / PConv（CVPR 2023）—— 只卷一部分通道，又快又准

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn

class Partial_conv3(nn.Module):
    """PConv：3×3 卷积只作用于一部分通道，其余通道直通。"""
    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv3 = dim // n_div              # 参与卷积的通道数（默认 C/4）
        self.dim_untouched = dim - self.dim_conv3  # 直通、不计算的通道数
        # 3×3 深度无关的普通卷积，但只吃 dim_conv3 个通道
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3,
                                       kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # 按通道切成"要算的"和"直通的"两部分
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)                # 只对 x1 做 3×3
        x = torch.cat((x1, x2), dim=1)             # x2 原样拼接回去
        return x

class Faster_Block(nn.Module):
    """FasterNet Block：PConv + 两个 1×1 卷积（PWConv）+ 残差。"""
    def __init__(self, dim, n_div=4, mlp_ratio=2, drop_path=0.0):
        super().__init__()
        self.conv3 = Partial_conv3(dim, n_div)                        # PConv
        self.conv1 = nn.Conv2d(dim, int(dim * mlp_ratio), 1, 1, 0, bias=False)  # 1×1 升维
        self.conv2 = nn.Conv2d(int(dim * mlp_ratio), dim, 1, 1, 0, bias=False)  # 1×1 降维
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
        self.drop_path = nn.Identity() if drop_path <= 0 else nn.Dropout(drop_path)

    def forward(self, x):
        shortcut = x                       # 残差分支
        x = self.conv3(x)                  # PConv：只卷部分通道
        x = self.bn(x)
        x = self.conv1(x)                  # 通道升维，增强表达
        x = self.act(x)
        x = self.conv2(x)                  # 通道降维
        x = shortcut + self.drop_path(x)   # 残差相加
        return x
