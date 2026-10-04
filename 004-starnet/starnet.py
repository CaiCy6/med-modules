# -*- coding: utf-8 -*-
"""
【医学图像分割模块】StarNet —— 星操作（元素级乘法）—— 把"相加"换成"相乘"的轻量卷积块

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn

class StarBlock(nn.Module):
    """StarNet 的核心模块（简化教学版）"""
    def __init__(self, dim, expand_ratio=4):
        super().__init__()
        # 记录输入通道数，用于最后的残差相加
        self.dim = dim
        # 深度可分离卷积：每个通道独立做 3x3 空间卷积，负责局部空间建模
        # groups=dim 表示逐通道卷积，参数量和计算量都很小
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        # 第一条 1x1 线性分支：把通道扩展到 expand_ratio 倍
        # 这里用 expand_ratio=4，给后面的逐元素相乘提供更宽的特征
        self.fc1 = nn.Conv2d(dim, dim * expand_ratio, kernel_size=1)
        # 第二条 1x1 线性分支：同样扩展到 expand_ratio 倍
        # 两条分支输出形状一致，才能做逐元素相乘
        self.fc2 = nn.Conv2d(dim, dim * expand_ratio, kernel_size=1)
        # 融合后的 1x1 映射：把扩展后的通道压回原始维度
        self.fc3 = nn.Conv2d(dim * expand_ratio, dim, kernel_size=1)
        # 激活函数，放在乘法之后引入额外非线性
        self.act = nn.GELU()

    def forward(self, x):
        # 保存输入，用于最后的残差连接
        identity = x
        # 先做深度可分离卷积，注入局部空间信息
        x = self.dwconv(x)
        # 两条 1x1 分支分别做线性变换，得到 a 和 b
        a = self.fc1(x)
        b = self.fc2(x)
        # 关键一步：逐元素相乘（star operation）
        # 相乘会产生输入的高阶交叉项，相当于隐式映射到高维非线性空间
        out = a * b
        # 激活，进一步增加非线性
        out = self.act(out)
        # 1x1 映射压回原始通道数
        out = self.fc3(out)
        # 残差连接：稳定训练，也让模块可以安全地插入已有网络
        return out + identity
