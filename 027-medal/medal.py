# -*- coding: utf-8 -*-
"""
【医学图像分割模块】Medal S —— Medal S: Spatio-Textual Prompt Model for Medical Segmentatio

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedalSBlock(nn.Module):
    """
    Medal S 的即插即用精修块。
    输入:
        x:    粗分割特征, shape (B, C, D, H, W)
        prompt: 提示特征, shape (B, C_p, D, H, W) 或 (B, C_p)
                文本提示通常是 (B, C_p)，空间提示是 (B, C_p, D, H, W)
    输出:
        精修后的特征, shape (B, C, D, H, W)
    """

    def __init__(self, in_channels, prompt_channels, hidden_channels=None, use_3d_conv=True):
        super().__init__()
        hidden_channels = hidden_channels or in_channels

        # 1) 通道级对齐：把提示特征投影到和体素特征同一个通道空间
        #    这是 Medal S 的关键一步，消除提示与图像特征的通道鸿沟
        self.prompt_proj = nn.Sequential(
            nn.Conv3d(prompt_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.GELU(),
        )

        # 2) 门控调制：用对齐后的提示生成逐通道的门控权重
        #    相当于让提示决定"哪些通道该被强调"
        self.gate = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # 3) 轻量 3D 卷积精修：在体素空间做邻域修正
        #    用 depthwise + pointwise 控制参数量，保持"轻量"
        if use_3d_conv:
            self.refine = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3,
                          padding=1, groups=in_channels, bias=False),  # depthwise
                nn.BatchNorm3d(in_channels),
                nn.GELU(),
                nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False),  # pointwise
                nn.BatchNorm3d(in_channels),
            )
        else:
            self.refine = nn.Identity()

        # 4) 残差融合：把精修结果和原始特征相加，保证不破坏主干信息
        self.out_proj = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x, prompt):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape

        # 如果提示是全局向量 (B, C_p)，先广播成体素形状
        if prompt.dim() == 2:
            prompt = prompt.view(B, -1, 1, 1, 1).expand(-1, -1, D, H, W)

        # 通道对齐
        p = self.prompt_proj(prompt)          # (B, C, D, H, W)

        # 门控调制：提示生成门控，乘到体素特征上
        g = self.gate(p)                      # (B, C, D, H, W)
        x_mod = x * g                         # 逐通道加权

        # 3D 卷积精修
        x_ref = self.refine(x_mod)            # (B, C, D, H, W)

        # 残差：原始特征 + 精修增量
        out = x + self.out_proj(x_ref)
        return out
