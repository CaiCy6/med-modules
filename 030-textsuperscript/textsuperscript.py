# -*- coding: utf-8 -*-
"""
【医学图像分割模块】M\textsuperscript{4}Fuse —— M\textsuperscript{4}Fuse: Lightweight State-Space MoE with a

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGate(nn.Module):
    """第一级：空间门控，压掉背景体素响应。"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        # 用 1x1 卷积把通道压到 1，得到一个空间重要性图
        # 这里不直接用 sigmoid(conv)，而是先降维再升维，减少参数
        hidden = max(channels // reduction, 4)
        self.conv = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=False),
            nn.InstanceNorm3d(hidden),   # 3D 医学图像 batch 通常很小，用 IN 比 BN 稳
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, 1, kernel_size=1, bias=False),
        )

    def forward(self, x):
        # x: (B, C, D, H, W)
        attn = torch.sigmoid(self.conv(x))   # (B, 1, D, H, W)，每个体素一个权重
        return x * attn                      # 逐体素加权，背景被压低


class ChannelGate(nn.Module):
    """第二级：通道门控，对齐跨尺度语义。"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        # 输入是两路特征拼接后的通道数（2C），输出 C 个通道权重
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, enc, up):
        # enc, up: (B, C, D, H, W)
        # 全局平均池化，把空间维压掉，只保留通道统计量
        b, c = enc.shape[:2]
        enc_g = enc.mean(dim=(2, 3, 4))          # (B, C)
        up_g = up.mean(dim=(2, 3, 4))            # (B, C)
        z = torch.cat([enc_g, up_g], dim=1)      # (B, 2C)
        w = torch.sigmoid(self.mlp(z))           # (B, C)，每个通道一个权重
        w = w.view(b, c, 1, 1, 1)                # 广播回空间维
        return enc * w                           # 对编码器特征做通道重加权


class CrossScaleGatingBridge(nn.Module):
    """跨尺度门控桥：即插即用的 skip 融合模块。

    输入：enc_feat (B, C, D, H, W)，up_feat (B, C, D, H, W)
    输出：融合后的特征 (B, C, D, H, W)，空间尺寸不变
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.spatial_gate = SpatialGate(channels, reduction)
        self.channel_gate = ChannelGate(channels, reduction)
        # 融合后的 1x1 卷积，把两路信息压回原通道数
        self.fuse = nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm3d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, enc_feat, up_feat):
        # 第一步：空间去噪，压掉编码器浅层的背景响应
        enc_denoised = self.spatial_gate(enc_feat)
        # 第二步：通道对齐，用解码器语义指导编码器特征的通道选择
        enc_aligned = self.channel_gate(enc_denoised, up_feat)
        # 第三步：拼接 + 1x1 卷积融合，输出和输入同尺寸同通道
        out = torch.cat([enc_aligned, up_feat], dim=1)
        out = self.fuse(out)
        out = self.norm(out)
        return self.act(out)
