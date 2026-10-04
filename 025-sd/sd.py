# -*- coding: utf-8 -*-
"""
【医学图像分割模块】SD —— SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Ima

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SD(nn.Module):
    """
    SD: Stable Diffusion 特征注入块（即插即用）
    输入:
        seg_feat: 分割网络当前特征, 形状 (B, C_seg, H, W)
        sd_feat:  扩散模型中间层特征, 形状 (B, C_sd, H_sd, W_sd)
    输出:
        融合后的特征, 形状 (B, C_seg, H, W)
    """

    def __init__(self, c_seg, c_sd, c_mid=128, gate_init=0.0):
        super().__init__()
        # 把分割特征压到统一中间通道
        self.proj_seg = nn.Conv2d(c_seg, c_mid, kernel_size=1, bias=False)
        # 把扩散特征压到统一中间通道
        self.proj_sd = nn.Conv2d(c_sd, c_mid, kernel_size=1, bias=False)
        # 归一化，稳定两路特征的尺度差异
        self.norm_seg = nn.BatchNorm2d(c_mid)
        self.norm_sd = nn.BatchNorm2d(c_mid)
        # 可学习门控：控制扩散特征注入强度，初始为 gate_init
        # 用 Parameter 而不是固定系数，让网络自己学该借多少
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        # 融合后的轻量残差卷积，恢复表达能力
        self.fuse = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(inplace=True),
        )
        # 输出投影回分割特征通道，方便直接替换原特征
        self.proj_out = nn.Conv2d(c_mid, c_seg, kernel_size=1, bias=False)

    def forward(self, seg_feat, sd_feat):
        # 记录分割特征原始尺寸，用于最后对齐
        h, w = seg_feat.shape[-2:]
        # 扩散特征空间尺寸可能不同，双线性插值对齐到分割特征尺寸
        if sd_feat.shape[-2:] != (h, w):
            sd_feat = F.interpolate(
                sd_feat, size=(h, w), mode="bilinear", align_corners=False
            )
        # 两路各自投影到统一通道并归一化
        s = self.norm_seg(self.proj_seg(seg_feat))
        d = self.norm_sd(self.proj_sd(sd_feat))
        # 门控加权融合：gate 可正可负，网络自行调节注入方向与强度
        fused = s + self.gate * d
        # 残差卷积增强
        fused = self.fuse(fused)
        # 投影回原通道
        out = self.proj_out(fused)
        # 残差连接：即使门控没学好，最差也退化成恒等映射
        return seg_feat + out
