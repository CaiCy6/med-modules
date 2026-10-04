# -*- coding: utf-8 -*-
"""
【医学图像分割模块】MedCAGD —— MedCAGD: Context-Aware Gated Decoder for Efficient Medical I

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedCAGD(nn.Module):
    """
    Context-Aware Gated Decoder 子模块（即插即用版）
    输入：dec_feat 当前解码特征 [B, C, H, W]
          ctx_feat 上下文特征（编码器 skip 或深层特征）[B, C, H, W]
    输出：门控融合后的特征 [B, C, H, W]
    要求：两路输入通道数相同，空间尺寸相同（不同则内部自动对齐）
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        # 通道注意力分支：先全局池化，再用两个全连接层学通道权重
        # reduction 控制瓶颈比例，默认 8，参数量很小
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                      # [B,C,H,W] -> [B,C,1,1]
            nn.Conv2d(channels, channels // reduction, 1), # 降维，压缩通道
            nn.ReLU(inplace=True),                         # 非线性
            nn.Conv2d(channels // reduction, channels, 1), # 升维，恢复通道
            nn.Sigmoid()                                   # 输出 0~1 的门控值
        )

        # 空间注意力分支：用 1x1 卷积把两路特征压成单通道，学空间权重
        # 输入是 dec_feat 和 ctx_feat 的拼接，所以是 2*channels -> 1
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 2, 1, kernel_size=1),     # 融合两路信息到单通道
            nn.Sigmoid()                                   # 输出 0~1 的空间门控图
        )

        # 融合后的精炼卷积，保证输出特征表达能力
        self.refine = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # 可学习的残差缩放系数，初始为 0，训练初期等价于恒等映射
        # 这样插入预训练网络时不会破坏原有行为，非常关键
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, dec_feat, ctx_feat):
        # 如果空间尺寸不一致，用双线性插值把上下文特征对齐到解码特征
        if dec_feat.shape[-2:] != ctx_feat.shape[-2:]:
            ctx_feat = F.interpolate(
                ctx_feat, size=dec_feat.shape[-2:],
                mode='bilinear', align_corners=False
            )

        # ---- 通道门控 ----
        # 用上下文特征的全局信息生成通道权重
        c_gate = self.channel_gate(ctx_feat)              # [B,C,1,1]
        # 对解码特征做通道加权，同时用 1-c_gate 保留原始信息
        dec_weighted = dec_feat * c_gate + dec_feat * (1 - c_gate)  # 等价于 dec_feat，这里保留结构清晰
        # 实际有效的是下面这步：用门控调制上下文，再注入解码特征
        ctx_modulated = ctx_feat * c_gate

        # ---- 空间门控 ----
        # 拼接两路特征，学出空间上的融合权重
        concat = torch.cat([dec_feat, ctx_modulated], dim=1)  # [B,2C,H,W]
        s_gate = self.spatial_gate(concat)                     # [B,1,H,W]

        # 空间门控融合：s_gate 控制上下文注入比例
        fused = dec_feat * (1 - s_gate) + ctx_modulated * s_gate

        # ---- 精炼 + 残差 ----
        out = self.refine(fused)
        # alpha 初始为 0，训练中逐渐学习注入多少门控信息
        return dec_feat + self.alpha * out
