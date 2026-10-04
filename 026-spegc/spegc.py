# -*- coding: utf-8 -*-
"""
【医学图像分割模块】SPEGC —— SPEGC: Continual Test-Time Adaptation via Semantic-Prompt-En

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPEGC(nn.Module):
    """
    Semantic-Prompt-Enhanced Graph Clustering 模块（即插即用版）。
    输入:  (B, C, H, W)
    输出:  (B, C, H, W)  形状不变，可直接接回主干。
    """

    def __init__(self, channels, num_prompts=8, reduction=4, temperature=0.1):
        super().__init__()
        self.channels = channels
        self.num_prompts = num_prompts          # 语义提示向量的个数（语义锚点数量）
        self.temperature = temperature          # 图聚类 softmax 的温度系数，越小越"硬"

        # ---- 语义提示分支 ----
        # 一组可学习的提示向量，形状 (num_prompts, channels)，充当语义锚点
        self.prompts = nn.Parameter(torch.randn(num_prompts, channels) * 0.02)

        # 把提示向量投影成 query，把输入特征投影成 key/value，做提示-特征交互
        self.to_q = nn.Conv2d(channels, channels, 1)   # 特征 -> query
        self.to_k = nn.Linear(channels, channels)      # 提示 -> key
        self.to_v = nn.Linear(channels, channels)      # 提示 -> value

        # ---- 特征精炼 ----
        # 交互后的语义增强特征与原特征融合，用 1x1 卷积降维再升维，控制参数量
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, 1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )

        # ---- 图聚类聚合 ----
        # 用可学习的缩放因子控制聚类残差强度，初始化为小值，保证训练初期接近恒等映射
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W  # 空间位置数量，即图节点数

        # ===== 1. 语义提示增强 =====
        # 特征投影为 query: (B, C, H, W) -> (B, N, C)
        q = self.to_q(x).flatten(2).transpose(1, 2)          # (B, N, C)

        # 提示向量投影为 key / value: (P, C) -> (P, C)
        k = self.to_k(self.prompts)                          # (P, C)
        v = self.to_v(self.prompts)                          # (P, C)

        # 提示-特征注意力：每个空间位置对每个语义锚点的响应
        # (B, N, C) @ (C, P) -> (B, N, P)
        attn = torch.matmul(q, k.t()) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)                       # 在提示维度归一化

        # 用响应加权聚合提示向量，得到语义增强特征: (B, N, P) @ (P, C) -> (B, N, C)
        semantic = torch.matmul(attn, v)                     # (B, N, C)
        semantic = semantic.transpose(1, 2).reshape(B, C, H, W)  # 还原成特征图

        # 与原特征拼接后融合，得到语义增强后的特征
        enhanced = self.fuse(torch.cat([x, semantic], dim=1))    # (B, C, H, W)

        # ===== 2. 图聚类聚合 =====
        # 把增强特征展平为节点: (B, C, N)
        feat = enhanced.flatten(2)                           # (B, C, N)

        # 计算节点间相似度图: (B, N, N)，用点积相似度
        # 先做 L2 归一化，让相似度落在 [-1, 1]，数值更稳
        feat_norm = F.normalize(feat, dim=1)                 # (B, C, N)
        sim = torch.matmul(feat_norm.transpose(1, 2), feat_norm)  # (B, N, N)

        # 温度缩放 + softmax，得到软聚类分配矩阵（每个节点对各节点的归属权重）
        adj = F.softmax(sim / self.temperature, dim=-1)      # (B, N, N)

        # 图聚类消息传递：按归属权重聚合节点特征
        clustered = torch.matmul(feat, adj.transpose(1, 2))  # (B, C, N)
        clustered = clustered.reshape(B, C, H, W)            # 还原成特征图

        # ===== 3. 残差输出 =====
        # gamma 初始为 0，训练初期输出等于 enhanced，随训练逐步引入聚类修正
        out = enhanced + self.gamma * clustered
        return out
