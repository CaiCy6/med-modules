# -*- coding: utf-8 -*-
"""
【医学图像分割模块】Uni —— Uni-Encoder Meets Multi-Encoders: Representation Before Fusi

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UniBlock(nn.Module):
    """
    Uni 表示块（即插即用版）
    输入:  (B, C, H, W)  特征图
    输出:  (B, C, H, W)  形状不变，内容被全局表示增强
    核心:  patch 化 -> 随机掩码 -> ViT 编码 -> 全局 token -> 广播回原图
    """

    def __init__(self, channels, patch_size=4, embed_dim=256,
                 depth=4, num_heads=8, mlp_ratio=4.0, mask_ratio=0.5):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # 1) patch 嵌入：把每个 patch 的像素展平后线性映射到 embed_dim
        #    这里用卷积实现，stride=patch_size 即等价于不重叠切 patch
        self.patch_embed = nn.Conv2d(
            channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # 2) 可学习的全局 token，类似 ViT 的 class token
        #    它负责在编码后聚合整张图的全局信息
        self.global_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.global_token, std=0.02)

        # 3) 标准 Transformer 编码器，处理 patch 序列
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 4) 输出投影：把全局 token 映射回 channels，用于广播相加
        self.out_proj = nn.Linear(embed_dim, channels)

        # 5) 归一化，稳定训练
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        # --- 步骤 1: patch 化 ---
        # (B, C, H, W) -> (B, embed_dim, H/p, W/p)
        feat = self.patch_embed(x)
        # 展平成序列: (B, embed_dim, N) -> (B, N, embed_dim)
        feat = feat.flatten(2).transpose(1, 2)
        N = feat.shape[1]  # patch 数量

        # --- 步骤 2: 随机掩码（仅训练时启用）---
        # 掩码的作用是逼模型从残缺信息里恢复语义，
        # 从而对「模态缺失」这种信息不全的情况更鲁棒
        if self.training and self.mask_ratio > 0:
            # 每个样本独立采样要保留的 patch 索引
            keep = max(1, int(N * (1 - self.mask_ratio)))
            # 生成随机分数，取分数最高的 keep 个作为可见 patch
            noise = torch.rand(B, N, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_keep = ids_shuffle[:, :keep]
            # 按索引 gather 出可见 patch
            feat = torch.gather(
                feat, 1,
                ids_keep.unsqueeze(-1).expand(-1, -1, feat.shape[-1])
            )

        # --- 步骤 3: 拼接全局 token ---
        # 把 global_token 复制 B 份，拼到序列最前面
        gt = self.global_token.expand(B, -1, -1)
        feat = torch.cat([gt, feat], dim=1)  # (B, 1+keep, embed_dim)

        # --- 步骤 4: Transformer 编码 ---
        feat = self.encoder(feat)
        feat = self.norm(feat)

        # --- 步骤 5: 取全局 token 作为整图表示 ---
        global_feat = feat[:, 0]  # (B, embed_dim)

        # --- 步骤 6: 投影回 channels 并广播回原特征图 ---
        # (B, embed_dim) -> (B, C) -> (B, C, 1, 1)
        global_feat = self.out_proj(global_feat)
        global_feat = global_feat[:, :, None, None]

        # 残差相加：保留原始局部特征，叠加全局上下文
        return x + global_feat
