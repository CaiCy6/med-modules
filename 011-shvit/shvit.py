# -*- coding: utf-8 -*-
"""
【医学图像分割模块】SHViT —— 单头视觉 Transformer —— 砍掉多头冗余，做出又快又准的高效主干

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn


class SingleHeadAttention(nn.Module):
    """SHViT 的单头注意力块（简化教学版）。

    核心：只用一个注意力头做全局建模，同时并行一个深度可分离卷积分支做局部建模，
    两路结果相加后过一层投影。这样既避免了多头冗余，又不丢局部细节。
    """

    def __init__(self, dim, num_heads=1, sr_ratio=2, kernel_size=3):
        super().__init__()
        assert num_heads == 1, "本模块只保留单头，头数固定为 1"
        self.dim = dim
        self.sr_ratio = sr_ratio  # 全局分支的空间下采样比例，用来压注意力矩阵

        # ---- 全局分支：单头 QKV ----
        # 一个头，所以 qkv 的通道数就是 dim，不再做多头切分
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1)

        # 对 K/V 做空间下采样，减少注意力矩阵规模（内存高效的关键）
        self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
        self.norm = nn.GroupNorm(1, dim)  # 用 GN 做归一化，batch 小时比 BN 稳

        # ---- 局部分支：深度可分离卷积，建模局部窗口 ----
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2,
                      groups=dim, bias=False),   # 深度卷积，逐通道做局部聚合
            nn.GroupNorm(1, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),  # 逐点卷积，做通道混合
        )

        # ---- 输出投影 ----
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # ===== 全局分支 =====
        q = self.q(x)                      # [B, C, H, W]
        kv = self.kv(x)                    # [B, 2C, H, W]
        k, v = kv.chunk(2, dim=1)          # 各 [B, C, H, W]

        if self.sr_ratio > 1:
            k = self.sr(k)                 # 下采样 K
            v = self.sr(v)                 # 下采样 V，注意力矩阵随之变小
        k = self.norm(k)
        v = self.norm(v)

        # 展平成序列，单头注意力：softmax(Q K^T / sqrt(d)) V
        Bk, Ck, Hk, Wk = k.shape
        q_flat = q.flatten(2).transpose(1, 2)          # [B, HW, C]
        k_flat = k.flatten(2).transpose(1, 2)          # [B, HkWk, C]
        v_flat = v.flatten(2).transpose(1, 2)          # [B, HkWk, C]

        scale = C ** -0.5
        attn = torch.softmax(q_flat @ k_flat.transpose(1, 2) * scale, dim=-1)  # [B, HW, HkWk]
        global_out = attn @ v_flat                     # [B, HW, C]
        global_out = global_out.transpose(1, 2).reshape(B, C, H, W)

        # ===== 局部分支 =====
        local_out = self.local(x)          # [B, C, H, W]

        # 两路并行相加，再投影输出
        out = self.proj(global_out + local_out)
        return out
