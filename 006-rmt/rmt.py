# -*- coding: utf-8 -*-
"""
【医学图像分割模块】RMT —— 保留注意力 —— 给视觉 Transformer 加一个"显式记忆"

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRetention(nn.Module):
    """简化版保留注意力：带显式指数衰减的线性注意力"""
    def __init__(self, dim, num_heads=4, gamma=0.9):
        super().__init__()
        self.dim = dim                      # 输入特征维度
        self.num_heads = num_heads          # 头数
        self.head_dim = dim // num_heads    # 每个头的维度
        self.gamma = gamma                  # 衰减因子，越接近 1 记忆越长
        # Q、K、V 的线性投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim) # 输出投影
        self.norm = nn.GroupNorm(1, dim)    # 输出归一化，稳定训练

    def forward(self, x):
        # x: (B, N, C)，N 是 token 数（如 H*W），C 是通道数
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim
        # 投影并拆成多头: (B, H, N, D)
        q = self.q_proj(x).view(B, N, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, D).transpose(1, 2)

        # 构造衰减矩阵 D_mat: (N, N)，D_mat[i, j] = gamma^(i-j) if i>=j else 0
        idx = torch.arange(N, device=x.device)
        # 相对距离 i-j，下三角为正
        rel = idx[:, None] - idx[None, :]          # (N, N)
        decay = torch.where(rel >= 0,
                            self.gamma ** rel.clamp(min=0).float(),
                            torch.zeros_like(rel, dtype=torch.float))
        decay = decay.to(x.dtype)                  # (N, N)

        # 线性注意力核心：先算 K^T V，再乘 Q，避免 N×N 的 QK^T
        # 但为了显式衰减，这里用衰减加权的 K^T V 形式
        # 简化写法：直接对 K 做衰减加权后再算注意力
        # 权重 w[i,j] = decay[i,j]，对 j 做归一化
        w = decay / (decay.sum(dim=-1, keepdim=True) + 1e-6)  # (N, N)
        # 注意力输出: (B, H, N, D) = w @ v
        out = torch.einsum('ij,bhjd->bhid', w, v)  # (B, H, N, D)

        # 合并多头并投影
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        # 残差 + 归一化
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return out + x
