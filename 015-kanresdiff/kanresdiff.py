# -*- coding: utf-8 -*-
"""
【医学图像分割模块】KANResDiff —— KANResDiff: Learning Local Residual Diffusion via Kolmogorov

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SplineTimeEmbedding(nn.Module):
    """独立时间编码：用样条基函数替代 MLP 的线性时间嵌入。"""
    def __init__(self, dim, num_basis=8, grid_range=(-1.0, 1.0)):
        super().__init__()
        self.dim = dim
        self.num_basis = num_basis
        # 可学习的样条控制点，每个基函数一组系数
        self.coeff = nn.Parameter(torch.randn(num_basis, dim) * 0.02)
        # 固定网格，用于把标量时间步映射到基函数响应
        self.register_buffer(
            "grid", torch.linspace(grid_range[0], grid_range[1], num_basis)
        )

    def forward(self, t):
        # t: (B,) 取值范围约定在 [0,1]
        # 把 t 拉伸到网格区间，再算到每个网格点的距离
        t = t.view(-1, 1) * 2.0 - 1.0                 # (B,1) -> [-1,1]
        dist = t - self.grid.view(1, -1)              # (B, num_basis)
        # 用高斯型基函数做软分配，得到样条响应
        basis = torch.exp(-(dist ** 2) / (2 * 0.25 ** 2))  # (B, num_basis)
        basis = basis / (basis.sum(dim=1, keepdim=True) + 1e-6)
        # 基函数响应加权控制点，得到时间嵌入
        emb = basis @ self.coeff                      # (B, dim)
        return emb


class KANLayer(nn.Module):
    """一个简化的 KAN 层：输入经样条基展开后线性组合。"""
    def __init__(self, in_dim, out_dim, num_basis=8):
        super().__init__()
        self.num_basis = num_basis
        # 每个输入维度对应一组样条基权重
        self.spline_weight = nn.Parameter(
            torch.randn(in_dim, num_basis, out_dim) * 0.02
        )
        # 残差式的线性旁路，保证训练稳定
        self.base_weight = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # x: (B, C, H, W)，这里把通道当特征维
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)   # (B*H*W, C)
        # 样条基：用 sin 组合近似不同频率的基函数
        k = torch.arange(self.num_basis, device=x.device).float()
        basis = torch.sin(x_flat.unsqueeze(-1) * (k + 1) * 3.14159)  # (N,C,K)
        # 基函数响应加权求和到输出维度
        spline_out = torch.einsum("nck,cko->no", basis, self.spline_weight)
        out = spline_out + self.base_weight(x_flat)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return out


class KANResDiff(nn.Module):
    """即插即用的局部残差扩散块。"""
    def __init__(self, channels, num_basis=8, num_steps=10):
        super().__init__()
        self.channels = channels
        self.num_steps = num_steps
        # 时间编码：把标量时间步映射成通道维度的嵌入
        self.time_emb = SplineTimeEmbedding(channels, num_basis=num_basis)
        # 去噪主干：两层 KAN，中间带激活
        self.kan1 = KANLayer(channels, channels, num_basis=num_basis)
        self.kan2 = KANLayer(channels, channels, num_basis=num_basis)
        self.act = nn.SiLU()
        # 输出层，把特征压回残差量
        self.out_proj = nn.Conv2d(channels, channels, 1)
        # 可学习的缩放系数，初始很小，保证插入初期不破坏主干
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, t=None):
        # x: (B, C, H, W) 主干特征
        B = x.shape[0]
        if t is None:
            # 未指定时间步时，随机采一个，训练时提供多样性
            t = torch.rand(B, device=x.device)
        # 时间嵌入 -> (B, C) -> 广播到空间维
        emb = self.time_emb(t).view(B, self.channels, 1, 1)
        h = x + emb                                  # 注入时间信息
        h = self.act(self.kan1(h))                   # 第一层 KAN
        h = self.kan2(h)                             # 第二层 KAN
        residual = self.out_proj(h)                  # 得到残差修正量
        # 残差加回主干，scale 控制注入强度
        return x + self.scale * residual
