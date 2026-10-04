# -*- coding: utf-8 -*-
"""
【医学图像分割模块】Latent —— Latent-to-Latent Flow for Volumetric Stochastic Segmentation

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """把标量时间 t 编码成向量，供网络感知当前处于 flow 的哪一步。"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [B]，取值在 [0, 1]
        half = self.dim // 2
        # 不同频率的正弦基，频率随维度指数增长
        freqs = torch.exp(
            -torch.arange(half, device=t.device, dtype=t.dtype)
            * (torch.log(torch.tensor(10000.0, device=t.device)) / (half - 1))
        )
        args = t[:, None] * freqs[None]  # [B, half]
        # 拼接 sin 和 cos，得到 [B, dim]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class VelocityNet(nn.Module):
    """预测向量场 v(z_t, t)。这里用几层 MLP，潜空间维度小，够用。"""
    def __init__(self, latent_dim, hidden_dim=256, time_dim=64):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        # 输入是潜编码 + 时间编码
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),  # 输出与潜编码同维
        )

    def forward(self, z_t, t):
        # z_t: [B, latent_dim]，t: [B]
        te = self.time_embed(t)          # [B, time_dim]
        x = torch.cat([z_t, te], dim=-1) # 拼接
        return self.net(x)               # 预测速度 [B, latent_dim]


class Latent(nn.Module):
    """
    即插即用的潜空间 flow matching 模块。
    训练：传入编码器给出的潜编码 z1，内部采样 z0 和 t，回归速度。
    推理：从先验采样 z0，积分若干步得到 z1。
    """
    def __init__(self, latent_dim, hidden_dim=256, time_dim=64, steps=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.steps = steps  # 推理时的积分步数，步数越多越准但越慢
        self.velocity = VelocityNet(latent_dim, hidden_dim, time_dim)

    def forward(self, z1):
        """训练前向：返回 flow matching 损失。z1: [B, latent_dim]"""
        B = z1.shape[0]
        device = z1.device
        # 1) 从标准高斯先验采样起点 z0
        z0 = torch.randn_like(z1)
        # 2) 在 [0,1] 上均匀采样时间 t
        t = torch.rand(B, device=device)
        # 3) 构造线性插值路径 z_t = (1-t) z0 + t z1
        t_ = t[:, None]
        z_t = (1 - t_) * z0 + t_ * z1
        # 4) 目标速度就是路径的导数：d z_t / dt = z1 - z0
        target = z1 - z0
        # 5) 网络预测速度，用 MSE 回归
        pred = self.velocity(z_t, t)
        loss = F.mse_loss(pred, target)
        return loss

    @torch.no_grad()
    def sample(self, n, device):
        """推理：从先验采样 n 个潜编码，积分得到目标分布样本。"""
        # 起点：标准高斯
        z = torch.randn(n, self.latent_dim, device=device)
        dt = 1.0 / self.steps
        # 欧拉法沿速度场积分，从 t=0 走到 t=1
        for i in range(self.steps):
            t = torch.full((n,), i * dt, device=device)
            v = self.velocity(z, t)
            z = z + v * dt
        return z  # [n, latent_dim]
