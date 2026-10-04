# -*- coding: utf-8 -*-
"""
【医学图像分割模块】UBone3D —— UBone3D: Physics-Rectified Conditional Flow Matching for Ana

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import math


class SinusoidalPosEmb(nn.Module):
    """标准正弦时间嵌入，把标量时间 t 映射成高维向量。"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) 取值范围 [0,1]
        device = t.device
        half = self.dim // 2
        # 不同频率，保证不同时间步可区分
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None] * freqs[None]  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return emb


class PhysicsConditionEncoder(nn.Module):
    """把物理伪影先验编码成条件向量。

    输入是每个点的物理统计量，例如：
      - 局部厚度（表面增厚程度）
      - 沿声束方向的强度方差（条纹）
      - 邻域点数（丢点程度）
    输出与点特征同维的条件嵌入。
    """
    def __init__(self, phys_dim=3, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(phys_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, phys):
        # phys: (B, N, phys_dim)
        return self.net(phys)  # (B, N, out_dim)


class PhysicsRectifiedCFMBlock(nn.Module):
    """UBone3D 的可插拔核心：物理校正条件流匹配块。

    作用：给定残缺点云坐标 x0 和物理条件 c，
         预测位移场 v(x_t, t, c)，用于把点推向完整表面。
    """
    def __init__(self, feat_dim=128, phys_dim=3, hidden=256):
        super().__init__()
        self.feat_dim = feat_dim
        # 点坐标编码：3 维坐标 -> feat_dim
        self.xyz_enc = nn.Sequential(
            nn.Linear(3, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim),
        )
        # 时间嵌入
        self.time_emb = SinusoidalPosEmb(feat_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.SiLU(),
        )
        # 物理条件编码
        self.phys_enc = PhysicsConditionEncoder(phys_dim, feat_dim)
        # 主干：融合坐标 + 时间 + 物理条件，回归位移
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim * 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),  # 输出 3D 位移向量
        )

    def forward(self, x_t, t, phys):
        """
        x_t : (B, N, 3) 当前时刻的点坐标
        t   : (B,)      时间步，标量
        phys: (B, N, phys_dim) 物理条件
        返回 v : (B, N, 3) 预测位移场
        """
        B, N, _ = x_t.shape
        # 坐标特征
        f_xyz = self.xyz_enc(x_t)                       # (B, N, feat_dim)
        # 时间特征，广播到每个点
        f_t = self.time_mlp(self.time_emb(t))           # (B, feat_dim)
        f_t = f_t[:, None, :].expand(B, N, self.feat_dim)
        # 物理条件特征
        f_c = self.phys_enc(phys)                       # (B, N, feat_dim)
        # 拼接后回归位移
        h = torch.cat([f_xyz, f_t, f_c], dim=-1)        # (B, N, 3*feat_dim)
        v = self.trunk(h)                               # (B, N, 3)
        return v

    @torch.no_grad()
    def sample(self, x0, phys, steps=10):
        """推理：从残缺点 x0 出发，用欧拉法积分 ODE 得到补全点。

        x0  : (B, N, 3) 残缺点云
        phys: (B, N, phys_dim) 物理条件
        steps: 积分步数，越大越精细
        """
        x = x0
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            v = self.forward(x, t, phys)  # 当前位移场
            x = x + v * dt                # 欧拉前进一步
        return x
