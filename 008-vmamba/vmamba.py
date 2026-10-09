# -*- coding: utf-8 -*-
"""
【医学图像分割模块】VMamba —— 视觉状态空间（SS2D 四向扫描）—— 把 Mamba 的线性复杂度搬进视觉主干

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedSS2D(nn.Module):
    """简化版 2D 选择性扫描模块（教学用，非官方实现）"""

    def __init__(self, dim, d_state=16):
        super().__init__()
        self.dim = dim
        self.d_state = d_state

        # 用 1x1 卷积把输入投影成扫描所需的几组量：
        # x 是主干信息，delta 控制状态更新步长，B/C 是状态空间的门控
        self.proj = nn.Conv2d(dim, dim * 3, kernel_size=1)
        # 把 delta 映射到正数，保证递推稳定
        self.dt_proj = nn.Linear(dim, dim)
        # 四条扫描路径的输出融合
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def _scan_one_direction(self, x):
        """沿一个方向做一维选择性扫描的简化递推。
        x: (B, C, H, W)，这里按行优先拉平成序列处理。
        """
        B, C, H, W = x.shape
        # 拉平成序列：(B, C, L)，L = H*W
        seq = x.flatten(2)                      # (B, C, L)
        seq = seq.transpose(1, 2)               # (B, L, C)

        # 简化：用一个可学习的标量衰减代替完整的 A 矩阵
        # 真实实现里 A 是 d_state 维的对角矩阵
        decay = torch.sigmoid(self.dt_proj(seq))  # (B, L, C)

        # 顺序递推：h_t = decay * h_{t-1} + (1 - decay) * x_t
        # 这是状态空间递推的极简形式，保留“选择性”的直觉
        h = torch.zeros_like(seq[:, 0])        # (B, C)
        outs = []
        for t in range(seq.shape[1]):
            d = decay[:, t]                     # (B, C)
            h = d * h + (1 - d) * seq[:, t]     # 状态更新
            outs.append(h)
        out = torch.stack(outs, dim=1)          # (B, L, C)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out

    def forward(self, x):
        B, C, H, W = x.shape

        # 生成四条扫描路径：原图 + 三种翻转/转置组合
        paths = [
            x,                                  # 左上 -> 右下
            torch.flip(x, dims=[2]),            # 上下翻转
            torch.flip(x, dims=[3]),            # 左右翻转
            torch.flip(x, dims=[2, 3]),         # 中心对称
        ]

        # 每条路径各自扫描，再翻回来对齐到原坐标系
        outs = []
        for i, p in enumerate(paths):
            o = self._scan_one_direction(p)
            if i == 1:
                o = torch.flip(o, dims=[2])
            elif i == 2:
                o = torch.flip(o, dims=[3])
            elif i == 3:
                o = torch.flip(o, dims=[2, 3])
            outs.append(o)

        # 四条路径求和，得到融合了四向上下文的特征
        out = sum(outs)
        return self.out_proj(out)
