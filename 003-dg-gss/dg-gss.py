# -*- coding: utf-8 -*-
"""
【医学图像分割模块】DG-GSS / 方向-组图选择性扫描（arXiv 2026）—— 让轻量分割网络"扫"得更聪明

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn

class SelectiveScan2D(nn.Module):
    """最简版 2D 选择性扫描：对 4 个方向做"逐位置的状态递推"，再合并。
    真实实现（VSS/VMamba）会做并行扫描与门控；这里用可读的递推表达思路。
    """
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)          # 逐位置输入投影
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)  # 局部前处理

    def _scan(self, x, order):
        # order: 'lr' | 'rl' | 'tb' | 'bt'，按行/列正反序做状态递推
        B, C, H, W = x.shape
        if order in ('lr', 'rl'):
            seq = x.permute(0, 1, 2, 3).reshape(B, C, H * W)         # 逐行
        else:
            seq = x.permute(0, 1, 3, 2).reshape(B, C, H * W)         # 逐列
        if order in ('rl', 'bt'):
            seq = seq.flip(-1)                                        # 反向
        h = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(seq.shape[-1]):                                # 逐位置递推（教学用，慢）
            h = 0.9 * h + 0.1 * seq[..., t]                           # 简化的状态更新（真实为 A/B/C/Δ 参数化）
            outs.append(h)
        y = torch.stack(outs, dim=-1)
        if order in ('rl', 'bt'):
            y = y.flip(-1)
        return y

    def forward(self, x):                        # x: [B, C, H, W]
        x = self.dw(x)
        dirs = ['lr', 'rl', 'tb', 'bt']
        ys = [self._scan(x, d) for d in dirs]    # 四个方向 [B, C, HW]
        y = torch.stack(ys, dim=1)               # [B, 4, C, HW]：把"方向"当一维
        y = y.reshape(x.shape[0], 4, x.shape[1], x.shape[2], x.shape[3])
        return y                                 # 返回"每方向"的响应，交给图模块


class DirectionGroupGraph(nn.Module):
    """方向-组图消息传递：节点 = (方向, 分组)，边 = 同方向 / 同分组 / 自连接。
    用轻量注意力在图上聚合，实现方向与分组之间的结构化信息交换。
    """
    def __init__(self, dim, n_groups=4, n_dirs=4):
        super().__init__()
        self.n_dirs, self.n_groups = n_dirs, n_groups
        self.group_dim = dim // n_groups
        self.node_proj = nn.Linear(dim, dim)
        self.gat = nn.MultiheadAttention(dim, num_heads=n_dirs, batch_first=True)

    def build_adj(self, dev):
        """构造邻接矩阵：i、j 节点相连若 同方向 或 同分组 或 i==j。"""
        n = self.n_dirs * self.n_groups
        A = torch.eye(n, device=dev)
        for i in range(n):
            di, gi = divmod(i, self.n_groups)          # 节点 i 的（方向, 分组）
            for j in range(n):
                dj, gj = divmod(j, self.n_groups)
                if di == dj or gi == gj:                # 同方向 或 同分组 → 连边
                    A[i, j] = 1
        return A

    def forward(self, per_dir):                        # per_dir: [B, 4, C, H, W]
        B, D, C, H, W = per_dir.shape
        # 每个方向先按通道分组，得到 "方向×分组" 的节点特征
        x = per_dir.reshape(B, D, self.n_groups, self.group_dim, H, W)
        nodes = x.mean(dim=(-1, -2)).reshape(B, D * self.n_groups, self.group_dim)  # [B, N, gd]
        nodes = self.node_proj(torch.nn.functional.pad(nodes, (0, C - self.group_dim)))
        A = self.build_adj(per_dir.device)             # [N, N]
        mask = (A == 0)                                # 用作 attention 掩码（不相连则屏蔽）
        agg, _ = self.gat(nodes, nodes, nodes, attn_mask=mask)   # 图上注意力聚合
        return agg.reshape(B, D, self.n_groups, self.group_dim)  # 回到 (方向, 分组)


class DGGSSBlock(nn.Module):
    """DG-GSS Block：选择性扫描 → 方向-组图交互 → 融合回特征图。"""
    def __init__(self, dim, n_groups=4):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.scan = SelectiveScan2D(dim)
        self.graph = DirectionGroupGraph(dim, n_groups=n_groups)
        self.fuse = nn.Conv2d(dim, dim, 1)             # 方向/分组融合
        self.out = nn.Conv2d(dim, dim, 1)              # 输出投影

    def forward(self, x):                              # x: [B, C, H, W]
        residual = x
        x = self.norm(x)
        per_dir = self.scan(x)                         # [B, 4, C, H, W]
        agg = self.graph(per_dir)                      # [B, 4, G, C/G] 图聚合结果
        # 把"方向×分组"的聚合结果加权回各方向特征
        w = agg.reshape(agg.shape[0], agg.shape[1], -1)
        w = torch.softmax(w, dim=-1)                   # 方向/分组权重
        y = (per_dir.flatten(2) * w.mean(-1, keepdim=True)).reshape_as(per_dir)
        y = y.sum(dim=1)                               # 合并方向 → [B, C, H, W]
        y = self.fuse(y)
        return residual + self.out(y)                  # 残差接住，即插即用
