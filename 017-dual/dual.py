# -*- coding: utf-8 -*-
"""
【医学图像分割模块】Dual —— Dual-Adaptive SAM3: Hierarchical Routing over Low-Rank Exper

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankExpert(nn.Module):
    """单个低秩专家：先降维再升维，等价于一个低秩约束的线性变换。"""

    def __init__(self, dim, rank):
        super().__init__()
        # 降维投影：dim -> rank，把特征压到低秩子空间
        self.down = nn.Linear(dim, rank, bias=False)
        # 升维投影：rank -> dim，再映射回原空间
        self.up = nn.Linear(rank, dim, bias=False)
        # 低秩分支初始化为接近 0，保证插入时不破坏原网络行为
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        # x: (B, N, C)，N 为空间位置数（H*W 或 token 数）
        return self.up(self.down(x))


class Dual(nn.Module):
    """低秩专家 + 层级路由的即插即用模块。

    输入输出形状一致 (B, C, H, W)，可直接替换任意卷积/注意力块。
    """

    def __init__(self, dim, num_experts=4, rank=8, reduction=4):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts

        # 一组低秩专家，每个专家参数量为 2*dim*rank，远小于全量微调
        self.experts = nn.ModuleList(
            [LowRankExpert(dim, rank) for _ in range(num_experts)]
        )

        # 路由网络：全局池化 -> 小 MLP -> 每个专家一个权重
        hidden = max(dim // reduction, num_experts)
        self.router = nn.Sequential(
            nn.Linear(dim, hidden),   # 压缩到隐层
            nn.GELU(),                # 非线性，让路由能表达输入相关的偏好
            nn.Linear(hidden, num_experts),  # 输出 num_experts 个 logit
        )

        # 输出前的归一化，稳定训练
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # 记录原始输入，用于残差
        identity = x

        # 支持 (B, C, H, W) 和 (B, N, C) 两种布局
        if x.dim() == 4:
            B, C, H, W = x.shape
            # 转成 (B, N, C) 以便用 Linear 处理
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        else:
            B, N, C = x.shape
            H = W = None

        # 路由：对空间维做平均池化，得到每个样本的全局描述子
        pooled = x.mean(dim=1)                # (B, C)
        logits = self.router(pooled)          # (B, num_experts)
        weights = F.softmax(logits, dim=-1)   # (B, num_experts)，稠密权重

        # 所有专家输出堆叠：每个专家对 (B, N, C) 做低秩变换
        expert_outs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # (B, num_experts, N, C)

        # 用路由权重对专家输出做加权求和（稠密组合，无 top-k 分发）
        w = weights.view(B, self.num_experts, 1, 1)  # 广播用
        out = (expert_outs * w).sum(dim=1)           # (B, N, C)

        # 归一化 + 残差，保证模块可安全插入
        out = self.norm(out) + identity

        # 还原回原始布局
        if H is not None:
            out = out.transpose(1, 2).reshape(B, C, H, W)
        return out
