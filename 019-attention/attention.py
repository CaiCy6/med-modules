# -*- coding: utf-8 -*-
"""
【医学图像分割模块】Attention —— Attention-Based Prototype Calibration for Multi-Rater Few-Sh

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPrototypeCalibration(nn.Module):
    """
    注意力原型校准模块（即插即用）。
    输入:
        consensus: [B, C]        共识原型（所有标注者平均或融合得到）
        rater_protos: [B, N, C]  N 个标注者各自的原型
    输出:
        calibrated: [B, N, C]    校准后的 rater-specific 原型
    """

    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim 必须能被 num_heads 整除"

        # 共识原型投影成 query：用来「查询」每个标注者的偏差
        self.q_proj = nn.Linear(dim, dim)
        # 标注者原型投影成 key / value：提供偏差信息
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        # 输出投影，把多头结果融合回原维度
        self.out_proj = nn.Linear(dim, dim)
        # 偏差缩放系数，初始化为 0，保证训练初期等价于「不校准」
        self.gamma = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, consensus, rater_protos):
        B, N, C = rater_protos.shape

        # 共识原型扩展成 query: [B, 1, C] -> [B, H, 1, D]
        q = self.q_proj(consensus).view(B, 1, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [B, H, 1, D]

        # 标注者原型投影成 key / value: [B, N, C] -> [B, H, N, D]
        k = self.k_proj(rater_protos).view(B, N, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)  # [B, H, N, D]
        v = self.v_proj(rater_protos).view(B, N, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)  # [B, H, N, D]

        # 缩放点积注意力：共识原型对每个标注者原型分配权重
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)          # [B, H, 1, N]
        attn = self.dropout(attn)

        # 加权聚合标注者信息: [B, H, 1, D]
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, 1, C)  # [B, 1, C]
        out = self.out_proj(out)

        # 残差式校准：以共识原型为锚点，加上带缩放的注意力偏差
        # gamma 初始为 0，训练中自适应学习校准强度
        calibrated = rater_protos + self.gamma * out  # [B, N, C]
        return calibrated
