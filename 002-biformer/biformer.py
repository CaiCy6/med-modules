# -*- coding: utf-8 -*-
"""
【医学图像分割模块】BiFormer / 双层路由注意力（CVPR 2023）—— 只算"该算的地方"的稀疏注意力

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn

class BiLevelRoutingAttention(nn.Module):
    """BiFormer 双层路由注意力（简化教学版）。
    ① 区域级路由：每个区域只保留最相关的 topk 个区域；
    ② 区域级 token-to-token 注意力：只在选中的区域里算注意力。
    """
    def __init__(self, dim, num_heads=8, n_win=7, topk=4):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.dim = dim
        self.num_heads = num_heads
        self.n_win = n_win                      # 把特征图切成 n_win × n_win 个区域
        self.topk = topk                        # 每个区域只和 topk 个最相关区域做注意力
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5      # 缩放因子，防止点积过大
        self.qkv = nn.Linear(dim, dim * 3)      # 一次投影出 q/k/v
        self.proj = nn.Linear(dim, dim)         # 输出投影
        # 深度卷积做局部位置编码（LePE），补回被稀疏化丢失的局部信息
        self.lepe = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)                   # 假设方形特征图
        n = self.n_win
        rs = (H // n) * (W // n)                # 每个区域内的 token 数

        # ① 生成 q/k/v：[B, heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # ② 按区域重塑：[B, heads, R, rs, head_dim]，R = n*n 个区域
        def to_region(t):
            return t.reshape(B, self.num_heads, n * n, rs, self.head_dim)
        q_r, k_r, v_r = to_region(q), to_region(k), to_region(v)

        # ③ 区域"代表向量" = 区域内 token 平均，用来算区域间相关度
        q_rep = q_r.mean(dim=3)                 # [B, heads, R, head_dim]
        k_rep = k_r.mean(dim=3)                 # [B, heads, R, head_dim]
        aff = (q_rep * self.scale) @ k_rep.transpose(-1, -2)   # [B, heads, R, R] 区域相关度

        # ④ 双层路由的关键：每个区域只留 topk 个最相关区域
        topk = min(self.topk, n * n)
        idx = aff.topk(topk, dim=-1).indices    # [B, heads, R, topk]

        # ⑤ 按路由索引，把选中的 k/v 区域聚起来：[B, heads, R, topk, rs, head_dim]
        def gather_region(t):
            d = t.shape[-1]
            t_exp = t.unsqueeze(3).expand(B, self.num_heads, n * n, topk, rs, d)
            index = idx[..., None, None].expand(B, self.num_heads, n * n, topk, rs, d)
            return torch.gather(t_exp, 2, index)
        k_g = gather_region(k_r).reshape(B, self.num_heads, n * n, topk * rs, self.head_dim)
        v_g = gather_region(v_r).reshape(B, self.num_heads, n * n, topk * rs, self.head_dim)

        # ⑥ 只在"本区域 token × 选中区域 key"之间做注意力
        attn = (q_r * self.scale) @ k_g.transpose(-1, -2)          # [B, heads, R, rs, topk*rs]
        attn = attn.softmax(dim=-1)
        out = attn @ v_g                                            # [B, heads, R, rs, head_dim]

        # ⑦ 还原回 token 序列：[B, N, C]
        out = out.reshape(B, self.num_heads, N, self.head_dim).permute(0, 2, 1, 3).reshape(B, N, C)

        # ⑧ 加局部位置编码（LePE）+ 输出投影
        img = out.transpose(1, 2).reshape(B, C, H, W)              # [B, C, H, W]
        img = img + self.lepe(img)                                  # 补局部位置信息
        out = img.flatten(2).transpose(1, 2)                        # [B, N, C]
        out = self.proj(out)
        return out
