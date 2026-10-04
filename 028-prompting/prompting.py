# -*- coding: utf-8 -*-
"""
【医学图像分割模块】Prompting Segment Anything Model with Do —— Prompting Segment Anything Model with Domain-Adaptive Protot

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainAdaptivePrototypePrompt(nn.Module):
    """
    域自适应原型提示块（DAPP 的可插拔版本）。
    输入: 主干特征 x, 形状 [B, C, H, W]
    输出: 被 prompt 调制后的特征, 形状 [B, C, H, W]
    """

    def __init__(self, channels, num_prototypes=8, prompt_dim=128, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_prototypes = num_prototypes

        # 原型库: 每个原型是一段可学习向量, 维度等于特征通道数
        # 用 nn.Parameter 而不是 buffer, 因为要参与梯度更新
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, channels) * 0.02)

        # 把特征投影到匹配空间, 避免直接在高维通道上算相似度
        self.feat_proj = nn.Conv2d(channels, prompt_dim, kernel_size=1)

        # 把聚合后的原型投影成 prompt token
        self.prompt_proj = nn.Sequential(
            nn.Linear(channels, prompt_dim),
            nn.GELU(),
            nn.Linear(prompt_dim, prompt_dim),
        )

        # cross-attention: query 来自特征, key/value 来自 prompt
        self.attn = nn.MultiheadAttention(
            embed_dim=prompt_dim, num_heads=num_heads, batch_first=True
        )

        # 输出投影回原通道, 并做残差
        self.out_proj = nn.Linear(prompt_dim, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1) 特征投影到匹配空间: [B, C, H, W] -> [B, D, H, W]
        f = self.feat_proj(x)

        # 2) 展平成 token 序列: [B, D, H*W] -> [B, H*W, D]
        f_flat = f.flatten(2).transpose(1, 2)

        # 3) 原型投影到同一匹配空间: [P, C] -> [P, D]
        proto = self.prototypes  # [P, C]

        # 4) 计算每个 token 与每个原型的相似度: [B, H*W, P]
        #    用余弦相似度, 对尺度不敏感, 训练更稳
        f_norm = F.normalize(f_flat, dim=-1)
        p_norm = F.normalize(proto, dim=-1)
        sim = torch.einsum("bnd,pd->bnp", f_norm, p_norm)  # [B, N, P]

        # 5) softmax 得到软分配权重, 温度系数控制锐度
        weight = F.softmax(sim * 10.0, dim=-1)  # [B, N, P]

        # 6) 加权聚合原型, 得到每个位置的域自适应原型: [B, N, C]
        #    weight: [B, N, P], proto: [P, C]
        adaptive_proto = torch.einsum("bnp,pc->bnc", weight, proto)

        # 7) 生成 prompt token: [B, N, D]
        prompt = self.prompt_proj(adaptive_proto)

        # 8) cross-attention: 特征作 query, prompt 作 key/value
        attn_out, _ = self.attn(f_flat, prompt, prompt)  # [B, N, D]

        # 9) 投影回原通道并 reshape 回特征图: [B, N, C] -> [B, C, H, W]
        attn_out = self.out_proj(attn_out)  # [B, N, C]
        attn_out = attn_out.transpose(1, 2).reshape(B, C, H, W)

        # 10) 残差 + LayerNorm(在通道维上), 保证训练稳定
        out = x + attn_out
        out = out.permute(0, 2, 3, 1)  # [B, H, W, C]
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)  # [B, C, H, W]
        return out
