# -*- coding: utf-8 -*-
"""
【医学图像分割模块】From Adaptation to Generalization —— From Adaptation to Generalization: Adaptive Visual Prompting

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class APEXBlock(nn.Module):
    """
    自适应 Prompt 提取模块（APEX-Block）
    输入: 特征图 x, 形状 [B, C, H, W]
    输出: 被 prompt 调制后的特征图, 形状 [B, C, H, W]
    """

    def __init__(self, channels, num_prompts=16, prompt_dim=None, mode="add"):
        super().__init__()
        # prompt 维度默认等于特征通道数，方便直接做加性注入
        prompt_dim = prompt_dim or channels
        self.channels = channels
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim
        self.mode = mode  # 注入方式: "add" 或 "film"

        # 可学习的 prompt memory: [N, D]，训练时和网络一起优化
        self.memory = nn.Parameter(torch.randn(num_prompts, prompt_dim) * 0.02)

        # 把输入特征投影成检索 query，维度对齐到 prompt_dim
        self.query_proj = nn.Linear(channels, prompt_dim)

        # 温度系数，控制检索的"尖锐"程度，可学习
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        if mode == "film":
            # FiLM 模式: 从 prompt 生成 scale 和 shift
            self.film = nn.Linear(prompt_dim, channels * 2)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1) 全局平均池化得到输入描述子 [B, C]
        desc = x.mean(dim=(2, 3))

        # 2) 投影成 query [B, D]
        query = self.query_proj(desc)

        # 3) 归一化后与 memory 做点积相似度 [B, N]
        query = F.normalize(query, dim=-1)
        mem = F.normalize(self.memory, dim=-1)
        logits = query @ mem.t() * self.logit_scale

        # 4) softmax 得到每个 prompt 槽位的权重
        weights = F.softmax(logits, dim=-1)

        # 5) 加权聚合出输入专属 prompt [B, D]
        prompt = weights @ self.memory

        # 6) 注入到特征上
        if self.mode == "add":
            # 加性注入: prompt 广播到空间维度后相加
            out = x + prompt[:, :, None, None]
        else:
            # FiLM 注入: prompt 生成逐通道 scale/shift
            gamma_beta = self.film(prompt)          # [B, 2C]
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gamma = gamma[:, :, None, None]
            beta = beta[:, :, None, None]
            out = x * (1 + gamma) + beta

        return out
