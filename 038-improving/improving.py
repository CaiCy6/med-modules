# -*- coding: utf-8 -*-
"""
【医学图像分割模块】Improving Cross —— Improving Cross-Site Whole-Heart Segmentation

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiteRoutingAdapter(nn.Module):
    """
    站点/模态特征条件化路由适配块（即插即用）。
    输入:  (B, C, D, H, W)  或  (B, C, H, W)
    输出:  与输入同形状
    只做通道级调制，不改空间尺寸和通道数。
    """

    def __init__(self, channels, num_experts=4, reduction=8):
        super().__init__()
        self.channels = channels
        self.num_experts = num_experts

        # 1) 域描述子：全局平均池化 -> (B, C)，无需参数
        # 2) 共享的指纹压缩层，把 C 维压到 C//reduction，降低后续计算量
        hidden = max(channels // reduction, 8)  # 防止通道太小时压成 0
        self.fingerprint = nn.Sequential(
            nn.Linear(channels, hidden),   # 压缩域指纹
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),   # 还原回通道维度，供调制使用
        )

        # 3) 路由头：从域指纹预测每个专家的权重 (B, num_experts)
        self.router = nn.Linear(channels, num_experts)

        # 4) 每个专家一组逐通道 scale/shift 参数，初始化为恒等映射
        #    scale 初始为 1，shift 初始为 0，保证插入初期不破坏原网络
        self.expert_scale = nn.Parameter(torch.ones(num_experts, channels))
        self.expert_shift = nn.Parameter(torch.zeros(num_experts, channels))

        # 5) 残差缩放系数，可学习，初始很小，让模块从「接近恒等」开始
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 记录输入形状，兼容 2D / 3D
        is_3d = x.dim() == 5

        # 全局平均池化得到域指纹: (B, C)
        if is_3d:
            desc = x.mean(dim=(2, 3, 4))
        else:
            desc = x.mean(dim=(2, 3))

        # 指纹变换，得到条件向量
        cond = self.fingerprint(desc)              # (B, C)

        # 路由权重: (B, num_experts)，softmax 保证加权和为 1
        route = F.softmax(self.router(desc), dim=1)  # (B, num_experts)

        # 按路由权重融合各专家的 scale / shift
        #   route: (B, E), expert_scale: (E, C) -> (B, C)
        scale = route @ self.expert_scale          # (B, C)
        shift = route @ self.expert_shift          # (B, C)

        # 用条件向量进一步调制 scale/shift，让调制依赖当前输入
        scale = scale * (1.0 + cond)               # 条件化增益
        shift = shift + cond                       # 条件化偏置

        # 变形到 (B, C, 1, 1[, 1]) 以便广播到特征图
        if is_3d:
            scale = scale.view(-1, self.channels, 1, 1, 1)
            shift = shift.view(-1, self.channels, 1, 1, 1)
        else:
            scale = scale.view(-1, self.channels, 1, 1)
            shift = shift.view(-1, self.channels, 1, 1)

        # 仿射调制 + 残差连接，gamma 控制模块整体强度
        out = x * scale + shift
        return x + self.gamma * out
