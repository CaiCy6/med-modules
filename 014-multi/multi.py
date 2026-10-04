# -*- coding: utf-8 -*-
"""
【医学图像分割模块】Multi —— Multi-Stage Prompt-Guided Feature Modulation for Generalizab

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn


class Multi(nn.Module):
    """
    提示引导的特征调制块（Prompt-Guided Feature Modulation Block）。
    输入输出形状完全一致，可直接插在任意卷积层之后。
    """

    def __init__(self, channels, num_prompts=8, reduction=4):
        """
        channels   : 输入特征图的通道数，必须和主干该层一致
        num_prompts: 提示向量的个数，控制调制系数的表达能力
        reduction  : 生成调制系数时的通道压缩比，控制参数量
        """
        super().__init__()
        self.channels = channels
        self.num_prompts = num_prompts

        # 可学习的提示向量组，形状 [num_prompts, channels]
        # 每个提示向量长度等于通道数，代表一种"通道重要性模式"
        self.prompts = nn.Parameter(torch.randn(num_prompts, channels) * 0.02)

        # 把提示向量聚合成通道级调制系数的轻量网络
        # 先降维再升维，减少参数，同时引入非线性
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),   # 降维
            nn.ReLU(inplace=True),         # 非线性
            nn.Linear(hidden, channels),   # 升回原通道数
            nn.Sigmoid(),                  # 输出 0~1 的调制系数
        )

        # 残差补偿卷积，对调制后的特征做一次轻量修正
        # 用 1x1 卷积，不改变空间尺寸，只做通道混合
        self.compensate = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)  # 稳定训练

    def forward(self, x):
        """
        x: [B, C, H, W]
        返回: [B, C, H, W]，形状与输入完全一致
        """
        B, C, H, W = x.shape

        # 1) 用提示向量的均值作为"全局通道先验"
        #    这里对 num_prompts 维度求均值，得到 [C] 的向量
        #    也可以改成加权求和，但均值最稳、最不容易过拟合
        prior = self.prompts.mean(dim=0)          # [C]

        # 2) 通过轻量网络生成通道调制系数
        #    prior 扩展成 batch 维度，得到 [B, C]
        scale = self.fc(prior.unsqueeze(0).expand(B, -1))  # [B, C]

        # 3) 把系数 reshape 成 [B, C, 1, 1]，方便逐通道相乘
        scale = scale.view(B, C, 1, 1)

        # 4) 通道调制：原特征逐通道乘以系数
        modulated = x * scale

        # 5) 残差补偿：调制后的特征过 1x1 卷积 + BN
        comp = self.bn(self.compensate(modulated))

        # 6) 残差相加，保证信息不丢失
        return x + comp


# 快速自测：形状是否一致
if __name__ == "__main__":
    block = Multi(channels=64)
    feat = torch.randn(2, 64, 32, 32)
    out = block(feat)
    print(out.shape)  # 期望 torch.Size([2, 64, 32, 32])
