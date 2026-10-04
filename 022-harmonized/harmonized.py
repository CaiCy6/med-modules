# -*- coding: utf-8 -*-
"""
【医学图像分割模块】Harmonized Feature Conditioning and Freq —— Harmonized Feature Conditioning and Frequency-Prompt Persona

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyPrompt(nn.Module):
    """频域个性化分支：在频谱上做可学习的增益调制。"""

    def __init__(self, channels, prompt_size=8):
        super().__init__()
        # 可学习的频率 prompt，形状 [C, prompt_size, prompt_size]
        # 用一个小尺寸频谱模板，再插值到实际特征尺寸，避免参数量随分辨率爆炸
        self.prompt = nn.Parameter(torch.ones(channels, prompt_size, prompt_size))
        self.prompt_size = prompt_size
        # 逐通道的可学习缩放，初始为 0，保证插入初期不破坏原网络行为
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # 1) 实数 FFT，得到复数频谱；rfft2 只返回非冗余的一半，省显存
        x_freq = torch.fft.rfft2(x, norm='ortho')  # [B, C, H, W//2+1]

        # 2) 把可学习 prompt 插值到当前频谱尺寸
        prompt = self.prompt.unsqueeze(0)  # [1, C, p, p]
        prompt = F.interpolate(
            prompt, size=x_freq.shape[-2:], mode='bilinear', align_corners=False
        )  # [1, C, H, W//2+1]

        # 3) 频谱乘以 prompt（复数乘法，prompt 为实数增益）
        x_freq = x_freq * prompt

        # 4) 逆变换回空域
        x_out = torch.fft.irfft2(x_freq, s=(H, W), norm='ortho')

        # 5) 残差回注，scale 初始为 0，训练中逐渐学出频域修正量
        return x + self.scale * x_out


class Harmonizer(nn.Module):
    """即插即用模块：特征条件化 + 频域个性化。"""

    def __init__(self, channels, reduction=8, prompt_size=8):
        super().__init__()
        # ---- 条件化分支：从全局上下文预测 gamma / beta ----
        hidden = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)          # 全局平均池化，拿到通道描述子
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),          # 降维
            nn.GELU(),
            nn.Conv2d(hidden, channels * 2, 1),      # 一次性预测 gamma 和 beta
        )
        # 初始化为接近恒等：gamma≈1, beta≈0
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

        # ---- 频域分支 ----
        self.freq = FrequencyPrompt(channels, prompt_size)

        # 输出前的轻量融合，稳定训练
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        # x: [B, C, H, W]
        # 1) 条件化：预测逐通道 gamma / beta
        ctx = self.gap(x)                            # [B, C, 1, 1]
        params = self.fc(ctx)                        # [B, 2C, 1, 1]
        gamma, beta = params.chunk(2, dim=1)         # 各 [B, C, 1, 1]
        gamma = 1.0 + gamma                          # 以 1 为中心，初始接近恒等
        x_cond = gamma * x + beta                    # FiLM 调制

        # 2) 频域个性化：抑制设备/采集伪影
        x_freq = self.freq(x_cond)

        # 3) 融合并残差回注，保证梯度顺畅
        out = self.proj(x_freq)
        return x + out
