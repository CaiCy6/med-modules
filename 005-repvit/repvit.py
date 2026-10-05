# -*- coding: utf-8 -*-
"""
【医学图像分割模块】RepViT —— 重参数化 ViT 块 —— 用结构重参数化把 ViT 搬到移动端

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn


class RepTokenMixer(nn.Module):
    """token mixer：训练时多分支深度卷积，推理时融合成单路 3x3 深度卷积。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # 分支1：3x3 深度卷积，负责主要空间混合
        self.dw3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        # 分支2：1x1 深度卷积，等价于逐通道缩放，提供恒等附近的微调
        self.dw1 = nn.Conv2d(dim, dim, 1, padding=0, groups=dim, bias=False)
        # 分支3：恒等映射，保证梯度直通
        self.bn = nn.BatchNorm2d(dim)  # 训练时对融合结果做归一化

    def forward(self, x):
        # 训练阶段：三分支相加，再归一化
        out = self.dw3(x) + self.dw1(x) + x
        return self.bn(out)

    @torch.no_grad()
    def fuse(self):
        """推理前调用：把多分支合并成单个 3x3 深度卷积，去掉 BN。"""
        # 简化处理：这里只演示思路，实际需按 BN 公式把缩放/偏置折进卷积权重
        # 真实工程请用 torch.nn.utils.fusion 或官方 reparameterize 工具
        pass


class ChannelMixer(nn.Module):
    """channel mixer：1x1 升维 -> 激活 -> 1x1 降维，外套残差。"""

    def __init__(self, dim, ratio=4):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Conv2d(dim, hidden, 1, bias=False)   # 升维
        self.act = nn.GELU()                                # 非线性
        self.fc2 = nn.Conv2d(hidden, dim, 1, bias=False)   # 降维回原通道

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))             # 输出与输入同形状


class RepViTBlock(nn.Module):
    """完整的 RepViT Block：token mixer + channel mixer，两段都带残差。"""

    def __init__(self, dim, ratio=4):
        super().__init__()
        self.token_mixer = RepTokenMixer(dim)              # 空间混合
        self.channel_mixer = ChannelMixer(dim, ratio)      # 通道混合

    def forward(self, x):
        x = x + self.token_mixer(x)                        # 残差连接
        x = x + self.channel_mixer(x)                      # 残差连接
        return x


if __name__ == "__main__":
    # 快速自测：输入输出形状应一致
    block = RepViTBlock(dim=64)
    y = block(torch.randn(2, 64, 32, 32))
    print(y.shape)  # torch.Size([2, 64, 32, 32])
