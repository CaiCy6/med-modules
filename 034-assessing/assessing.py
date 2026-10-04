# -*- coding: utf-8 -*-
"""
【医学图像分割模块】Assessing nnU —— Assessing nnU-Net Generalization across Brain Tumor Populati

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn


class TestTimeMirroring(nn.Module):
    """
    测试时镜像集成（TTM）即插即用包装器。

    用法：把任意已训练好的分割网络 model 传进来，
    推理时直接调用 forward，内部自动做镜像平均。
    训练阶段建议关闭（self.training 为 True 时直接走原网络）。
    """

    def __init__(self, model, dims=(-1,), enable_in_train=False):
        super().__init__()
        self.model = model                 # 被包装的任意分割网络（U-Net / nnU-Net / SwinUNet 等）
        self.dims = dims                   # 沿哪些空间维度翻转，脑部常用最后一维（左右轴）
        self.enable_in_train = enable_in_train  # 训练时是否也启用，默认关闭省算力

    def _flip(self, x):
        # 按指定维度翻转，torch.flip 不改变张量形状，只重排元素
        return torch.flip(x, dims=self.dims)

    @torch.no_grad()
    def forward(self, x):
        # 训练模式下（且未强制开启）直接走原网络，避免拖慢训练
        if self.training and not self.enable_in_train:
            return self.model(x)

        # 分支 A：原始输入直接前向
        out_a = self.model(x)

        # 分支 B：先翻转输入，前向后再翻回来，保证与 out_a 空间对齐
        out_b = self._flip(self.model(self._flip(x)))

        # 两分支逐元素平均，得到镜像集成结果
        return (out_a + out_b) / 2.0
