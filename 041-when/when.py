# -*- coding: utf-8 -*-
"""
【医学图像分割模块】When Adaptation Hurts —— When Adaptation Hurts: Connecting Representational Drift to

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RepresentationalDriftConstraint(nn.Module):
    """
    表征漂移约束块 (RDC)
    即插即用：挂在任意一对 (student_feat, teacher_feat) 上，
    输出一个标量对齐损失，加回总 loss 即可。
    推理时不需要调用本模块。
    """

    def __init__(self, layer_weights=None, mode="cosine", eps=1e-6):
        super().__init__()
        # 逐层权重：浅层大、深层小，默认给 4 层 encoder 的配置
        # 如果层数不同，传入等长的 list 即可
        if layer_weights is None:
            layer_weights = [1.0, 0.75, 0.5, 0.25]
        # 注册成 buffer，随模型一起搬到 GPU，但不参与梯度更新
        self.register_buffer(
            "layer_weights",
            torch.tensor(layer_weights, dtype=torch.float32),
        )
        self.mode = mode          # "cosine" 或 "mse"
        self.eps = eps            # 防止除零

    def _pair_loss(self, s, t):
        """计算单层 student/teacher 特征的对齐损失"""
        # 特征形状通常是 (B, C, H, W)，先展平成 (B, C, H*W)
        if s.dim() == 4:
            s = s.flatten(2)
            t = t.flatten(2)
        # 沿通道维做 L2 归一化，消除尺度差异
        s = F.normalize(s, dim=1, eps=self.eps)
        t = F.normalize(t, dim=1, eps=self.eps)

        if self.mode == "cosine":
            # 余弦相似度：越接近 1 越好，损失取 1 - cos
            cos = (s * t).sum(dim=1)          # (B, H*W)
            return (1.0 - cos).mean()
        else:
            # 归一化后的 MSE，等价于 2 - 2*cos 的单调变换
            return F.mse_loss(s, t)

    def forward(self, student_feats, teacher_feats):
        """
        student_feats: list[Tensor]，可训练分支各层特征
        teacher_feats: list[Tensor]，冻结分支各层特征（同层数、同空间尺寸）
        返回：加权后的标量对齐损失
        """
        assert len(student_feats) == len(teacher_feats), \
            "student/teacher 特征层数必须一致"
        n = len(student_feats)
        # 若传入层数与预设权重长度不符，退化为均匀权重
        if n != self.layer_weights.numel():
            weights = torch.ones(n, device=student_feats[0].device)
        else:
            weights = self.layer_weights[:n].to(student_feats[0].device)

        total = 0.0
        for i, (s, t) in enumerate(zip(student_feats, teacher_feats)):
            # teacher 特征不参与梯度，detach 掉更省显存
            total = total + weights[i] * self._pair_loss(s, t.detach())
        # 按权重和归一化，保证不同层数下损失量级可比
        return total / weights.sum()
