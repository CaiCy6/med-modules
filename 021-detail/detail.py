# -*- coding: utf-8 -*-
"""
【医学图像分割模块】Detail Consistent Stage —— Detail Consistent Stage-Wise Distillation for Efficient 3D M

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWaveletDecompose(nn.Module):
    """
    固定 Haar 小波分解，不参与训练。
    输入: (B, C, D, H, W) 或 (B, C, H, W)
    输出: LL, LH, HL, HH 四个分量，空间尺寸各减半。
    用 depthwise conv 实现，等价于对每个通道独立做小波变换。
    """

    def __init__(self):
        super().__init__()
        # 3D Haar 的四个滤波器，形状 (2,2,2)，分别对应 LL / LH / HL / HH
        # 这里用 2x2x2 的核，stride=2，实现一次下采样分解
        ll = torch.ones(1, 1, 2, 2, 2) / 8.0          # 低频近似：全 1 平均
        lh = torch.tensor([[[[[1, 1], [1, 1]], [[-1, -1], [-1, -1]]]]],
                          dtype=torch.float32) / 8.0   # 沿深度方向的差分
        hl = torch.tensor([[[[[1, 1], [-1, -1]], [[1, 1], [-1, -1]]]]],
                          dtype=torch.float32) / 8.0   # 沿高度方向的差分
        hh = torch.tensor([[[[[1, -1], [1, -1]], [[-1, 1], [-1, 1]]]]],
                          dtype=torch.float32) / 8.0   # 对角方向差分
        # 堆成 (4,1,2,2,2)，作为 depthwise 卷积核
        kernels = torch.cat([ll, lh, hl, hh], dim=0)   # (4,1,2,2,2)
        self.register_buffer("kernels", kernels)       # 注册为 buffer，不训练

    def forward(self, x):
        # x: (B, C, D, H, W)
        c = x.shape[1]
        # 把卷积核复制到每个通道，做 depthwise 卷积
        weight = self.kernels.repeat(c, 1, 1, 1, 1)    # (4C,1,2,2,2)
        # padding=0, stride=2，输出 (B, 4C, D/2, H/2, W/2)
        out = F.conv3d(x, weight, stride=2, groups=c)
        # 拆回四个分量，每个 (B, C, D/2, H/2, W/2)
        ll, lh, hl, hh = torch.chunk(out, 4, dim=1)
        return ll, lh, hl, hh


class DetailConsistentStage(nn.Module):
    """
    单个编码器阶段上的细节一致性模块。
    训练期使用：对齐教师与学生在小波高频分量上的一致性。
    推理期可整体丢弃，不影响学生网络前向。
    """

    def __init__(self, student_channels, teacher_channels, use_proj=True):
        super().__init__()
        self.wavelet = HaarWaveletDecompose()
        # 学生通道数可能与教师不同，用 1x1 卷积投影对齐
        if use_proj and student_channels != teacher_channels:
            self.proj = nn.Conv3d(student_channels, teacher_channels, 1)
        else:
            self.proj = nn.Identity()

    def forward(self, feat_student, feat_teacher):
        # 学生特征先投影到教师通道数
        fs = self.proj(feat_student)
        ft = feat_teacher

        # 各自做小波分解，只取三个高频细节分量
        _, s_lh, s_hl, s_hh = self.wavelet(fs)
        _, t_lh, t_hl, t_hh = self.wavelet(ft)

        # 逐方向计算一致性损失（这里用 L1，对细节更敏感）
        loss = 0.0
        for s_d, t_d in [(s_lh, t_lh), (s_hl, t_hl), (s_hh, t_hh)]:
            loss = loss + F.l1_loss(s_d, t_d.detach())  # 教师梯度不回传
        return loss / 3.0


class MultiStageDetailConsistency(nn.Module):
    """
    把多个阶段的 DetailConsistentStage 打包，统一算加权损失。
    student_channels / teacher_channels 是各阶段通道数列表。
    """

    def __init__(self, student_channels, teacher_channels, stage_weights=None):
        super().__init__()
        assert len(student_channels) == len(teacher_channels)
        self.stages = nn.ModuleList([
            DetailConsistentStage(sc, tc)
            for sc, tc in zip(student_channels, teacher_channels)
        ])
        # 各阶段损失权重，默认等权
        if stage_weights is None:
            stage_weights = [1.0] * len(student_channels)
        self.stage_weights = stage_weights

    def forward(self, student_feats, teacher_feats):
        # student_feats / teacher_feats: 各阶段特征列表
        total = 0.0
        for stage, w, sf, tf in zip(
            self.stages, self.stage_weights, student_feats, teacher_feats
        ):
            total = total + w * stage(sf, tf)
        return total
