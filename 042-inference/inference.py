# -*- coding: utf-8 -*-
"""
【医学图像分割模块】Inference —— Inference-Time Orthogonal Seeding Enables Geometry-Aligned 3

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalSeedingInference(nn.Module):
    """
    推理期正交播种模块（即插即用，无可学习参数）。

    思路：
      1. 对体数据沿三个正交方向（轴位/冠位/矢状位）分别做逐切片分割；
      2. 每个方向都从"离目标区域最近的种子切片"出发，向两侧传播；
      3. 三路结果映射回同一体素网格后，按"传播距离最短优先"做体素级融合。

    被包装的 slice_fn 约定：
      输入 (B, 1, H, W) 的单切片张量，输出 (B, C, H, W) 的 logits/prob。
      这样它可以是一个 2D U-Net，也可以是任何逐切片模型。
    """

    def __init__(self, slice_fn, num_classes=2, fuse="nearest_seed"):
        super().__init__()
        self.slice_fn = slice_fn          # 被包装的逐切片分割网络（2D U-Net 等）
        self.num_classes = num_classes    # 类别数，含背景
        self.fuse = fuse                  # 融合策略：nearest_seed / mean

    # ---------- 基础工具：对单个方向的体数据做逐切片分割 ----------
    def _segment_along_axis(self, vol, axis):
        """
        vol: (B, 1, D, H, W) 体数据
        axis: 0/1/2 对应 D/H/W 三个方向，表示"沿哪个轴切片"
        返回: (B, C, D, H, W) 的逐切片分割结果
        """
        # 把目标轴挪到最前面，方便按切片遍历
        v = vol.movedim(2 + axis, 2)              # (B, 1, N, h, w)
        B, _, N, h, w = v.shape
        outs = []
        for i in range(N):                        # 逐切片调用 2D 网络
            sl = v[:, :, i]                       # (B, 1, h, w)
            logit = self.slice_fn(sl)             # (B, C, h, w)
            outs.append(logit)
        out = torch.stack(outs, dim=2)            # (B, C, N, h, w)
        # 还原回原始轴顺序
        out = out.movedim(2, 2 + axis)            # (B, C, D, H, W)
        return out

    # ---------- 计算每个体素到种子的"传播距离" ----------
    def _propagation_distance(self, mask, axis):
        """
        mask: (B, 1, D, H, W) 前景掩膜（用于定位种子所在切片）
        axis: 传播方向
        返回: (B, 1, D, H, W) 每个体素沿该方向的传播距离（离最近前景切片多远）
        """
        # 沿 axis 求每个切片是否含前景 -> (B, 1, N)
        dims = [d for d in (2, 3, 4) if d != 2 + axis]
        has_fg = (mask > 0.5).float().amax(dim=dims)      # (B, 1, N)
        N = has_fg.shape[-1]
        idx = torch.arange(N, device=mask.device).view(1, 1, N)
        # 前景切片的位置，非前景处填一个大数，取最近前景切片的距离
        big = N + 1
        pos = torch.where(has_fg > 0.5, idx, torch.full_like(idx, big))
        # 前向最近距离
        fwd = torch.cummin(pos, dim=-1).values
        # 反向最近距离
        rev = torch.flip(torch.cummin(torch.flip(pos, dims=[-1]),
                                      dim=-1).values, dims=[-1])
        nearest = torch.minimum(fwd, rev)                 # (B, 1, N)
        dist = (idx - nearest).abs().float()              # 到最近前景切片的距离
        # 广播回体素形状
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = N
        dist = dist.view(shape).expand_as(mask)
        return dist

    # ---------- 主前向：三方向播种 + 几何对齐融合 ----------
    def forward(self, vol, seed_mask):
        """
        vol:       (B, 1, D, H, W) 输入体数据
        seed_mask: (B, 1, D, H, W) 稀疏种子掩膜（只标了少量切片），用于定位种子
        返回:      (B, C, D, H, W) 融合后的分割结果
        """
        probs, dists = [], []
        for axis in range(3):                     # 三个正交方向各跑一遍
            logit = self._segment_along_axis(vol, axis)   # (B, C, D, H, W)
            prob = torch.softmax(logit, dim=1)            # 转成概率，便于融合
            dist = self._propagation_distance(seed_mask, axis)  # (B,1,D,H,W)
            probs.append(prob)
            dists.append(dist)

        probs = torch.stack(probs, dim=0)          # (3, B, C, D, H, W)
        dists = torch.stack(dists, dim=0)          # (3, B, 1, D, H, W)

        if self.fuse == "mean":
            # 简单平均：最省事，但没利用"哪个方向更可信"
            fused = probs.mean(dim=0)
        else:
            # nearest_seed：每个体素选传播距离最短的那一路结果
            # 距离越小说明该方向离种子越近、误差越小
            w = torch.softmax(-dists, dim=0)       # (3, B, 1, D, H, W)
            fused = (probs * w).sum(dim=0)         # 加权融合 -> (B, C, D, H, W)

        return fused
