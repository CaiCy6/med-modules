# -*- coding: utf-8 -*-
"""
【医学图像分割模块】Evaluating the Effects of Inter —— Evaluating the Effects of Inter-Observer and Model Variabili

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionConsistencyHead(nn.Module):
    """
    决策一致性评估头（即插即用）。
    输入：分割概率图 probs (B, C, *spatial)，以及每个区域对应的二值 mask。
    输出：区域级决策指标（阳性判定、与参考的一致性统计）。
    不参与反向传播，默认 eval 模式使用。
    """

    def __init__(self, num_regions, region_threshold=0.5, aggregate="mean"):
        super().__init__()
        self.num_regions = num_regions          # 解剖学定义的 rPCI 区域个数
        self.region_threshold = region_threshold  # 区域级阳性判定阈值
        self.aggregate = aggregate              # 区域内概率聚合方式：mean / max

    def _aggregate_region(self, prob, region_mask):
        """
        prob: (B, *spatial) 单通道概率图
        region_mask: (*spatial) 该区域的二值 mask（0/1）
        返回: (B,) 每个样本在该区域内的聚合概率
        """
        mask = region_mask.float()
        # 展平空间维，方便做加权聚合
        p = prob.flatten(start_dim=1)                 # (B, N)
        m = mask.flatten().unsqueeze(0)               # (1, N)
        if self.aggregate == "mean":
            # 区域内概率均值；分母加 eps 防止空区域除零
            num = (p * m).sum(dim=1)
            den = m.sum(dim=1).clamp(min=1e-6)
            return num / den
        elif self.aggregate == "max":
            # 区域内概率最大值；空区域用 -inf 填充后取 max 会出问题，这里用 masked_fill
            neg_inf = torch.finfo(p.dtype).min
            masked = p.masked_fill(m == 0, neg_inf)
            return masked.max(dim=1).values
        else:
            raise ValueError(f"未知聚合方式: {self.aggregate}")

    @torch.no_grad()
    def forward(self, probs, region_masks, reference=None):
        """
        probs: (B, C, *spatial) 分割概率图，C 为类别数
        region_masks: list，长度 num_regions，每个元素是 (*spatial) 的二值 mask
        reference: 可选，(B, num_regions) 的参考阳性判定（0/1），用于算一致性
        返回: dict，包含区域聚合概率、阳性判定、以及（若给了 reference）一致性统计
        """
        # 取前景类（假设索引 1 为病灶/阳性类），得到 (B, *spatial)
        fg_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

        region_probs = []
        for r in range(self.num_regions):
            rp = self._aggregate_region(fg_prob, region_masks[r])  # (B,)
            region_probs.append(rp)
        region_probs = torch.stack(region_probs, dim=1)            # (B, num_regions)

        # 区域级硬判定：超过阈值即判为阳性
        pred_pos = (region_probs >= self.region_threshold).long()  # (B, num_regions)

        out = {"region_probs": region_probs, "pred_pos": pred_pos}

        if reference is not None:
            ref = reference.long()
            # 逐区域统计 TP / FP / FN / TN
            tp = ((pred_pos == 1) & (ref == 1)).sum().item()
            fp = ((pred_pos == 1) & (ref == 0)).sum().item()
            fn = ((pred_pos == 0) & (ref == 1)).sum().item()
            tn = ((pred_pos == 0) & (ref == 0)).sum().item()
            # 决策翻转率：模型判定与参考不一致的比例
            flip_rate = ((pred_pos != ref).float().mean().item())
            out.update({
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "sensitivity": tp / max(tp + fn, 1),
                "specificity": tn / max(tn + fp, 1),
                "flip_rate": flip_rate,
            })
        return out


class SoftDecisionLoss(nn.Module):
    """
    可选：把决策一致性软化成可微辅助损失，训练时用。
    用 sigmoid 近似硬阈值，让区域聚合概率向参考判定靠拢。
    """

    def __init__(self, num_regions, region_threshold=0.5, temperature=0.1):
        super().__init__()
        self.num_regions = num_regions
        self.region_threshold = region_threshold
        self.temperature = temperature  # 温度越低越接近硬阈值

    def forward(self, probs, region_masks, reference):
        fg_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        region_probs = []
        for r in range(self.num_regions):
            mask = region_masks[r].float().flatten().unsqueeze(0)
            p = fg_prob.flatten(start_dim=1)
            rp = (p * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            region_probs.append(rp)
        region_probs = torch.stack(region_probs, dim=1)  # (B, num_regions)

        # 软判定：sigmoid((p - thr) / T)，T 小则接近阶跃
        soft = torch.sigmoid((region_probs - self.region_threshold) / self.temperature)
        # 与参考判定做 BCE
        return F.binary_cross_entropy(soft.clamp(1e-6, 1 - 1e-6), reference.float())
