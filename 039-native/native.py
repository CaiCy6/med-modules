# -*- coding: utf-8 -*-
"""
【医学图像分割模块】Native —— Native-Space 3D CarveMix for Multi-Site T1w Stroke Segmentat

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NativeCarveMix(nn.Module):
    """
    Native-Space 3D CarveMix 可插拔增强块。
    输入: image (B, 1, D, H, W), label (B, 1, D, H, W)
    输出: 增强后的 image, label，形状不变。
    只在训练时启用。
    """

    def __init__(self,
                 patch_size=(16, 32, 32),   # 抠出的病灶 patch 尺寸 (D,H,W)
                 num_carve=1,               # 每个样本雕刻几次
                 p=0.5,                     # 该 batch 触发增强的概率
                 min_lesion_vox=50,         # 病灶体素数低于此值就跳过
                 host_margin=8):            # 宿主区离病灶至少留多少体素，避免贴回原地
        super().__init__()
        self.patch_size = patch_size
        self.num_carve = num_carve
        self.p = p
        self.min_lesion_vox = min_lesion_vox
        self.host_margin = host_margin

    @torch.no_grad()
    def _sample_lesion_center(self, label):
        """在病灶体素里随机挑一个中心点，返回 (d,h,w)。"""
        # label: (1,1,D,H,W)，取非零坐标
        coords = torch.nonzero(label[0, 0] > 0.5, as_tuple=False)  # (N,3)
        if coords.shape[0] < self.min_lesion_vox:
            return None
        idx = torch.randint(0, coords.shape[0], (1,)).item()
        return coords[idx].tolist()  # [d,h,w]

    @torch.no_grad()
    def _find_host_center(self, label, lesion_center):
        """在健康脑区里找一个和病灶 patch 尺寸匹配的宿主中心。"""
        D, H, W = label.shape[-3:]
        pd, ph, pw = self.patch_size
        # 宿主中心必须让 patch 完整落在图内
        d_lo, d_hi = pd // 2, D - pd // 2
        h_lo, h_hi = ph // 2, H - ph // 2
        w_lo, w_hi = pw // 2, W - pw // 2
        if d_lo >= d_hi or h_lo >= h_hi or w_lo >= w_hi:
            return None

        # 随机试若干次，找一个 patch 内没有病灶、且离原病灶足够远的中心
        for _ in range(20):
            d = torch.randint(d_lo, d_hi, (1,)).item()
            h = torch.randint(h_lo, h_hi, (1,)).item()
            w = torch.randint(w_lo, w_hi, (1,)).item()
            # 距离约束：别贴回病灶附近
            if abs(d - lesion_center[0]) < self.host_margin and \
               abs(h - lesion_center[1]) < self.host_margin and \
               abs(w - lesion_center[2]) < self.host_margin:
                continue
            # 宿主 patch 内不能已有病灶，否则标签会冲突
            host_patch = label[0, 0,
                               d - pd // 2:d + pd // 2,
                               h - ph // 2:h + ph // 2,
                               w - pw // 2:w + pw // 2]
            if host_patch.sum() > 0:
                continue
            return [d, h, w]
        return None

    @torch.no_grad()
    def _carve_once(self, image, label):
        """对单个样本做一次雕刻混合，返回新 image, label。"""
        lesion_center = self._sample_lesion_center(label)
        if lesion_center is None:
            return image, label
        host_center = self._find_host_center(label, lesion_center)
        if host_center is None:
            return image, label

        pd, ph, pw = self.patch_size
        lc, hc = lesion_center, host_center

        # 抠出病灶 patch（image 和 label 同步抠）
        def _slice(c):
            return (slice(c[0] - pd // 2, c[0] + pd // 2),
                    slice(c[1] - ph // 2, c[1] + ph // 2),
                    slice(c[2] - pw // 2, c[2] + pw // 2))

        ls = _slice(lc)
        hs = _slice(hc)

        img_patch = image[0, :, ls[0], ls[1], ls[2]].clone()
        lbl_patch = label[0, :, ls[0], ls[1], ls[2]].clone()

        # 原生空间直接粘贴：不重采样、不做强度变换
        image[0, :, hs[0], hs[1], hs[2]] = img_patch
        label[0, :, hs[0], hs[1], hs[2]] = lbl_patch

        return image, label

    def forward(self, image, label):
        # 训练时才增强；eval 模式直接原样返回
        if not self.training:
            return image, label
        if torch.rand(1).item() > self.p:
            return image, label

        image = image.clone()
        label = label.clone()
        B = image.shape[0]
        for b in range(B):
            for _ in range(self.num_carve):
                img_b = image[b:b + 1]
                lbl_b = label[b:b + 1]
                img_b, lbl_b = self._carve_once(img_b, lbl_b)
                image[b:b + 1] = img_b
                label[b:b + 1] = lbl_b
        return image, label
