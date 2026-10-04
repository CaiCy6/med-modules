# -*- coding: utf-8 -*-
"""
【医学图像分割模块】NeuroSeg Meets DINOv3 —— NeuroSeg Meets DINOv3: Transferring 2D Self-Supervised Visua

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn


def inflate_2d_to_3d_weight(w2d: torch.Tensor, kernel_depth: int = 3,
                            mode: str = "copy") -> torch.Tensor:
    """
    把 2D 卷积权重膨胀成 3D 卷积权重。

    参数:
        w2d: 形状 (C_out, C_in, kH, kW) 的 2D 卷积权重
        kernel_depth: 目标 3D 卷积的深度核大小 kD
        mode: "copy" 沿深度轴复制; "center" 只在中心层放原权重, 其余置零

    返回:
        形状 (C_out, C_in, kD, kH, kW) 的 3D 权重
    """
    assert w2d.dim() == 4, "输入必须是 4D 的 2D 卷积权重"
    c_out, c_in, kh, kw = w2d.shape

    if mode == "copy":
        # 在深度维新增一维, 然后沿该维复制 kD 次
        w3d = w2d.unsqueeze(2).repeat(1, 1, kernel_depth, 1, 1)
        # 关键: 除以 kD 做归一化, 保证输出响应量级与 2D 时一致
        w3d = w3d / kernel_depth
    elif mode == "center":
        # 只在深度方向中心层保留原权重, 其余层为零
        w3d = torch.zeros(c_out, c_in, kernel_depth, kh, kw,
                          dtype=w2d.dtype, device=w2d.device)
        center = kernel_depth // 2
        w3d[:, :, center, :, :] = w2d
    else:
        raise ValueError(f"不支持的 mode: {mode}")

    return w3d


def inflate_2d_to_3d_bias(b2d: torch.Tensor, kernel_depth: int = 3) -> torch.Tensor:
    """
    偏置项不随深度膨胀, 直接沿用即可。
    这里保留函数是为了接口统一, 方便后续扩展。
    """
    return b2d.clone()


class InflatedConv3d(nn.Conv3d):
    """
    一个可以直接替换 nn.Conv3d 的即插即用卷积层。
    支持从 2D 权重初始化, 前向计算与普通 Conv3d 完全一致。
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        # 如果传入的是整数, 统一成三元组
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        super().__init__(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias)

    @torch.no_grad()
    def load_from_2d(self, w2d: torch.Tensor, b2d: torch.Tensor = None,
                     mode: str = "copy"):
        """
        用 2D 权重初始化本层。

        参数:
            w2d: (C_out, C_in, kH, kW) 的 2D 权重
            b2d: 可选, (C_out,) 的 2D 偏置
            mode: 膨胀模式, 见 inflate_2d_to_3d_weight
        """
        kd = self.weight.shape[2]  # 本层深度核大小
        w3d = inflate_2d_to_3d_weight(w2d, kernel_depth=kd, mode=mode)

        # 形状必须严格匹配, 否则说明通道数或核大小对不上
        assert w3d.shape == self.weight.shape, \
            f"膨胀后权重形状 {w3d.shape} 与目标层 {self.weight.shape} 不匹配"

        self.weight.copy_(w3d)

        if b2d is not None and self.bias is not None:
            assert b2d.shape == self.bias.shape, "偏置形状不匹配"
            self.bias.copy_(b2d)


def inflate_vit_linear_to_3d(w2d: torch.Tensor, depth_tokens: int) -> torch.Tensor:
    """
    针对 ViT 线性层的膨胀: 把 2D patch 的投影权重沿深度方向扩展。
    这里给出一个简化版, 实际使用时需根据 patch 划分方式调整。

    参数:
        w2d: (out_dim, in_dim) 的线性层权重, in_dim 对应 2D patch 展平
        depth_tokens: 深度方向 token 数
    """
    out_dim, in_dim = w2d.shape
    # 假设 in_dim = C * kH * kW, 这里只做示意性的复制扩展
    # 真实场景需要按 patch 结构 reshape 后再沿深度复制
    w3d = w2d.unsqueeze(0).repeat(depth_tokens, 1, 1)
    return w3d
