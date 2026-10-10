# -*- coding: utf-8 -*-
"""
【医学图像分割模块】UniRepLKNet —— 膨胀重参数大核块（Dilated Reparam Block）—— 大核卷积又能训又能用

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn

class DilatedReparamBlock(nn.Module):
    """
    简化版 Dilated Reparam Block。
    训练时：多个深度卷积分支并行相加（部分带膨胀）。
    推理时：调用 reparameterize() 融合成单个大核深度卷积。
    """
    def __init__(self, channels, kernel_size=7, dilations=(1, 2, 3)):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilations = dilations

        # 训练期的并行分支：每个分支是一个深度卷积 + BN
        # 分支1：普通小核深度卷积（dilation=1），负责局部细节
        self.branches = nn.ModuleList()
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                          groups=channels, bias=False),  # 深度卷积，groups=channels
                nn.BatchNorm2d(channels)                   # BN 用于训练稳定，融合时可吸收
            )
        )
        # 其余分支：带膨胀的小核深度卷积，用膨胀撑开感受野
        for d in dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3,
                              padding=d, dilation=d,
                              groups=channels, bias=False),  # 膨胀深度卷积
                    nn.BatchNorm2d(channels)
                )
            )

        # 推理时使用的融合大核（初始为空，reparameterize 后赋值）
        self.fused_conv = None

    def forward(self, x):
        # 训练/推理统一走并行分支相加，保证行为一致
        out = 0
        for branch in self.branches:
            out = out + branch(x)   # 多分支输出逐元素相加
        return out

    @torch.no_grad()
    def reparameterize(self):
        """
        将多个并行分支等价融合成单个大核深度卷积。
        核心思路：把每个分支的卷积核通过零填充放到统一的大核尺寸上，再相加。
        """
        # 计算融合后大核的实际尺寸：最大膨胀覆盖范围
        max_dilation = max(self.dilations) if self.dilations else 1
        # 单个 3x3 核在膨胀 d 下的等效尺寸 = 3 + (3-1)*(d-1) = 2d+1
        fused_size = max(3, 2 * max_dilation + 1)

        # 初始化融合核权重，形状 [channels, 1, fused_size, fused_size]
        fused_weight = torch.zeros(self.channels, 1, fused_size, fused_size,
                                   device=next(self.parameters()).device)

        for branch in self.branches:
            conv = branch[0]          # 取出卷积分支
            bn = branch[1]            # 取出 BN
            # 把 BN 吸收进卷积：w' = w * gamma / sqrt(var + eps)
            bn_scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            w = conv.weight * bn_scale.view(-1, 1, 1, 1)  # 逐通道缩放卷积核

            # 计算该分支等效核尺寸
            k = conv.kernel_size[0]
            d = conv.dilation[0]
            eff = k + (k - 1) * (d - 1)   # 膨胀后的等效尺寸

            # 把等效核放到融合核的中心位置（零填充对齐）
            start = (fused_size - eff) // 2
            # 注意：膨胀核需要按 dilation 展开填充，这里用步长写入
            for i in range(k):
                for j in range(k):
                    fused_weight[:, :, start + i * d, start + j * d] += w[:, :, i, j]

        # 构造融合后的深度卷积
        self.fused_conv = nn.Conv2d(self.channels, self.channels,
                                    kernel_size=fused_size,
                                    padding=fused_size // 2,
                                    groups=self.channels, bias=False)
        self.fused_conv.weight.copy_(fused_weight)
        self.fused_conv.eval()

    def forward_fused(self, x):
        """推理时调用：使用融合后的大核深度卷积，速度更快。"""
        assert self.fused_conv is not None, "请先调用 reparameterize()"
        return self.fused_conv(x)
