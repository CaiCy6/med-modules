# 插入示例（塞进你的网络）

import torch.nn as nn
from torch import Tensor

class UNetBlock(nn.Module):
    """一个简化的 U-Net 卷积块，用 DilatedReparamBlock 替换普通 3x3 卷积。"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.drb = DilatedReparamBlock(out_ch, kernel_size=7, dilations=(1, 2, 3))  # 插入点
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn(self.conv1(x)))
        x = self.drb(x)          # 大核深度卷积，扩大感受野
        return self.relu(x)

# 使用：直接替换 U-Net 编码器/瓶颈里的普通卷积块
# block = UNetBlock(64, 128)
