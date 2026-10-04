# 插入示例（塞进你的网络）

import torch
import torch.nn as nn

# 假设你已定义好上面的 StarBlock
# 在 U-Net 瓶颈层替换原来的卷积块
class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 先用 1x1 调整通道，再交给 StarBlock 做非线性建模
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.star = StarBlock(out_ch)          # 直接插入 StarNet 模块
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.proj(x)
        x = self.star(x)                       # 即插即用
        return self.norm(x)
