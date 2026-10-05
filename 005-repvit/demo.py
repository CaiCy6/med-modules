# 插入示例（塞进你的网络）

import torch.nn as nn
from repvit_block import RepViTBlock  # 假设上面的类放在这个文件里

class UNetEncoderStage(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        # 在下采样后用 RepViT Block 提纯特征
        self.block = RepViTBlock(dim=out_ch)

    def forward(self, x):
        x = self.down(x)
        return self.block(x)
