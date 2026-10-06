# 插入示例（塞进你的网络）

# 假设你在 U-Net 瓶颈层有一个特征图 x: (B, C, H, W)
from torch import nn

class BottleneckWithRMT(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rmt = SimpleRetention(dim=channels, num_heads=4, gamma=0.9)

    def forward(self, x):
        B, C, H, W = x.shape
        # 展平成 token 序列: (B, H*W, C)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.rmt(tokens)                 # 保留注意力
        # 还原回特征图
        out = tokens.transpose(1, 2).view(B, C, H, W)
        return out

# 用法：替换掉 U-Net 瓶颈层的普通卷积或自注意力
# bottleneck = BottleneckWithRMT(512)
