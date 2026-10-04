# 插入示例（塞进你的网络）

class UNetWithHadBalance(nn.Module):
    def __init__(self, in_ch=1, num_classes=2, base=64):
        super().__init__()
        # ... 你的编码器/解码器定义 ...
        self.bottleneck = nn.Conv2d(base * 8, base * 8, 3, padding=1)
        # 只加这一行
        self.had = HadBalance(base * 8)

    def forward(self, x):
        # ... 编码下采样到 feat ...
        feat = self.bottleneck(feat)
        feat = self.had(feat)   # 即插即用，形状不变
        # ... 继续解码 ...
        return out
