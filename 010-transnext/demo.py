# 插入示例（塞进你的网络）

# 假设你在写一个 U-Net 的编码器 block
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        # 把简化版聚合注意力插在卷积之后
        self.agg_attn = SimplifiedAggregatedAttention(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.agg_attn(x)   # 即插即用，形状不变
        return x
