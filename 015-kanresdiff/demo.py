# 插入示例（塞进你的网络）

class DecoderBlockWithKANResDiff(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # 即插即用模块，通道数对齐到 out_ch
        self.kan_resdiff = KANResDiff(out_ch)

    def forward(self, x, t=None):
        x = self.conv(x)
        x = self.kan_resdiff(x, t)   # 一行插入
        return x
