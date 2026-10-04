# 插入示例（塞进你的网络）

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # 在卷积块输出后接一个 CC，通道数对齐 out_ch
        self.cc = CC(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.cc(x)   # 形状不变，直接返回
        return x
