# 插入示例（塞进你的网络）

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # 在卷积之后插入 Multi 模块
        self.multi = Multi(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.multi(x)   # 一行调用，形状不变
        return x
