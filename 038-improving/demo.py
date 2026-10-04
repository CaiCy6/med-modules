# 插入示例（塞进你的网络）

class ConvBlockWithAdapter(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
        )
        # 在卷积块之后插入适配块，通道数对齐 out_ch
        self.adapter = SiteRoutingAdapter(out_ch, num_experts=4)

    def forward(self, x):
        x = self.conv(x)
        x = self.adapter(x)   # 一行接入，形状不变
        return x
