# 插入示例（塞进你的网络）

class UNetBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        # 在瓶颈层插入 SS2D，给深层特征补全局上下文
        self.ss2d = SimplifiedSS2D(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = x + self.ss2d(x)   # 残差接法，稳定训练
        return x
