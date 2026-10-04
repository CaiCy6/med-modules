# 插入示例（塞进你的网络）

class UNetWithDAPP(nn.Module):
    def __init__(self, in_ch=1, num_classes=2, base_ch=64):
        super().__init__()
        # ... 这里省略 encoder / decoder 定义 ...
        self.bottleneck_ch = base_ch * 8
        # 只加这一行, 就完成了插入
        self.dapp = DomainAdaptivePrototypePrompt(self.bottleneck_ch)

    def forward(self, x):
        # ... encoder 前向, 得到 bottleneck 特征 ...
        feat = self.bottleneck(x)          # [B, C, H, W]
        feat = self.dapp(feat)             # 一行调用, 形状不变
        # ... decoder 继续 ...
        return self.decoder(feat)
