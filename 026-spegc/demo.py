# 插入示例（塞进你的网络）

class UNetWithSPEGC(nn.Module):
    def __init__(self, in_ch=1, num_classes=2, base_ch=64):
        super().__init__()
        # ... 这里省略编码器、解码器定义 ...
        self.bottleneck_ch = base_ch * 8          # 瓶颈层通道数
        self.spegc = SPEGC(self.bottleneck_ch)    # 实例化即插即用模块

    def forward(self, x):
        # ... 编码器下采样 ...
        feat = self.encoder(x)                    # 到达瓶颈层特征
        feat = self.spegc(feat)                   # 一行插入，形状不变
        # ... 解码器上采样 ...
        return self.decoder(feat)
