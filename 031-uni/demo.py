# 插入示例（塞进你的网络）

class UNetWithUni(nn.Module):
    def __init__(self, in_ch=4, num_classes=4, base_ch=64):
        super().__init__()
        # 假设 self.encoder / self.decoder 是你原有的 U-Net 主干
        self.encoder = YourEncoder(in_ch, base_ch)
        self.decoder = YourDecoder(base_ch, num_classes)

        # 瓶颈处通道数通常是 base_ch * 8
        bottleneck_ch = base_ch * 8
        # 插入 Uni 块，形状不变，直接串
        self.uni = UniBlock(channels=bottleneck_ch, patch_size=4, embed_dim=256)

    def forward(self, x):
        # 编码器逐级下采样
        feats = self.encoder(x)
        # 瓶颈特征
        bottleneck = feats[-1]
        # 插入 Uni：增强全局表示
        bottleneck = self.uni(bottleneck)
        feats[-1] = bottleneck
        # 解码器上采样
        return self.decoder(feats)
