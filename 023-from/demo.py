# 插入示例（塞进你的网络）

class UNetWithAPEX(nn.Module):
    def __init__(self, base_unet, channels=512):
        super().__init__()
        self.backbone = base_unet
        # 在 bottleneck 后插一个 APEX-Block
        self.apex = APEXBlock(channels=channels, num_prompts=16, mode="add")

    def forward(self, x):
        feats = self.backbone.encoder(x)      # 编码器输出
        feats = self.apex(feats)              # 一行插入，形状不变
        return self.backbone.decoder(feats)   # 继续解码
