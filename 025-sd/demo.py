# 插入示例（塞进你的网络）

class UNetWithSD(nn.Module):
    def __init__(self, base_unet, c_seg, c_sd):
        super().__init__()
        self.backbone = base_unet          # 你的原 U-Net
        self.sd = SD(c_seg, c_sd)          # 插入 SD 块

    def forward(self, x, sd_feat):
        # 编码器走到瓶颈
        feat = self.backbone.encoder(x)
        # 在瓶颈处注入扩散特征
        feat = self.sd(feat, sd_feat)
        # 继续解码
        out = self.backbone.decoder(feat)
        return out
