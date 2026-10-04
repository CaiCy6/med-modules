# 插入示例（塞进你的网络）

class UNetWithTAdapter(nn.Module):
    def __init__(self, base_unet, channels=512):
        super().__init__()
        self.unet = base_unet
        # 在 bottleneck 后插入 T 模块
        self.t_adapter = TemporalGatedAdapter(channels)

    def forward(self, x):
        # x: (B, T, C_in, H, W)，T 张相邻切片
        B, T = x.shape[:2]
        # 把序列拍平送进 encoder，逐帧提特征
        feats = self.unet.encoder(x.reshape(B * T, *x.shape[2:]))  # (B*T, C, h, w)
        C, h, w = feats.shape[1:]
        feats = feats.reshape(B, T, C, h, w)                       # 还原时间维
        feats = self.t_adapter(feats)                              # (B, C, h, w) 门控修正
        return self.unet.decoder(feats)                            # 接回 decoder
