# 插入示例（塞进你的网络）

class UNetWithM2C(nn.Module):
    def __init__(self, unet, in_channels, concept_dim=256):
        super().__init__()
        self.unet = unet
        # 即插即用：只加这一个模块
        self.m2c = MaskToConcept(in_channels, concept_dim=concept_dim)
        # 把概念嵌入映射回通道数，用于 FiLM 调制
        self.film = nn.Linear(concept_dim, in_channels * 2)

    def forward(self, x, ref_mask):
        # 编码器拿到 bottleneck 特征
        feat = self.unet.encoder(x)          # [B, C, H, W]

        # 用参考掩码生成概念嵌入
        concept = self.m2c(feat, ref_mask)   # [B, concept_dim]

        # FiLM：概念嵌入 -> 缩放和平移参数，调制 bottleneck 特征
        gamma_beta = self.film(concept)      # [B, C*2]
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma[:, :, None, None]      # 广播到空间维
        beta = beta[:, :, None, None]
        feat = feat * (1 + gamma) + beta     # 条件调制

        # 解码器继续
        out = self.unet.decoder(feat)
        return out
