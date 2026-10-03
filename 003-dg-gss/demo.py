# 插入示例（塞进你的网络）

# 例：用 DG-GSS 替换 U-Net 深层编码器块（低分辨率、高通道处）
self.deep_block = DGGSSBlock(dim=256, n_groups=4)

def forward(self, x):              # x: [B, C, H, W]
    x = self.deep_block(x)         # 替换原来的注意力块 / SSM 块，无需改接口
    return x
