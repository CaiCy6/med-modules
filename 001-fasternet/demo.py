# 插入示例（塞进你的网络）

# 例：用 FasterNet Block 替换 U-Net 编码器某个 stage 里的重卷积块
self.block = Faster_Block(dim=256, n_div=4, mlp_ratio=2)

def forward(self, x):
    x = self.down(x)
    x = self.block(x)          # ← 替换原来的 DoubleConv / ResBlock
    return x
