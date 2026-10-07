# 插入示例（塞进你的网络）

# 假设你的 U-Net 瓶颈层特征图是 (B, 512, 16, 16)
bottleneck = nn.Sequential(
    nn.Conv2d(256, 512, 3, padding=1),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True),
    GFNetBlock(dim=512, h=16, w=16),   # 直接插在瓶颈
    GFNetBlock(dim=512, h=16, w=16),
)
