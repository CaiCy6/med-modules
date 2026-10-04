# 插入示例（塞进你的网络）

class UNetWithGET(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        # ... 你的 encoder / decoder 定义 ...
        self.get = GETBlock(channels=base_ch * 8)  # 最底层通道数

    def forward(self, x):
        # ... encoder 下采样 ...
        feat = self.get(feat)  # 一行插入，形状不变
        # ... decoder 上采样 ...
        return out
