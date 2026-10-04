# 插入示例（塞进你的网络）

class UNetBottleneck(nn.Module):
    def __init__(self, in_ch=512, out_ch=512):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        # 在瓶颈处插入 Dual，通道数与特征一致
        self.dual = Dual(dim=out_ch, num_experts=4, rank=8)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dual(x)   # 一行接入，形状不变
        return x
