# 插入示例（塞进你的网络）

# 原始写法：直接 concat
x = torch.cat([upsample(x), skip], dim=1)
x = self.conv(x)
