# 插入示例（塞进你的网络）

# 原来的写法
x = torch.cat([skip_feat, up_feat], dim=1)   # 通道翻倍
x = self.conv_block(x)
