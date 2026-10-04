# 插入示例（塞进你的网络）

# 原来的写法：直接 concat
x = torch.cat([up_feat, enc_feat], dim=1)
x = self.conv_block(x)
