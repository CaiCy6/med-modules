# 插入示例（塞进你的网络）

# 初始化一次，放在 __init__ 里
self.harmonizer = Harmonizer(channels=256)

# 在 forward 里，拿到编码器特征后直接套一层
feat = self.encoder_stage3(x)      # [B, 256, H/8, W/8]
feat = self.harmonizer(feat)       # 形状不变，直接往下走
