# 插入示例（塞进你的网络）

# 假设 dinov3_conv2d_weight 是你从 DINOv3 里取出的某个 2D 卷积权重
# 形状为 (C_out, C_in, 3, 3)
dinov3_conv2d_weight = torch.randn(64, 3, 3, 3)  # 这里用随机数占位

# 原来的一层
# conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)

# 替换成可膨胀的版本
conv1 = InflatedConv3d(3, 64, kernel_size=3, padding=1)

# 用 2D 权重初始化
conv1.load_from_2d(dinov3_conv2d_weight, mode="copy")

# 之后照常接进你的 U-Net
# x = conv1(x)
