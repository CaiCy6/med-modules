# 插入示例（塞进你的网络）

# 初始化时（__init__ 里）
self.latent_dim = 128
self.to_latent = nn.Linear(C * D * H * W, self.latent_dim)   # 压到潜空间
self.from_latent = nn.Linear(self.latent_dim, C * D * H * W) # 还原回瓶颈
self.latent = Latent(self.latent_dim)

# 训练时（forward 里）
z1 = self.to_latent(bottleneck.flatten(1))   # 编码器瓶颈 -> 潜编码
loss_flow = self.latent(z1)                  # flow matching 损失，加进总 loss
# 训练阶段解码器仍用真实的 z1，保证分割精度
bottleneck = self.from_latent(z1).view_as(bottleneck)

# 推理时（采样多个分割结果）
z_samples = self.latent.sample(n=5, device=bottleneck.device)  # 采 5 个
for z in z_samples:
    b = self.from_latent(z).view(B, C, D, H, W)
    mask = self.decoder(b)   # 每个 z 得到一个分割结果
