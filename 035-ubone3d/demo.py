# 插入示例（塞进你的网络）

# 1. 实例化模块（放在 __init__ 里）
self.cfm_block = PhysicsRectifiedCFMBlock(feat_dim=128, phys_dim=3)

# 2. 前向里：分割 mask -> 点云 -> 补全
# seg_logits: (B, 1, D, H, W) 分割输出
mask = (torch.sigmoid(seg_logits) > 0.5).float()
# 取前景体素中心作为点云（简化写法，实际可用 grid_sample 取亚体素坐标）
coords = torch.nonzero(mask[0, 0]).float()  # (N, 3)
coords = coords[None]                        # (B=1, N, 3)
# 物理条件：这里用局部厚度/强度方差，示例给全零
phys = torch.zeros_like(coords)              # (B, N, 3)
# 补全
completed = self.cfm_block.sample(coords, phys, steps=10)  # (B, N, 3)
