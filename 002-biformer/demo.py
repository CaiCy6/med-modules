# 插入示例（塞进你的网络）

# 例：用 BRA 替换 U-Net 瓶颈层的全局注意力
self.attn = BiLevelRoutingAttention(dim=256, num_heads=8, n_win=7, topk=4)

def forward(self, x):              # x: [B, N, C]
    x = x + self.attn(x)           # 残差接住，替换原来的 SelfAttention / TransformerBlock
    return x
