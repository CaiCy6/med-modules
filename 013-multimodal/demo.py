# 插入示例（塞进你的网络）

# 初始化一次即可
router_refine = MultimodalRoutingRegionRefinement(
    visual_dim=512,      # 你的 bottleneck 通道数
    text_dim=768,        # 你的文本编码器输出维度
    hidden_dim=128,
    num_routes=4,
).to(device)

# 前向时插在 bottleneck 之后、解码器之前
feat = encoder(x)                    # (B, 512, H/32, W/32)
text_tokens = text_encoder(text)     # (B, L, 768)
feat = router_refine(feat, text_tokens)   # 形状不变，直接接解码器
out = decoder(feat, skips)           # 后续完全不用改
