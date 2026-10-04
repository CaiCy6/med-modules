# 插入示例（塞进你的网络）

# 假设你有一个现成的 U-Net
unet = UNet(in_channels=1, num_classes=2)
# 实例化评估头，区域数按你的 rPCI 定义来，比如 13 个区域
eval_head = DecisionConsistencyHead(num_regions=13, region_threshold=0.5)

unet.eval()
with torch.no_grad():
    logits = unet(x)                       # (B, 2, H, W, D)
    probs = torch.softmax(logits, dim=1)   # 概率图
    metrics = eval_head(probs, region_masks, reference=ref_pos)
    print(metrics["flip_rate"], metrics["sensitivity"])
