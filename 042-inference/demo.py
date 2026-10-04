# 插入示例（塞进你的网络）

unet2d = MyUNet2D(in_ch=1, num_classes=2)      # 你原有的逐切片网络
unet2d.eval()                                   # 推理模式

# 用即插即用模块把它的推理过程包起来，网络本身一行都不用改
model = OrthogonalSeedingInference(
    slice_fn=unet2d,                            # 被包装的 2D 网络
    num_classes=2,
    fuse="nearest_seed",                        # 用最短传播距离加权
).cuda().eval()

with torch.no_grad():
    vol = vol.cuda()                            # (1, 1, D, H, W)
    seed = seed.cuda()                          # (1, 1, D, H, W) 稀疏种子
    out = model(vol, seed)                      # (1, C, D, H, W) 融合结果
pred = out.argmax(dim=1)                        # 取类别
