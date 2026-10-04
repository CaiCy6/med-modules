# 插入示例（塞进你的网络）

# 假设你已经有一个训练好的 U-Net
from my_unet import UNet

model = UNet(in_channels=4, num_classes=4)
model.load_state_dict(torch.load("unet_brats.pth"))
model.eval()

# 一行套上 TTM，其余代码完全不用改
model = TestTimeMirroring(model, dims=(-1,))

# 推理照旧
with torch.no_grad():
    logits = model(x)          # 内部已自动做镜像平均
pred = logits.argmax(dim=1)
