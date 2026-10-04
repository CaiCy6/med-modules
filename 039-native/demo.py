# 插入示例（塞进你的网络）

# 1. 实例化增强块
native = NativeCarveMix(patch_size=(16, 32, 32), p=0.5).cuda()

# 2. 训练循环里，前向之前插一行
model.train()
native.train()          # 关键：训练时打开
for image, label in train_loader:
    image, label = image.cuda(), label.cuda()
    image, label = native(image, label)   # ← 就这一行
    pred = model(image)
    loss = criterion(pred, label)
    loss.backward()
    optimizer.step()

# 3. 验证时关掉
model.eval()
native.eval()           # 关键：eval 时自动跳过
with torch.no_grad():
    for image, label in val_loader:
        pred = model(image.cuda())
