# 插入示例（塞进你的网络）

# 1. 实例化 RDC，λ 是正则强度，从 0.1 开始调
rdc = RepresentationalDriftConstraint(
    layer_weights=[1.0, 0.75, 0.5, 0.25],
    mode="cosine",
).cuda()
lambda_rdc = 0.1

# 2. 准备一个冻结的 teacher encoder（用预训练权重初始化后冻结）
teacher_encoder = build_encoder(pretrained=True)
teacher_encoder.eval()
for p in teacher_encoder.parameters():
    p.requires_grad = False

# 3. 训练循环里，同一输入分别过两支
student_feats = student_encoder(x)          # list，长度 4
with torch.no_grad():
    teacher_feats = teacher_encoder(x)      # list，长度 4

# 4. 正常算分割损失
seg_loss = criterion(pred, target)

# 5. 加一项对齐损失，回传
loss = seg_loss + lambda_rdc * rdc(student_feats, teacher_feats)
loss.backward()
optimizer.step()
