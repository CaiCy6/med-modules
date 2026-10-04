# 插入示例（塞进你的网络）

# 1. 初始化模块（通道数按你网络实际填）
dcd = MultiStageDetailConsistency(
    student_channels=[32, 64, 128, 256],
    teacher_channels=[32, 64, 128, 256],
    stage_weights=[1.0, 1.0, 1.0, 1.0],
).cuda()

# 2. 训练循环里，取两个网络的编码器特征
student_feats = student_net.encode(x)   # 返回各阶段特征列表
with torch.no_grad():
    teacher_feats = teacher_net.encode(x)

# 3. 算分割主损失 + 细节一致性损失
seg_loss = criterion(student_logits, label)
dcd_loss = dcd(student_feats, teacher_feats)
loss = seg_loss + 0.5 * dcd_loss        # 权重按经验调

loss.backward()
optimizer.step()
