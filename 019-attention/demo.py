# 插入示例（塞进你的网络）

# 原来的流程：聚合出共识原型和标注者原型
consensus = masked_avg_pool(support_feat, support_mask)          # [B, C]
rater_protos = torch.stack(rater_proto_list, dim=1)              # [B, N, C]

# === 插入校准模块 ===
calib = AttentionPrototypeCalibration(dim=C, num_heads=4).to(device)
rater_protos = calib(consensus, rater_protos)                    # [B, N, C]

# 后续匹配逻辑完全不用改
logits = match_query_to_prototypes(query_feat, rater_protos)
