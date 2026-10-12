# 插入示例（塞进你的网络）

import torch
import torch.nn as nn
from torch import Tensor

# 假设你有一个 U-Net 的瓶颈层特征 x: [B, 256, 16, 16]
sha = SingleHeadAttention(dim=256, num_heads=1, sr_ratio=2, kernel_size=3)

x: Tensor = torch.randn(2, 256, 16, 16)
y = sha(x)          # 输出形状不变: [2, 256, 16, 16]
print(y.shape)
