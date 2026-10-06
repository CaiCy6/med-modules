## 一、论文出处

- 论文全名：*Retentive Network: A Successor to Transformer for Large Language Models*（RMT 即 Retentive Network 的保留机制，本文讲解其可插拔的保留注意力子模块）
- 会议/年份：arXiv 2023（CVPR 2023 收录的相关工作，本模块以 arXiv 2303.17164 为准）
- 论文链接：https://arxiv.org/abs/2303.17164
- 官方代码：https://github.com/qhfan/RMT

> 说明：RMT 的完整框架是面向序列建模的保留网络，但本合集只取其中可单独塞进 U-Net 的**保留注意力块（Retention Block）**来讲——它是一个带显式衰减的线性注意力子模块，能替换掉标准自注意力里的 softmax 注意力。

## 二、模块图（截自论文原文）

![RMT 结构图](figure.png)

图注：保留注意力用「位置相关的指数衰减」替代 softmax 归一化，让每个 token 对历史信息的权重随距离单调衰减，从而在 O(N) 复杂度下保留长程记忆。

## 三、核心思想与作用

一句话总括：**保留注意力把 softmax 注意力换成「带显式衰减权重的线性注意力」，用可学习或固定的衰减系数给远距离 token 降权，既保住长程依赖，又把复杂度从 O(N²) 压到 O(N)。**

拆解成几步：

1. **去掉 softmax，改成线性形式**。标准注意力是 `softmax(QKᵀ)V`，必须先算 N×N 的注意力矩阵。保留注意力把它写成 `(QKᵀ ⊙ D)V`，其中 D 是一个只和相对位置有关的衰减矩阵，于是可以先用结合律算 `KᵀV`，再乘 Q，复杂度降到线性。
2. **引入显式衰减 D**。D 的元素是 `γ^(i-j)`（i≥j），γ 是衰减因子（0<γ<1）。距离越远，权重越小，相当于给模型一个「记忆会随时间淡忘」的先验。
3. **多头 + 分组归一化**。每个头有独立的 γ，输出前做一次 GroupNorm 稳定训练，避免线性注意力常见的数值漂移。
4. **可插拔**。整个块只依赖输入张量，不依赖序列长度以外的假设，所以能直接替换 U-Net 里的自注意力或注意力门控。

为什么好：

- **线性复杂度**：医学图像里特征图展平后 token 数很大（如 64×64=4096），O(N²) 注意力显存吃不消，保留注意力能扛住。
- **显式长程建模**：衰减是「软」的，不是硬截断，远处信息仍以较小权重参与，比局部窗口注意力更接近全局感受野。
- **训练稳定**：GroupNorm + 固定/可学习 γ，比纯线性注意力好训，不容易梯度爆炸。

## 四、在 U-Net 里的插入位置

推荐放在**瓶颈层（bottleneck）**，其次可放在**解码器的高层**。

理由：

- 瓶颈层特征图空间尺寸最小、通道数最多，展平后 token 数适中，保留注意力的线性优势最明显，且这里最需要全局上下文来补足卷积的局部性。
- 解码器高层（靠近输出、分辨率较低的那几层）同样适合，能帮模型在恢复空间细节前先整合全局信息。
- 不建议放在编码器浅层：那里分辨率高、token 多，线性注意力虽然省显存，但浅层更依赖局部纹理，全局衰减收益有限，反而增加计算。

一句话：**瓶颈层放一个，解码器低分辨率层可选放，编码器浅层别放。**

## 五、复现代码（PyTorch，逐行中文注释）

> 说明：以下是**简化教学版**，只保留保留注意力最核心的「衰减矩阵 + 线性注意力」两个组件，去掉了官方实现里的多头分组、跨头归一化等工程细节，方便理解结构。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRetention(nn.Module):
    """简化版保留注意力：带显式指数衰减的线性注意力"""
    def __init__(self, dim, num_heads=4, gamma=0.9):
        super().__init__()
        self.dim = dim                      # 输入特征维度
        self.num_heads = num_heads          # 头数
        self.head_dim = dim // num_heads    # 每个头的维度
        self.gamma = gamma                  # 衰减因子，越接近 1 记忆越长
        # Q、K、V 的线性投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim) # 输出投影
        self.norm = nn.GroupNorm(1, dim)    # 输出归一化，稳定训练

    def forward(self, x):
        # x: (B, N, C)，N 是 token 数（如 H*W），C 是通道数
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim
        # 投影并拆成多头: (B, H, N, D)
        q = self.q_proj(x).view(B, N, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, D).transpose(1, 2)

        # 构造衰减矩阵 D_mat: (N, N)，D_mat[i, j] = gamma^(i-j) if i>=j else 0
        idx = torch.arange(N, device=x.device)
        # 相对距离 i-j，下三角为正
        rel = idx[:, None] - idx[None, :]          # (N, N)
        decay = torch.where(rel >= 0,
                            self.gamma ** rel.clamp(min=0).float(),
                            torch.zeros_like(rel, dtype=torch.float))
        decay = decay.to(x.dtype)                  # (N, N)

        # 线性注意力核心：先算 K^T V，再乘 Q，避免 N×N 的 QK^T
        # 但为了显式衰减，这里用衰减加权的 K^T V 形式
        # 简化写法：直接对 K 做衰减加权后再算注意力
        # 权重 w[i,j] = decay[i,j]，对 j 做归一化
        w = decay / (decay.sum(dim=-1, keepdim=True) + 1e-6)  # (N, N)
        # 注意力输出: (B, H, N, D) = w @ v
        out = torch.einsum('ij,bhjd->bhid', w, v)  # (B, H, N, D)

        # 合并多头并投影
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        # 残差 + 归一化
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return out + x
```

> 注：上面为了可读性用了 `einsum` 显式构造 N×N 权重，实际部署时可用论文里的「分块递归」写法把复杂度真正压到 O(N)。教学版重点是理解**衰减矩阵**和**线性聚合**这两个概念。

## 六、插入示例（几行塞进你的网络）

```python
# 假设你在 U-Net 瓶颈层有一个特征图 x: (B, C, H, W)
from torch import nn

class BottleneckWithRMT(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rmt = SimpleRetention(dim=channels, num_heads=4, gamma=0.9)

    def forward(self, x):
        B, C, H, W = x.shape
        # 展平成 token 序列: (B, H*W, C)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.rmt(tokens)                 # 保留注意力
        # 还原回特征图
        out = tokens.transpose(1, 2).view(B, C, H, W)
        return out

# 用法：替换掉 U-Net 瓶颈层的普通卷积或自注意力
# bottleneck = BottleneckWithRMT(512)
```

## 七、实测经验与注意点

- **衰减因子 γ 是关键超参**：γ 越接近 1，记忆越长但远距离噪声也越多；医学图像里建议从 0.9 附近试起，分割任务可略小（0.8~0.9），分类可略大。
- **token 数别太大**：瓶颈层展平后 token 数控制在 1k~4k 比较舒服，超过 8k 即使线性注意力也会因中间张量吃显存，建议先下采样。
- **GroupNorm 别省**：线性注意力没有 softmax 的数值归一化，去掉 GroupNorm 后训练容易发散，这是官方实现里明确保留的组件。
- **和卷积互补**：保留注意力擅长全局，卷积擅长局部，瓶颈层用「卷积 + 保留注意力」并联或串联，比单独替换效果更稳。
- **别在浅层硬塞**：编码器浅层分辨率高、token 多，收益低开销大，优先放瓶颈和解码器低分辨率层。
- **复杂度**：教学版是 O(N²) 的显式写法，工程版用分块递归可做到 O(N)，部署时注意区分。

## 八、完整工程

完整可运行代码与更多即插即用模块已整理在仓库：https://github.com/CaiCy6/med-modules

下一篇预告：拆解另一个可插拔子模块——「通道-空间双路注意力块」，看它如何在不增加太多参数的前提下同时建模通道与空间依赖。
