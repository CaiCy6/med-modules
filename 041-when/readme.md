## 一、论文出处

- 论文：When Adaptation Hurts: Connecting Representational Drift to OOD Failures in MedSAM Fine-Tuning
- 会议：MICCAI 2026
- 链接：https://arxiv.org/abs/2608.21300v2

先说清楚这篇论文的定位：它本身是一篇**分析型工作**，研究的是 MedSAM 在六种微调策略（全模型微调、encoder-only LoRA、shallow/deep VPT、decoder-only 微调、全量微调）下，跨医学影像 benchmark 的泛化行为，核心结论是"适配会漂移，漂移会伤 OOD"。

但本合集收的是**可插拔子模块**，不是完整框架。所以我要做的，是从这篇论文里把那个真正能单独塞进 U-Net 的东西抽出来——也就是论文用来**度量并抑制表征漂移**的那个模块：**表征漂移约束块（Representational Drift Constraint, RDC）**。它做的事很朴素：在微调时，把当前特征和冻结的预训练特征对齐，用一个小模块把"漂移量"算出来并回传成损失。它不改变主干结构，可以挂在任意一层特征后面。

## 二、模块图（截自论文原文）

![When Adaptation Hurts 结构图](figure.png)

图注解读：左侧是冻结的预训练编码器（teacher 分支），右侧是可训练的适配分支（student）。同一个输入分别过两支，在若干中间层抽出特征，RDC 模块对每一对特征算一个对齐项，汇总后作为正则损失加回总损失。注意它**不参与前向推理**——推理时只走 student 分支，RDC 整个被丢掉，所以是零推理开销的即插即用件。

## 三、核心思想与作用

论文的观察是：MedSAM 这类基础模型在 zero/few-shot 下泛化好，是因为它的表征空间已经覆盖了多模态、多域的医学图像。一旦你在某个小数据集上做微调，参数一动，表征就开始"漂移"（representational drift）——在训练域上指标涨，但在没见过的域上掉。漂移越狠，OOD 掉得越惨。

RDC 的作用就是给这个漂移**加一个显式的刹车**：

- 它不阻止你微调，而是要求微调后的中间特征**别离预训练特征太远**；
- 用余弦相似度或归一化后的 MSE 度量"远不远"，比直接 L2 更稳，因为医学图像特征尺度差异大；
- 逐层加权，浅层权重大（浅层承载的是通用纹理/边缘，漂移代价最高），深层权重小（深层本来就该适应任务）。

为什么有效：它把"泛化"从一个事后指标变成了一个**训练时的可优化目标**。你不需要换 backbone、不需要改数据增强，只在 loss 里加一项，就能把 OOD 性能往回拉。这也是它作为即插即用模块的价值——**它是挂在 loss 上的模块，不是挂在网络结构上的模块**，插入成本极低。

## 四、在 U-Net 里的插入位置

RDC 的插入方式和普通卷积块不一样，它需要**成对**的特征。典型做法：

1. 复制一份 U-Net 的 encoder，权重冻结，作为 teacher 分支（或者直接用预训练权重初始化后冻结）；
2. 在 encoder 的每个 stage 输出后（下采样前）抽特征，teacher 和 student 各抽一份；
3. 把每一对特征送进 RDC，得到逐层对齐损失；
4. 总损失 = 分割损失 + λ · 对齐损失。

插入位置建议：
- **encoder 的 4 个 stage 输出**：这是漂移最敏感的地方，也是论文重点分析的对象；
- **bottleneck**：可选，权重给小一点；
- **decoder 不建议加**：decoder 本来就是任务特定的，约束它反而限制拟合能力。

如果你的显存紧张，teacher 分支可以用 `torch.no_grad()` 跑，或者干脆只保留 encoder 的前两层做 teacher。

## 五、复现代码（PyTorch，逐行中文注释）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RepresentationalDriftConstraint(nn.Module):
    """
    表征漂移约束块 (RDC)
    即插即用：挂在任意一对 (student_feat, teacher_feat) 上，
    输出一个标量对齐损失，加回总 loss 即可。
    推理时不需要调用本模块。
    """

    def __init__(self, layer_weights=None, mode="cosine", eps=1e-6):
        super().__init__()
        # 逐层权重：浅层大、深层小，默认给 4 层 encoder 的配置
        # 如果层数不同，传入等长的 list 即可
        if layer_weights is None:
            layer_weights = [1.0, 0.75, 0.5, 0.25]
        # 注册成 buffer，随模型一起搬到 GPU，但不参与梯度更新
        self.register_buffer(
            "layer_weights",
            torch.tensor(layer_weights, dtype=torch.float32),
        )
        self.mode = mode          # "cosine" 或 "mse"
        self.eps = eps            # 防止除零

    def _pair_loss(self, s, t):
        """计算单层 student/teacher 特征的对齐损失"""
        # 特征形状通常是 (B, C, H, W)，先展平成 (B, C, H*W)
        if s.dim() == 4:
            s = s.flatten(2)
            t = t.flatten(2)
        # 沿通道维做 L2 归一化，消除尺度差异
        s = F.normalize(s, dim=1, eps=self.eps)
        t = F.normalize(t, dim=1, eps=self.eps)

        if self.mode == "cosine":
            # 余弦相似度：越接近 1 越好，损失取 1 - cos
            cos = (s * t).sum(dim=1)          # (B, H*W)
            return (1.0 - cos).mean()
        else:
            # 归一化后的 MSE，等价于 2 - 2*cos 的单调变换
            return F.mse_loss(s, t)

    def forward(self, student_feats, teacher_feats):
        """
        student_feats: list[Tensor]，可训练分支各层特征
        teacher_feats: list[Tensor]，冻结分支各层特征（同层数、同空间尺寸）
        返回：加权后的标量对齐损失
        """
        assert len(student_feats) == len(teacher_feats), \
            "student/teacher 特征层数必须一致"
        n = len(student_feats)
        # 若传入层数与预设权重长度不符，退化为均匀权重
        if n != self.layer_weights.numel():
            weights = torch.ones(n, device=student_feats[0].device)
        else:
            weights = self.layer_weights[:n].to(student_feats[0].device)

        total = 0.0
        for i, (s, t) in enumerate(zip(student_feats, teacher_feats)):
            # teacher 特征不参与梯度，detach 掉更省显存
            total = total + weights[i] * self._pair_loss(s, t.detach())
        # 按权重和归一化，保证不同层数下损失量级可比
        return total / weights.sum()
```

## 六、插入示例（几行塞进你的网络）

假设你有一个标准 U-Net，encoder 有 4 个 stage。下面演示怎么把 RDC 挂上去：

```python
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
```

就这五行。推理时把 teacher 分支和 RDC 全部删掉，模型结构和原来一模一样。

## 七、实测经验与注意点

- **λ 别开大**。RDC 是正则不是主损失，λ 从 0.05~0.1 起调。开太大（比如 1.0）会直接把模型锁死在预训练表征上，训练域指标都上不去，等于没微调。
- **浅层权重大是对的，但别极端**。我试过 [1, 1, 1, 1] 和 [1, 0.75, 0.5, 0.25]，后者在跨域测试上更稳；但如果把浅层权重拉到 5 以上，细节分割会变糊。
- **teacher 分支的显存开销**。最省的做法是 `torch.no_grad()` + 只保留 encoder，别把 decoder 也复制一份。如果还是爆显存，teacher 只跑前两层，后两层不约束。
- **cosine 比 MSE 稳**。医学图像不同模态的特征尺度差很多，直接 MSE 会被大尺度层主导，归一化后的 cosine 更公平。
- **它治的是 OOD，不是 ID**。如果你的测试集和训练集同分布，RDC 大概率只是让收敛慢一点，收益不明显。它的价值在跨中心、跨设备、跨模态的场景。
- **和 LoRA 叠加要小心**。LoRA 本身已经限制了参数更新量，再叠 RDC 可能过度约束。论文里 LoRA 和全量微调的漂移模式不同，建议先单独跑 RDC，再考虑组合。
- **别指望它救 prompt 质量**。MedSAM 这类 prompt-based 模型，prompt 烂的话 RDC 也拉不回来，它约束的是表征不是输入。

## 八、完整工程

把上面的东西拼成一个可直接跑的最小工程，目录结构建议：

```
rdc_unet/
├── models/
│   ├── unet.py            # 你的 U-Net 主干
│   └── rdc.py             # 本文的 RepresentationalDriftConstraint
├── train.py               # 训练脚本，含 teacher 分支与 loss 组合
├── configs/
│   └── default.yaml       # lambda_rdc、layer_weights、mode 等超参
└── README.md
```

`train.py` 的关键骨架：

```python
def train_one_epoch(model, teacher, rdc, loader, optimizer, criterion, lam):
    model.train()
    teacher.eval()
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        # student 前向，拿到分割输出和中间特征
        pred, s_feats = model(x, return_feats=True)
        # teacher 前向，只取中间特征，不建图
        with torch.no_grad():
            t_feats = teacher(x)
        seg_loss = criterion(pred, y)
        align_loss = rdc(s_feats, t_feats)
        loss = seg_loss + lam * align_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

工程上要注意两点：一是 `return_feats=True` 这个开关要加在你的 U-Net 里，把 encoder 各 stage 输出收集成 list 返回，不改动原有前向逻辑；二是 teacher 的权重加载要和 student 的初始化一致，否则对齐的是两个不相干的表征，RDC 会起反作用。推理阶段直接 `model.eval()` 正常前向即可，RDC 和 teacher 都不参与。
