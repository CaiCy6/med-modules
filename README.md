# med-modules · 医学图像分割「即插即用模块」复现合集

> 看得懂、抄得走、插得上。

收录 **CVPR / ICCV / ECCV / NeurIPS 等顶会** 医学图像分割常用的**即插即用模块**：每篇一个文件夹，含
**架构图（截自论文原文）+ PyTorch 复现代码（逐行中文注释）+ 讲解文档 + 插入示例**。

> 说明：本合集以**顶会正式录用**论文为主；个别条目若为预印本会据实标注，不作为收录模板。

## 目录

<!-- CATALOG:BEGIN -->
| # | 模块 | 会议 | 一句话 | 文件夹 |
|---|------|------|--------|--------|
| 001 | **FasterNet / PConv** | CVPR 2023 | 只对部分通道做 3×3 卷积，砍 FLOPs 也砍访存，实测更快 | [`001-fasternet`](./001-fasternet) |
| 002 | **BiFormer** | CVPR 2023 | 双层路由注意力，只算「该算的地方」 | [`002-biformer`](./002-biformer) |
| 003 | **DG-GSS** | arXiv 2026 | 方向-组图选择性扫描，让轻量分割网络「扫」得更聪明 | [`003-dg-gss`](./003-dg-gss) |
| 004 | **StarNet** | CVPR 2024 | 星操作（元素级乘法），把「相加」换成「相乘」 | [`004-starnet`](./004-starnet) |
| 005 | **RepViT** | CVPR 2024 |  | [`005-repvit`](./005-repvit) |
| 006 | **Rmt** | CVPR 2023 |  | [`006-rmt`](./006-rmt) |
| 013 | **Multimodal** | MICCAI 2026 | 图文动态路由+区域细化，每个图文对走自己的融合路径 | [`013-multimodal`](./013-multimodal) |
| 014 | **Multi** | MICCAI 2026 | 多阶段提示引导的特征调制，按提示逐级校准泛化分割 | [`014-multi`](./014-multi) |
| 015 | **Kanresdiff** | MICCAI 2026 | KAN 局部残差扩散，把残差学习塞进可插拔扩散块 | [`015-kanresdiff`](./015-kanresdiff) |
| 016 | **Medcagd** | ECCV 2026 | 上下文感知门控解码器，用门控按需聚合多尺度特征 | [`016-medcagd`](./016-medcagd) |
| 017 | **Dual** | MICCAI 2026 | 低秩专家上的双自适应层次路由，即插式参数高效适配 | [`017-dual`](./017-dual) |
| 018 | **Mask** | MICCAI 2026 | 掩码到概念的可自动提示模块，免训练提示分割 | [`018-mask`](./018-mask) |
| 019 | **Attention** | MICCAI 2026 | 注意力原型校准，修正多标注者少样本的分割偏差 | [`019-attention`](./019-attention) |
| 020 | **Hadbalance** | MICCAI 2026 | 即插即用的全局几何先验（Hadamard 平衡）约束块 | [`020-hadbalance`](./020-hadbalance) |
| 021 | **Detail** | MICCAI 2026 | 细节一致的分阶段蒸馏块，让轻量 3D 分割保持边界 | [`021-detail`](./021-detail) |
| 022 | **Harmonized** | CVPR 2026 | 谐调特征条件+频率提示，个性化特征调制块 | [`022-harmonized`](./022-harmonized) |
| 023 | **From** | CVPR 2026 | 自适应视觉提示，从「适配」走向「泛化」的提示模块 | [`023-from`](./023-from) |
| 024 | **Gated** | CVPR 2026 | 轻量时序门控适配器，给时序/视频分割加低成本适配 | [`024-gated`](./024-gated) |
| 025 | **Sd** | CVPR 2026 | 借稳定扩散做少样本医学分割的适配模块 | [`025-sd`](./025-sd) |
| 026 | **Spegc** | CVPR 2026 | 语义提示增强的持续测试时自适应块 | [`026-spegc`](./026-spegc) |
| 027 | **Medal** | CVPR 2025 | 时空文本提示模型，多模态医学分割的提示融合模块 | [`027-medal`](./027-medal) |
| 028 | **Prompting** | MICCAI 2024 | 域自适应原型提示 SAM 的可插拔提示块 | [`028-prompting`](./028-prompting) |
| 029 | **Cc** | ECCV 2024 | 交叉特征注意力+上下文的 SAM 增强模块 | [`029-cc`](./029-cc) |
| 030 | **Textsuperscript** |  | 轻量状态空间 MoE + 自适应融合的多模态融合块 | [`030-textsuperscript`](./030-textsuperscript) |
| 031 | **Uni** |  | 单编码器对多编码器的表征再融合模块 | [`031-uni`](./031-uni) |
| 032 | **Mambaliteunet** |  | 交叉门控自适应特征融合的 Mamba 轻量块 | [`032-mambaliteunet`](./032-mambaliteunet) |
| 033 | **Neuroseg** |  | 把 2D 自监督视觉先验迁移到 3D 分割的适配块 | [`033-neuroseg`](./033-neuroseg) |
| 034 | **Assessing** | MICCAI 2026 | 跨人群 nnU-Net 泛化评估与稳健性分析 | [`034-assessing`](./034-assessing) |
| 035 | **Ubone3d** | ECCV 2026 | 物理校正的条件流匹配，做解剖结构生成式分割 | [`035-ubone3d`](./035-ubone3d) |
| 036 | **Latent** | MICCAI 2026 | 潜空间到潜空间的流匹配，体积随机分割模块 | [`036-latent`](./036-latent) |
| 037 | **Evaluating** | MICCAI 2026 | 观测者/模型方差对分割评估影响的度量块 | [`037-evaluating`](./037-evaluating) |
| 038 | **Improving** | MICCAI 2026 | 跨中心全心分割的域适应改进模块 | [`038-improving`](./038-improving) |
| 039 | **Native** | MICCAI 2026 | 原生空间 3D CarveMix，多点脑卒中分割增广块 | [`039-native`](./039-native) |
| 040 | **Get** | ECCV 2026 | 生成式嵌入翻译，做医学图像分割的跨模态映射 | [`040-get`](./040-get) |
| 041 | **When** | MICCAI 2026 | 表征漂移与自适应代价的诊断/正则块 | [`041-when`](./041-when) |
| 042 | **Inference** | MICCAI 2026 | 推理时正交播种，几何对齐的免训练分割模块 | [`042-inference`](./042-inference) |
<!-- CATALOG:END -->

## 单个模块文件夹结构

```
001-fasternet/
├── readme.md       # 讲解：出处 / 核心思想 / 插哪 / 插入示例 / 实测注意点
├── figure.png      # 架构图（截自论文原文）
├── fasternet.py    # 复现代码（PyTorch，逐行中文注释）
├── demo.py         # 插入示例
└── meta.json       # 元信息（会议 / 论文链接 / 官方代码）
```

## 免责与来源

- 所有模块均**来自已公开发表的论文**，架构图截自论文原文，代码为**独立复现**，仅用于学习与科研。
- 版权归各论文原作者所有，官方实现见每个文件夹 `meta.json` 中的 `code` 字段。

## 关于

医学图像分割 / 算法代码 —— 公众号 **MediVision**。

> 📚 **模块索引**：[INDEX.md](INDEX.md)
