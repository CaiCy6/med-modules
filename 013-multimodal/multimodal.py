# -*- coding: utf-8 -*-
"""
【医学图像分割模块】Multimodal Routing and Region Refinement —— Multimodal Routing and Region Refinement for Language-Guided

复现代码（PyTorch）。出处见 readme.md。
自论文原文；仅用于学习/科研，版权归原作者。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalRoutingRegionRefinement(nn.Module):
    """
    即插即用模块：Multimodal Routing + Region Refinement
    输入：
        visual_feat: (B, C_v, H, W)  编码器最深层视觉特征
        text_feat:   (B, L, C_t)     文本 token 序列（如 PubMedBERT 输出）
    输出：
        refined:     (B, C_v, H, W)  形状与输入视觉特征一致，可直接接解码器
    """

    def __init__(self, visual_dim, text_dim, hidden_dim=128, num_routes=4):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.num_routes = num_routes

        # ---- 1. 把文本 token 压成一个全局文本向量 ----
        # 用注意力池化而不是简单平均，让关键 token（如 finding/location 词）权重更大
        self.text_score = nn.Linear(text_dim, 1)

        # ---- 2. 联合路由器：图文拼接后生成路由权重 ----
        # 输入是 [视觉全局向量 ; 文本全局向量]，输出 num_routes 个路由权重
        self.router = nn.Sequential(
            nn.Linear(visual_dim + text_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_routes),
        )

        # ---- 3. 每条路由对应一组视觉/文本更新方向 ----
        # 用低秩方式参数化，避免参数量爆炸：route 权重 -> 组合若干基向量
        self.visual_basis = nn.Parameter(torch.randn(num_routes, visual_dim) * 0.02)
        self.text_basis = nn.Parameter(torch.randn(num_routes, text_dim) * 0.02)

        # ---- 4. 区域细化：把路由后的多模态信息按空间位置重加权 ----
        # 文本向量投影到视觉通道空间，再和视觉特征做空间注意力
        self.text_to_visual = nn.Linear(text_dim, visual_dim)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(visual_dim, visual_dim, kernel_size=3, padding=1, groups=visual_dim),
            nn.GELU(),
            nn.Conv2d(visual_dim, visual_dim, kernel_size=1),
        )
        self.norm = nn.GroupNorm(num_groups=min(32, visual_dim), num_channels=visual_dim)

    def forward(self, visual_feat, text_feat, text_mask=None):
        B, C, H, W = visual_feat.shape

        # ---------- 文本全局池化（注意力池化） ----------
        # text_feat: (B, L, C_t) -> score: (B, L, 1)
        score = self.text_score(text_feat)
        if text_mask is not None:
            # mask 掉 padding token，避免它们污染池化结果
            score = score.masked_fill(text_mask.unsqueeze(-1) == 0, -1e4)
        attn = torch.softmax(score, dim=1)                 # (B, L, 1)
        text_global = (attn * text_feat).sum(dim=1)        # (B, C_t)

        # ---------- 视觉全局池化 ----------
        visual_global = visual_feat.mean(dim=(2, 3))       # (B, C_v)

        # ---------- 联合路由 ----------
        joint = torch.cat([visual_global, text_global], dim=-1)   # (B, C_v + C_t)
        route_logits = self.router(joint)                         # (B, num_routes)
        route_weights = torch.softmax(route_logits, dim=-1)       # (B, num_routes)

        # 用路由权重组合基向量，得到每个样本自己的视觉/文本更新方向
        visual_delta = route_weights @ self.visual_basis          # (B, C_v)
        text_delta = route_weights @ self.text_basis              # (B, C_t)

        # ---------- 用更新方向调制特征 ----------
        # 视觉：全局方向广播到空间，做残差式调制
        visual_mod = visual_feat + visual_delta[:, :, None, None]
        # 文本：同样做残差调制，供后续细化使用
        text_mod = text_global + text_delta                       # (B, C_t)

        # ---------- 区域细化 ----------
        # 文本向量投影到视觉通道空间，作为空间注意力的 query
        text_query = self.text_to_visual(text_mod)                # (B, C_v)
        # 和视觉特征逐通道做相似度，得到空间响应图
        spatial_attn = (visual_mod * text_query[:, :, None, None]).sum(dim=1, keepdim=True)
        spatial_attn = torch.sigmoid(spatial_attn)                # (B, 1, H, W)

        # 用空间响应图重加权，再过一个深度可分离卷积做局部细化
        refined = visual_mod * spatial_attn
        refined = self.spatial_conv(refined)
        refined = self.norm(refined + visual_feat)                # 残差 + 归一化，稳定训练

        return refined
