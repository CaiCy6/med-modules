# 插入示例（塞进你的网络）

class MyUNetWithMedalS(nn.Module):
    def __init__(self, backbone, num_classes, prompt_channels):
        super().__init__()
        self.backbone = backbone
        # 从 backbone 拿到最后一级 decoder 的通道数
        c = backbone.out_channels
        # 插入 Medal S 精修块
        self.medal_s = MedalSBlock(in_channels=c, prompt_channels=prompt_channels)
        # 最后的分割头
        self.head = nn.Conv3d(c, num_classes, kernel_size=1)

    def forward(self, x, prompt):
        feat = self.backbone(x)          # 粗特征 (B, C, D, H, W)
        feat = self.medal_s(feat, prompt)  # 精修
        return self.head(feat)           # logits
