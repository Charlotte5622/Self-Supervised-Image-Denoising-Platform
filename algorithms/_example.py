"""
═══════════════════════════════════════════════════════════
  算法接入示范文件（文件名以 _ 开头，不会被自动加载）
  复制此文件并去掉前缀 _ 即可使你的算法生效
═══════════════════════════════════════════════════════════

下面展示两种接入方式：
  A) 仅推理模型（最简单）
  B) 带自定义自监督训练逻辑的模型
"""

import torch
import torch.nn as nn

# ─── 从平台核心导入注册装饰器和训练接口 ───
from denoising_platform.core.models import register_model, SelfSupervisedTrainable


# ═══════════════════════════════════════════════════════════
#  示范 A：仅推理模型（最低接入门槛）
# ═══════════════════════════════════════════════════════════

@register_model('example_dncnn')
class ExampleDnCNN(nn.Module):
    """
    接入要求：
      1. 继承 nn.Module
      2. forward(x) 接收 [B, C, H, W]，返回同尺寸张量
      3. 用 @register_model('名称') 注册

    权重文件放在 models/weights/example_dncnn.pth
    平台会自动加载（如果存在的话）
    """

    def __init__(self, in_channels: int = 3, num_layers: int = 17, features: int = 64):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, features, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Conv2d(features, features, 3, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.Conv2d(features, in_channels, 3, padding=1, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.net(x)  # 残差学习


# ═══════════════════════════════════════════════════════════
#  示范 B：自定义自监督训练模型
# ═══════════════════════════════════════════════════════════

@register_model('example_n2v')
class ExampleNoise2Void(nn.Module, SelfSupervisedTrainable):
    """
    接入要求（在 A 基础上额外）：
      4. 同时继承 SelfSupervisedTrainable
      5. 实现 self_supervised_train_step(noisy_img, optimizer, epoch, **kwargs)
         返回 dict，至少包含 'loss' (float)

    实现此接口后，平台的"自监督训练模式"会调用你的方法
    而不是默认的棋盘掩码策略。
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(48, in_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))

    def self_supervised_train_step(
        self,
        noisy_img: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        **kwargs,
    ) -> dict:
        """
        Noise2Void 风格：随机遮盖像素，用周围像素预测被遮盖位置

        返回值规范：
          必须: {'loss': float}
          可选: {'denoised': Tensor, 'extra_metrics': dict}
        """
        self.train()
        B, C, H, W = noisy_img.shape

        # 生成随机盲点掩码（约 0.5% 的像素）
        mask = torch.rand(B, 1, H, W, device=noisy_img.device) < 0.005
        mask = mask.float()

        # 被遮盖位置用邻域均值替代作为输入
        kernel = torch.ones(1, 1, 3, 3, device=noisy_img.device) / 9.0
        smoothed = torch.nn.functional.conv2d(
            noisy_img, kernel.expand(C, -1, -1, -1), padding=1, groups=C
        )
        model_input = noisy_img * (1 - mask) + smoothed * mask

        output = self(model_input)
        loss = torch.nn.functional.mse_loss(output * mask, noisy_img * mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {'loss': loss.item()}
