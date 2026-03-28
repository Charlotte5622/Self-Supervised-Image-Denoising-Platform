"""
演示用深度学习模型 —— 用于验证平台是否正常工作
正式使用时可删除此文件，替换为你自己的算法
"""
import torch
import torch.nn as nn
from denoising_platform.core.models import register_model, SelfSupervisedTrainable


@register_model('dncnn', category='pretrained', display_name='DnCNN')
class DemoDnCNN(nn.Module):
    """DnCNN 残差去噪网络（预训练推理模式演示）"""

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
        return x - self.net(x)


@register_model('simpleunet', category='self_supervised', display_name='SimpleUNet')
class DemoUNet(nn.Module, SelfSupervisedTrainable):
    """轻量 U-Net（自监督训练模式演示）"""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.enc1 = self._block(in_channels, 48)
        self.enc2 = self._block(48, 48)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._block(48, 48)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = self._block(96, 96)
        self.dec1 = self._block(144, 96)
        self.final = nn.Conv2d(96, in_channels, kernel_size=1)

    @staticmethod
    def _block(ic, oc):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(oc, oc, 3, padding=1), nn.LeakyReLU(0.1, True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return x + self.final(d1)

    def self_supervised_train_step(self, noisy_img, optimizer, epoch, **kwargs):
        self.train()
        B, C, H, W = noisy_img.shape
        mask = torch.zeros(1, 1, H, W, device=noisy_img.device)
        mask[:, :, 0::2, 0::2] = 1
        mask[:, :, 1::2, 1::2] = 1
        mask_c = 1 - mask

        output = self(noisy_img * mask)
        loss = nn.functional.mse_loss(output * mask_c, noisy_img * mask_c)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {'loss': loss.item()}
