"""
algorithms/ —— 算法实现文件夹

在此文件夹中放置你的去噪算法 .py 文件，平台会自动扫描并注册。

规则：
  1. 文件名不要以 _ 开头（下划线开头的文件会被跳过）
  2. 在文件中 import 注册装饰器并使用：
       from denoising_platform.core.models import register_model
  3. 模型类必须继承 torch.nn.Module 并实现 forward(self, x)
  4. 可选：继承 SelfSupervisedTrainable 并实现自定义训练步

详见 _example.py 中的完整示范。
"""
