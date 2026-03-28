"""
模型管理层 —— 接口规范、分类注册机制和统一管理器

具体的网络架构请放在项目根目录的 algorithms/ 文件夹中，
通过 @register_model 装饰器自动注册即可接入平台。

═══════════════════════════════════════════════════════════
  三种算法类别 (category)：
═══════════════════════════════════════════════════════════

  1. 'pretrained'       —— 有预训练权重的深度学习模型，直接推理
  2. 'self_supervised'  —— 单图像自监督去噪，需先训练再去噪
  3. 'traditional'      —— 传统图像处理算法（高斯/中值/BM3D 等），无需权重

═══════════════════════════════════════════════════════════
  接口约定（你的算法必须遵守）：
═══════════════════════════════════════════════════════════

  1. 继承 nn.Module
  2. forward(self, x) 接收 [B, C, H, W] 张量 (float32, 值域 [0,1])
     返回同尺寸去噪后张量
  3. 使用 @register_model('name', category='...') 装饰器注册
  4. category='self_supervised' 的模型可额外实现 SelfSupervisedTrainable
"""
import abc
import importlib
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Callable, List, Any
import logging

logger = logging.getLogger(__name__)

VALID_CATEGORIES = ('pretrained', 'self_supervised', 'traditional')

# ═══════════════════════════════════════════════════════════
#  全局模型注册表
# ═══════════════════════════════════════════════════════════

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}
# 结构: { 'model_name': { 'cls': ModelClass, 'category': str, 'display_name': str } }


def register_model(
    name: str,
    category: str = 'pretrained',
    display_name: str = '',
) -> Callable:
    """
    模型注册装饰器 —— 在 algorithms/ 中使用

    Args:
        name: 唯一标识符，如 'dncnn', 'gaussian_filter'
        category: 算法类别
            - 'pretrained': 有预训练权重，直接推理
            - 'self_supervised': 单图自监督，需训练
            - 'traditional': 传统算法，无需权重无需训练
        display_name: 前端显示名称（可选，默认用 name）

    用法:
        @register_model('gaussian_filter', category='traditional', display_name='高斯滤波')
        class GaussianFilter(nn.Module):
            ...
    """
    if category not in VALID_CATEGORIES:
        raise ValueError(f"category 必须是 {VALID_CATEGORIES} 之一，收到: '{category}'")

    def decorator(cls: type) -> type:
        if name in MODEL_REGISTRY:
            logger.warning(f"模型 '{name}' 已存在，将被覆盖为 {cls.__name__}")
        MODEL_REGISTRY[name] = {
            'cls': cls,
            'category': category,
            'display_name': display_name or name,
        }
        logger.info(f"✓ 注册模型: '{name}' [{category}] → {cls.__name__}")
        return cls
    return decorator


# ═══════════════════════════════════════════════════════════
#  自监督训练策略接口（可选实现）
# ═══════════════════════════════════════════════════════════

class SelfSupervisedTrainable(abc.ABC):
    """
    可选 Mixin：category='self_supervised' 的模型如果有自定义训练流程，
    让模型类同时继承 nn.Module 和 SelfSupervisedTrainable。
    如不实现，平台使用默认的棋盘掩码策略训练。
    """

    @abc.abstractmethod
    def self_supervised_train_step(
        self,
        noisy_img: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        **kwargs,
    ) -> dict:
        """
        单步自监督训练，返回 dict 至少包含 'loss' (float)
        可选: 'denoised' (Tensor), 'extra_metrics' (dict)
        """
        ...


# ═══════════════════════════════════════════════════════════
#  算法自动发现
# ═══════════════════════════════════════════════════════════

def _discover_algorithms(algorithms_dir: Optional[Path] = None) -> None:
    """自动导入 algorithms/ 下所有非 _ 开头的 .py 文件"""
    if algorithms_dir is None:
        algorithms_dir = Path(__file__).resolve().parent.parent.parent / 'algorithms'

    if not algorithms_dir.is_dir():
        logger.warning(f"算法目录不存在: {algorithms_dir}")
        return

    import sys
    project_root = str(algorithms_dir.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    for py_file in sorted(algorithms_dir.glob('*.py')):
        if py_file.name.startswith('_'):
            continue
        module_name = f"algorithms.{py_file.stem}"
        try:
            importlib.import_module(module_name)
            logger.info(f"已加载算法模块: {module_name}")
        except Exception as e:
            logger.error(f"加载算法模块 {module_name} 失败: {e}", exc_info=True)


# ═══════════════════════════════════════════════════════════
#  模型管理器
# ═══════════════════════════════════════════════════════════

class ModelManager:
    """
    统一模型管理器

    用法:
        mgr = ModelManager(weights_dir=Path('models/weights'))
        output = mgr.inference('dncnn', input_tensor)
    """

    def __init__(self, weights_dir: Path, algorithms_dir: Optional[Path] = None):
        self.weights_dir = weights_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._cache: Dict[str, nn.Module] = {}

        _discover_algorithms(algorithms_dir)
        logger.info(
            f"ModelManager 初始化完成 | 设备: {self.device} | "
            f"已注册模型: {list(MODEL_REGISTRY.keys())}"
        )

    def _resolve_arch_key(self, model_name: str) -> str:
        if model_name in MODEL_REGISTRY:
            return model_name
        prefix = model_name.split('_')[0].lower()
        if prefix in MODEL_REGISTRY:
            return prefix
        return ''

    def _resolve_model(self, model_name: str) -> nn.Module:
        """按名称查找或加载模型，带内存缓存"""
        if model_name in self._cache:
            return self._cache[model_name]

        arch_key = self._resolve_arch_key(model_name)
        if not arch_key:
            raise ValueError(
                f"未知模型: '{model_name}'\n"
                f"已注册: {list(MODEL_REGISTRY.keys())}\n"
                f"请在 algorithms/ 中添加模型并用 @register_model 注册"
            )

        entry = MODEL_REGISTRY[arch_key]
        model = entry['cls']()

        custom_loaded = False
        if entry['category'] != 'traditional':
            custom_loader = getattr(model, 'load_pretrained_weights', None)
            if callable(custom_loader):
                try:
                    custom_loaded = bool(custom_loader(self.weights_dir, model_name))
                    if custom_loaded:
                        logger.info("模型 %s 使用自定义权重加载器", model_name)
                except Exception:
                    logger.error("自定义权重加载失败: %s", model_name, exc_info=True)

            if not custom_loaded:
                weight_path = self.weights_dir / f"{model_name}.pth"
                if weight_path.exists():
                    state = torch.load(weight_path, map_location=self.device, weights_only=True)
                    model.load_state_dict(state)
                    logger.info(f"已加载权重: {weight_path}")
                else:
                    logger.warning(f"权重文件不存在({weight_path})，使用随机初始化（演示模式）")

        model = model.to(self.device).eval()
        self._cache[model_name] = model
        return model

    @torch.no_grad()
    def inference(self, model_name: str, input_tensor: torch.Tensor) -> torch.Tensor:
        model = self._resolve_model(model_name)
        input_tensor = input_tensor.to(self.device)
        output = model(input_tensor)
        return output.clamp(0, 1).cpu()

    def get_fresh_model(self, arch: str = 'simpleunet') -> nn.Module:
        if arch not in MODEL_REGISTRY:
            raise ValueError(f"未知架构: '{arch}'，可用: {list(MODEL_REGISTRY.keys())}")
        return MODEL_REGISTRY[arch]['cls']().to(self.device)

    def is_custom_trainable(self, arch: str) -> bool:
        if arch not in MODEL_REGISTRY:
            return False
        return issubclass(MODEL_REGISTRY[arch]['cls'], SelfSupervisedTrainable)

    def get_category(self, model_name: str) -> str:
        key = self._resolve_arch_key(model_name)
        if key and key in MODEL_REGISTRY:
            return MODEL_REGISTRY[key]['category']
        return 'pretrained'

    def list_available(self) -> List[dict]:
        """返回所有可用模型的详细信息列表"""
        result = []
        seen = set()
        for name, entry in MODEL_REGISTRY.items():
            result.append({
                'name': name,
                'category': entry['category'],
                'display_name': entry['display_name'],
            })
            seen.add(name)
        if self.weights_dir.exists():
            for f in self.weights_dir.glob('*.pth'):
                if f.stem not in seen:
                    result.append({
                        'name': f.stem,
                        'category': 'pretrained',
                        'display_name': f.stem,
                    })
        result.sort(key=lambda x: (x['category'], x['name']))
        return result

    def list_by_category(self, category: str) -> List[dict]:
        return [m for m in self.list_available() if m['category'] == category]

    def clear_cache(self, model_name: Optional[str] = None) -> None:
        if model_name:
            self._cache.pop(model_name, None)
        else:
            self._cache.clear()
        torch.cuda.empty_cache()
