"""
Django 视图层 —— 图像去噪平台的所有 API 端点

端点列表：
  GET  /                      → 数字大屏主页面
  POST /api/upload/            → 上传图像
  POST /api/denoise/           → 预训练模型推理去噪
  POST /api/train/             → 启动自监督训练任务
  GET  /api/train/status/<id>/ → 查询训练进度（轮询）
  POST /api/compare/           → 多算法对比（需 Ground Truth）
  GET  /api/preset-images/     → 获取预置图像配对列表
  GET  /api/models/            → 获取可用模型列表（含分类信息）
"""
import json
import uuid
import threading
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from django.http import JsonResponse
from django.shortcuts import render
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .models import ModelManager, MODEL_REGISTRY
from .metrics import compute_psnr, compute_ssim

logger = logging.getLogger(__name__)

_manager: ModelManager = None
_training_tasks: dict = {}
UPLOAD_RETENTION_SECONDS = 12 * 60 * 60
PRESET_DATASET_LIMIT = 5

to_tensor = transforms.ToTensor()


def _get_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager(weights_dir=settings.MODEL_WEIGHTS_DIR)
    return _manager


# ═══════════════════════════════════════════════════════════
#  图像路径统一解析
# ═══════════════════════════════════════════════════════════

def _resolve_media_path(rel_path: str) -> Path:
    """
    将前端传来的相对路径解析为绝对路径
    支持: 'uploads/abc.png', 'preset/noisy/lena.png' 等
    """
    if not rel_path:
        return Path(settings.MEDIA_ROOT) / '__missing__'
    media_root = Path(settings.MEDIA_ROOT).resolve()
    candidate = (media_root / rel_path).resolve(strict=False)
    try:
        candidate.relative_to(media_root)
    except ValueError:
        return media_root / '__invalid__'
    return candidate


def _save_tensor_as_image(tensor: torch.Tensor, save_path: Path) -> None:
    img_np = tensor.squeeze(0).permute(1, 2, 0).numpy()
    img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_np).save(str(save_path))


def _load_image_as_tensor(image_path: Path) -> torch.Tensor:
    img = Image.open(image_path).convert('RGB')
    return to_tensor(img).unsqueeze(0)


def _is_upload_rel_path(rel_path: str) -> bool:
    return isinstance(rel_path, str) and rel_path.startswith('uploads/')


def _delete_upload_file(rel_path: str) -> bool:
    if not _is_upload_rel_path(rel_path):
        return False
    path = _resolve_media_path(rel_path)
    if path.exists() and path.is_file():
        path.unlink(missing_ok=True)
        return True
    return False


def _touch_upload_file(rel_path: str) -> None:
    if not _is_upload_rel_path(rel_path):
        return
    path = _resolve_media_path(rel_path)
    if path.exists():
        path.touch()


def _cleanup_stale_uploads() -> None:
    upload_dir = Path(settings.MEDIA_ROOT) / 'uploads'
    if not upload_dir.exists():
        return
    expire_before = time.time() - UPLOAD_RETENTION_SECONDS
    for file_path in upload_dir.iterdir():
        if not file_path.is_file():
            continue
        try:
            if file_path.stat().st_mtime < expire_before:
                file_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("清理上传缓存失败: %s", file_path)


def _iter_preset_datasets():
    preset_root = Path(settings.MEDIA_ROOT) / 'preset'
    preset_root.mkdir(parents=True, exist_ok=True)

    datasets = []
    for child in sorted(preset_root.iterdir()):
        if not child.is_dir() or child.name in {'noisy', 'clean'}:
            continue
        datasets.append({
            'id': child.name,
            'name': child.name,
            'root_dir': child,
            'noisy_dir': child / 'noisy',
            'clean_dir': child / 'clean',
        })

    if datasets:
        return datasets[:PRESET_DATASET_LIMIT]

    legacy_noisy = preset_root / 'noisy'
    legacy_clean = preset_root / 'clean'
    if legacy_noisy.exists() or legacy_clean.exists():
        datasets.append({
            'id': 'default',
            'name': '默认数据集（SIDD）',
            'root_dir': preset_root,
            'noisy_dir': legacy_noisy,
            'clean_dir': legacy_clean,
        })

    return datasets[:PRESET_DATASET_LIMIT]


def _preset_stem_candidates(stem: str) -> list[str]:
    """Generate candidate names for matching noisy/clean preset files."""
    if not stem:
        return []

    candidates = [stem]
    seen = {stem.lower()}

    def add(value: str) -> None:
        key = value.lower()
        if value and key not in seen:
            candidates.append(value)
            seen.add(key)

    suffix_groups = (
        ('_n', '_cl'),
        ('_real', '_mean'),
        ('_real',),
        ('_mean',),
        ('_cl',),
        ('_n',),
    )

    stem_lower = stem.lower()
    for group in suffix_groups:
        for suffix in group:
            if stem_lower.endswith(suffix):
                base = stem[:-len(suffix)]
                add(base)
                for alt in group:
                    add(base + alt)

    return candidates


def _match_preset_clean_name(noisy_stem: str, clean_name_map: dict[str, str]) -> Optional[str]:
    for candidate in _preset_stem_candidates(noisy_stem):
        matched = clean_name_map.get(candidate.lower())
        if matched:
            return matched
    return None


# ═══════════════════════════════════════════════════════════
#  页面渲染
# ═══════════════════════════════════════════════════════════

def index(request):
    return render(request, 'index.html')


# ═══════════════════════════════════════════════════════════
#  图像上传
# ═══════════════════════════════════════════════════════════

@csrf_exempt
@require_POST
def upload_image(request):
    _cleanup_stale_uploads()

    file = request.FILES.get('image')
    if not file:
        return JsonResponse({'error': '请选择要上传的图像文件'}, status=400)

    allowed = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    suffix = Path(file.name).suffix.lower()
    if suffix not in allowed:
        return JsonResponse({'error': f'不支持的格式: {suffix}'}, status=400)

    uid = uuid.uuid4().hex[:8]
    filename = f"{uid}{suffix}"
    upload_dir = Path(settings.MEDIA_ROOT) / 'uploads'
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / filename

    with open(save_path, 'wb') as f:
        for chunk in file.chunks():
            f.write(chunk)

    old_file = request.POST.get('replace', '')
    if old_file:
        _delete_upload_file(old_file)

    return JsonResponse({
        'success': True,
        'filename': f'uploads/{filename}',
        'url': f'{settings.MEDIA_URL}uploads/{filename}',
    })


@csrf_exempt
@require_POST
def cleanup_uploads(request):
    _cleanup_stale_uploads()

    payload = {}
    if request.body:
        try:
            payload = json.loads(request.body)
        except json.JSONDecodeError:
            payload = {}

    filenames = payload.get('filenames')
    if not isinstance(filenames, list):
        filenames = request.POST.getlist('filenames')

    deleted = 0
    for rel_path in filenames:
        deleted += int(_delete_upload_file(rel_path))

    return JsonResponse({'success': True, 'deleted': deleted})


# ═══════════════════════════════════════════════════════════
#  预训练模型推理去噪
# ═══════════════════════════════════════════════════════════

@csrf_exempt
@require_POST
def denoise_image(request):
    """
    POST body: { filename, model, ground_truth? }
    filename/ground_truth 均为相对于 media/ 的路径
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': '请求体必须是 JSON'}, status=400)

    filename = body.get('filename', '')
    model_name = body.get('model', 'dncnn')
    gt_filename = body.get('ground_truth', '')

    _cleanup_stale_uploads()
    _touch_upload_file(filename)
    _touch_upload_file(gt_filename)

    input_path = _resolve_media_path(filename)
    if not input_path.exists():
        return JsonResponse({'error': f'图像不存在: {filename}'}, status=404)

    mgr = _get_manager()
    model_category = mgr.get_category(model_name)
    if model_category == 'self_supervised':
        return JsonResponse({'error': '推理去噪模块不支持自监督训练算法，请切换到自监督训练模块。'}, status=400)
    input_tensor = _load_image_as_tensor(input_path)

    start_time = time.time()
    output_tensor = mgr.inference(model_name, input_tensor)
    elapsed = round(time.time() - start_time, 3)

    result_name = f"denoised_{uuid.uuid4().hex[:8]}.png"
    result_dir = Path(settings.MEDIA_ROOT) / 'results'
    result_dir.mkdir(parents=True, exist_ok=True)
    _save_tensor_as_image(output_tensor, result_dir / result_name)

    response = {
        'success': True,
        'result_url': f'{settings.MEDIA_URL}results/{result_name}',
        'input_url': f'{settings.MEDIA_URL}{filename}',
        'elapsed_ms': int(elapsed * 1000),
        'device': str(mgr.device),
    }

    if gt_filename:
        gt_path = _resolve_media_path(gt_filename)
        if gt_path.exists():
            response['gt_url'] = f'{settings.MEDIA_URL}{gt_filename}'
            gt_tensor = _load_image_as_tensor(gt_path)
            response['psnr_before'] = round(compute_psnr(input_tensor, gt_tensor), 2)
            response['ssim_before'] = round(compute_ssim(input_tensor, gt_tensor), 4)
            response['psnr_after'] = round(compute_psnr(output_tensor, gt_tensor), 2)
            response['ssim_after'] = round(compute_ssim(output_tensor, gt_tensor), 4)

    return JsonResponse(response)


# ═══════════════════════════════════════════════════════════
#  自监督训练（异步线程 + 轮询）
# ═══════════════════════════════════════════════════════════

def _default_train_step(model, input_tensor, optimizer, device):
    _, C, H, W = input_tensor.shape
    model.train()
    mask = torch.zeros(1, 1, H, W, device=device)
    mask[:, :, 0::2, 0::2] = 1
    mask[:, :, 1::2, 1::2] = 1
    mask_comp = 1 - mask
    output = model(input_tensor * mask)
    loss = torch.nn.functional.mse_loss(output * mask_comp, input_tensor * mask_comp)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {'loss': loss.item()}


def _self_supervised_train(task_id: str, image_path: Path, config: dict):
    task = _training_tasks[task_id]
    try:
        mgr = _get_manager()
        device = mgr.device
        total_epochs = config.get('epochs', 200)
        lr = config.get('lr', 1e-3)
        arch = config.get('arch', list(MODEL_REGISTRY.keys())[0] if MODEL_REGISTRY else 'simpleunet')

        input_tensor = _load_image_as_tensor(image_path).to(device)

        model = mgr.get_fresh_model(arch=arch)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
        use_custom = mgr.is_custom_trainable(arch)

        task.update({
            'status': 'running', 'total_epochs': total_epochs, 'losses': [],
            'train_mode': 'custom' if use_custom else 'default_checkerboard',
        })

        for epoch in range(1, total_epochs + 1):
            if task.get('cancel'):
                task['status'] = 'cancelled'
                return

            if use_custom:
                step_result = model.self_supervised_train_step(input_tensor, optimizer, epoch, total_epochs=total_epochs)
            else:
                step_result = _default_train_step(model, input_tensor, optimizer, device)
            scheduler.step()

            task['epoch'] = epoch
            task['losses'].append(round(step_result['loss'], 6))
            task['current_lr'] = round(scheduler.get_last_lr()[0], 8)
            if 'extra_metrics' in step_result:
                task['extra_metrics'] = step_result['extra_metrics']

            if epoch % max(1, total_epochs // 10) == 0 or epoch == total_epochs:
                model.eval()
                with torch.no_grad():
                    denoised = model(input_tensor).clamp(0, 1).cpu()
                inter_name = f"inter_{task_id}_e{epoch}.png"
                inter_dir = Path(settings.MEDIA_ROOT) / 'intermediate'
                inter_dir.mkdir(parents=True, exist_ok=True)
                _save_tensor_as_image(denoised, inter_dir / inter_name)
                task['intermediate_url'] = f'{settings.MEDIA_URL}intermediate/{inter_name}'

        model.eval()
        with torch.no_grad():
            final_output = model(input_tensor).clamp(0, 1).cpu()

        result_name = f"ss_result_{task_id}.png"
        result_dir = Path(settings.MEDIA_ROOT) / 'results'
        result_dir.mkdir(parents=True, exist_ok=True)
        _save_tensor_as_image(final_output, result_dir / result_name)
        task.update({'status': 'completed', 'result_url': f'{settings.MEDIA_URL}results/{result_name}'})

        gt_rel = config.get('ground_truth', '')
        if gt_rel:
            gt_path = _resolve_media_path(gt_rel)
            if gt_path.exists():
                gt_tensor = _load_image_as_tensor(gt_path)
                task['psnr_before'] = round(compute_psnr(input_tensor.cpu(), gt_tensor), 2)
                task['ssim_before'] = round(compute_ssim(input_tensor.cpu(), gt_tensor), 4)
                task['psnr'] = round(compute_psnr(final_output, gt_tensor), 2)
                task['ssim'] = round(compute_ssim(final_output, gt_tensor), 4)

    except Exception as e:
        logger.exception(f"训练任务 {task_id} 失败")
        task.update({'status': 'error', 'error': str(e)})


@csrf_exempt
@require_POST
def start_training(request):
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': '请求体必须是 JSON'}, status=400)

    filename = body.get('filename', '')
    gt_filename = body.get('ground_truth', '')
    _cleanup_stale_uploads()
    _touch_upload_file(filename)
    _touch_upload_file(gt_filename)
    image_path = _resolve_media_path(filename)
    if not image_path.exists():
        return JsonResponse({'error': f'图像不存在: {filename}'}, status=404)

    task_id = uuid.uuid4().hex[:12]
    _training_tasks[task_id] = {
        'status': 'initializing', 'epoch': 0,
        'total_epochs': body.get('epochs', 200), 'losses': [], 'filename': filename,
    }

    thread = threading.Thread(target=_self_supervised_train, args=(task_id, image_path, body), daemon=True)
    thread.start()

    return JsonResponse({'success': True, 'task_id': task_id, 'message': '自监督训练任务已启动'})


@require_GET
def training_status(request, task_id: str):
    task = _training_tasks.get(task_id)
    if not task:
        return JsonResponse({'error': '任务不存在'}, status=404)

    return JsonResponse({
        'task_id': task_id,
        'status': task.get('status', 'unknown'),
        'epoch': task.get('epoch', 0),
        'total_epochs': task.get('total_epochs', 0),
        'losses': task.get('losses', [])[-50:],
        'all_losses': task.get('losses', []),
        'current_lr': task.get('current_lr', 0),
        'intermediate_url': task.get('intermediate_url', ''),
        'result_url': task.get('result_url', ''),
        'psnr_before': task.get('psnr_before', None),
        'ssim_before': task.get('ssim_before', None),
        'psnr': task.get('psnr', None),
        'ssim': task.get('ssim', None),
        'error': task.get('error', ''),
    })


# ═══════════════════════════════════════════════════════════
#  预置配对图像
# ═══════════════════════════════════════════════════════════

@require_GET
def preset_images(request):
    """
    扫描 media/preset/<dataset>/{noisy,clean}/
    同名文件自动配对；兼容旧结构 media/preset/{noisy,clean}/
    """
    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    datasets = []
    for dataset in _iter_preset_datasets():
        root_dir = dataset['root_dir']
        noisy_dir = dataset['noisy_dir']
        clean_dir = dataset['clean_dir']

        active_noisy_dir = noisy_dir if noisy_dir.exists() else root_dir
        active_clean_dir = clean_dir if clean_dir.exists() else None

        clean_stems = {}
        if active_clean_dir and active_clean_dir.exists():
            for file_path in active_clean_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in exts:
                    clean_stems[file_path.stem.lower()] = file_path.name

        dataset_prefix = f"preset/{dataset['name']}"
        if dataset['id'] == 'default':
            dataset_prefix = 'preset'

        pairs = []
        if active_noisy_dir.exists():
            for file_path in sorted(active_noisy_dir.iterdir()):
                if not file_path.is_file() or file_path.suffix.lower() not in exts:
                    continue
                noisy_rel_dir = 'noisy' if active_noisy_dir == noisy_dir else ''
                clean_rel_dir = 'clean' if active_clean_dir == clean_dir else ''
                noisy_base = f'{dataset_prefix}/{noisy_rel_dir}'.rstrip('/')
                clean_base = f'{dataset_prefix}/{clean_rel_dir}'.rstrip('/')
                entry = {
                    'name': file_path.stem,
                    'noisy_file': f'{noisy_base}/{file_path.name}',
                    'noisy_url': f'{settings.MEDIA_URL}{noisy_base}/{file_path.name}',
                    'has_gt': False,
                }
                gt_name = _match_preset_clean_name(file_path.stem, clean_stems)
                if gt_name:
                    entry.update({
                        'has_gt': True,
                        'clean_file': f'{clean_base}/{gt_name}',
                        'clean_url': f'{settings.MEDIA_URL}{clean_base}/{gt_name}',
                    })
                pairs.append(entry)

        datasets.append({
            'id': dataset['id'],
            'name': dataset['name'],
            'count': len(pairs),
            'presets': pairs,
        })

    return JsonResponse({'datasets': datasets})


# ═══════════════════════════════════════════════════════════
#  模型列表
# ═══════════════════════════════════════════════════════════

@require_GET
def list_models(request):
    mgr = _get_manager()
    return JsonResponse({'models': mgr.list_available(), 'device': str(mgr.device)})


# ═══════════════════════════════════════════════════════════
#  多算法对比
# ═══════════════════════════════════════════════════════════

@csrf_exempt
@require_POST
def compare_models(request):
    """
    POST body: { filename, ground_truth, models[] }
    ground_truth 在对比模式下必需
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': '请求体必须是 JSON'}, status=400)

    filename = body.get('filename', '')
    gt_filename = body.get('ground_truth', '')
    model_names = body.get('models', [])

    _cleanup_stale_uploads()
    _touch_upload_file(filename)
    _touch_upload_file(gt_filename)

    if not gt_filename:
        return JsonResponse({'error': '多算法对比需要提供 Ground Truth'}, status=400)
    if not model_names:
        return JsonResponse({'error': '请至少选择一个算法'}, status=400)

    input_path = _resolve_media_path(filename)
    gt_path = _resolve_media_path(gt_filename)
    if not input_path.exists():
        return JsonResponse({'error': f'含噪图不存在: {filename}'}, status=404)
    if not gt_path.exists():
        return JsonResponse({'error': f'真值图不存在: {gt_filename}'}, status=404)

    mgr = _get_manager()
    input_tensor = _load_image_as_tensor(input_path)
    gt_tensor = _load_image_as_tensor(gt_path)

    baseline_psnr = round(compute_psnr(input_tensor, gt_tensor), 2)
    baseline_ssim = round(compute_ssim(input_tensor, gt_tensor), 4)

    results = []
    result_dir = Path(settings.MEDIA_ROOT) / 'results'
    result_dir.mkdir(parents=True, exist_ok=True)

    for mname in model_names:
        entry = {'model': mname, 'display_name': mname}
        reg = MODEL_REGISTRY.get(mname)
        if reg:
            entry['display_name'] = reg['display_name']
            entry['category'] = reg['category']
        try:
            t0 = time.time()
            output_tensor = mgr.inference(mname, input_tensor)
            elapsed = round(time.time() - t0, 3)
            rname = f"cmp_{mname}_{uuid.uuid4().hex[:6]}.png"
            _save_tensor_as_image(output_tensor, result_dir / rname)
            entry.update({
                'success': True,
                'psnr': round(compute_psnr(output_tensor, gt_tensor), 2),
                'ssim': round(compute_ssim(output_tensor, gt_tensor), 4),
                'elapsed_ms': int(elapsed * 1000),
                'result_url': f'{settings.MEDIA_URL}results/{rname}',
            })
        except Exception as e:
            logger.error(f"对比模型 {mname} 推理失败: {e}")
            entry.update({'success': False, 'error': str(e)})
        results.append(entry)

    return JsonResponse({
        'success': True,
        'baseline': {'psnr': baseline_psnr, 'ssim': baseline_ssim},
        'results': results,
        'input_url': f'{settings.MEDIA_URL}{filename}',
        'gt_url': f'{settings.MEDIA_URL}{gt_filename}',
    })
