"""全局 URL 路由配置"""
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from denoising_platform.core import views

urlpatterns = [
    # ─── 页面 ───
    path('', views.index, name='index'),

    # ─── API 接口 ───
    path('api/upload/', views.upload_image, name='api_upload'),
    path('api/upload/cleanup/', views.cleanup_uploads, name='api_upload_cleanup'),
    path('api/denoise/', views.denoise_image, name='api_denoise'),
    path('api/train/', views.start_training, name='api_train'),
    path('api/train/status/<str:task_id>/', views.training_status, name='api_train_status'),
    path('api/preset-images/', views.preset_images, name='api_preset_images'),
    path('api/models/', views.list_models, name='api_models'),
    path('api/compare/', views.compare_models, name='api_compare'),
]

# 开发模式下直接提供媒体文件服务
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])
