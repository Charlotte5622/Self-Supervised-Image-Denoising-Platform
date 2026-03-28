"""WSGI 入口"""
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'denoising_platform.settings')
application = get_wsgi_application()
