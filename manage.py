#!/usr/bin/env python
"""Django 管理脚本"""
import os
import sys

def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'denoising_platform.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "无法导入 Django，请确保已安装并在 PYTHONPATH 中可用。"
        ) from exc
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()
