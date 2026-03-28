# 实时自监督学习图像去噪处理平台

一个基于 Django 的图像去噪可视化平台，支持：

- 单模型推理去噪
- 自监督训练与 Loss 曲线展示
- 多算法对比、局部 ROI 指标分析、分布图分析
- 预置数据集选择、临时上传清理、模型动态发现

项目当前以文件系统管理数据，不依赖关系型数据库。
## 0. 项目UI预览
<img width="1912" height="994" alt="image" src="https://github.com/user-attachments/assets/5cf93446-8f93-4e9f-b1bf-8bfa0d095830" />
<img width="1912" height="994" alt="image" src="https://github.com/user-attachments/assets/010e6b08-738e-48d6-af93-3cefa959935b" />
<img width="1912" height="994" alt="image" src="https://github.com/user-attachments/assets/98bc884a-51ec-4fbf-bbdb-896f956c4f75" />
<img width="1912" height="994" alt="image" src="https://github.com/user-attachments/assets/8d7ed1cc-f724-4868-8b14-8dc688ed9472" />
<img width="1912" height="994" alt="image" src="https://github.com/user-attachments/assets/4aa226ba-bea0-4789-aeb2-0cb3c74abcaf" />
<img width="1912" height="994" alt="image" src="https://github.com/user-attachments/assets/b8732dbd-815c-4e67-b6b7-ca2dc9686a81" />



## 1. 目录说明

```text
Project_denosing/
├── algorithms/              # 各类去噪算法接口与模型封装
├── denoising_platform/      # Django 项目代码
├── media/                   # 预置图片、上传图片、推理结果、训练中间结果
├── models/weights/          # 预训练权重
├── static/                  # 前端静态资源
├── templates/               # Django 模板
├── manage.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## 2. 环境要求

建议环境：

- Python 3.10
- pip 23+
- Linux / macOS / Windows（Docker 推荐 Linux）

本项目无数据库迁移要求，但首次运行前请确保以下目录存在（Docker 会自动创建）：

- `media/uploads`
- `media/results`
- `media/intermediate`
- `media/preset`
- `models/weights`

## 3. 本地运行

### 3.1 创建虚拟环境

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3.2 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` 默认通过 PyTorch CPU 源安装 `torch` / `torchvision`，这样对大多数机器更容易直接跑起来。

如果你本机已经有 CUDA 版 PyTorch，也可以自行替换为你自己的 GPU 版本。

### 3.3 启动项目

```bash
python manage.py runserver 0.0.0.0:8000
```

浏览器访问：

```text
http://127.0.0.1:8000
```

### 3.4 可选环境变量

```bash
export DJANGO_DEBUG=True
export DJANGO_SECRET_KEY=dev-change-me
```

## 4. Docker 一键运行

### 4.1 准备环境变量

复制一份示例配置：

```bash
cp .env.example .env
```

如无特殊需求，默认配置即可。

### 4.2 启动

```bash
docker compose up --build
```

启动后访问：

```text
http://127.0.0.1:8000
```

### 4.3 Docker 挂载说明

`docker-compose.yml` 默认挂载：

- `./media:/app/media`
- `./models:/app/models`

这样做的好处是：

- 预训练权重不会丢
- 预置数据集不会丢
- 推理结果、训练中间结果、上传缓存都保留在宿主机

## 5. 模型权重与数据放置

### 5.1 预训练权重

请把权重文件放到：

```text
models/weights/
```

常见形式：

```text
models/weights/apbsn.pth
models/weights/cvfsid.pth
models/weights/mmbsn.pth
```

### 5.2 预置数据集

目前前端支持最多 5 个数据集入口，推荐目录结构：

```text
media/preset/<dataset_name>/
├── noisy/
└── clean/
```

例如：

```text
media/preset/SIDD/noisy/0001.png
media/preset/SIDD/clean/0001.png
```

系统会按同名文件自动配对。

## 6. 常用命令

### 本地开发

```bash
python manage.py runserver 0.0.0.0:8000
```

### Django 配置检查

```bash
python manage.py check
```

### Docker 启动

```bash
docker compose up --build
```

### Docker 后台运行

```bash
docker compose up --build -d
```

### 停止容器

```bash
docker compose down
```

## 7. 注意事项

- 项目默认无数据库，不需要执行 `migrate`
- Docker 镜像默认使用 CPU 版 PyTorch，兼容性更好，但训练/推理速度会慢于本机 GPU 环境
- 如果要在 Docker 中启用 GPU，建议基于你自己的 CUDA / PyTorch 镜像再做扩展
- `media/uploads` 中的临时上传文件会由系统自动清理
- 大型预训练权重和数据集不建议直接打进镜像，建议始终通过挂载目录提供

## 8. 交付建议

如果你要把项目交给别人，推荐一起提供：

- `models/weights/` 中所需权重
- `media/preset/` 中的示例数据
- 一份可直接执行的 `.env`

这样对方基本只需要执行：

```bash
docker compose up --build
```

就可以直接启动整个平台。


