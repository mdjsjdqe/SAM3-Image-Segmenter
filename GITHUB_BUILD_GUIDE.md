# SAM3 Image Segmenter — GitHub Actions 编译为 Windows EXE 完整指南

## 📋 项目依赖分析

| 依赖                     | 类型            | 说明                          |
| ------------------------ | --------------- | ----------------------------- |
| tkinter                  | 标准库          | Python 自带，无需安装         |
| numpy                    | 第三方          | 图像数据处理                  |
| Pillow (PIL)             | 第三方          | 图像加载/显示                 |
| tkinterdnd2              | 第三方(可选)    | 拖拽文件支持                  |
| sam3_engine              | 本地模块        | SAM3 分割引擎（需打包进 exe） |
| torch / segment_anything | sam3_engine依赖 | 深度学习推理（体积大）        |

## 🗂️ 需要添加的文件

### 1. requirements.txt

```
numpy
Pillow
tkinterdnd2
torch
segment-anything
```

### 2. .github/workflows/build-windows.yml

```yaml
name: Build Windows EXE

on:
  push:
    tags:
      - 'v*'          # 推送 v1.0.0 等标签时触发
  workflow_dispatch:    # 也支持手动触发

jobs:
  build:
    runs-on: windows-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pyinstaller

      - name: Build EXE with PyInstaller
        run: |
          pyinstaller --onefile --windowed ^
            --name "SAM3 Image Segmenter" ^
            --hidden-import=tkinterdnd2 ^
            --hidden-import=sam3_engine ^
            --hidden-import=segment_anything ^
            --hidden-import=torch ^
            --collect-data torch ^
            --collect-data segment_anything ^
            --icon=icon.ico ^
            main.py

      - name: Upload EXE artifact
        uses: actions/upload-artifact@v4
        with:
          name: SAM3-Windows-EXE
          path: dist/SAM3 Image Segmenter.exe

  release:
    needs: build
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')

    steps:
      - name: Download EXE artifact
        uses: actions/download-artifact@v4
        with:
          name: SAM3-Windows-EXE

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v2
        with:
          files: "SAM3 Image Segmenter.exe"
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### 3. SAM3 Image Segmenter.spec（可选，更精细的打包配置）

```python
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        # 如果有资源文件（图标、模型权重等），在这里添加
        # ('resources/icon.ico', 'resources'),
        # ('models/sam_vit_h.pth', 'models'),
    ],
    hiddenimports=[
        'tkinterdnd2',
        'sam3_engine',
        'segment_anything',
        'torch',
        'torchvision',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除不需要的大模块，减小体积
        'matplotlib',
        'scipy',
        'IPython',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SAM3 Image Segmenter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_dir=None,
    console=False,        # 不显示命令行窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',      # 应用图标
)
```

## 🚀 使用步骤

### Step 1: 创建 GitHub 仓库

```bash
# 在项目根目录
git init
git add main.py sam3_engine.py requirements.txt icon.ico
git add .github/workflows/build-windows.yml
git commit -m "Initial commit"
git remote add origin https://github.com/<your-username>/sam3-segmenter.git
git push -u origin main
```

### Step 2: 触发编译

**方式 A：推送版本标签**

```bash
git tag v1.0.0
git push origin v1.0.0
```

**方式 B：手动触发**

- 进入 GitHub 仓库 → Actions → Build Windows EXE → Run workflow

### Step 3: 下载 EXE

- 编译完成后，Actions 页面会出现 Artifacts 下载链接
- 如果是 tag 触发，还会自动创建 Release 页面，附带 EXE 下载

## ⚠️ 重要注意事项

### 1. 模型权重文件体积问题

SAM 模型权重（`sam_vit_h.pth` ≈ 2.4GB）太大，不适合打包进 EXE。

**推荐方案**：EXE 启动后由用户自行下载/指定模型路径（代码中已有此逻辑）

### 2. torch 体积问题

PyTorch CPU 版 ≈ 150MB，GPU 版 ≈ 2GB+。

**减小体积**：

```bash
# requirements.txt 中只安装 CPU 版
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

对应 workflow 修改：

```yaml
- name: Install PyTorch CPU-only
  run: |
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

- name: Install other dependencies
  run: |
    pip install -r requirements.txt
    pip install pyinstaller
```

### 3. tkinterdnd2 可选依赖

代码中已有 `HAS_DND` 降级处理，如果编译失败可以移除：

```yaml
# 移除 --hidden-import=tkinterdnd2
# requirements.txt 中也可以去掉
```

### 4. UPX 压缩

UPX 可以减小 EXE 体积 30-50%，但可能被杀毒软件误报。

**禁用 UPX**（更安全）：

```yaml
pyinstaller --onefile --windowed --noupx ...
```

### 5. 应用图标

需要准备 `icon.ico` 文件（256x256 像素），放在项目根目录。

## 📊 预估体积

| 组成                           | 大小       |
| ------------------------------ | ---------- |
| Python 解释器                  | ~30MB      |
| PyTorch CPU                    | ~150MB     |
| numpy + Pillow                 | ~15MB      |
| tkinterdnd2                    | ~2MB       |
| sam3_engine + segment_anything | ~5MB       |
| **总计（未压缩）**             | **~200MB** |
| **UPX 压缩后**                 | **~120MB** |

## 🔧 本地调试编译（不使用 GitHub）

在本地 Windows 机器上也可以先测试：

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name "SAM3 Image Segmenter" main.py
# 生成的 EXE 在 dist/ 目录下
```
