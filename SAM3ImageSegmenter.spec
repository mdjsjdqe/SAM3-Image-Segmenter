# -*- mode: python ; coding: utf-8 -*-
# SAM3 Image Segmenter — PyInstaller spec 文件
# 用于将 main.py + sam3_engine.py 打包为 Windows EXE

import os
import sys

block_cipher = None

# ── 分析入口 ──
a = analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        # tkinterdnd2 的数据文件（拖拽支持）
        ('tkinterdnd2', 'tkinterdnd2'),
    ],
    hiddenimports=[
        'sam3_engine',           # SAM 模型引擎
        'tkinterdnd2',           # 拖拽文件支持
        'tkinterdnd2.TkinterDnD',
        'tkinterdnd2.dnd',       # DND_FILES 常量
        'numpy',                 # 图像数据处理
        'PIL',                   # Pillow 图像处理
        'PIL._tkinter_finder',   # Pillow tkinter 兼容
        'torch',                 # PyTorch
        'torch.nn',              # 神经网络模块
        'torchvision',           # 视觉模型库
        'segment_anything',      # SAM 分割模型
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除不需要的大模块，减小 EXE 体积
        'matplotlib',
        'scipy',
        'pandas',
        'IPython',
        'jupyter',
        'notebook',
        'sphinx',
        'pytest',
        'unittest',
        'setuptools',
        'pip',
        'wheel',
        'distutils',
        'lib2to3',
        'tkinter.test',
        'test',
        'tests',
        '__pycache__',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ── 收集 PyTorch 运行时数据 ──
# torch 需要一些动态库和配置文件才能运行
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
torch_datas = collect_data_files('torch')
torch_hidden = collect_submodules('torch')
a.datas += torch_datas
a.hiddenimports += torch_hidden

# ── PYZ 压缩包 ──
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ── EXE 配置 ──
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SAM3ImageSegmenter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                # 使用 UPX 压缩减小体积（需 UPX 在 PATH 中）
    console=False,           # 不显示控制台窗口（GUI 应用）
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',         # 应用图标（由 generate_icon.py 生成）
    version_file=None,       # 可选：Windows 版本信息文件
)

# 注意：不生成 COLLECT（onefile 模式下 EXE 包含所有内容）
