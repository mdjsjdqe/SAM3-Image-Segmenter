# -*- mode: python ; coding: utf-8 -*-
# SAM3 Image Segmenter - PyInstaller spec file
# Supports Windows (.ico) and macOS (.icns)

import os
import sys

block_cipher = None

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect submodules
torch_hiddenimports = collect_submodules('torch')
sam_hiddenimports = collect_submodules('segment_anything')
dnd_hiddenimports = collect_submodules('tkinterdnd2')

# Select icon based on platform
if sys.platform == 'darwin':
    icon_file = 'icon.icns'
else:
    icon_file = 'icon.ico'

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('sam3_engine.py', '.'),
    ] + collect_data_files('torch') + collect_data_files('segment_anything') + collect_data_files('tkinterdnd2'),
    hiddenimports=[
        'sam3_engine',
        'tkinterdnd2',
        'tkinterdnd2.TkinterDnD',
        'torch',
        'torchvision',
        'segment_anything',
        'numpy',
        'PIL',
    ] + torch_hiddenimports + sam_hiddenimports + dnd_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'scipy',
        'pandas',
        'IPython',
        'notebook',
        'jupyter',
        'tkinter.test',
        'tensorboard',
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
    name='SAM3ImageSegmenter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)
