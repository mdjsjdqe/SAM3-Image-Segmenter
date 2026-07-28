# -*- mode: python ; coding: utf-8 -*-
# SAM3 Image Segmenter - PyInstaller spec file

import os
import sys

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('sam3_engine.py', '.'),
    ],
    hiddenimports=[
        'sam3_engine',
        'tkinterdnd2',
        'torch',
        'torchvision',
        'segment_anything',
        'numpy',
        'PIL',
    ],
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
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Collect torch data and submodules
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
a.datas += collect_data_files('torch')
a.datas += collect_data_files('tkinterdnd2')
a.hiddenimports += collect_submodules('torch')

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
    icon='icon.ico',
)
