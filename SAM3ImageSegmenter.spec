# -*- mode: python ; coding: utf-8 -*-
# SAM3 Image Segmenter - PyInstaller spec file

import os
import sys

block_cipher = None

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect data files and ensure 3-tuple format (dest, src, typecode)
def safe_collect_data(package):
    result = []
    for item in collect_data_files(package):
        if len(item) == 2:
            result.append((item[0], item[1], 'DATA'))
        else:
            result.append(item)
    return result

# Collect all submodules
torch_hiddenimports = collect_submodules('torch')
sam_hiddenimports = collect_submodules('segment_anything')
dnd_hiddenimports = collect_submodules('tkinterdnd2')

# Collect all data files
torch_datas = safe_collect_data('torch')
sam_datas = safe_collect_data('segment_anything')
dnd_datas = safe_collect_data('tkinterdnd2')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('sam3_engine.py', '.'),
    ] + torch_datas + sam_datas + dnd_datas,
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
    icon='icon.ico',
)
