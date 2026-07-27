# -*- mode: python ; coding: utf-8 -*-
"""
SAM3 Image Segmenter - PyInstaller spec 文件
用于将 Python 应用打包为 macOS .app

使用方法:
    bash build.sh
"""

import sys
import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_all

block_cipher = None

# ── 收集隐式依赖 ──
hidden_imports = [
    'sam3_engine',
    'tkinter', 'tkinter.filedialog', 'tkinter.messagebox', 'tkinter.ttk',
    'PIL', 'PIL.Image', 'PIL.ImageTk', 'PIL.ImageDraw', 'PIL.ImageFont',
    'numpy', 'torch', 'torch.nn', 'torchvision',
    'ultralytics', 'ultralytics.models', 'ultralytics.models.sam',
    'ultralytics.nn', 'ultralytics.utils', 'ultralytics.data', 'ultralytics.engine',
    'cv2', 'tqdm', 'yaml', 'requests', 'scipy', 'scipy.ndimage',
    'matplotlib', 'packaging', 'psutil',
    # 标准库核心（绝不能排除）
    'importlib', 'importlib.metadata', 'importlib.resources',
    'importlib.util', 'importlib.machinery', 'importlib.abc',
    'importlib._bootstrap', 'importlib._bootstrap_external',
    'pkgutil', 'inspect', 'ast', 'dis', 'token', 'tokenize',
    'codecs', 'encodings', 'encodings.utf_8', 'encodings.ascii',
    'encodings.idna', 'encodings.utf_16', 'encodings.utf_32',
    'encodings.latin_1', 'encodings.cp1252',
    'collections', 'collections.abc', 'functools', 'itertools',
    'operator', 'typing', 'dataclasses', 'enum', 'abc',
    'contextlib', 'copy', 'gettext', 'io', 'os', 'sys',
    'pathlib', 're', 'struct', 'textwrap', 'threading',
    'queue', 'subprocess', 'shutil', 'tempfile', 'glob',
    'hashlib', 'hmac', 'random', 'math', 'decimal',
    'urllib', 'urllib.parse', 'json', 'csv',
    'xml.etree', 'xml.etree.ElementTree',
    'logging', 'logging.handlers', 'warnings',
    'weakref', 'atexit', 'traceback', 'types',
    'platform', 'signal', 'posixpath', 'genericpath',
    'stat', 'errno', 'fcntl', 'select',
    'concurrent', 'concurrent.futures',
    'multiprocessing', 'multiprocessing.spawn',
    'distutils', 'distutils.version',
    'configparser', 'zipfile', 'tarfile',
]

# 收集第三方包的所有子模块
for pkg in ['ultralytics', 'torch', 'torchvision', 'PIL', 'numpy', 'cv2', 'clip', 'scipy', 'matplotlib', 'timm']:
    try:
        hidden_imports += collect_submodules(pkg)
    except Exception:
        pass

# ── 收集数据文件 ──
datas = []

# 应用图标
icon_path = os.path.join(SPECPATH, 'icon.icns')
if os.path.exists(icon_path):
    datas.append((icon_path, '.'))

# Ultralytics 配置和字体
for pkg in ['ultralytics', 'torch', 'torchvision', 'timm']:
    try:
        for f in collect_data_files(pkg):
            datas.append(f)
    except Exception:
        pass

# ── 排除列表（只排除确定不需要的大型包）──
# ⚠️ 绝不能排除任何标准库模块！torch/ultralytics 间接依赖大量标准库
excludes = [
    'IPython', 'jupyter', 'notebook', 'pytest',
    'setuptools', 'pip', 'wheel',
    'pydoc', 'lib2to3', 'xmlrpc',
    'curses', 'readline', 'rlcompleter',
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SAM3 Image Segmenter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon='icon.icns',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='SAM3 Image Segmenter',
)

app = BUNDLE(
    coll,
    name='SAM3 Image Segmenter.app',
    icon='icon.icns',
    bundle_identifier='com.sunny.sam3-segmenter',
    info_plist={
        'CFBundleName': 'SAM3 Image Segmenter',
        'CFBundleDisplayName': 'SAM3 Image Segmenter',
        'CFBundleVersion': '1.3.0',
        'CFBundleShortVersionString': '1.3.0',
        'CFBundleIdentifier': 'com.sunny.sam3-segmenter',
        'CFBundlePackageType': 'APPL',
        'CFBundleSignature': 'SAM3',
        'LSMinimumSystemVersion': '10.15',
        'NSHighResolutionCapable': True,
        'NSSupportsAutomaticGraphicsSwitching': True,
        'CFBundleDocumentTypes': [
            {
                'CFBundleTypeName': 'Image File',
                'CFBundleTypeRole': 'Viewer',
                'LSItemContentTypes': [
                    'public.png',
                    'public.jpeg',
                    'com.microsoft.bmp',
                    'public.tiff',
                ],
                'CFBundleTypeExtensions': ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'],
            }
        ],
        'NSRequiresAquaSystemAppearance': False,
    },
)
