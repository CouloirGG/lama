# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for POE2 Price Overlay.
Builds a single-folder distribution with all dependencies bundled.

Build command:
    pyinstaller build.spec --noconfirm

Output: dist/POE2PriceOverlay/
"""

import sys
from pathlib import Path

block_cipher = None
app_dir = Path(SPECPATH)

a = Analysis(
    [str(app_dir / 'src' / 'main.py')],
    pathex=[str(app_dir / 'src')],
    binaries=[],
    datas=[
        # Include src modules
        (str(app_dir / 'src' / 'config.py'), 'src'),
        (str(app_dir / 'src' / 'screen_capture.py'), 'src'),
        (str(app_dir / 'src' / 'ocr_engine.py'), 'src'),
        (str(app_dir / 'src' / 'item_parser.py'), 'src'),
        (str(app_dir / 'src' / 'price_cache.py'), 'src'),
        (str(app_dir / 'src' / 'overlay.py'), 'src'),
    ],
    hiddenimports=[
        'mss',
        'mss.windows',
        'cv2',
        'numpy',
        'PIL',
        'pytesseract',
        'requests',
        'tkinter',
        'ctypes',
        'win32gui',
        'win32con',
        'win32api',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'scipy',
        'pandas',
        'notebook',
        'IPython',
        'pytest',
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
    [],
    exclude_binaries=True,
    name='POE2PriceOverlay',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Keep console for now (useful for debugging)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon=str(app_dir / 'assets' / 'icon.ico'),  # Uncomment when icon exists
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='POE2PriceOverlay',
)
