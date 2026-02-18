# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for POE2 Price Overlay.
Single-exe architecture: app.py is the entry point.
  - Default mode: launches FastAPI server + pywebview dashboard
  - --overlay-worker: runs the overlay subprocess (main.py:main)

Build command:
    pyinstaller build.spec --noconfirm --clean

Output: dist/POE2PriceOverlay/
"""

from pathlib import Path

block_cipher = None
app_dir = Path(SPECPATH)

# All .py source files to bundle as data (so imports work in frozen mode)
py_sources = [
    (str(app_dir / f), '.')
    for f in app_dir.glob('*.py')
    if f.name not in ('build.spec',)
]

a = Analysis(
    [str(app_dir / 'app.py')],
    pathex=[str(app_dir)],
    binaries=[],
    datas=[
        *py_sources,
        (str(app_dir / 'dashboard.html'), '.'),
        (str(app_dir / 'VERSION'), '.'),
        # Include .filter template if present
        *((str(f), '.') for f in app_dir.glob('*.filter') if 'updated' not in f.name),
    ],
    hiddenimports=[
        # FastAPI + uvicorn
        'fastapi', 'fastapi.middleware', 'fastapi.middleware.cors',
        'fastapi.responses', 'pydantic',
        'uvicorn', 'uvicorn.config', 'uvicorn.main',
        'uvicorn.lifespan', 'uvicorn.lifespan.on',
        'uvicorn.loops', 'uvicorn.loops.auto', 'uvicorn.loops.asyncio',
        'uvicorn.protocols', 'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto', 'uvicorn.protocols.http.h11_impl',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.protocols.websockets.wsproto_impl',
        'uvicorn.logging',
        'starlette', 'starlette.routing', 'starlette.responses',
        'starlette.middleware', 'starlette.middleware.cors',
        'anyio', 'anyio._backends', 'anyio._backends._asyncio',
        'h11', 'wsproto',
        # pywebview
        'webview', 'webview.platforms', 'webview.platforms.edgechromium',
        'clr_loader', 'pythonnet',
        # Win32 APIs
        'win32gui', 'win32con', 'win32api', 'win32process',
        'ctypes', 'ctypes.wintypes',
        # Project modules
        'bundle_paths', 'config', 'main',
        'server', 'watchlist',
        'item_detection', 'item_parser', 'mod_parser',
        'price_cache', 'overlay', 'trade_client',
        'filter_updater', 'mod_database', 'calibration',
        'calibration_harvester', 'bug_reporter',
        'clipboard_reader', 'screen_capture',
        # Standard lib
        'requests', 'tkinter', 'numpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pytest', 'cv2', 'pytesseract', 'mss',
        'matplotlib', 'scipy', 'pandas',
        'notebook', 'IPython',
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
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
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
