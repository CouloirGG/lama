# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for LAMA (Live Auction Market Assessor).
Single-exe architecture: app.py is the entry point.
  - Default mode: launches FastAPI server + pywebview dashboard
  - --overlay-worker: runs the overlay subprocess (main.py:main)

Build command (from project root):
    pyinstaller scripts/build.spec --noconfirm --clean

Output: dist/LAMA/
"""

from pathlib import Path

block_cipher = None
project_dir = Path(SPECPATH).parent  # SPECPATH is scripts/, go up to project root
src_dir = project_dir / 'src'

# All .py source files to bundle as data (so imports work in frozen mode)
py_sources = [
    (str(src_dir / f), '.')
    for f in src_dir.glob('*.py')
]

a = Analysis(
    [str(src_dir / 'app.py')],
    pathex=[str(src_dir)],
    binaries=[],
    datas=[
        *py_sources,
        (str(project_dir / 'resources' / 'dashboard.html'), 'resources'),
        (str(project_dir / 'resources' / 'VERSION'), 'resources'),
        # Include .filter template if present
        *((str(f), 'resources') for f in (project_dir / 'resources').glob('*.filter') if 'updated' not in f.name),
        # Include calibration shards if present
        *((str(f), 'resources') for f in (project_dir / 'resources').glob('*.json.gz')),
        # Include image assets
        *((str(f), 'resources/img') for f in (project_dir / 'resources' / 'img').glob('*.png') if (project_dir / 'resources' / 'img').exists()),
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
    name='LAMA',
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
    name='LAMA',
)
