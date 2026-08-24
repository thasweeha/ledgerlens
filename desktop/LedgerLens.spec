# PyInstaller spec for the LedgerLens desktop app.
# Build:  pyinstaller desktop/LedgerLens.spec --noconfirm
import os
import sys

block_cipher = None

hidden = [
    "backend.app",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "email.mime.multipart",
    "anyio._backends._asyncio",
]

a = Analysis(
    [os.path.join("..", "desktop", "launcher.py")],
    pathex=[os.path.join("..")],
    binaries=[],
    datas=[
        (os.path.join("..", "ui"), "ui"),
        (os.path.join("..", "models", "hf"), "models" + os.sep + "hf"),
    ],
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
        "pytest",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LedgerLens",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="LedgerLens",
)

app = None
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="LedgerLens.app",
        bundle_identifier="com.ledgerlens.desktop",
    )
