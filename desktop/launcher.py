"""
LedgerLens Desktop Launcher.
Starts the FastAPI server locally and opens the web UI in the default browser.
Designed to run both from source and as a PyInstaller-frozen .app bundle.

Offline guarantee: HF_HOME points to a user-writable cache seeded once from
model weights bundled inside the app, so TrOCR never needs internet.
"""
import os
import sys
import time
import shutil
import socket
import threading
import webbrowser
from pathlib import Path

APP_NAME = "LedgerLens"
PREFERRED_PORT = 8000

BUNDLED_MODELS_ENV = "LEDGERLENS_BUNDLED_HF_HOME"


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


def _user_data_dir() -> Path:
    home = Path.home()
    if sys.platform == "darwin":
        base = home / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share"))
    d = base / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _seed_model_cache() -> None:
    """Copy bundled HuggingFace cache to a writable location on first run."""
    bundled = os.environ.get(BUNDLED_MODELS_ENV)
    if not bundled and getattr(sys, "frozen", False):
        bundled = str(_base_dir() / "models" / "hf")
    if not bundled:
        return
    bundled = Path(bundled)
    target = _user_data_dir() / "hf"
    marker = target / ".seeded"
    if marker.exists():
        return
    if bundled.exists():
        print(f"[LedgerLens] Preparing local OCR engine (first run only)...")
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(bundled, target)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok")


def _redirect_output_to_log() -> None:
    """Windowed bundles have no stdout/stderr; keep logs in the data dir."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    log_path = _user_data_dir() / "ledgerlens.log"
    log_file = open(log_path, "a", buffering=1)
    sys.stderr = sys.stdout = log_file


def _configure_environment() -> None:
    _redirect_output_to_log()
    _seed_model_cache()
    hf_home = _user_data_dir() / "hf"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _free_port() -> int:
    for candidate in range(PREFERRED_PORT, PREFERRED_PORT + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", candidate))
                return candidate
            except OSError:
                continue
    raise RuntimeError("No free port found")


def _wait_for_server(port: int, timeout_s: float = 120.0) -> bool:
    import urllib.request

    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/api/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main() -> None:
    _configure_environment()

    base = _base_dir()
    os.chdir(base)

    port = _free_port()
    print(f"[LedgerLens] Starting local server on port {port}...")

    import uvicorn

    config = uvicorn.Config(
        "backend.app:app",
        host="127.0.0.1",
        port=port,
        log_level="warning",
        reload=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    if _wait_for_server(port):
        webbrowser.open(f"http://127.0.0.1:{port}/")
    else:
        print("[LedgerLens] Server failed to start in time.")

    try:
        while server.should_exit is False and thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
