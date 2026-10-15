import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
CONFIG = json.loads((ROOT / "config.json").read_text())
if os.environ.get("DUPLEX_AVATAR_CONFIG"):
    CONFIG.update(json.loads((ROOT / os.environ["DUPLEX_AVATAR_CONFIG"]).read_text()))
CONFIG.update(json.loads(os.environ.get("DUPLEX_AVATAR_OVERRIDES", "{}")))
BRAIN_URL = f"http://127.0.0.1:{CONFIG['brain_port']}"
AVATAR_URL = f"http://127.0.0.1:{CONFIG['avatar_port']}"
BRAIN_WS = f"ws://127.0.0.1:{CONFIG['brain_port']}/backend"
ASR_URL = f"http://127.0.0.1:{CONFIG['asr_port']}"


def image_path():
    return WORKSPACE / CONFIG["avatar_image"]


def tls_paths():
    """Project-owned certificates; absolute paths are accepted for external TLS."""
    return (ROOT / CONFIG.get("tls_cert_path", "runtime/certs/cert.pem"),
            ROOT / CONFIG.get("tls_key_path", "runtime/certs/key.pem"))
