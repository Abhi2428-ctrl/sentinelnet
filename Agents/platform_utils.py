"""
SentinelNet v2.0 — Cross-Platform Utilities
============================================
Central home for ALL OS-specific logic.
Every other module imports from here.

Covers:
  - OS detection
  - File paths via pathlib (Fix 4)
  - Admin/root check that works on all OS (Fix 2)
  - File permissions chmod/icacls (Fix 5)
  - Line-ending safe file open (Fix 8)
  - AES-256 credential encryption (Fix 6)
  - Machine fingerprint key derivation
  - Screenshot: mss > ImageGrab > scrot (Fix 1)
  - Network interface detection per OS (Fix 2)
  - Package manager detection apt/dnf/pacman (Fix 3)
  - Browser open per OS
  - Model weights save/load (Fix 14)
  - Log rotation (Fix 16)
  - Startup cache (Fix 15)
"""

import os
import sys
import platform
import subprocess
import hashlib
import json
import gc
import tempfile
import threading
from pathlib import Path
from typing import Optional, List, Tuple, Dict

# ── OS constants ───────────────────────────────────────────
OS       = platform.system()    # 'Windows' | 'Linux' | 'Darwin'
IS_WIN   = OS == "Windows"
IS_LINUX = OS == "Linux"
IS_MAC   = OS == "Darwin"

# ── Base directories — always via pathlib (Fix 4) ─────────
BASE_DIR   = Path(__file__).parent.parent.resolve()
DATA_DIR   = BASE_DIR / "data"
LOGS_DIR   = BASE_DIR / "logs"
CERTS_DIR  = BASE_DIR / "certs"
AGENTS_DIR = BASE_DIR / "agents"
MODELS_DIR = DATA_DIR / "models"

for _d in [DATA_DIR, LOGS_DIR, CERTS_DIR,
           LOGS_DIR / "audit", LOGS_DIR / "system",
           MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════
# ADMIN / ROOT  (Fix 2 — safe on all OS)
# ══════════════════════════════════════════════════════════

def is_admin() -> bool:
    """Check admin/root without crashing on any OS."""
    if IS_WIN:
        try:
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False
    else:
        try:
            return os.geteuid() == 0
        except AttributeError:
            return False


# ══════════════════════════════════════════════════════════
# FILE PERMISSIONS  (Fix 5)
# ══════════════════════════════════════════════════════════

def secure_file(path: Path):
    """
    Restrict a sensitive file to owner read/write only.
    Linux/Mac: chmod 600
    Windows:   icacls restrict to current user
    """
    path = Path(path)
    if not path.exists():
        return
    if IS_WIN:
        try:
            username = os.environ.get("USERNAME", "")
            if username:
                subprocess.run(
                    ["icacls", str(path), "/inheritance:r",
                     "/grant:r", f"{username}:(R,W)"],
                    capture_output=True, check=False
                )
        except Exception:
            pass
    else:
        try:
            path.chmod(0o600)
        except Exception:
            pass


def open_text(path, mode: str = "r", **kwargs):
    """
    Open text file with consistent line endings on all OS. (Fix 8)
    Always uses newline='\n' — no \r\n on Windows.
    """
    return open(Path(path), mode, newline="\n",
                encoding="utf-8", **kwargs)


# ══════════════════════════════════════════════════════════
# MACHINE FINGERPRINT for AES key derivation
# ══════════════════════════════════════════════════════════

def machine_fingerprint() -> bytes:
    """
    Stable 32-byte machine-specific key.
    Derived from hardware IDs — never stored anywhere.
    Re-derived identically each run on the same machine.
    """
    parts = []

    if IS_WIN:
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography"
            )
            guid, _ = winreg.QueryValueEx(key, "MachineGuid")
            winreg.CloseKey(key)
            parts.append(guid)
        except Exception:
            pass
        parts.append(os.environ.get("COMPUTERNAME", ""))
        parts.append(os.environ.get("USERNAME", ""))

    elif IS_MAC:
        try:
            result = subprocess.run(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                capture_output=True, text=True, timeout=3
            )
            for line in result.stdout.splitlines():
                if "IOPlatformUUID" in line:
                    parts.append(line.split('"')[-2])
                    break
        except Exception:
            pass
        parts.append(platform.node())

    else:
        for mid_path in ["/etc/machine-id", "/var/lib/dbus/machine-id"]:
            try:
                parts.append(Path(mid_path).read_text().strip())
                break
            except Exception:
                pass
        parts.append(platform.node())

    parts.append(sys.executable)
    combined = "|".join(p for p in parts if p)
    if not combined:
        combined = f"sentinelnet-{platform.node()}-{OS}"
    return hashlib.sha256(combined.encode()).digest()


# ══════════════════════════════════════════════════════════
# AES-256 CREDENTIAL ENCRYPTION  (Fix 6)
# ══════════════════════════════════════════════════════════

def encrypt_credentials(data: dict) -> str:
    """
    Encrypt dict using AES-256-GCM with machine fingerprint key.
    Returns base64-encoded string. Key never stored.
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        import base64, secrets as _s
        key    = machine_fingerprint()
        nonce  = _s.token_bytes(12)
        aesgcm = AESGCM(key)
        ct     = aesgcm.encrypt(nonce, json.dumps(data).encode(), None)
        return base64.b64encode(nonce + ct).decode()
    except Exception:
        # Fallback: store plain JSON (still protected by OS file perms)
        return json.dumps(data)


def decrypt_credentials(payload: str) -> dict:
    """Decrypt credentials. Returns {} on any failure."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        import base64
        raw    = base64.b64decode(payload.encode())
        nonce  = raw[:12]
        ct     = raw[12:]
        aesgcm = AESGCM(machine_fingerprint())
        pt     = aesgcm.decrypt(nonce, ct, None)
        return json.loads(pt.decode())
    except Exception:
        try:
            return json.loads(payload)   # plain JSON fallback
        except Exception:
            return {}


def save_credentials(path: Path, data: dict):
    """Encrypt and save credentials; restrict file permissions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_text(path, "w") as f:
        f.write(encrypt_credentials(data))
    secure_file(path)


def load_credentials(path: Path) -> dict:
    """Load and decrypt credentials file."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open_text(path) as f:
            return decrypt_credentials(f.read().strip())
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════
# SCREENSHOT CAPTURE  (Fix 1) — mss > ImageGrab > scrot
# ══════════════════════════════════════════════════════════

_screenshot_backend: Optional[str] = None
_backend_lock = threading.Lock()


def _detect_screenshot_backend() -> str:
    global _screenshot_backend
    with _backend_lock:
        if _screenshot_backend:
            return _screenshot_backend

        # 1. mss — best cross-platform (Windows + Linux + Mac)
        try:
            import mss as _mss
            _screenshot_backend = "mss"
            return "mss"
        except ImportError:
            pass

        # 2. PIL ImageGrab — Windows and Mac only
        if IS_WIN or IS_MAC:
            try:
                from PIL import ImageGrab
                _test = ImageGrab.grab(bbox=(0, 0, 2, 2))
                _screenshot_backend = "imagegrab"
                return "imagegrab"
            except Exception:
                pass

        # 3. scrot — Linux with X11
        if IS_LINUX:
            if subprocess.run(["which","scrot"],
                              capture_output=True).returncode == 0:
                _screenshot_backend = "scrot"
                return "scrot"

        # 4. xwd — another Linux X11 option
        if IS_LINUX:
            if subprocess.run(["which","xwd"],
                              capture_output=True).returncode == 0:
                _screenshot_backend = "xwd"
                return "xwd"

        _screenshot_backend = "none"
        return "none"


def take_screenshot(max_width: int = 1920) -> Optional["PIL.Image.Image"]:
    """
    Cross-platform screenshot. Returns PIL Image or None.
    Automatically downscales if wider than max_width to save RAM.
    """
    backend = _detect_screenshot_backend()
    img = None

    try:
        if backend == "mss":
            import mss
            from PIL import Image
            with mss.mss() as sct:
                mon = sct.monitors[1]        # primary monitor
                raw = sct.grab(mon)
                img = Image.frombytes(
                    "RGB", (raw.width, raw.height),
                    raw.bgra, "raw", "BGRX"
                )

        elif backend == "imagegrab":
            from PIL import ImageGrab
            img = ImageGrab.grab()

        elif backend == "scrot":
            from PIL import Image
            tmp = Path(tempfile.mktemp(suffix=".png"))
            r = subprocess.run(["scrot", str(tmp)],
                               capture_output=True, timeout=5)
            if r.returncode == 0 and tmp.exists():
                img = Image.open(tmp)
                img.load()
                tmp.unlink(missing_ok=True)

        elif backend == "xwd":
            from PIL import Image
            tmp = Path(tempfile.mktemp(suffix=".xwd"))
            r = subprocess.run(
                ["xwd", "-root", "-silent", "-out", str(tmp)],
                capture_output=True, timeout=5
            )
            if r.returncode == 0 and tmp.exists():
                img = Image.open(tmp)
                img.load()
                tmp.unlink(missing_ok=True)

    except Exception:
        pass

    if img is not None:
        w, h = img.size
        if w > max_width:
            ratio = max_width / w
            img = img.resize((max_width, int(h * ratio)), resample=1)

    return img


def screenshot_available() -> bool:
    return _detect_screenshot_backend() != "none"


def screenshot_backend_name() -> str:
    return _detect_screenshot_backend()


# ══════════════════════════════════════════════════════════
# NETWORK INTERFACE DETECTION  (Fix 2)
# ══════════════════════════════════════════════════════════

def _is_skip_interface(name: str) -> bool:
    """True if this interface should be skipped (loopback, virtual, etc.)"""
    skip_exact    = {"lo", "lo0", "any"}
    skip_prefixes = ("lo","docker","br-","veth","virbr","vmnet",
                     "vboxnet","tun","tap","utun","awdl","llw",
                     "gif","stf","p2p","XHC","pktap","dummy")
    n = name.lower()
    if n in skip_exact:
        return True
    return any(n.startswith(p) for p in skip_prefixes)


def get_best_network_interface() -> Optional[str]:
    """
    Find the best real network interface for packet capture.
    Uses psutil for scoring if available, falls back to OS heuristics.
    """
    try:
        import psutil
        stats = psutil.net_if_stats()
        addrs = psutil.net_if_addrs()
        scores: Dict[str, int] = {}
        for iface, stat in stats.items():
            if not stat.isup or _is_skip_interface(iface):
                continue
            score = 0
            for addr in addrs.get(iface, []):
                if addr.family == 2:   # AF_INET
                    score += 10
                    ip = addr.address
                    if ip.startswith(("192.168.", "10.", "172.")):
                        score += 5
            if stat.speed >= 100:
                score += 3
            scores[iface] = score
        if scores:
            return max(scores, key=lambda k: scores[k])
    except ImportError:
        pass

    # OS-specific fallback candidates
    if IS_WIN:
        candidates = ["Ethernet","Wi-Fi","Local Area Connection",
                      "Wireless Network Connection","Ethernet0"]
    elif IS_MAC:
        candidates = ["en0","en1","en2"]
    else:
        candidates = ["eth0","ens33","ens3","enp3s0","enp0s3",
                      "wlan0","wlp2s0","eno1","eno0"]

    try:
        from scapy.all import get_if_list
        available = get_if_list()
        for c in candidates:
            if c in available:
                return c
        for iface in available:
            if not _is_skip_interface(iface):
                return iface
    except Exception:
        pass

    return None


def list_network_interfaces() -> List[dict]:
    """List all network interfaces with details."""
    interfaces = []
    try:
        import psutil
        for iface, stat in psutil.net_if_stats().items():
            addrs = psutil.net_if_addrs().get(iface, [])
            ipv4  = next((a.address for a in addrs if a.family == 2), None)
            interfaces.append({
                "name":  iface,
                "up":    stat.isup,
                "speed": stat.speed,
                "ipv4":  ipv4,
                "skip":  _is_skip_interface(iface),
            })
    except ImportError:
        try:
            from scapy.all import get_if_list
            for iface in get_if_list():
                interfaces.append({
                    "name": iface, "up": True,
                    "speed": 0, "ipv4": None,
                    "skip": _is_skip_interface(iface),
                })
        except Exception:
            pass
    return interfaces


# ══════════════════════════════════════════════════════════
# PACKAGE MANAGER DETECTION  (Fix 3)
# ══════════════════════════════════════════════════════════

def detect_package_manager() -> Optional[str]:
    """Detect Linux package manager."""
    if not IS_LINUX:
        return None
    for pm in ["apt","dnf","yum","pacman","zypper","apk"]:
        if subprocess.run(["which", pm],
                          capture_output=True).returncode == 0:
            return pm
    return None


def libpcap_install_command() -> str:
    """Return exact OS-specific install command for packet capture."""
    if IS_WIN:
        return (
            "Download Npcap from https://npcap.com/#download\n"
            "Run as Administrator, check 'WinPcap API-compatible mode'"
        )
    elif IS_MAC:
        return (
            "libpcap is built into macOS — no install needed.\n"
            "To enable capture:\n"
            "  Option A: sudo ./start.sh\n"
            "  Option B: sudo chmod +r /dev/bpf*  (persists until reboot)\n"
            "  Option C (macOS Ventura+): System Settings → Privacy & Security\n"
            "            → Local Network → enable Terminal"
        )
    else:
        pm   = detect_package_manager()
        cmds = {
            "apt":    "sudo apt install -y libpcap-dev tcpdump",
            "dnf":    "sudo dnf install -y libpcap-devel tcpdump",
            "yum":    "sudo yum install -y libpcap-devel tcpdump",
            "pacman": "sudo pacman -S libpcap",
            "zypper": "sudo zypper install libpcap-devel",
            "apk":    "sudo apk add libpcap-dev tcpdump",
        }
        base = cmds.get(pm, (
            "# Ubuntu/Debian:\n"
            "sudo apt install -y libpcap-dev tcpdump\n"
            "# Fedora/RHEL:\n"
            "sudo dnf install -y libpcap-devel tcpdump\n"
            "# Arch:\n"
            "sudo pacman -S libpcap"
        ))
        return (
            base + "\n\n"
            "OR give Python raw socket capability (no sudo on each run):\n"
            "  sudo setcap cap_net_raw+eip $(which python3)"
        )


# ══════════════════════════════════════════════════════════
# BROWSER / PYTHON COMMAND
# ══════════════════════════════════════════════════════════

def open_browser(url: str):
    """Open browser cross-platform."""
    try:
        if IS_WIN:
            os.startfile(url)
        elif IS_MAC:
            subprocess.Popen(["open", url])
        else:
            subprocess.Popen(["xdg-open", url],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    except Exception:
        pass


def python_cmd() -> str:
    """Return the correct python executable for this environment."""
    return sys.executable or ("python" if IS_WIN else "python3")


def macos_version() -> Tuple[int, int]:
    """Return macOS version as (major, minor) tuple."""
    if not IS_MAC:
        return (0, 0)
    try:
        parts = platform.mac_ver()[0].split(".")
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except Exception:
        return (0, 0)


# ══════════════════════════════════════════════════════════
# MODEL WEIGHTS PERSISTENCE  (Fix 14)
# ══════════════════════════════════════════════════════════

def model_path(agent_name: str) -> Path:
    safe = agent_name.replace(" ", "_").lower()
    return MODELS_DIR / f"{safe}.npz"


def save_model(agent_name: str, weights: dict):
    """Persist agent weights to disk so learning survives restarts."""
    path = model_path(agent_name)
    try:
        import numpy as np
        np.savez_compressed(str(path), **weights)
    except Exception:
        pass


def load_model(agent_name: str) -> Optional[dict]:
    """Load persisted agent weights. Returns None if not found."""
    path = model_path(agent_name)
    if not path.exists():
        return None
    try:
        import numpy as np
        d = np.load(str(path), allow_pickle=True)
        return {k: d[k] for k in d.files}
    except Exception:
        return None


# ══════════════════════════════════════════════════════════
# LOG ROTATION  (Fix 16)
# ══════════════════════════════════════════════════════════

def get_monthly_log_path(subdir: str, prefix: str,
                          suffix: str = ".jsonl") -> Path:
    """Monthly rotating log path: logs/<subdir>/<prefix>_YYYY_MM.jsonl"""
    from datetime import datetime
    now  = datetime.utcnow()
    name = f"{prefix}_{now.year:04d}_{now.month:02d}{suffix}"
    d    = LOGS_DIR / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def cleanup_old_logs(subdir: str, max_age_days: int = 365):
    """Delete log files older than max_age_days."""
    import time
    log_dir = LOGS_DIR / subdir
    if not log_dir.exists():
        return
    cutoff = time.time() - (max_age_days * 86400)
    for f in log_dir.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            try:
                f.unlink()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════
# STARTUP CACHE  (Fix 15)
# ══════════════════════════════════════════════════════════

STARTUP_CACHE_PATH = DATA_DIR / "startup_cache.json"
STARTUP_CACHE_TTL  = 86400   # 24 hours


def load_startup_cache() -> Optional[dict]:
    """Return cached startup results if still valid (same OS, < 24h old)."""
    import time
    if not STARTUP_CACHE_PATH.exists():
        return None
    try:
        with open_text(STARTUP_CACHE_PATH) as f:
            cache = json.load(f)
        if time.time() - cache.get("cached_at", 0) > STARTUP_CACHE_TTL:
            return None
        if cache.get("system") != OS:
            return None
        return cache
    except Exception:
        return None


def save_startup_cache(results: dict):
    """Cache startup results so next boot is instant."""
    import time
    try:
        cache = dict(results)
        cache["cached_at"] = time.time()
        with open_text(STARTUP_CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2, default=str)
    except Exception:
        pass


def invalidate_startup_cache():
    """Force full re-check on next boot."""
    try:
        STARTUP_CACHE_PATH.unlink(missing_ok=True)
    except Exception:
        pass
