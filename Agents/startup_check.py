"""
SentinelNet v2.0 — Startup Dependency Checker (Cross-Platform)
===============================================================
Runs before everything else on boot.
Gives EXACT OS-specific fix instructions.
Results cached 24h so subsequent starts are instant.

Checks:
  1.  Python version (3.8+)
  2.  Admin/root rights
  3.  Packet capture driver (Npcap / libpcap per OS)
  4.  Required Python packages
  5.  Optional packages (opencv, mss, psutil, watchdog, certifi)
  6.  Network interfaces
  7.  Port availability (8000 / 8443)
  8.  Disk space (200 MB min)
  9.  RAM (512 MB min)
  10. TLS certificate
  11. Screen capture capability
  12. Windows Defender false-positive check (Windows only)
  13. macOS screen-recording permission (macOS only)
  14. Linux display / headless detection (Linux only)
"""

import os, sys, platform, subprocess, socket, shutil, importlib, json
from pathlib import Path
from typing import Dict, List, Optional

from agents.platform_utils import (
    OS, IS_WIN, IS_LINUX, IS_MAC,
    BASE_DIR, DATA_DIR, CERTS_DIR,
    is_admin, secure_file, open_text,
    screenshot_available, screenshot_backend_name,
    libpcap_install_command, detect_package_manager,
    macos_version, list_network_interfaces,
    load_startup_cache, save_startup_cache,
)

MIN_PYTHON  = (3, 8)
MIN_RAM_MB  = 512
MIN_DISK_MB = 200

REQUIRED_PACKAGES = [
    ("fastapi",      "fastapi",      "pip install fastapi"),
    ("uvicorn",      "uvicorn",      "pip install uvicorn"),
    ("numpy",        "numpy",        "pip install numpy"),
    ("PIL",          "Pillow",       "pip install Pillow"),
    ("cryptography", "cryptography", "pip install cryptography"),
    ("scapy",        "scapy",        "pip install scapy"),
]

OPTIONAL_PACKAGES = [
    ("cv2",     "opencv-python", "pip install opencv-python",
     "Face detection in video calls — +10-15% accuracy"),
    ("psutil",  "psutil",        "pip install psutil",
     "CPU/RAM monitoring and auto-pause"),
    ("mss",     "mss",           "pip install mss",
     "Best cross-platform screen capture (Linux+Mac+Win)"),
    ("watchdog","watchdog",      "pip install watchdog",
     "File-system watcher for auto deepfake scanning"),
    ("certifi", "certifi",       "pip install certifi",
     "SSL cert bundle — prevents email failures on Linux"),
]


# ── Individual checks ──────────────────────────────────────

def check_python_version() -> dict:
    cur = sys.version_info[:2]
    ok  = cur >= MIN_PYTHON
    return {
        "name": "Python Version", "ok": ok,
        "value": f"{cur[0]}.{cur[1]}.{sys.version_info[2]}",
        "needed": f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}+",
        "fix": "Download from https://python.org" if not ok else "",
        "severity": "CRITICAL" if not ok else "OK",
    }


def check_admin_rights() -> dict:
    admin = is_admin()
    if admin:
        return {"name": "Admin / Root Rights", "ok": True,
                "value": "Yes", "severity": "OK", "fix": ""}
    if IS_WIN:
        fix  = "Right-click START_WINDOWS.bat → Run as Administrator"
        note = "Live capture unavailable without admin rights on Windows"
    elif IS_MAC:
        fix  = "Run: sudo ./start.sh"
        note = "macOS requires root or /dev/bpf* permission for packet capture"
    else:
        fix  = ("sudo ./start.sh\n"
                "OR (once, no sudo needed after):\n"
                "  sudo setcap cap_net_raw+eip $(which python3)")
        note = "Linux requires root or cap_net_raw capability for packet capture"
    return {
        "name": "Admin / Root Rights", "ok": False,
        "value": "No", "fix": fix, "note": note,
        "severity": "WARNING",
        "impact": "SYNTHETIC mode active — real network threats not detected",
    }


def check_packet_capture_driver() -> dict:
    if IS_WIN:   return _check_npcap()
    elif IS_MAC: return _check_libpcap_mac()
    else:        return _check_libpcap_linux()


def _check_npcap() -> dict:
    found, version = False, "Not found"
    # Check directory
    if Path("C:/Windows/System32/Npcap").exists():
        found, version = True, "Installed (C:/Windows/System32/Npcap)"
    # Check registry
    if not found:
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Npcap")
            v, _ = winreg.QueryValueEx(key, ""); winreg.CloseKey(key)
            found, version = True, f"v{v} (registry)"
        except Exception: pass
    # Probe via scapy
    if not found:
        try:
            from scapy.arch.windows import get_windows_if_list
            if get_windows_if_list(): found, version = True, "Detected via scapy"
        except Exception: pass

    fix_steps = [] if found else [
        "1. Go to https://npcap.com/#download",
        "2. Download latest Npcap installer",
        "3. Run installer as Administrator",
        "4. Check 'WinPcap API-compatible mode'",
        "5. Restart SentinelNet as Administrator",
    ]
    return {
        "name": "Npcap Driver (Windows)", "ok": found,
        "value": version,
        "needed": "Required for live packet capture on Windows",
        "fix": "Download from https://npcap.com/#download" if not found else "",
        "fix_steps": fix_steps,
        "download": "https://npcap.com/#download" if not found else "",
        "severity": "WARNING" if not found else "OK",
        "impact": "SYNTHETIC mode — no real traffic" if not found else "LIVE capture available",
    }


def _check_libpcap_linux() -> dict:
    found, value = False, "Not found"
    lib_paths = [
        "/usr/lib/libpcap.so", "/usr/lib/libpcap.so.0",
        "/usr/lib/x86_64-linux-gnu/libpcap.so.0.8",
        "/usr/lib/aarch64-linux-gnu/libpcap.so.0.8",
        "/usr/local/lib/libpcap.so",
    ]
    for p in lib_paths:
        if Path(p).exists():
            found, value = True, f"Found: {p}"; break
    if not found:
        r = subprocess.run(["which","tcpdump"], capture_output=True)
        if r.returncode == 0:
            found, value = True, f"tcpdump: {r.stdout.decode().strip()}"

    admin   = is_admin()
    has_cap = _check_setcap()
    ok      = found and (admin or has_cap)
    fix     = libpcap_install_command() if not found else (
        "Run with sudo: sudo ./start.sh\n"
        "OR: sudo setcap cap_net_raw+eip $(which python3)"
        if not (admin or has_cap) else ""
    )
    pm = detect_package_manager() or "apt"
    return {
        "name": f"libpcap (Linux/{pm})", "ok": ok, "value": value,
        "fix": fix, "severity": "WARNING" if not ok else "OK",
        "impact": "SYNTHETIC mode" if not ok else "LIVE capture available",
        "is_root": admin, "has_setcap": has_cap,
    }


def _check_libpcap_mac() -> dict:
    found, value = False, "Not found"
    for p in ["/usr/lib/libpcap.dylib", "/usr/lib/libpcap.A.dylib",
              "/usr/local/lib/libpcap.dylib"]:
        if Path(p).exists():
            found, value = True, f"Built-in macOS libpcap: {p}"; break
    if not found:
        r = subprocess.run(["which","tcpdump"], capture_output=True)
        if r.returncode == 0: found, value = True, "macOS built-in libpcap"

    admin  = is_admin()
    bpf_ok = _check_bpf_permissions()
    ver    = macos_version()
    ok     = found and (admin or bpf_ok)
    fix    = ""
    if not ok:
        fix = (
            "Option A (easiest): sudo ./start.sh\n"
            "Option B (until reboot): sudo chmod +r /dev/bpf*\n"
        )
        if ver >= (13, 0):
            fix += ("Option C: System Settings → Privacy & Security\n"
                    "          → Local Network → enable Terminal")
    return {
        "name": f"libpcap (macOS {ver[0]}.{ver[1]} — built-in)",
        "ok": ok, "value": value, "fix": fix,
        "severity": "WARNING" if not ok else "OK",
        "impact": "SYNTHETIC mode" if not ok else "LIVE capture available",
        "is_root": admin, "bpf_ok": bpf_ok,
    }


def _check_setcap() -> bool:
    try:
        r = subprocess.run(["getcap", sys.executable],
                           capture_output=True, text=True)
        return "cap_net_raw" in r.stdout
    except Exception:
        return False


def _check_bpf_permissions() -> bool:
    try:
        return any(os.access(str(f), os.R_OK)
                   for f in Path("/dev").glob("bpf*"))
    except Exception:
        return False


def check_required_packages() -> List[dict]:
    results = []
    for imp_name, pkg_name, install_cmd in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(imp_name)
            ver = getattr(mod, "__version__", "installed")
            results.append({"name": f"Package: {pkg_name}", "ok": True,
                             "value": ver, "fix": "", "severity": "OK"})
        except ImportError:
            results.append({"name": f"Package: {pkg_name}", "ok": False,
                             "value": "NOT INSTALLED", "fix": install_cmd,
                             "severity": "CRITICAL"})
    return results


def check_optional_packages() -> List[dict]:
    results = []
    for imp_name, pkg_name, install_cmd, benefit in OPTIONAL_PACKAGES:
        try:
            mod = importlib.import_module(imp_name)
            ver = getattr(mod, "__version__", "installed")
            results.append({"name": f"Optional: {pkg_name}", "ok": True,
                             "value": ver, "benefit": benefit,
                             "fix": "", "severity": "OK"})
        except ImportError:
            results.append({"name": f"Optional: {pkg_name}", "ok": False,
                             "value": "Not installed", "benefit": benefit,
                             "fix": install_cmd, "severity": "OPTIONAL"})
    return results


def check_port_availability() -> List[dict]:
    results = []
    for port in [8000, 8443]:
        in_use = False
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.3)
            in_use = (s.connect_ex(("127.0.0.1", port)) == 0)
            s.close()
        except Exception: pass
        label = "HTTPS Dashboard" if port == 8443 else "HTTP Dashboard"
        results.append({
            "name": f"Port {port} ({label})",
            "ok": not in_use,
            "value": "CONFLICT — already in use!" if in_use else "Available",
            "fix": f"Kill process using port {port}" if in_use else "",
            "severity": "CRITICAL" if in_use else "OK",
        })
    return results


def check_disk_space() -> dict:
    try:
        free_mb = shutil.disk_usage(BASE_DIR).free // (1024*1024)
        ok = free_mb >= MIN_DISK_MB
        return {"name": "Disk Space", "ok": ok,
                "value": f"{free_mb:,} MB free",
                "fix": "Free up disk space" if not ok else "",
                "severity": "WARNING" if not ok else "OK"}
    except Exception:
        return {"name": "Disk Space", "ok": True,
                "value": "Unable to check", "severity": "OK", "fix": ""}


def check_ram() -> dict:
    try:
        import psutil
        mem     = psutil.virtual_memory()
        free_mb = mem.available // (1024*1024)
        ok      = free_mb >= MIN_RAM_MB
        return {"name": "Available RAM", "ok": ok,
                "value": f"{free_mb:,} MB free ({mem.percent:.0f}% used)",
                "fix": "Close other apps to free RAM" if not ok else "",
                "severity": "WARNING" if not ok else "OK"}
    except ImportError:
        return {"name": "Available RAM", "ok": True,
                "value": "psutil not installed — cannot check",
                "severity": "OK", "fix": ""}


def check_tls_certificate() -> dict:
    cert = CERTS_DIR / "sentinelnet.crt"
    key  = CERTS_DIR / "sentinelnet.key"
    if not cert.exists() or not key.exists():
        return {"name": "TLS Certificate", "ok": True,
                "value": "Will be auto-generated on startup",
                "severity": "OK", "fix": ""}
    try:
        import ssl
        from datetime import datetime
        info      = ssl._ssl._test_decode_cert(str(cert))
        not_after = info.get("notAfter", "")
        if not_after:
            exp       = datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z")
            days_left = (exp - datetime.utcnow()).days
            ok        = days_left > 7
            secure_file(key)
            return {
                "name": "TLS Certificate", "ok": ok,
                "value": f"Valid — {days_left} days remaining",
                "fix": "Regenerate in Security tab" if not ok else "",
                "severity": "WARNING" if not ok else "OK"}
    except Exception: pass
    secure_file(key)
    return {"name": "TLS Certificate", "ok": True,
            "value": "Present", "severity": "OK", "fix": ""}


def check_screen_capture() -> dict:
    avail   = screenshot_available()
    backend = screenshot_backend_name()
    if avail:
        return {"name": "Screen Capture", "ok": True,
                "value": f"Working ({backend})",
                "severity": "OK", "fix": ""}
    if IS_WIN:
        fix = "pip install mss"
    elif IS_MAC:
        ver = macos_version()
        fix = ("pip install mss\n"
               "System Settings → Privacy & Security\n"
               "→ Screen Recording → enable Terminal")
        if ver >= (14, 0):
            fix += "\n(macOS Sonoma: also check Full Disk Access)"
    else:
        fix = ("pip install mss\n"
               "OR: sudo apt install scrot\n"
               "Note: DISPLAY env variable must be set for X11 methods")
    return {
        "name": "Screen Capture", "ok": False,
        "value": "Unavailable",
        "fix": fix, "severity": "WARNING",
        "note": "Video call monitor will NOT work without screen capture",
    }


def check_network_interfaces() -> dict:
    try:
        ifaces = list_network_interfaces()
        active = [i for i in ifaces if i["up"] and not i["skip"]]
        return {
            "name": "Network Interfaces",
            "ok": len(active) > 0,
            "value": (f"{len(active)} active: " +
                      ", ".join(i["name"] for i in active[:3])),
            "interfaces": ifaces,
            "severity": "WARNING" if not active else "OK",
            "fix": "No active network interfaces found" if not active else "",
        }
    except Exception:
        return {"name": "Network Interfaces", "ok": True,
                "value": "Unable to enumerate", "severity": "OK", "fix": ""}


def check_windows_defender() -> Optional[dict]:
    """Windows only: warn if SentinelNet folder not excluded from Defender."""
    if not IS_WIN: return None
    try:
        r = subprocess.run(
            ["powershell","-Command",
             "Get-MpPreference | Select-Object -ExpandProperty ExclusionPath"],
            capture_output=True, text=True, timeout=5
        )
        excluded_paths = [p.strip().lower() for p in r.stdout.strip().splitlines() if p.strip()]
        base_lower = str(BASE_DIR).lower()
        # Check if BASE_DIR or ANY of its parent folders is excluded
        is_excluded = any(
            base_lower.startswith(ep) or base_lower == ep
            for ep in excluded_paths
        )
        if not is_excluded:
            return {
                "name": "Windows Defender Exclusion",
                "ok": False,
                "value": "SentinelNet folder not excluded",
                "fix": (
                    f"Windows Security → Virus & threat protection\n"
                    f"→ Manage settings → Exclusions → Add folder:\n"
                    f"  {BASE_DIR}\n"
                    f"This prevents Defender flagging scapy as 'Trojan:Python/Casdet'"
                ),
                "severity": "WARNING",
                "note": "Defender may block scapy with a false positive",
            }
    except Exception: pass
    return None


def check_macos_privacy() -> Optional[dict]:
    """macOS only: check Screen Recording permission."""
    if not IS_MAC: return None
    ver = macos_version()
    if ver < (10, 15): return None
    try:
        from agents.platform_utils import take_screenshot
        img = take_screenshot()
        if img is None: raise Exception("None returned")
        return {"name": "macOS Screen Recording", "ok": True,
                "value": "Permission granted", "severity": "OK", "fix": ""}
    except Exception:
        fix = ("System Settings → Privacy & Security\n"
               "→ Screen Recording → enable Terminal\n"
               "Then restart SentinelNet")
        if ver >= (14, 0):
            fix += "\n\nmacOS Sonoma: also check Local Network permission"
        return {
            "name": "macOS Screen Recording", "ok": False,
            "value": "Permission NOT granted",
            "fix": fix, "severity": "WARNING",
            "note": "Video call monitor requires Screen Recording permission",
        }


def check_linux_display() -> Optional[dict]:
    """Linux only: check display availability for screenshots."""
    if not IS_LINUX: return None
    display  = os.environ.get("DISPLAY", "")
    wayland  = os.environ.get("WAYLAND_DISPLAY", "")
    if display or wayland:
        return {"name": "Linux Display", "ok": True,
                "value": f"DISPLAY={display or wayland}",
                "severity": "OK", "fix": ""}
    try:
        import mss
        return {"name": "Linux Display", "ok": True,
                "value": "Headless (mss works without X11)",
                "severity": "OK", "fix": ""}
    except ImportError: pass
    return {
        "name": "Linux Display", "ok": False,
        "value": "No DISPLAY — headless without mss",
        "fix": ("pip install mss  # works headless\n"
                "OR set DISPLAY=:0 if X11 is running"),
        "severity": "WARNING",
        "note": "Video call monitor needs display or mss library",
    }


# ══════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════

class StartupChecker:

    def __init__(self):
        self.results: dict   = {}
        self.all_checks: list = []

    def run_all(self, use_cache: bool = True) -> dict:
        if use_cache:
            cached = load_startup_cache()
            if cached:
                self.results    = cached
                self.all_checks = cached.get("checks", [])
                return cached

        checks: List[dict] = []
        checks.append(check_python_version())
        checks.append(check_admin_rights())
        checks.append(check_packet_capture_driver())
        checks.extend(check_required_packages())
        checks.extend(check_port_availability())
        checks.append(check_disk_space())
        checks.append(check_ram())
        checks.append(check_tls_certificate())
        checks.append(check_screen_capture())
        checks.append(check_network_interfaces())
        checks.extend(check_optional_packages())

        for extra_fn in [check_windows_defender,
                         check_macos_privacy,
                         check_linux_display]:
            r = extra_fn()
            if r: checks.append(r)

        self.all_checks = checks

        criticals = [c for c in checks
                     if c.get("severity") == "CRITICAL" and not c["ok"]]
        warnings  = [c for c in checks
                     if c.get("severity") == "WARNING"  and not c["ok"]]
        optionals = [c for c in checks
                     if c.get("severity") == "OPTIONAL" and not c["ok"]]

        cap   = next((c for c in checks if any(
            k in c["name"] for k in ["Npcap","libpcap","Packet Capture"]
        )), None)
        admin = next((c for c in checks
                      if c["name"] == "Admin / Root Rights"), None)
        live  = bool(cap and cap["ok"] and admin and admin["ok"])

        if criticals:
            overall, msg = "RED",    f"{len(criticals)} critical — SentinelNet may not start"
        elif warnings:
            overall, msg = "YELLOW", f"{len(warnings)} warning(s) — some features limited"
        else:
            overall, msg = "GREEN",  "All systems ready"

        self.results = {
            "overall_status":  overall,
            "status_message":  msg,
            "capture_mode":    "LIVE" if live else "SYNTHETIC",
            "live_available":  live,
            "system":          OS,
            "python_version":  f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "checks":          checks,
            "critical_count":  len(criticals),
            "warning_count":   len(warnings),
            "optional_count":  len(optionals),
            "criticals":       criticals,
            "warnings":        warnings,
            "optionals":       optionals,
            "opencv_available":any(c["ok"] and "opencv" in c["name"].lower() for c in checks),
            "mss_available":   any(c["ok"] and "mss" in c["name"].lower()    for c in checks),
            "screen_capture":  any(c["ok"] and c["name"] == "Screen Capture" for c in checks),
            "screen_backend":  screenshot_backend_name(),
        }
        save_startup_cache(self.results)
        return self.results

    def print_report(self):
        r = self.results
        W = 66
        print()
        print("=" * W)
        print("  SENTINELNET v2.0 - STARTUP CHECK")
        print("=" * W)
        print(f"  OS      : {OS} {platform.release()}")
        print(f"  Python  : {r['python_version']}")
        print(f"  Arch    : {platform.machine()}")
        print("=" * W)
        icon = {"GREEN":"[OK]","YELLOW":"[!] ","RED":"[X] "}.get(r["overall_status"],"[?] ")
        print(f"  {icon} {r['overall_status']} - {r['status_message']}")
        print()
        if r["live_available"]:
            print("  [OK] CAPTURE MODE : LIVE")
        else:
            print("  [!]  CAPTURE MODE : SYNTHETIC")
            print()
            print(f"  To enable LIVE capture on {OS}:")
            for line in libpcap_install_command().splitlines():
                print(f"    {line}")
        print()
        print("-" * W)
        if r["criticals"]:
            print(f"  [X] CRITICAL ({len(r['criticals'])}):")
            for c in r["criticals"]:
                print(f"     [X] {c['name']}: {c['value']}")
                for l in c.get("fix","").split("\n"):
                    if l.strip(): print(f"        {l}")
            print()
        if r["warnings"]:
            print(f"  [!] WARNINGS ({len(r['warnings'])}):")
            for w in r["warnings"]:
                print(f"     [!] {w['name']}: {w['value']}")
                for l in w.get("fix","").split("\n")[:3]:
                    if l.strip(): print(f"        {l}")
                if w.get("note"): print(f"        Note: {w['note']}")
            print()
        if r["optionals"]:
            print(f"  [+] OPTIONAL ({len(r['optionals'])}) - improves capabilities:")
            for o in r["optionals"]:
                print(f"     [+] {o['name'].replace('Optional: ','')}  ->  {o['fix']}")
            print()
        ok_n = sum(1 for c in r["checks"] if c["ok"] and c.get("severity")=="OK")
        print(f"  ✅ READY  ({ok_n} checks passed)")
        print()
        print("-" * W)
        print("  Dashboard : http://localhost:8000")
        print("  API Docs  : http://localhost:8000/docs")
        print("=" * W)
        print()


# ── Module-level singleton ─────────────────────────────────
_instance: Optional[StartupChecker] = None
_results:  Optional[dict]           = None


def run_startup_check(print_report: bool = True,
                      use_cache:    bool = True) -> dict:
    global _instance, _results
    _instance = StartupChecker()
    _results  = _instance.run_all(use_cache=use_cache)
    if print_report:
        _instance.print_report()
    return _results


def get_check_results() -> Optional[dict]:
    return _results


def get_capture_mode() -> str:
    return (_results.get("capture_mode", "SYNTHETIC")
            if _results else "SYNTHETIC")
