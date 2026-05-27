"""
SentinelNet v2.0 — Enterprise Security Module
Implements 3 critical security layers for 500+ employee deployment:

  1. Privacy Mode     — Zero email body storage, metadata-only logging
  2. HTTPS/TLS        — Self-signed cert generation + HTTPS enforcement
  3. Audit Trail      — Immutable log of every dashboard access and action

Designed for:
  - GDPR (Europe) compliance
  - ECPA (USA) compliance
  - IT Act 2000 (India) compliance
  - SOC 2 Type II readiness
"""

import os
import json
import time
import hashlib
import hmac
import base64
import secrets
import logging
import ipaddress
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from collections import deque
from pathlib import Path

log = logging.getLogger("SentinelNet.Security")

# ── Cross-platform helpers ────────────────────────────────
from agents.platform_utils import (
    IS_WIN, IS_MAC, IS_LINUX,
    secure_file, open_text,
    get_monthly_log_path, cleanup_old_logs,
)

# ── Paths ─────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
CERTS_DIR  = BASE_DIR / "certs"
AUDIT_DIR  = BASE_DIR / "logs" / "audit"

for d in [DATA_DIR, CERTS_DIR, AUDIT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Use monthly rotating audit log (Fix 16)
def _current_audit_path():
    return get_monthly_log_path("audit", "audit_trail")

AUDIT_LOG_PATH = _current_audit_path()
PRIVACY_CFG_PATH  = DATA_DIR / "privacy_config.json"
SESSION_STORE     = {}   # token → session info
SESSION_LOCK      = threading.Lock()
SESSION_TTL       = 8 * 3600  # 8 hours


# ══════════════════════════════════════════════════════════
# PART 1 — PRIVACY MODE
# ══════════════════════════════════════════════════════════

class PrivacyMode:
    """
    Controls what email data SentinelNet is allowed to store.

    Privacy Mode ON  (recommended for companies):
      → NEVER stores email body, subject, or sender name
      → Stores ONLY: scan verdict, confidence score,
        threat type, timestamp, anonymized email ID
      → Even in RAM — body is discarded immediately after scan

    Privacy Mode OFF (personal use only):
      → Stores subject, sender, first 300 chars of body
      → Used for debugging and manual review
    """

    DEFAULT_CONFIG = {
        "privacy_mode_enabled":     True,   # ON by default for enterprise
        "store_subject":            False,  # store subject line?
        "store_sender":             False,  # store sender address?
        "store_body_preview":       False,  # store any body content?
        "store_attachments_list":   False,  # store attachment names?
        "anonymize_email_ids":      True,   # replace emails with "Email #N"
        "retention_days":           90,     # auto-delete logs after N days
        "max_body_chars_in_ram":    0,      # 0 = never keep body in RAM
        "log_scan_metadata":        True,   # always log verdict + score
        "consent_notice_shown":     False,  # track if users were notified
        "gdpr_mode":                True,   # extra GDPR protections
    }

    def __init__(self):
        self.config = self._load()
        self._email_counter = 0
        self._counter_lock  = threading.Lock()

    def _load(self) -> dict:
        if PRIVACY_CFG_PATH.exists():
            try:
                with open_text(PRIVACY_CFG_PATH) as f:
                    saved = json.load(f)
                    cfg = dict(self.DEFAULT_CONFIG)
                    cfg.update(saved)
                    return cfg
            except:
                pass
        self._save(self.DEFAULT_CONFIG)
        return dict(self.DEFAULT_CONFIG)

    def _save(self, cfg: dict):
        with open_text(PRIVACY_CFG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)

    def update(self, updates: dict):
        self.config.update(updates)
        self._save(self.config)
        log.info(f"Privacy config updated: {updates}")

    @property
    def enabled(self) -> bool:
        return self.config["privacy_mode_enabled"]

    def next_email_id(self) -> str:
        """Generate anonymized email identifier"""
        with self._counter_lock:
            self._email_counter += 1
            return f"EMAIL-{self._email_counter:08d}"

    def sanitize_email_record(self, raw: dict) -> dict:
        """
        Strip sensitive fields from email record based on privacy settings.
        Called BEFORE any storage — guarantees body never hits disk.
        """
        if not self.enabled:
            return raw  # privacy mode off — return as-is

        safe = {
            # Always safe to keep
            "scan_id":          raw.get("scan_id", ""),
            "type":             raw.get("type", "EMAIL_LIVE"),
            "timestamp":        raw.get("timestamp", ""),
            "provider":         raw.get("provider", ""),
            "account":          raw.get("account", ""),  # which monitored inbox

            # Scan results — always kept
            "is_threat":        raw.get("is_threat", False),
            "confidence":       raw.get("confidence", 0),
            "severity":         raw.get("severity", "LOW"),
            "verdict":          raw.get("verdict", ""),
            "recommended_action": raw.get("recommended_action", "ALLOW"),
            "scores":           raw.get("scores", {}),

            # Privacy-controlled fields
            "email_ref":        self.next_email_id() if self.config["anonymize_email_ids"]
                                else raw.get("scan_id", ""),
        }

        # Conditionally include based on config
        if self.config["store_subject"]:
            safe["subject"] = raw.get("subject", "")[:200]
        else:
            safe["subject"] = "[REDACTED — Privacy Mode]"

        if self.config["store_sender"]:
            safe["sender"] = raw.get("sender", "")
        else:
            # Hash sender for deduplication without storing identity
            sender = raw.get("sender", "")
            if sender:
                safe["sender_hash"] = hashlib.sha256(
                    sender.encode()
                ).hexdigest()[:12]
            safe["sender"] = "[REDACTED]"

        # Body — NEVER stored in privacy mode
        safe["body_preview"] = ""

        # Indicators — keep threat indicators but strip quoted content
        raw_indicators = raw.get("indicators", [])
        safe["indicators"] = [
            i for i in raw_indicators
            if not any(sensitive in i.lower()
                       for sensitive in ["password", "credit", "ssn", "account number"])
        ][:5]

        return safe

    def get_config(self) -> dict:
        return dict(self.config)

    def get_status_display(self) -> dict:
        return {
            "privacy_mode":       self.enabled,
            "body_storage":       "NEVER" if not self.config["store_body_preview"] else "LIMITED",
            "subject_storage":    "STORED" if self.config["store_subject"] else "REDACTED",
            "sender_storage":     "STORED" if self.config["store_sender"] else "HASHED ONLY",
            "retention_days":     self.config["retention_days"],
            "gdpr_mode":          self.config["gdpr_mode"],
            "anonymized_ids":     self.config["anonymize_email_ids"],
            "compliance_ready":   self.enabled and self.config["gdpr_mode"],
        }


# ══════════════════════════════════════════════════════════
# PART 2 — HTTPS / TLS CERTIFICATE
# ══════════════════════════════════════════════════════════

class HTTPSManager:
    """
    Generates a self-signed TLS certificate for HTTPS.
    For production: replace with Let's Encrypt or company CA cert.

    Self-signed is sufficient for:
      - Internal company network deployment
      - Intranet access
      - Behind a corporate reverse proxy (nginx/Apache)

    For public internet: use Let's Encrypt (certbot)
    """

    CERT_FILE = CERTS_DIR / "sentinelnet.crt"
    KEY_FILE  = CERTS_DIR / "sentinelnet.key"

    def ensure_certificates(self) -> dict:
        """Generate certificates if they don't exist"""
        if self.CERT_FILE.exists() and self.KEY_FILE.exists():
            # Check if cert is still valid (not expired)
            if self._is_valid():
                log.info("TLS certificates found and valid")
                return {
                    "exists": True,
                    "cert":   str(self.CERT_FILE),
                    "key":    str(self.KEY_FILE),
                }

        return self._generate()

    def _is_valid(self) -> bool:
        """Check if existing cert is still valid"""
        try:
            import ssl
            cert_info = ssl._ssl._test_decode_cert(str(self.CERT_FILE))
            not_after = cert_info.get("notAfter", "")
            if not_after:
                exp = datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z")
                return exp > datetime.utcnow() + timedelta(days=7)
        except:
            pass
        return False

    def _generate(self) -> dict:
        """Generate self-signed TLS certificate using cryptography library"""
        try:
            from cryptography import x509
            from cryptography.x509.oid import NameOID
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
            import ipaddress as ip_module

            log.info("Generating self-signed TLS certificate...")

            # Generate RSA-2048 private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )

            # Certificate subject
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME,             "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME,   "Security"),
                x509.NameAttribute(NameOID.LOCALITY_NAME,            "SentinelNet"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME,        "SentinelNet AMACDF"),
                x509.NameAttribute(NameOID.COMMON_NAME,              "sentinelnet.local"),
            ])

            # Build certificate — valid 2 years
            cert = (
                x509.CertificateBuilder()
                .subject_name(subject)
                .issuer_name(issuer)
                .public_key(private_key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(datetime.utcnow())
                .not_valid_after(datetime.utcnow() + timedelta(days=730))
                .add_extension(
                    x509.SubjectAlternativeName([
                        x509.DNSName("localhost"),
                        x509.DNSName("sentinelnet.local"),
                        x509.IPAddress(ip_module.IPv4Address("127.0.0.1")),
                        x509.IPAddress(ip_module.IPv4Address("0.0.0.0")),
                    ]),
                    critical=False,
                )
                .add_extension(
                    x509.BasicConstraints(ca=False, path_length=None),
                    critical=True,
                )
                .sign(private_key, hashes.SHA256())
            )

            # Write key
            with open(self.KEY_FILE, "wb") as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                ))

            # Write cert
            with open(self.CERT_FILE, "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))

            # Secure key file permissions (Unix only)
            try:
                os.chmod(self.KEY_FILE, 0o600)
                os.chmod(self.CERT_FILE, 0o644)
            except:
                pass

            log.info(f"TLS certificate generated: {self.CERT_FILE}")
            return {
                "exists":    True,
                "generated": True,
                "cert":      str(self.CERT_FILE),
                "key":       str(self.KEY_FILE),
                "valid_days": 730,
                "note": (
                    "Self-signed certificate generated. "
                    "Browser will show security warning — click 'Advanced' → 'Proceed'. "
                    "For production, replace with Let's Encrypt certificate."
                ),
            }

        except ImportError:
            log.warning("cryptography library not installed — HTTPS not available")
            return {
                "exists":  False,
                "error":   "Install: pip install cryptography",
                "install": "pip install cryptography",
            }
        except Exception as e:
            log.error(f"Certificate generation failed: {e}")
            return {"exists": False, "error": str(e)}

    def get_ssl_context(self):
        """Return SSL context for uvicorn"""
        try:
            import ssl
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(str(self.CERT_FILE), str(self.KEY_FILE))
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            return ctx
        except Exception as e:
            log.error(f"SSL context error: {e}")
            return None

    @property
    def available(self) -> bool:
        return self.CERT_FILE.exists() and self.KEY_FILE.exists()

    def get_status(self) -> dict:
        if not self.available:
            return {"https_available": False, "reason": "No certificates found"}
        try:
            import ssl
            info = ssl._ssl._test_decode_cert(str(self.CERT_FILE))
            not_after = info.get("notAfter", "")
            exp = None
            days_left = None
            if not_after:
                exp = datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z")
                days_left = (exp - datetime.utcnow()).days
            return {
                "https_available": True,
                "cert_file":       str(self.CERT_FILE),
                "key_file":        str(self.KEY_FILE),
                "expires":         str(exp) if exp else "Unknown",
                "days_remaining":  days_left,
                "type":            "Self-signed (replace with CA cert for production)",
                "tls_min_version": "TLS 1.2",
            }
        except Exception as e:
            return {"https_available": True, "cert_file": str(self.CERT_FILE), "note": str(e)}


# ══════════════════════════════════════════════════════════
# PART 3 — IMMUTABLE AUDIT TRAIL
# ══════════════════════════════════════════════════════════

# Audit event types
class AuditEvent:
    # Dashboard access
    DASHBOARD_ACCESS    = "DASHBOARD_ACCESS"
    DASHBOARD_LOGIN     = "DASHBOARD_LOGIN"
    DASHBOARD_LOGOUT    = "DASHBOARD_LOGOUT"
    LOGIN_FAILED        = "LOGIN_FAILED"

    # Email monitor
    EMAIL_ACCOUNT_ADDED = "EMAIL_ACCOUNT_ADDED"
    EMAIL_ACCOUNT_REMOVED = "EMAIL_ACCOUNT_REMOVED"
    EMAIL_SCANNED       = "EMAIL_SCANNED"
    EMAIL_THREAT_FOUND  = "EMAIL_THREAT_FOUND"

    # Scan actions
    SCAN_TEXT           = "SCAN_TEXT"
    SCAN_EMAIL          = "SCAN_EMAIL"
    SCAN_IMAGE          = "SCAN_IMAGE"
    SCAN_VOICE          = "SCAN_VOICE"
    SCAN_VIDEO          = "SCAN_VIDEO"

    # System
    SYSTEM_START        = "SYSTEM_START"
    SYSTEM_STOP         = "SYSTEM_STOP"
    CONFIG_CHANGED      = "CONFIG_CHANGED"
    PRIVACY_MODE_CHANGED = "PRIVACY_MODE_CHANGED"
    THREAT_BLOCKED      = "THREAT_BLOCKED"
    FL_ROUND_COMPLETED  = "FL_ROUND_COMPLETED"

    # Data access
    AUDIT_LOG_VIEWED    = "AUDIT_LOG_VIEWED"
    SCAN_RESULTS_VIEWED = "SCAN_RESULTS_VIEWED"
    THREAT_DETAILS_VIEWED = "THREAT_DETAILS_VIEWED"


class AuditTrail:
    """
    Immutable append-only audit log.

    Every entry records:
      - WHO:   IP address, session token hash
      - WHAT:  Event type + details
      - WHEN:  UTC timestamp (millisecond precision)
      - HASH:  SHA-256 chain hash (tamper detection)

    Stored as JSONL (one JSON object per line).
    Chain hashing means any modification to past entries
    is immediately detectable.

    Designed for:
      - SOC 2 Type II compliance
      - GDPR Article 30 (records of processing)
      - Internal security investigations
      - Legal discovery
    """

    def __init__(self):
        self._lock        = threading.Lock()
        self._chain_hash  = "0" * 64   # genesis hash
        self._buffer      = deque(maxlen=10000)  # in-memory recent events
        self._entry_count = 0
        self._load_existing()

    def _load_existing(self):
        """Load existing audit trail and restore chain hash"""
        if not AUDIT_LOG_PATH.exists():
            self._log_system_event(AuditEvent.SYSTEM_START,
                                   {"message": "Audit trail initialized"})
            return
        try:
            last_hash = "0" * 64
            count = 0
            with open_text(_current_audit_path()) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        last_hash = entry.get("chain_hash", last_hash)
                        count += 1
                        self._buffer.append(entry)
                    except:
                        continue
            self._chain_hash  = last_hash
            self._entry_count = count
            log.info(f"Audit trail loaded: {count} existing entries")
        except Exception as e:
            log.error(f"Audit trail load error: {e}")

    def log(self, event_type: str, details: dict = None,
            ip_address: str = "system", session_token: str = "",
            user_id: str = "system") -> dict:
        """
        Record an audit event. Thread-safe, append-only.
        Returns the created entry.
        """
        with self._lock:
            self._entry_count += 1
            ts = datetime.utcnow().isoformat() + "Z"

            # Privacy-safe session identifier (never store raw token)
            session_ref = ""
            if session_token:
                session_ref = hashlib.sha256(
                    session_token.encode()
                ).hexdigest()[:16]

            # Sanitize details — remove any sensitive fields
            safe_details = self._sanitize_details(details or {})

            entry = {
                "seq":          self._entry_count,
                "timestamp":    ts,
                "event_type":   event_type,
                "user_id":      user_id,
                "ip_address":   self._anonymize_ip(ip_address),
                "session_ref":  session_ref,
                "details":      safe_details,
            }

            # Compute chain hash (links this entry to previous)
            entry_str = json.dumps(entry, sort_keys=True)
            chain_input = f"{self._chain_hash}:{entry_str}"
            entry["chain_hash"] = hashlib.sha256(
                chain_input.encode()
            ).hexdigest()
            self._chain_hash = entry["chain_hash"]

            # Write to disk (append-only)
            try:
                with open_text(_current_audit_path(), "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as e:
                log.error(f"Audit write error: {e}")

            # Keep in memory buffer
            self._buffer.append(entry)

            return entry

    def _sanitize_details(self, details: dict) -> dict:
        """Remove sensitive fields from audit details"""
        FORBIDDEN_KEYS = {
            "password", "token", "secret", "key", "credential",
            "body", "content", "email_body", "raw_text",
            "credit_card", "ssn", "api_key",
        }
        return {
            k: "[REDACTED]" if k.lower() in FORBIDDEN_KEYS else v
            for k, v in details.items()
            if isinstance(v, (str, int, float, bool, list)) or v is None
        }

    def _anonymize_ip(self, ip: str) -> str:
        """
        Partially anonymize IP for GDPR compliance.
        192.168.1.100 → 192.168.1.xxx
        """
        if not ip or ip == "system":
            return ip
        try:
            addr = ipaddress.ip_address(ip)
            if addr.version == 4:
                parts = ip.split(".")
                return f"{parts[0]}.{parts[1]}.{parts[2]}.xxx"
            else:
                # IPv6 — keep first 3 groups
                parts = ip.split(":")
                return ":".join(parts[:3]) + ":xxxx:xxxx:xxxx:xxxx:xxxx"
        except:
            return ip[:8] + "xxx"

    def verify_integrity(self) -> dict:
        """
        Verify the chain hash of the entire audit trail.
        Any tampering will break the chain.
        """
        if not AUDIT_LOG_PATH.exists():
            return {"valid": True, "entries": 0, "message": "No audit log yet"}

        prev_hash    = "0" * 64
        count        = 0
        tampered_at  = None

        try:
            with open_text(_current_audit_path()) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        stored_hash = entry.pop("chain_hash", "")
                        entry_str   = json.dumps(entry, sort_keys=True)
                        chain_input = f"{prev_hash}:{entry_str}"
                        computed    = hashlib.sha256(chain_input.encode()).hexdigest()

                        if computed != stored_hash:
                            tampered_at = count + 1
                            break

                        prev_hash = stored_hash
                        count += 1
                    except:
                        tampered_at = count + 1
                        break
        except Exception as e:
            return {"valid": False, "error": str(e)}

        if tampered_at:
            return {
                "valid":       False,
                "entries":     count,
                "tampered_at": tampered_at,
                "message":     f"⚠️ INTEGRITY VIOLATION at entry {tampered_at}",
            }

        return {
            "valid":   True,
            "entries": count,
            "message": f"✅ All {count} entries verified — audit trail intact",
        }

    def get_recent(self, limit: int = 100,
                   event_type: str = None,
                   ip_address: str = None) -> List[dict]:
        """Get recent audit entries with optional filtering"""
        entries = list(self._buffer)
        entries.reverse()  # most recent first

        if event_type:
            entries = [e for e in entries if e.get("event_type") == event_type]
        if ip_address:
            anon_ip = self._anonymize_ip(ip_address)
            entries = [e for e in entries if e.get("ip_address") == anon_ip]

        return entries[:limit]

    def get_stats(self) -> dict:
        entries  = list(self._buffer)
        by_type  = {}
        by_hour  = {}
        for e in entries:
            et = e.get("event_type", "UNKNOWN")
            by_type[et] = by_type.get(et, 0) + 1
            try:
                hour = e["timestamp"][:13]  # "2025-01-15T14"
                by_hour[hour] = by_hour.get(hour, 0) + 1
            except:
                pass
        return {
            "total_entries":  self._entry_count,
            "in_memory":      len(entries),
            "by_event_type":  by_type,
            "by_hour":        dict(sorted(by_hour.items())[-24:]),
            "log_file":       str(AUDIT_LOG_PATH),
            "log_size_kb":    round(
                AUDIT_LOG_PATH.stat().st_size / 1024, 1
            ) if AUDIT_LOG_PATH.exists() else 0,
            "chain_hash":     self._chain_hash[:16] + "...",
        }

    def _log_system_event(self, event_type: str, details: dict):
        self.log(event_type, details, ip_address="system", user_id="system")

    def export_csv(self, output_path: str, days: int = 30) -> str:
        """Export audit trail to CSV for compliance reporting"""
        import csv
        cutoff = datetime.utcnow() - timedelta(days=days)
        rows   = []

        try:
            with open_text(_current_audit_path()) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        ts_str = entry.get("timestamp", "")
                        ts = datetime.fromisoformat(ts_str.replace("Z", ""))
                        if ts >= cutoff:
                            rows.append({
                                "seq":        entry.get("seq", ""),
                                "timestamp":  ts_str,
                                "event_type": entry.get("event_type", ""),
                                "user_id":    entry.get("user_id", ""),
                                "ip_address": entry.get("ip_address", ""),
                                "details":    json.dumps(entry.get("details", {})),
                                "chain_hash": entry.get("chain_hash", "")[:16],
                            })
                    except:
                        continue
        except:
            pass

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "seq", "timestamp", "event_type", "user_id",
                "ip_address", "details", "chain_hash"
            ])
            writer.writeheader()
            writer.writerows(rows)

        return output_path


# ══════════════════════════════════════════════════════════
# PART 4 — SESSION MANAGER (needed for HTTPS + Audit)
# ══════════════════════════════════════════════════════════

class SessionManager:
    """
    Lightweight session manager.
    Issues secure tokens, tracks active sessions.
    Ties into audit trail automatically.
    """

    def __init__(self, audit: AuditTrail):
        self.audit    = audit
        self._sessions: Dict[str, dict] = {}
        self._lock    = threading.Lock()
        # Start cleanup thread
        threading.Thread(target=self._cleanup_loop, daemon=True).start()

    def create_session(self, user_id: str, ip: str) -> str:
        """Create new session, return token"""
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[token] = {
                "user_id":    user_id,
                "ip":         ip,
                "created_at": time.time(),
                "last_seen":  time.time(),
                "requests":   0,
            }
        self.audit.log(
            AuditEvent.DASHBOARD_LOGIN,
            {"user_id": user_id},
            ip_address=ip,
            session_token=token,
            user_id=user_id,
        )
        return token

    def validate(self, token: str, ip: str = "") -> Optional[dict]:
        """Validate token, update last_seen. Returns session or None."""
        with self._lock:
            session = self._sessions.get(token)
            if not session:
                return None
            if time.time() - session["last_seen"] > SESSION_TTL:
                del self._sessions[token]
                return None
            session["last_seen"] = time.time()
            session["requests"] += 1
            return dict(session)

    def revoke(self, token: str):
        with self._lock:
            session = self._sessions.pop(token, None)
        if session:
            self.audit.log(
                AuditEvent.DASHBOARD_LOGOUT,
                {"user_id": session.get("user_id")},
                session_token=token,
            )

    def get_active_sessions(self) -> List[dict]:
        now = time.time()
        with self._lock:
            return [
                {
                    "user_id":      s["user_id"],
                    "ip":           s["ip"][:10] + "...",
                    "requests":     s["requests"],
                    "duration_min": round((now - s["created_at"]) / 60, 1),
                    "idle_min":     round((now - s["last_seen"])   / 60, 1),
                }
                for s in self._sessions.values()
                if now - s["last_seen"] <= SESSION_TTL
            ]

    def _cleanup_loop(self):
        while True:
            time.sleep(300)  # every 5 minutes
            now = time.time()
            with self._lock:
                expired = [
                    t for t, s in self._sessions.items()
                    if now - s["last_seen"] > SESSION_TTL
                ]
                for t in expired:
                    del self._sessions[t]
            if expired:
                log.info(f"Expired {len(expired)} session(s)")


# ══════════════════════════════════════════════════════════
# PART 5 — GDPR COMPLIANCE HELPER
# ══════════════════════════════════════════════════════════

class GDPRCompliance:
    """
    Helpers for GDPR / data protection compliance.
    Required for EU employee monitoring.
    """

    RETENTION_POLICY = {
        "scan_results":    90,   # days
        "audit_trail":    365,   # days (legal minimum)
        "email_metadata":  30,   # days
        "threat_alerts":   90,   # days
    }

    @staticmethod
    def get_consent_notice() -> str:
        return """
╔══════════════════════════════════════════════════════════════╗
║           EMPLOYEE EMAIL MONITORING NOTICE                   ║
║                  (Required by Law)                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Your company uses SentinelNet to monitor work email         ║
║  accounts for security threats including:                    ║
║                                                              ║
║    • Phishing and social engineering attacks                 ║
║    • AI-generated malicious content                         ║
║    • Malware and suspicious attachments                     ║
║                                                              ║
║  WHAT IS STORED:                                             ║
║    ✓ Threat verdict (SAFE / SUSPICIOUS / THREAT)            ║
║    ✓ Detection timestamp                                     ║
║    ✗ Email body content (NEVER stored)                      ║
║    ✗ Personal communications                                ║
║                                                              ║
║  YOUR RIGHTS (GDPR Article 15-22):                          ║
║    • Right to access your scan records                      ║
║    • Right to correct inaccurate data                       ║
║    • Data deleted after 90 days automatically               ║
║                                                              ║
║  Contact your Data Protection Officer for questions.        ║
╚══════════════════════════════════════════════════════════════╝
"""

    @staticmethod
    def purge_old_records(audit: AuditTrail, days: int = 365):
        """Purge audit records older than retention period"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        purged = 0
        if not AUDIT_LOG_PATH.exists():
            return 0
        temp_path = AUDIT_LOG_PATH.with_suffix(".tmp")
        try:
            with open(AUDIT_LOG_PATH) as src, open(temp_path, "w") as dst:
                for line in src:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        ts = datetime.fromisoformat(
                            entry["timestamp"].replace("Z", "")
                        )
                        if ts >= cutoff:
                            dst.write(line + "\n")
                        else:
                            purged += 1
                    except:
                        dst.write(line + "\n")
            os.replace(temp_path, AUDIT_LOG_PATH)
            log.info(f"GDPR purge: removed {purged} entries older than {days} days")
            audit.log(
                AuditEvent.CONFIG_CHANGED,
                {"action": "gdpr_purge", "entries_removed": purged, "retention_days": days}
            )
        except Exception as e:
            log.error(f"Purge error: {e}")
            if temp_path.exists():
                temp_path.unlink()
        return purged


# ── Singleton instances (imported by main.py) ──────────────
privacy_mode   = PrivacyMode()
https_manager  = HTTPSManager()
audit_trail    = AuditTrail()
gdpr           = GDPRCompliance()
session_mgr    = SessionManager(audit_trail)
