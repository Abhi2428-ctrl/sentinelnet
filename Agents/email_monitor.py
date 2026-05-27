"""
SentinelNet v2.0 — Real-Time Email Monitor
Connects to Gmail, Outlook/Office365, and custom IMAP servers.
Scans every incoming email automatically using the AI detector engine.

Actions on threat:
  - Send alert notification (dashboard + optional SMTP alert email)
  - Log to SentinelNet dashboard

Setup per provider:
  Gmail       → Enable IMAP + App Password (settings.google.com/apppasswords)
  Outlook     → Enable IMAP + App Password (account.microsoft.com/security)
  Custom IMAP → Host / Port / Username / Password
"""

import imaplib
import email
import email.header
import email.utils
import threading
import time
import json

# ── Cross-platform helpers ────────────────────────────────
from agents.platform_utils import (
    DATA_DIR, open_text,
    save_credentials, load_credentials,
)
import os
import re
import ssl
import smtplib
import queue
import logging
from datetime import datetime
from typing import Optional, Dict, List, Callable, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("SentinelNet.Email")

# ── Provider Presets ──────────────────────────────────────
PROVIDER_PRESETS = {
    "gmail": {
        "name":       "Gmail",
        "imap_host":  "imap.gmail.com",
        "imap_port":  993,
        "smtp_host":  "smtp.gmail.com",
        "smtp_port":  587,
        "use_ssl":    True,
        "setup_url":  "https://myaccount.google.com/apppasswords",
        "setup_note": (
            "1. Go to settings.google.com/apppasswords\n"
            "2. Generate App Password for 'Mail'\n"
            "3. Use that 16-char password here (not your Google password)"
        ),
    },
    "outlook": {
        "name":       "Outlook / Office 365",
        "imap_host":  "outlook.office365.com",
        "imap_port":  993,
        "smtp_host":  "smtp.office365.com",
        "smtp_port":  587,
        "use_ssl":    True,
        "setup_url":  "https://account.microsoft.com/security",
        "setup_note": (
            "1. Go to account.microsoft.com/security\n"
            "2. Enable 2FA, then create App Password\n"
            "3. Use that password here"
        ),
    },
    "yahoo": {
        "name":       "Yahoo Mail",
        "imap_host":  "imap.mail.yahoo.com",
        "imap_port":  993,
        "smtp_host":  "smtp.mail.yahoo.com",
        "smtp_port":  587,
        "use_ssl":    True,
        "setup_url":  "https://login.yahoo.com/account/security",
        "setup_note": "Generate App Password from Yahoo Account Security settings.",
    },
    "custom": {
        "name":       "Custom IMAP",
        "imap_host":  "",
        "imap_port":  993,
        "smtp_host":  "",
        "smtp_port":  587,
        "use_ssl":    True,
        "setup_note": "Enter your mail server's IMAP host, port, username, and password.",
    },
}


# ── Email Account Config ──────────────────────────────────
class EmailAccountConfig:
    def __init__(self, provider: str, email_address: str, password: str,
                 imap_host: str = "", imap_port: int = 993,
                 smtp_host: str = "", smtp_port: int = 587,
                 alert_to: str = ""):
        preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["custom"])
        self.provider       = provider
        self.name           = preset["name"]
        self.email_address  = email_address
        self.password       = password
        self.imap_host      = imap_host or preset["imap_host"]
        self.imap_port      = imap_port or preset["imap_port"]
        self.smtp_host      = smtp_host or preset.get("smtp_host", "")
        self.smtp_port      = smtp_port or preset.get("smtp_port", 587)
        self.use_ssl        = preset.get("use_ssl", True)
        self.alert_to       = alert_to or email_address  # where to send alert emails
        self.enabled        = True
        self.last_error     = ""
        self.emails_scanned = 0
        self.threats_found  = 0
        self.connected      = False

    def to_dict(self) -> dict:
        return {
            "provider":       self.provider,
            "name":           self.name,
            "email_address":  self.email_address,
            "imap_host":      self.imap_host,
            "imap_port":      self.imap_port,
            "enabled":        self.enabled,
            "connected":      self.connected,
            "emails_scanned": self.emails_scanned,
            "threats_found":  self.threats_found,
            "last_error":     self.last_error,
        }

    @staticmethod
    def from_dict(d: dict) -> "EmailAccountConfig":
        return EmailAccountConfig(
            provider      = d.get("provider", "custom"),
            email_address = d["email_address"],
            password      = d["password"],
            imap_host     = d.get("imap_host", ""),
            imap_port     = d.get("imap_port", 993),
            smtp_host     = d.get("smtp_host", ""),
            smtp_port     = d.get("smtp_port", 587),
            alert_to      = d.get("alert_to", ""),
        )


# ── Parsed Email ──────────────────────────────────────────
class ParsedEmail:
    def __init__(self):
        self.uid         = ""
        self.message_id  = ""
        self.subject     = ""
        self.sender      = ""
        self.sender_name = ""
        self.recipients  = []
        self.date        = ""
        self.body_text   = ""
        self.body_html   = ""
        self.headers     = {}
        self.attachments = []
        self.raw_size    = 0

    def to_dict(self) -> dict:
        return {
            "uid":         self.uid,
            "message_id":  self.message_id,
            "subject":     self.subject[:200],
            "sender":      self.sender,
            "sender_name": self.sender_name,
            "date":        self.date,
            "body_preview": self.body_text[:500],
            "attachments": [a["name"] for a in self.attachments],
            "size_bytes":  self.raw_size,
        }


# ── Single Account Monitor ────────────────────────────────
class AccountMonitor:
    """
    Monitors one email account via IMAP IDLE (real-time push).
    Falls back to polling every 30s if IDLE not supported.
    """
    IDLE_TIMEOUT    = 29 * 60  # 29 minutes (RFC 2177 recommends < 30min)
    POLL_INTERVAL   = 30       # seconds between polls if IDLE unavailable

    def __init__(self, config: EmailAccountConfig,
                 on_new_email: Callable[[ParsedEmail, EmailAccountConfig], None]):
        self.config       = config
        self.on_new_email = on_new_email
        self._imap        = None
        self._thread      = None
        self._running     = False
        self._paused      = False   # set True when Phishing Guardian is paused
        self._seen_uids   = set()
        self._idle_support = False

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._monitor_loop, daemon=True,
            name=f"monitor-{self.config.provider}"
        )
        self._thread.start()
        log.info(f"[{self.config.name}] Monitor started for {self.config.email_address}")

    def stop(self):
        self._running = False
        self._disconnect()

    # ── Connection ────────────────────────────────────────
    def pause(self):
        """Pause email scanning (Phishing Guardian paused)."""
        self._paused = True
        print("[EmailMonitor] Paused — Phishing Guardian disabled")

    def resume(self):
        """Resume email scanning (Phishing Guardian re-enabled)."""
        self._paused = False
        print("[EmailMonitor] Resumed — Phishing Guardian re-enabled")


    def _connect(self) -> bool:
        try:
            # Use certifi bundle for consistent SSL on all OS (Fix 11)
            try:
                import certifi
                ctx = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                ctx = ssl.create_default_context()
            if self.config.use_ssl:
                self._imap = imaplib.IMAP4_SSL(
                    self.config.imap_host, self.config.imap_port, ssl_context=ctx
                )
            else:
                self._imap = imaplib.IMAP4(self.config.imap_host, self.config.imap_port)
                self._imap.starttls(ssl_context=ctx)

            self._imap.login(self.config.email_address, self.config.password)
            self._imap.select("INBOX")

            # Check IDLE support
            caps = self._imap.capabilities
            self._idle_support = "IDLE" in (caps or [])

            self.config.connected = True
            self.config.last_error = ""
            log.info(f"[{self.config.name}] Connected ({'IDLE' if self._idle_support else 'POLL'} mode)")
            return True

        except imaplib.IMAP4.error as e:
            self.config.last_error = f"Auth failed: {e}"
            log.error(f"[{self.config.name}] Auth error: {e}")
        except ConnectionRefusedError:
            self.config.last_error = f"Cannot reach {self.config.imap_host}:{self.config.imap_port}"
            log.error(f"[{self.config.name}] Connection refused")
        except ssl.SSLError as e:
            self.config.last_error = f"SSL error: {e}"
            log.error(f"[{self.config.name}] SSL error: {e}")
        except Exception as e:
            self.config.last_error = str(e)
            log.error(f"[{self.config.name}] Connection error: {e}")

        self.config.connected = False
        return False

    def _disconnect(self):
        if self._imap:
            try:
                self._imap.close()
                self._imap.logout()
            except:
                pass
            self._imap = None
        self.config.connected = False

    # ── Main Monitor Loop ─────────────────────────────────
    def _monitor_loop(self):
        retry_delay = 10
        while self._running:
            # Respect pause state (Phishing Guardian paused)
            if getattr(self, '_paused', False):
                import time as _t; _t.sleep(2)
                continue
            if not self._connect():
                log.warning(f"[{self.config.name}] Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 300)  # exponential backoff
                continue

            retry_delay = 10  # reset on success

            # Fetch existing unseen emails first
            self._fetch_unseen()

            # Then monitor for new ones
            if self._idle_support:
                self._idle_loop()
            else:
                self._poll_loop()

    def _idle_loop(self):
        """Use IMAP IDLE for real-time push notifications"""
        log.info(f"[{self.config.name}] Starting IDLE real-time monitoring")
        while self._running and self.config.connected:
            try:
                # Send IDLE command
                tag = self._imap._new_tag().decode()
                self._imap.send(f"{tag} IDLE\r\n".encode())
                resp = self._imap.readline()
                if b"+ idling" not in resp.lower() and b"+ idle" not in resp.lower():
                    # Server doesn't really support IDLE
                    self._idle_support = False
                    return

                # Wait for EXISTS notification (new email)
                self._imap.sock.settimeout(self.IDLE_TIMEOUT)
                start = time.time()

                while self._running and (time.time() - start) < self.IDLE_TIMEOUT:
                    try:
                        line = self._imap.readline()
                        if not line:
                            break
                        line_str = line.decode("utf-8", errors="ignore")
                        # New email notification
                        if "EXISTS" in line_str or "RECENT" in line_str:
                            # Send DONE to exit IDLE
                            self._imap.send(b"DONE\r\n")
                            self._imap.readline()  # read OK response
                            self._fetch_unseen()
                            break
                    except (socket_timeout, OSError):
                        break

                # Re-send IDLE after timeout
                try:
                    self._imap.send(b"DONE\r\n")
                    self._imap.readline()
                except:
                    pass

            except Exception as e:
                log.warning(f"[{self.config.name}] IDLE error: {e}")
                self._disconnect()
                break

    def _poll_loop(self):
        """Poll for new emails every N seconds (fallback)"""
        log.info(f"[{self.config.name}] Polling every {self.POLL_INTERVAL}s")
        while self._running and self.config.connected:
            time.sleep(self.POLL_INTERVAL)
            if not self._running:
                break
            try:
                self._imap.select("INBOX")
                self._fetch_unseen()
            except Exception as e:
                log.warning(f"[{self.config.name}] Poll error: {e}")
                self._disconnect()
                break

    # ── Fetch & Parse ─────────────────────────────────────
    def _fetch_unseen(self):
        """Fetch all UNSEEN emails and trigger scanning"""
        try:
            status, data = self._imap.search(None, "UNSEEN")
            if status != "OK" or not data[0]:
                return
            uids = data[0].split()
            new_uids = [u for u in uids if u not in self._seen_uids]
            if not new_uids:
                return
            log.info(f"[{self.config.name}] {len(new_uids)} new email(s) to scan")
            for uid in new_uids[-20:]:  # max 20 at once
                if not self._running:
                    break
                parsed = self._fetch_and_parse(uid)
                if parsed:
                    self._seen_uids.add(uid)
                    self.config.emails_scanned += 1
                    self.on_new_email(parsed, self.config)
        except Exception as e:
            log.warning(f"[{self.config.name}] Fetch error: {e}")

    def _fetch_and_parse(self, uid: bytes) -> Optional[ParsedEmail]:
        """Fetch a single email by UID and parse it"""
        try:
            status, data = self._imap.fetch(uid, "(RFC822)")
            if status != "OK" or not data or not data[0]:
                return None
            raw = data[0][1] if isinstance(data[0], tuple) else None
            if not raw:
                return None
            return self._parse_raw(uid.decode(), raw)
        except Exception as e:
            log.warning(f"[{self.config.name}] Parse error for UID {uid}: {e}")
            return None

    def _parse_raw(self, uid: str, raw: bytes) -> ParsedEmail:
        """Parse raw RFC822 email bytes into ParsedEmail"""
        msg    = email.message_from_bytes(raw)
        parsed = ParsedEmail()
        parsed.uid      = uid
        parsed.raw_size = len(raw)

        # Subject
        subj = msg.get("Subject", "")
        try:
            parts = email.header.decode_header(subj)
            parsed.subject = "".join(
                p.decode(enc or "utf-8", errors="replace") if isinstance(p, bytes) else p
                for p, enc in parts
            )
        except:
            parsed.subject = str(subj)

        # Sender
        from_raw = msg.get("From", "")
        try:
            name, addr = email.utils.parseaddr(from_raw)
            parsed.sender      = addr
            parsed.sender_name = name
        except:
            parsed.sender = from_raw

        # Date
        parsed.date       = msg.get("Date", "")
        parsed.message_id = msg.get("Message-ID", "")

        # Headers (SPF / DKIM / ARC)
        for h in ["Received-SPF", "Authentication-Results", "DKIM-Signature",
                  "X-Spam-Status", "X-Spam-Score", "ARC-Authentication-Results"]:
            val = msg.get(h, "")
            if val:
                parsed.headers[h] = val[:500]

        # Body + attachments
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                disp  = str(part.get("Content-Disposition", ""))
                if "attachment" in disp:
                    fname = part.get_filename() or "unknown"
                    parsed.attachments.append({
                        "name": fname,
                        "type": ctype,
                        "size": len(part.get_payload(decode=True) or b""),
                    })
                elif ctype == "text/plain" and not parsed.body_text:
                    try:
                        charset = part.get_content_charset() or "utf-8"
                        parsed.body_text = part.get_payload(decode=True).decode(
                            charset, errors="replace"
                        )[:5000]
                    except:
                        pass
                elif ctype == "text/html" and not parsed.body_html:
                    try:
                        charset = part.get_content_charset() or "utf-8"
                        html = part.get_payload(decode=True).decode(
                            charset, errors="replace"
                        )
                        # Strip HTML tags for text analysis
                        parsed.body_html = html[:5000]
                        if not parsed.body_text:
                            parsed.body_text = re.sub(r"<[^>]+>", " ", html)[:3000]
                    except:
                        pass
        else:
            try:
                charset = msg.get_content_charset() or "utf-8"
                parsed.body_text = msg.get_payload(decode=True).decode(
                    charset, errors="replace"
                )[:5000]
            except:
                pass

        return parsed


# ── Alert Notifier ────────────────────────────────────────
class AlertNotifier:
    """Sends alert notification emails when a threat is detected"""

    def __init__(self, smtp_host: str, smtp_port: int,
                 username: str, password: str, from_addr: str):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username  = username
        self.password  = password
        self.from_addr = from_addr

    def send_alert(self, to_addr: str, scan_result: dict,
                   original_email: ParsedEmail) -> bool:
        """Send threat alert email notification"""
        try:
            confidence_pct = round(scan_result.get("confidence", 0) * 100)
            severity       = scan_result.get("severity", "MEDIUM")
            verdict        = scan_result.get("verdict", "SUSPICIOUS")
            action         = scan_result.get("recommended_action", "REVIEW")
            indicators     = scan_result.get("indicators", [])[:5]

            sev_emoji = {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡","LOW":"🟢"}.get(severity,"⚠️")

            subject = (
                f"{sev_emoji} SentinelNet Alert: {severity} Threat — "
                f"{original_email.subject[:50]}"
            )

            indicators_html = "".join(
                f"<li style='margin:3px 0;font-family:monospace;font-size:12px'>{i}</li>"
                for i in indicators
            )

            body_html = f"""
<html><body style="font-family:Arial,sans-serif;background:#f5f5f5;padding:20px">
<div style="max-width:600px;margin:auto;background:white;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.15)">
  <div style="background:#1a1a2e;padding:20px;text-align:center">
    <h1 style="color:#00f5ff;margin:0;font-size:20px;letter-spacing:3px">🛡 SENTINELNET</h1>
    <p style="color:#7aa8c4;margin:4px 0;font-size:11px">AUTONOMOUS CYBER DEFENSE SYSTEM</p>
  </div>
  <div style="padding:24px">
    <div style="background:{'#fff0f0' if severity in ['CRITICAL','HIGH'] else '#fffbe6'};
                border-left:4px solid {'#ff2d55' if severity=='CRITICAL' else '#ff6b35' if severity=='HIGH' else '#ffd60a'};
                padding:12px 16px;border-radius:4px;margin-bottom:20px">
      <strong style="font-size:16px">{sev_emoji} {severity} THREAT DETECTED</strong>
      <p style="margin:4px 0;font-size:13px;color:#444">{verdict} — {confidence_pct}% confidence</p>
      <p style="margin:4px 0;font-size:12px;color:#888">Recommended action: <strong>{action}</strong></p>
    </div>

    <h3 style="margin:0 0 10px;font-size:13px;color:#333;text-transform:uppercase;letter-spacing:1px">Original Email</h3>
    <table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:16px">
      <tr><td style="padding:5px 0;color:#888;width:90px">Subject</td>
          <td style="padding:5px 0;font-weight:bold">{original_email.subject[:100]}</td></tr>
      <tr><td style="padding:5px 0;color:#888">From</td>
          <td style="padding:5px 0">{original_email.sender_name} &lt;{original_email.sender}&gt;</td></tr>
      <tr><td style="padding:5px 0;color:#888">Received</td>
          <td style="padding:5px 0">{original_email.date}</td></tr>
      <tr><td style="padding:5px 0;color:#888">Size</td>
          <td style="padding:5px 0">{round(original_email.raw_size/1024, 1)} KB</td></tr>
    </table>

    <h3 style="margin:0 0 8px;font-size:13px;color:#333;text-transform:uppercase;letter-spacing:1px">Threat Indicators</h3>
    <ul style="background:#f8f8f8;border-radius:4px;padding:10px 10px 10px 24px;margin:0 0 16px">
      {indicators_html if indicators_html else "<li>See SentinelNet dashboard for full details</li>"}
    </ul>

    <div style="text-align:center;padding:12px;background:#f0f8ff;border-radius:6px">
      <p style="margin:0;font-size:12px;color:#555">
        View full details in the <strong>SentinelNet Dashboard</strong> at<br/>
        <a href="http://localhost:8000" style="color:#00a8cc">http://localhost:8000</a>
        → Audit Logs tab
      </p>
    </div>

    <p style="margin:16px 0 0;font-size:11px;color:#aaa;text-align:center">
      SentinelNet AMACDF v2.0 · Automated Threat Detection · Do not reply to this email
    </p>
  </div>
</div>
</body></html>"""

            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = f"SentinelNet Alert <{self.from_addr}>"
            msg["To"]      = to_addr
            msg.attach(MIMEText(
                f"SentinelNet Alert: {severity} threat in email from {original_email.sender}\n"
                f"Verdict: {verdict} ({confidence_pct}%)\nAction: {action}\n"
                f"See dashboard: http://localhost:8000", "plain"
            ))
            msg.attach(MIMEText(body_html, "html"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_addr, to_addr, msg.as_string())

            log.info(f"Alert sent to {to_addr} for: {original_email.subject[:50]}")
            return True

        except Exception as e:
            log.warning(f"Alert send failed: {e}")
            return False


# ── Config File Manager ───────────────────────────────────
CONFIG_PATH = DATA_DIR / "email_accounts.json"

def save_accounts(accounts: List[EmailAccountConfig]):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    data = [{
        "provider":      a.provider,
        "email_address": a.email_address,
        "password":      a.password,   # stored locally only
        "imap_host":     a.imap_host,
        "imap_port":     a.imap_port,
        "smtp_host":     a.smtp_host,
        "smtp_port":     a.smtp_port,
        "alert_to":      a.alert_to,
        "enabled":       a.enabled,
    } for a in accounts]
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)

def load_accounts() -> List[EmailAccountConfig]:
    if not os.path.exists(CONFIG_PATH):
        return []
    try:
        with open(CONFIG_PATH) as f:
            data = json.load(f)
        return [EmailAccountConfig.from_dict(d) for d in data]
    except Exception as e:
        log.warning(f"Failed to load accounts: {e}")
        return []


# ── Master Email Monitor ──────────────────────────────────
class RealTimeEmailMonitor:
    """
    Orchestrates monitoring of ALL configured email accounts simultaneously.
    Each account runs in its own thread.
    Calls the AI scanner on every new email.
    """

    def __init__(self, scan_callback: Callable[[dict], None]):
        """
        scan_callback: called with scan result dict for every new email
        """
        self.scan_callback = scan_callback
        self.accounts:  List[EmailAccountConfig]  = []
        self.monitors:  List[AccountMonitor]       = []
        self.notifiers: Dict[str, AlertNotifier]   = {}
        self.running    = False
        self._results   = []  # recent scan results
        self._result_q  = queue.Queue(maxsize=500)

        # Import the AI scanner
        try:
            from agents.ai_detectors import EmailScanner
            self._scanner = EmailScanner()
        except ImportError:
            self._scanner = None
            log.warning("EmailScanner not available — using basic analysis")

        # Load saved accounts
        self.accounts = load_accounts()
        log.info(f"Loaded {len(self.accounts)} saved account(s)")

    def add_account(self, config: EmailAccountConfig) -> bool:
        """Add and start monitoring a new account"""
        # Check for duplicates
        for a in self.accounts:
            if a.email_address == config.email_address:
                log.warning(f"Account already exists: {config.email_address}")
                return False
        self.accounts.append(config)
        save_accounts(self.accounts)
        if self.running:
            self._start_monitor(config)
        return True

    def remove_account(self, email_address: str) -> bool:
        """Remove and stop monitoring an account"""
        self.accounts = [a for a in self.accounts if a.email_address != email_address]
        save_accounts(self.accounts)
        # Stop associated monitor
        self.monitors = [
            m for m in self.monitors
            if m.config.email_address != email_address or not m.stop() or False
        ]
        return True

    def start(self):
        """Start monitoring all configured accounts"""
        self.running = True
        for account in self.accounts:
            if account.enabled:
                self._start_monitor(account)
        log.info(f"Email monitor started: {len(self.monitors)} account(s) active")

    def stop(self):
        """Stop all monitors"""
        self.running = False
        for monitor in self.monitors:
            monitor.stop()
        self.monitors.clear()
        log.info("Email monitor stopped")

    def pause(self):
        """Pause all account monitors (called when Phishing Guardian is paused)."""
        for monitor in self.monitors:
            monitor._paused = True
        log.info("[EmailMonitor] Paused — Phishing Guardian disabled")

    def resume(self):
        """Resume all account monitors (called when Phishing Guardian is resumed)."""
        for monitor in self.monitors:
            monitor._paused = False
        log.info("[EmailMonitor] Resumed — Phishing Guardian re-enabled")


    def _start_monitor(self, config: EmailAccountConfig):
        """Start a single account monitor thread"""
        monitor = AccountMonitor(config, self._on_new_email)
        self.monitors.append(monitor)
        monitor.start()

    def _on_new_email(self, parsed: ParsedEmail, config: EmailAccountConfig):
        """Called by AccountMonitor for every new email — runs AI scan"""
        try:
            log.info(
                f"[{config.name}] New email: '{parsed.subject[:50]}' "
                f"from {parsed.sender}"
            )

            # Run AI scanner
            if self._scanner:
                scan_result = self._scanner.analyze(
                    subject  = parsed.subject,
                    sender   = parsed.sender,
                    body     = parsed.body_text,
                    headers  = parsed.headers,
                )
            else:
                scan_result = self._basic_scan(parsed)

            # Build full result record
            result = {
                "scan_id":      f"LIVE-{int(time.time()*1000)}",
                "type":         "EMAIL_LIVE",
                "timestamp":    datetime.now().isoformat(),
                "account":      config.email_address,
                "provider":     config.provider,
                "subject":      parsed.subject[:200],
                "sender":       parsed.sender,
                "sender_name":  parsed.sender_name,
                "date":         parsed.date,
                "body_preview": parsed.body_text[:300],
                "attachments":  [a["name"] for a in parsed.attachments],
                "headers":      {
                    "spf":  parsed.headers.get("Received-SPF", ""),
                    "dkim": parsed.headers.get("Authentication-Results", ""),
                    "spam": parsed.headers.get("X-Spam-Status", ""),
                },
                **scan_result,
            }

            self._results.insert(0, result)
            if len(self._results) > 500:
                self._results = self._results[:500]

            config.emails_scanned += 1

            # Handle threat
            if scan_result.get("confidence", 0) >= 0.40:
                config.threats_found += 1
                log.warning(
                    f"[{config.name}] THREAT DETECTED: "
                    f"{scan_result.get('verdict')} "
                    f"({round(scan_result.get('confidence',0)*100)}%) "
                    f"in '{parsed.subject[:50]}'"
                )
                # Send alert notification
                self._send_alert_notification(config, scan_result, parsed)

            # Push to main system via callback
            self.scan_callback(result)

        except Exception as e:
            log.error(f"Email processing error: {e}")

    def _send_alert_notification(self, config: EmailAccountConfig,
                                  scan_result: dict, parsed: ParsedEmail):
        """Send email alert notification"""
        if not config.smtp_host or not config.alert_to:
            log.info(f"Alert logged (no SMTP configured for notifications)")
            return
        try:
            notifier_key = config.email_address
            if notifier_key not in self.notifiers:
                self.notifiers[notifier_key] = AlertNotifier(
                    smtp_host = config.smtp_host,
                    smtp_port = config.smtp_port,
                    username  = config.email_address,
                    password  = config.password,
                    from_addr = config.email_address,
                )
            notifier = self.notifiers[notifier_key]
            # Send in background thread
            threading.Thread(
                target=notifier.send_alert,
                args=(config.alert_to, scan_result, parsed),
                daemon=True
            ).start()
        except Exception as e:
            log.warning(f"Alert notification error: {e}")

    def _basic_scan(self, parsed: ParsedEmail) -> dict:
        """Minimal scan when full scanner unavailable"""
        suspicious_words = ["urgent","verify","click here","suspended","password",
                            "congratulations","won","bitcoin","wire transfer"]
        body_lower = parsed.body_text.lower()
        hits = sum(1 for w in suspicious_words if w in body_lower)
        conf = min(1.0, hits * 0.15)
        return {
            "is_threat":           conf >= 0.40,
            "confidence":          conf,
            "severity":            "HIGH" if conf >= 0.6 else "MEDIUM" if conf >= 0.4 else "LOW",
            "verdict":             "SUSPICIOUS" if conf >= 0.4 else "LIKELY SAFE",
            "recommended_action":  "FLAG" if conf >= 0.4 else "ALLOW",
            "indicators":          [f"Suspicious keyword found: {w}" for w in suspicious_words if w in body_lower][:5],
            "scores":              {"basic_keywords": conf},
        }

    def get_status(self) -> dict:
        return {
            "running":        self.running,
            "total_accounts": len(self.accounts),
            "active_monitors": sum(1 for m in self.monitors if m._running),
            "accounts":       [a.to_dict() for a in self.accounts],
            "total_scanned":  sum(a.emails_scanned for a in self.accounts),
            "total_threats":  sum(a.threats_found  for a in self.accounts),
            "recent_results": self._results[:20],
            "providers_supported": list(PROVIDER_PRESETS.keys()),
        }

    def get_results(self, limit: int = 100) -> List[dict]:
        return self._results[:limit]

    def test_connection(self, config: EmailAccountConfig) -> dict:
        """Test IMAP connection without starting monitor"""
        try:
            # Use certifi bundle for consistent SSL on all OS (Fix 11)
            try:
                import certifi
                ctx = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                ctx = ssl.create_default_context()
            imap = imaplib.IMAP4_SSL(config.imap_host, config.imap_port, ssl_context=ctx)
            imap.login(config.email_address, config.password)
            status, data = imap.select("INBOX")
            msg_count = int(data[0]) if data[0] else 0
            caps = imap.capabilities
            idle_support = "IDLE" in (caps or [])
            imap.logout()
            return {
                "success":      True,
                "inbox_count":  msg_count,
                "idle_support": idle_support,
                "message":      f"Connected! {msg_count} emails in inbox. "
                                f"{'Real-time IDLE' if idle_support else 'Poll mode'} will be used.",
            }
        except imaplib.IMAP4.error as e:
            return {
                "success": False,
                "message": f"Authentication failed: {e}",
                "hint":    PROVIDER_PRESETS.get(config.provider, {}).get("setup_note", ""),
            }
        except Exception as e:
            return {"success": False, "message": str(e)}
