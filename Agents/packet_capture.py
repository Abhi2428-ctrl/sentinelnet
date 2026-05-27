"""
SentinelNet v2.0 — Real Packet Capture Engine
Captures live network traffic and converts packets into
observation vectors for the DQN agents.

Requires: scapy (auto-installed by start scripts)
Fallback: synthetic mode if scapy/permissions unavailable
"""

import time
import math
import threading
import collections
import socket
import struct
import os
import sys
import random
import ipaddress
from datetime import datetime
from typing import Optional, Callable, Dict, List, Tuple
import numpy as np

# ── Cross-platform helpers ────────────────────────────────
from agents.platform_utils import (
    is_admin, get_best_network_interface,
    save_model, load_model,
)

# ── Try importing scapy ────────────────────────────────────
try:
    from scapy.all import (
        sniff, IP, TCP, UDP, ICMP, DNS, DNSQR, Raw,
        get_if_list, conf as scapy_conf
    )
    from scapy.layers.http import HTTP, HTTPRequest, HTTPResponse
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

# ── High-risk country IP ranges (simplified) ──────────────
HIGH_RISK_PREFIXES = [
    "1.0.1.", "1.0.2.", "1.1.1.",       # CN
    "5.8.37.", "5.45.", "5.61.",         # RU
    "175.45.", "175.42.", "210.52.",     # KP
    "91.108.", "91.105.",                # IR
]

# ── Known malicious ports ──────────────────────────────────
SUSPICIOUS_PORTS = {
    4444, 4445, 4446,        # Metasploit default
    1337, 31337,             # Elite / backdoors
    6660, 6661, 6667, 6697, # IRC C2
    9001, 9030,              # Tor
    23, 2323,                # Telnet
    5900, 5901,              # VNC (unencrypted)
    3389,                    # RDP brute force target
    445, 139,                # SMB (EternalBlue)
    1433, 3306, 5432,        # DB exposure
    6379, 27017,             # Redis/MongoDB open
    8888, 9999, 12345,       # Generic backdoors
}

COMMON_PORTS = {80, 443, 53, 22, 25, 587, 993, 995, 8080, 8443}

# ── Flow tracker ──────────────────────────────────────────
class FlowRecord:
    """Tracks per-flow statistics for feature extraction"""
    def __init__(self):
        self.packet_count = 0
        self.byte_count = 0
        self.start_time = time.time()
        self.last_time = time.time()
        self.syn_count = 0
        self.ack_count = 0
        self.rst_count = 0
        self.fin_count = 0
        self.dns_queries = 0
        self.failed_conns = 0
        self.payload_sizes = []
        self.inter_arrival = []
        self._last_pkt_time = time.time()

    def update(self, pkt_size: int, flags: dict, is_dns: bool = False):
        now = time.time()
        self.inter_arrival.append(now - self._last_pkt_time)
        self._last_pkt_time = now
        self.packet_count += 1
        self.byte_count += pkt_size
        self.last_time = now
        if flags.get("S"): self.syn_count += 1
        if flags.get("A"): self.ack_count += 1
        if flags.get("R"): self.rst_count += 1
        if flags.get("F"): self.fin_count += 1
        if is_dns: self.dns_queries += 1
        self.payload_sizes.append(pkt_size)
        if len(self.payload_sizes) > 100:
            self.payload_sizes.pop(0)
        if len(self.inter_arrival) > 100:
            self.inter_arrival.pop(0)

    @property
    def duration(self): return time.time() - self.start_time

    @property
    def packet_rate(self): return self.packet_count / max(self.duration, 0.001)

    @property
    def byte_rate(self): return self.byte_count / max(self.duration, 0.001)

    @property
    def payload_entropy(self):
        if not self.payload_sizes: return 0.0
        total = sum(self.payload_sizes)
        if total == 0: return 0.0
        probs = [s / total for s in self.payload_sizes]
        return -sum(p * math.log2(p + 1e-9) for p in probs) / 8.0

    @property
    def syn_ratio(self):
        return self.syn_count / max(self.packet_count, 1)


# ── Feature Extractor ─────────────────────────────────────
class FeatureExtractor:
    """
    Converts raw packet data into 16-dim observation vectors
    matching the RL agent input format (Eq. 14).
    """
    def __init__(self):
        self.flows: Dict[str, FlowRecord] = {}
        self.unique_src_ips: collections.deque = collections.deque(maxlen=1000)
        self.failed_auth_ips: collections.Counter = collections.Counter()
        self.dns_query_counts: collections.Counter = collections.Counter()
        self._lock = threading.Lock()
        self._cleanup_interval = 60
        self._last_cleanup = time.time()

    def _flow_key(self, src_ip, dst_ip, dst_port, proto):
        return f"{src_ip}:{dst_ip}:{dst_port}:{proto}"

    def _cleanup_old_flows(self):
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        with self._lock:
            stale = [k for k, v in self.flows.items() if now - v.last_time > 120]
            for k in stale:
                del self.flows[k]
        self._last_cleanup = now

    def geo_risk_score(self, ip: str) -> float:
        """Simple prefix-based geo risk (0=safe, 1=high risk)"""
        for prefix in HIGH_RISK_PREFIXES:
            if ip.startswith(prefix):
                return 0.85 + random.uniform(0, 0.15)
        try:
            addr = ipaddress.ip_address(ip)
            if addr.is_private or addr.is_loopback:
                return 0.05
        except:
            pass
        return random.uniform(0.1, 0.4)

    def extract(self, pkt_info: dict) -> Tuple[np.ndarray, dict]:
        """
        Convert a parsed packet dict → 16-dim observation vector.

        Feature map (matches FEATURE_NAMES in rl_agents.py):
        [0]  packet_rate        [8]  syn_ratio
        [1]  byte_rate          [9]  failed_auth
        [2]  protocol_enc       [10] geo_risk
        [3]  port_anomaly       [11] time_of_day
        [4]  payload_entropy    [12] prev_action (0=none)
        [5]  connection_duration[13] agent_load
        [6]  unique_ips         [14] peer_alert_count
        [7]  dns_ratio          [15] threat_history
        """
        self._cleanup_old_flows()

        src = pkt_info.get("src_ip", "0.0.0.0")
        dst = pkt_info.get("dst_ip", "0.0.0.0")
        dport = pkt_info.get("dst_port", 0)
        proto = pkt_info.get("proto", "TCP")
        size = pkt_info.get("size", 64)
        flags = pkt_info.get("flags", {})
        is_dns = pkt_info.get("is_dns", False)

        fk = self._flow_key(src, dst, dport, proto)
        with self._lock:
            if fk not in self.flows:
                self.flows[fk] = FlowRecord()
            flow = self.flows[fk]
            flow.update(size, flags, is_dns)
            self.unique_src_ips.append(src)

        if flags.get("R") or flags.get("F"):
            self.failed_auth_ips[src] += 1
        if is_dns:
            domain = pkt_info.get("dns_query", "")
            self.dns_query_counts[domain] += 1

        # Normalize features → [0, 1]
        obs = np.zeros(16, dtype=np.float32)

        # [0] packet_rate: normalize against 10k pps as saturation
        obs[0] = min(1.0, flow.packet_rate / 10000.0)

        # [1] byte_rate: normalize against 100 MB/s
        obs[1] = min(1.0, flow.byte_rate / (100 * 1024 * 1024))

        # [2] protocol encoding: TCP=0.3, UDP=0.6, ICMP=0.9, other=0.1
        proto_map = {"TCP": 0.3, "UDP": 0.6, "ICMP": 0.9}
        obs[2] = proto_map.get(proto, 0.1)

        # [3] port anomaly: 1.0 if suspicious port, 0.5 if unknown, 0.1 if common
        if dport in SUSPICIOUS_PORTS:
            obs[3] = 0.9 + random.uniform(0, 0.1)
        elif dport not in COMMON_PORTS and dport > 1024:
            obs[3] = 0.4 + random.uniform(0, 0.2)
        else:
            obs[3] = 0.05 + random.uniform(0, 0.1)

        # [4] payload entropy
        obs[4] = min(1.0, flow.payload_entropy)

        # [5] connection duration: normalize against 300s
        obs[5] = min(1.0, flow.duration / 300.0)

        # [6] unique source IPs seen recently (normalize against 500)
        obs[6] = min(1.0, len(set(list(self.unique_src_ips)[-100:])) / 500.0)

        # [7] DNS ratio: fraction of flows that are DNS queries
        total_flows = max(len(self.flows), 1)
        dns_flows = sum(1 for f in self.flows.values() if f.dns_queries > 0)
        obs[7] = dns_flows / total_flows

        # [8] SYN ratio
        obs[8] = min(1.0, flow.syn_ratio)

        # [9] failed auth / RST ratio
        total_failed = sum(self.failed_auth_ips.values())
        obs[9] = min(1.0, total_failed / 1000.0)

        # [10] geo risk
        obs[10] = self.geo_risk_score(src)

        # [11] time of day (normalized 0-1)
        obs[11] = (time.time() % 86400) / 86400.0

        # [12] prev_action: placeholder (updated by agent)
        obs[12] = 0.0

        # [13] agent load: flow count normalized
        obs[13] = min(1.0, len(self.flows) / 5000.0)

        # [14] peer_alert_count: placeholder (updated by coordinator)
        obs[14] = 0.0

        # [15] threat history: recent suspicious flows ratio
        suspicious = sum(1 for f in self.flows.values() if f.syn_ratio > 0.7)
        obs[15] = min(1.0, suspicious / max(total_flows, 1))

        # Determine which agent modality this packet best fits
        agent_id = self._classify_modality(pkt_info, obs)

        meta = {
            "src_ip": src,
            "dst_ip": dst,
            "dst_port": dport,
            "src_port": pkt_info.get("src_port", 0),
            "proto": proto,
            "size": size,
            "flow_packets": flow.packet_count,
            "flow_bytes": flow.byte_count,
            "duration": round(flow.duration, 3),
            "syn_ratio": round(flow.syn_ratio, 3),
            "packet_rate": round(flow.packet_rate, 2),
            "agent_id": agent_id,
            "is_dns": is_dns,
            "dns_query": pkt_info.get("dns_query", ""),
        }
        return obs, meta

    def _classify_modality(self, pkt_info: dict, obs: np.ndarray) -> str:
        """Route packet to most appropriate RL agent"""
        proto = pkt_info.get("proto", "TCP")
        dport = pkt_info.get("dst_port", 0)
        is_dns = pkt_info.get("is_dns", False)

        # DNS exfiltration → network agent
        if is_dns and obs[7] > 0.6:
            return "network"
        # High entropy binary → binary agent
        if obs[4] > 0.8 and proto == "TCP":
            return "binary"
        # Email ports → phishing agent
        if dport in {25, 465, 587, 993, 995, 143, 110}:
            return "phishing"
        # HTTP/HTTPS with high byte rate → deepfake / network
        if dport in {80, 443, 8080} and obs[1] > 0.5:
            return "deepfake"
        # High SYN ratio → network (port scan / DDoS)
        if obs[8] > 0.6:
            return "network"
        # Suspicious ports → binary agent
        if dport in SUSPICIOUS_PORTS:
            return "binary"
        return "network"


# ── Threat Classifier ─────────────────────────────────────
class ThreatClassifier:
    """
    Rule-based threat type and severity classifier.
    Works on top of the extracted feature vector.
    Real threats from your network — not simulated.
    In LIVE mode: shows ALL traffic including normal browsing, DNS, HTTPS etc.
    """
    THRESHOLDS = {
        "syn_flood":        {"syn_ratio": 0.8,  "packet_rate": 500},
        "port_scan":        {"unique_ips": 0.3, "port_anomaly": 0.5},
        "dns_exfil":        {"dns_ratio": 0.7,  "payload_entropy": 0.6},
        "c2_beacon":        {"connection_duration": 0.5, "byte_rate": 0.1},
        "data_exfil":       {"byte_rate": 0.7,  "payload_entropy": 0.7},
        "brute_force":      {"failed_auth": 0.4, "packet_rate": 0.3},
        "suspicious_port":  {"port_anomaly": 0.8},
    }

    # Normal traffic type labels by port/protocol
    NORMAL_TRAFFIC_TYPES = {
        80:   ("HTTP Request",         "LOW",  0.05),
        443:  ("HTTPS Connection",     "LOW",  0.04),
        8080: ("HTTP Alt Port",        "LOW",  0.08),
        8443: ("HTTPS Alt Port",       "LOW",  0.06),
        53:   ("DNS Query",            "LOW",  0.06),
        67:   ("DHCP Request",         "LOW",  0.03),
        68:   ("DHCP Response",        "LOW",  0.03),
        123:  ("NTP Sync",             "LOW",  0.04),
        22:   ("SSH Connection",       "MEDIUM", 0.30),
        23:   ("Telnet Connection",    "HIGH",  0.65),
        21:   ("FTP Connection",       "MEDIUM", 0.35),
        25:   ("SMTP Mail",            "MEDIUM", 0.28),
        587:  ("SMTP Submission",      "LOW",  0.12),
        993:  ("IMAPS Mail",           "LOW",  0.08),
        995:  ("POP3S Mail",           "LOW",  0.08),
        3389: ("RDP Remote Desktop",   "HIGH",  0.70),
        5900: ("VNC Remote Access",    "HIGH",  0.65),
        1433: ("SQL Server",           "MEDIUM", 0.45),
        3306: ("MySQL Database",       "MEDIUM", 0.42),
        5432: ("PostgreSQL",           "MEDIUM", 0.40),
        6379: ("Redis",                "MEDIUM", 0.38),
        27017:("MongoDB",              "MEDIUM", 0.40),
        4444: ("Suspicious Port 4444", "HIGH",  0.72),
        1337: ("Hacker Port 1337",     "HIGH",  0.68),
    }

    def classify(self, obs: np.ndarray, meta: dict) -> Tuple[str, str, float]:
        """
        Returns (threat_type, severity, anomaly_score)
        In LIVE mode: classifies ALL traffic including normal browsing.
        """
        f = {
            "syn_ratio": obs[8], "packet_rate": obs[0],
            "unique_ips": obs[6], "port_anomaly": obs[3],
            "dns_ratio": obs[7], "payload_entropy": obs[4],
            "connection_duration": obs[5], "byte_rate": obs[1],
            "failed_auth": obs[9],
        }

        dst_port = meta.get("dst_port", 0)
        src_port = meta.get("src_port", 0)
        proto    = meta.get("proto", "TCP")
        is_dns   = meta.get("is_dns", False)

        scores = {}

        # SYN Flood — needs extremely high ratio AND sustained packet rate
        if f["syn_ratio"] > 0.90 and f["packet_rate"] > 0.30:
            scores["DDoS / SYN Flood"] = f["syn_ratio"] * 0.6 + f["packet_rate"] * 0.4

        # Port scan — needs high anomaly AND many unique IPs
        if f["port_anomaly"] > 0.70 and f["unique_ips"] > 0.40:
            scores["Port Scan"] = f["port_anomaly"] * 0.5 + f["unique_ips"] * 0.5

        # DNS exfiltration — requires BOTH high dns_ratio AND high entropy
        if f["dns_ratio"] > 0.8 and f["payload_entropy"] > 0.75:
            scores["DNS Exfiltration"] = f["dns_ratio"] * 0.5 + f["payload_entropy"] * 0.5

        # C2 beaconing — needs sustained connection + very regular timing
        if f["connection_duration"] > 0.6 and 0.05 < f["byte_rate"] < 0.2:
            scores["C2 Beacon"] = f["connection_duration"] * 0.6 + (1 - f["byte_rate"]) * 0.4

        # Data exfiltration — needs very high byte rate AND entropy
        if f["byte_rate"] > 0.75 and f["payload_entropy"] > 0.75:
            scores["Data Exfiltration"] = f["byte_rate"] * 0.6 + f["payload_entropy"] * 0.4

        # Brute force — needs multiple failed auths
        if f["failed_auth"] > 0.5 and f["packet_rate"] > 0.2:
            scores["Brute Force Attack"] = f["failed_auth"] * 0.8 + f["packet_rate"] * 0.2

        # Suspicious port — only truly suspicious ports
        if dst_port in SUSPICIOUS_PORTS:
            scores["Suspicious Port Access"] = 0.7 + obs[10] * 0.3

        # Lateral movement — needs both high unique IPs and port anomaly
        if f["unique_ips"] > 0.6 and f["port_anomaly"] > 0.5:
            scores["Lateral Movement"] = f["unique_ips"] * 0.5 + f["port_anomaly"] * 0.5

        if scores:
            # Known attack pattern detected
            threat_type   = max(scores, key=scores.get)
            anomaly_score = scores[threat_type]
            if anomaly_score >= 0.75:   severity = "CRITICAL"
            elif anomaly_score >= 0.55: severity = "HIGH"
            elif anomaly_score >= 0.35: severity = "MEDIUM"
            else:                       severity = "LOW"
            return threat_type, severity, round(float(anomaly_score), 4)

        # ── No attack pattern — classify as normal traffic by port/protocol ──
        # DNS query
        if is_dns or dst_port == 53 or src_port == 53:
            dns_q = meta.get("dns_query", "")
            label = f"DNS Query" + (f": {dns_q[:30]}" if dns_q else "")
            return label, "LOW", 0.06

        # Known port → human-readable label
        for port in (dst_port, src_port):
            if port in self.NORMAL_TRAFFIC_TYPES:
                label, sev, score = self.NORMAL_TRAFFIC_TYPES[port]
                return label, sev, score

        # Unknown port — flag as mildly anomalous
        if dst_port > 1024 and dst_port not in (8080, 8443, 9000, 9090):
            anomaly = (obs[0]*0.15 + obs[3]*0.25 + obs[4]*0.15 +
                       obs[8]*0.2  + obs[9]*0.15 + obs[10]*0.1)
            if anomaly > 0.30:
                return f"Unknown Port {dst_port}", "MEDIUM", round(float(anomaly), 4)

        # ICMP / other protocol
        if proto == "ICMP":
            return "ICMP Ping", "LOW", 0.05

        # Generic normal traffic
        anomaly = (obs[0]*0.15 + obs[3]*0.25 + obs[4]*0.15 +
                   obs[8]*0.2  + obs[9]*0.15 + obs[10]*0.1)
        return "Network Traffic", "LOW", round(float(max(anomaly, 0.03)), 4)


# ── Live Packet Capture ───────────────────────────────────
class PacketCapture:
    """
    Captures real network packets using scapy.
    Falls back to enhanced synthetic mode if:
      - scapy not installed
      - no root/admin privileges
      - no network interface found
    """
    def __init__(self, callback: Callable, interface: Optional[str] = None,
                 bpf_filter: str = "ip", max_packets: int = 0):
        self.callback = callback
        self.interface = interface
        self.bpf_filter = bpf_filter
        self.max_packets = max_packets
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self.extractor = FeatureExtractor()
        self.classifier = ThreatClassifier()
        self.mode = "unknown"
        self.stats = {
            "captured": 0, "processed": 0, "errors": 0,
            "threats_found": 0, "normal_traffic": 0,
        }

    def _check_privileges(self) -> bool:
        """Check if we have root/admin privileges for raw capture"""
        return is_admin()

    def _get_best_interface(self) -> Optional[str]:
        """Find best network interface for this OS"""
        if not SCAPY_AVAILABLE:
            return None
        return get_best_network_interface()

    def _parse_scapy_packet(self, pkt) -> Optional[dict]:
        """Parse a scapy packet into our normalized dict format"""
        try:
            if not pkt.haslayer(IP):
                return None

            ip = pkt[IP]
            info = {
                "src_ip": ip.src,
                "dst_ip": ip.dst,
                "proto": "OTHER",
                "dst_port": 0,
                "src_port": 0,
                "size": len(pkt),
                "flags": {},
                "is_dns": False,
                "dns_query": "",
                "timestamp": time.time(),
            }

            if pkt.haslayer(TCP):
                tcp = pkt[TCP]
                info["proto"] = "TCP"
                info["dst_port"] = tcp.dport
                info["src_port"] = tcp.sport
                flags = tcp.flags
                info["flags"] = {
                    "S": bool(flags & 0x02),  # SYN
                    "A": bool(flags & 0x10),  # ACK
                    "R": bool(flags & 0x04),  # RST
                    "F": bool(flags & 0x01),  # FIN
                    "P": bool(flags & 0x08),  # PSH
                }
            elif pkt.haslayer(UDP):
                udp = pkt[UDP]
                info["proto"] = "UDP"
                info["dst_port"] = udp.dport
                info["src_port"] = udp.sport

                # Check DNS
                if pkt.haslayer(DNS):
                    info["is_dns"] = True
                    if pkt.haslayer(DNSQR):
                        try:
                            info["dns_query"] = pkt[DNSQR].qname.decode("utf-8", errors="replace")
                        except:
                            pass
            elif pkt.haslayer(ICMP):
                info["proto"] = "ICMP"

            return info
        except Exception:
            return None

    def _process_packet_info(self, pkt_info: dict):
        """Extract features, classify, and call back to main engine"""
        try:
            obs, meta = self.extractor.extract(pkt_info)
            threat_type, severity, anomaly_score = self.classifier.classify(obs, meta)

            self.stats["processed"] += 1

            # In LIVE mode: show EVERYTHING — normal browsing, DNS, HTTPS, all traffic
            # Only drop in synthetic mode to avoid flooding with noise
            # Every real packet from your network is shown and classified
            if self.mode == "SYNTHETIC" and severity == "LOW" and anomaly_score < 0.25:
                self.stats["normal_traffic"] += 1
                return  # Synthetic mode only: skip truly boring packets

            self.stats["threats_found"] += 1
            self.callback({
                "obs": obs,
                "meta": meta,
                "threat_type": threat_type,
                "severity": severity,
                "anomaly_score": anomaly_score,
                "source": "LIVE_CAPTURE",
            })
        except Exception as e:
            self.stats["errors"] += 1

    # ── Scapy live capture ─────────────────────────────────
    def _scapy_capture_loop(self):
        def pkt_callback(pkt):
            if not self.running:
                return
            self.stats["captured"] += 1
            pkt_info = self._parse_scapy_packet(pkt)
            if pkt_info:
                self._process_packet_info(pkt_info)

        iface = self.interface or self._get_best_interface()
        sniff(
            iface=iface,
            filter=self.bpf_filter,
            prn=pkt_callback,
            store=False,
            stop_filter=lambda _: not self.running,
            count=self.max_packets or 0,
        )

    # ── Enhanced synthetic fallback ────────────────────────
    def _synthetic_loop(self):
        """
        Realistic synthetic traffic when live capture unavailable.
        Simulates normal + attack traffic patterns.
        Wrapped in watchdog — auto-restarts if Windows Defender or any crash kills it.
        """
        while self.running:
            try:
                self._run_synthetic_once()
            except Exception as e:
                print(f"[SentinelNet] ⚠ Synthetic loop crashed ({e}) — restarting in 2s")
                time.sleep(2)

    def _run_synthetic_once(self):
        """Core synthetic loop body."""
        ATTACK_SCENARIOS = [
            # (prob, threat_type, severity, feature_mods)  ← prob raised for visible feed
            (0.30, "DDoS / SYN Flood",       "CRITICAL", {"syn_ratio":0.92, "packet_rate":0.88}),
            (0.35, "Port Scan",               "HIGH",     {"port_anomaly":0.85, "unique_ips":0.7}),
            (0.25, "C2 Beacon",               "HIGH",     {"connection_duration":0.75, "byte_rate":0.15}),
            (0.20, "DNS Exfiltration",        "HIGH",     {"dns_ratio":0.82, "payload_entropy":0.75}),
            (0.20, "Data Exfiltration",       "CRITICAL", {"byte_rate":0.85, "payload_entropy":0.8}),
            (0.28, "Brute Force Attack",      "HIGH",     {"failed_auth":0.78, "packet_rate":0.5}),
            (0.22, "Lateral Movement",        "MEDIUM",   {"unique_ips":0.65, "port_anomaly":0.55}),
            (0.25, "Suspicious Port Access",  "MEDIUM",   {"port_anomaly":0.95}),
            (0.15, "Ransomware Dropper",      "CRITICAL", {"payload_entropy":0.95, "byte_rate":0.6}),
            (0.12, "Polymorphic Malware",     "CRITICAL", {"payload_entropy":0.98, "failed_auth":0.5}),
        ]

        base_ips = [f"192.168.1.{i}" for i in range(2, 50)]
        ext_ips = [f"{random.randint(1,254)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
                   for _ in range(20)]
        malicious_ips = [f"5.45.{random.randint(1,254)}.{random.randint(1,254)}" for _ in range(5)] + \
                        [f"175.45.{random.randint(1,254)}.{random.randint(1,254)}" for _ in range(5)]

        t = 0
        while self.running:
            t += 1
            # Time-based attack bursts (simulate real attack windows)
            attack_hour = (t // 300) % 24
            attack_multiplier = 3.0 if attack_hour in [2, 3, 14, 15] else 1.0

            # Normal traffic background
            for _ in range(random.randint(5, 20)):
                if not self.running: break
                pkt_info = {
                    "src_ip": random.choice(base_ips),
                    "dst_ip": random.choice(ext_ips),
                    "proto": random.choice(["TCP", "TCP", "TCP", "UDP"]),
                    "dst_port": random.choice([80, 443, 53, 443, 8080]),
                    "src_port": random.randint(1024, 65535),
                    "size": random.randint(64, 1500),
                    "flags": {"S": False, "A": True, "R": False, "F": False, "P": True},
                    "is_dns": random.random() < 0.05,
                    "dns_query": "",
                    "timestamp": time.time(),
                }
                self.stats["captured"] += 1
                # Skip ~80% of normal traffic (was 95% — too aggressive, caused empty feed)
                if random.random() < 0.80:
                    self.stats["normal_traffic"] += 1
                    continue
                self._process_packet_info(pkt_info)

            # Inject attack scenarios
            for prob, ttype, sev, feat_mods in ATTACK_SCENARIOS:
                if not self.running: break
                if random.random() < prob * attack_multiplier:
                    src = random.choice(malicious_ips + ext_ips)
                    dst = random.choice(base_ips)
                    dport = (random.choice(list(SUSPICIOUS_PORTS))
                             if feat_mods.get("port_anomaly", 0) > 0.7
                             else random.choice([80, 443, 22, 3389, 445]))

                    pkt_info = {
                        "src_ip": src, "dst_ip": dst,
                        "proto": "TCP" if random.random() > 0.3 else "UDP",
                        "dst_port": dport, "src_port": random.randint(1024, 65535),
                        "size": random.randint(64, 9000),
                        "flags": {"S": feat_mods.get("syn_ratio", 0.1) > 0.5,
                                  "A": random.random() > 0.5,
                                  "R": random.random() < 0.1,
                                  "F": False, "P": random.random() > 0.5},
                        "is_dns": "DNS" in ttype,
                        "dns_query": f"{''.join(random.choices('abcdefghijklmnop',k=20))}.evil.com" if "DNS" in ttype else "",
                        "timestamp": time.time(),
                    }
                    self.stats["captured"] += 1

                    # Build observation with attack characteristics
                    obs = np.zeros(16, dtype=np.float32)
                    feat_map = {
                        "packet_rate": 0, "byte_rate": 1, "port_anomaly": 3,
                        "payload_entropy": 4, "connection_duration": 5,
                        "unique_ips": 6, "dns_ratio": 7, "syn_ratio": 8,
                        "failed_auth": 9,
                    }
                    # Base noise
                    obs[:] = np.random.uniform(0.05, 0.25, 16)
                    obs[10] = self.extractor.geo_risk_score(src)
                    obs[11] = (time.time() % 86400) / 86400
                    for feat, val in feat_mods.items():
                        if feat in feat_map:
                            obs[feat_map[feat]] = val + random.uniform(-0.05, 0.05)

                    obs = np.clip(obs, 0, 1)
                    anomaly = float(np.mean([feat_mods.get(k, 0) for k in feat_mods]))

                    self.stats["processed"] += 1
                    self.stats["threats_found"] += 1

                    # Determine agent
                    meta = {
                        "src_ip": src, "dst_ip": dst, "dst_port": dport,
                        "proto": pkt_info["proto"], "size": pkt_info["size"],
                        "flow_packets": random.randint(1, 500),
                        "flow_bytes": random.randint(64, 1024*1024),
                        "duration": round(random.uniform(0.1, 120), 3),
                        "syn_ratio": feat_mods.get("syn_ratio", 0.1),
                        "packet_rate": feat_mods.get("packet_rate", 0.1) * 10000,
                        "agent_id": self.extractor._classify_modality(pkt_info, obs),
                        "is_dns": pkt_info["is_dns"],
                    }
                    self.callback({
                        "obs": obs, "meta": meta,
                        "threat_type": ttype, "severity": sev,
                        "anomaly_score": round(anomaly, 4),
                        "source": "SYNTHETIC_REALISTIC",
                    })

            # First tick: no sleep so events appear instantly on startup
            if t == 1:
                time.sleep(0.1)
            else:
                time.sleep(random.uniform(0.8, 1.5))

    def start(self):
        """Start capture — automatically picks best mode"""
        self.running = True

        can_capture = (
            SCAPY_AVAILABLE and
            self._check_privileges() and
            self._get_best_interface() is not None
        )

        if can_capture:
            self.mode = "LIVE"
            self._thread = threading.Thread(target=self._scapy_capture_loop, daemon=True)
            print(f"[SentinelNet] 🟢 LIVE capture on interface: {self.interface or self._get_best_interface()}")
        else:
            self.mode = "SYNTHETIC"
            self._thread = threading.Thread(target=self._synthetic_loop, daemon=True)
            reason = []
            if not SCAPY_AVAILABLE:       reason.append("scapy not installed")
            if not self._check_privileges(): reason.append("no root/admin privileges")
            print(f"[SentinelNet] 🟡 SYNTHETIC mode ({', '.join(reason)})")
            print(f"[SentinelNet]    → Run as sudo/admin for LIVE capture")

        self._thread.start()
        return self.mode

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)

    def get_stats(self) -> dict:
        return {
            **self.stats,
            "mode": self.mode,
            "active_flows": len(self.extractor.flows),
            "scapy_available": SCAPY_AVAILABLE,
            "has_privileges": self._check_privileges(),
            "interface": self.interface or self._get_best_interface(),
        }
