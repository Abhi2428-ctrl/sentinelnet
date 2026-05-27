"""
SentinelNet v2.0 — Network Discovery Agent
ARP + ping sweep to find all devices on the local network.
Works without scapy — uses socket + subprocess (arp -a / nmap fallback).
"""

import sys
sys.dont_write_bytecode = True

import socket, subprocess, platform, re, threading, time, ipaddress
from datetime import datetime
from typing import List, Dict, Optional

# ── Known vendor OUI prefixes (first 3 bytes of MAC) ──────
OUI_MAP = {
    "b8:27:eb": "Raspberry Pi",        "dc:a6:32": "Raspberry Pi",
    "e4:5f:01": "Raspberry Pi",        "28:cd:c1": "Raspberry Pi",
    "00:50:56": "VMware",              "00:0c:29": "VMware",
    "00:1a:11": "Google",              "f4:f5:d8": "Google",
    "54:60:09": "Google/Chromecast",   "6c:ad:f8": "Google",
    "a4:77:33": "Google",              "48:d6:d5": "Google",
    "00:17:88": "Philips Hue",         "ec:b5:fa": "Philips Hue",
    "b0:be:83": "Xiaomi",              "78:11:dc": "Xiaomi",
    "64:09:80": "Xiaomi",              "28:6c:07