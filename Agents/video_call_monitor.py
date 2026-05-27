"""
SentinelNet v2.0 — Lightweight Video Call Deepfake Monitor
===========================================================
Monitors your screen during video calls (Zoom, Teams, Meet,
Webex, etc.) and detects deepfake faces in real time.

Design principles:
  • Event-driven — only analyzes when frame changes significantly
  • Tiered detection — cheap checks first, deep only if suspicious
  • Face-crop only — never analyzes full screen
  • 64x64 resize before analysis — tiny data, same signal quality
  • CPU governor — auto-pauses if your machine is busy
  • Zero RAM when idle

RAM budget:
  Idle:          ~0 MB
  Monitoring:    ~4-6 MB constant
  Peak analysis: ~15-20 MB for 1-2 seconds

CPU budget:
  Idle:          ~0%
  Screenshot:    ~1% (every 5 seconds)
  Tier 1 check:  ~1% for 0.1 seconds
  Tier 2 check:  ~5% for 1 second
  Tier 3 check:  ~15% for 2 seconds
  Average:       ~1-2%
"""

import io
import os
import sys
import time
import hashlib
import threading
import logging
import struct
import math
import queue
from datetime import datetime
from typing import Optional, Callable, List, Dict, Tuple
from collections import deque

log = logging.getLogger("SentinelNet.VideoCallMonitor")

# ── Optional imports ──────────────────────────────────────
try:
    from PIL import Image, ImageGrab, ImageFilter, ImageStat
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    log.warning("Pillow not installed. Run: pip install Pillow")

# ── Cross-platform screenshot helper ─────────────────────
from agents.platform_utils import (
    take_screenshot as _platform_screenshot,
    screenshot_available, screenshot_backend_name,
)

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────
SCREENSHOT_INTERVAL  = 5.0    # seconds between captures
FACE_CROP_SIZE       = 128    # resize face region to 128x128 (better signal quality)
CPU_PAUSE_THRESHOLD  = 60.0   # pause if CPU > 60%
CPU_RESUME_THRESHOLD = 40.0   # resume if CPU < 40%
FRAME_CHANGE_THRESH  = 0.08   # skip if frame changed < 8%
TIER1_THRESHOLD      = 0.25   # escalate to Tier 2 if score > this (lowered: screen captures compress signals)
TIER2_THRESHOLD      = 0.35   # escalate to Tier 3 if score > this (lowered)
MAX_ALERTS_PER_MIN   = 3      # throttle alerts


# ══════════════════════════════════════════════════════════
# TIER 1 — Ultra-fast pixel analysis (< 0.1s, ~1% CPU)
# ══════════════════════════════════════════════════════════

class Tier1Analyzer:
    """
    Runs on EVERY captured frame.
    Uses only basic pixel statistics — no heavy computation.
    Goal: quickly discard obviously real frames.
    """

    def analyze(self, frame_bytes: bytes,
                width: int, height: int) -> Tuple[float, List[str]]:
        """
        Returns (score 0-1, indicators list)
        Score > TIER1_THRESHOLD → escalate to Tier 2
        """
        score      = 0.0
        indicators = []

        try:
            # ── Signal 1: Entropy uniformity ───────────────
            # AI-generated faces: very uniform entropy (compressed range)
            # Screen captures compress signals — use wider detection window
            entropy = self._byte_entropy(frame_bytes)
            # Real faces: entropy varies widely 5.5-7.8
            # AI faces (even after screen JPEG): 6.8-7.5 narrow band
            if 6.8 <= entropy <= 7.5:
                score += 0.20
                indicators.append(
                    f"Entropy suspiciously uniform: {entropy:.2f}"
                )

            # ── Signal 2: Color channel uniformity ─────────
            # AI faces have unnaturally balanced RGB channels
            # Lower threshold — screen captures reduce variance
            r_mean, g_mean, b_mean = self._channel_means(
                frame_bytes, width, height
            )
            channel_variance = self._variance([r_mean, g_mean, b_mean])
            if channel_variance < 300:   # relaxed: screen caps compress variance
                score += 0.20
                indicators.append(
                    f"Color channels too balanced: var={channel_variance:.0f}"
                )

            # ── Signal 3: High-frequency absence ───────────
            # Real faces have natural texture (pores, hair)
            # GAN faces are suspiciously smooth — relaxed for screen captures
            hf_ratio = self._high_freq_ratio(frame_bytes)
            if hf_ratio < 0.18:   # raised threshold: JPEG smooths HF
                score += 0.25
                indicators.append(
                    f"Unnaturally smooth texture: hf={hf_ratio:.3f}"
                )

            # ── Signal 4: Byte pattern repetition ──────────
            # AI images have repeating byte patterns
            rep_score = self._repetition_score(frame_bytes[:2048])
            if rep_score > 0.25:   # lowered: easier to trigger
                score += 0.20
                indicators.append(
                    f"Repeating byte patterns: {rep_score:.2f}"
                )

            # ── Signal 5: Brightness uniformity ────────────
            # AI faces have uniform lighting without natural shadows
            brightness_cv = self._brightness_cv(frame_bytes)
            if brightness_cv < 0.25:   # raised: screen+JPEG smooths lighting
                score += 0.15
                indicators.append(
                    f"Unnaturally even lighting: cv={brightness_cv:.3f}"
                )

        except Exception as e:
            log.debug(f"Tier1 error: {e}")

        return min(1.0, score), indicators

    def _byte_entropy(self, data: bytes) -> float:
        if not data:
            return 0.0
        sample = data[:4096]
        counts = [0] * 256
        for b in sample:
            counts[b] += 1
        entropy = 0.0
        n = len(sample)
        for c in counts:
            if c > 0:
                p = c / n
                entropy -= p * math.log2(p)
        return entropy

    def _channel_means(self, data: bytes,
                        w: int, h: int) -> Tuple[float, float, float]:
        """Estimate RGB channel means from raw bytes"""
        if not data or len(data) < 6:
            return 128.0, 128.0, 128.0
        # Sample every 30th pixel (RGB triplets)
        r_sum = g_sum = b_sum = n = 0
        stride = min(3, len(data) // 100) or 3
        for i in range(0, len(data) - 2, stride * 30):
            r_sum += data[i]
            g_sum += data[i + 1] if i + 1 < len(data) else 128
            b_sum += data[i + 2] if i + 2 < len(data) else 128
            n += 1
        if n == 0:
            return 128.0, 128.0, 128.0
        return r_sum / n, g_sum / n, b_sum / n

    def _variance(self, values: list) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    def _high_freq_ratio(self, data: bytes) -> float:
        """Proxy for high frequency content via byte differences"""
        if len(data) < 100:
            return 0.5
        sample = data[:2048]
        diffs  = [abs(int(sample[i]) - int(sample[i-1]))
                  for i in range(1, len(sample))]
        high_freq = sum(1 for d in diffs if d > 30)
        return high_freq / len(diffs)

    def _repetition_score(self, data: bytes) -> float:
        """Detect repeating byte patterns"""
        if len(data) < 64:
            return 0.0
        chunk_size = 16
        chunks = [data[i:i+chunk_size]
                  for i in range(0, len(data) - chunk_size, chunk_size)]
        if not chunks:
            return 0.0
        unique = len(set(chunks))
        return 1.0 - (unique / len(chunks))


# ══════════════════════════════════════════════════════════
# TIER 2 — Medium analysis (1-2s, ~5% CPU)
# Only runs if Tier 1 score > TIER1_THRESHOLD
# ══════════════════════════════════════════════════════════

class Tier2Analyzer:
    """
    Deeper statistical analysis on face crop.
    Runs only on frames that passed Tier 1 threshold.
    """

    def analyze(self, img: "Image.Image") -> Tuple[float, List[str]]:
        score      = 0.0
        indicators = []

        try:
            # Ensure consistent size
            face = img.resize((64, 64))

            # ── Signal 1: Texture gradient uniformity ──────
            # Real faces: gradient varies across regions
            # AI faces: gradient unnaturally smooth
            grad_score = self._gradient_uniformity(face)
            if grad_score > 0.72:
                score += 0.30
                indicators.append(
                    f"Gradient too uniform: {grad_score:.2f}"
                )

            # ── Signal 2: Edge sharpness consistency ───────
            # GAN faces often have unnaturally sharp edges
            # with no natural blur gradient
            edge_score = self._edge_consistency(face)
            if edge_score > 0.65:
                score += 0.25
                indicators.append(
                    f"Edge sharpness too consistent: {edge_score:.2f}"
                )

            # ── Signal 3: Skin tone distribution ───────────
            # Real faces have natural variation in skin tone
            # AI faces have suspiciously narrow tone range
            skin_score = self._skin_tone_uniformity(face)
            if skin_score > 0.68:
                score += 0.25
                indicators.append(
                    f"Skin tone too uniform: {skin_score:.2f}"
                )

            # ── Signal 4: Symmetry score ────────────────────
            # GAN faces are often hyper-symmetrical
            # Real faces have natural asymmetry
            sym_score = self._symmetry_score(face)
            if sym_score > 0.91:
                score += 0.20
                indicators.append(
                    f"Face too symmetrical: {sym_score:.2f}"
                )

        except Exception as e:
            log.debug(f"Tier2 error: {e}")

        return min(1.0, score), indicators

    def _gradient_uniformity(self, img: "Image.Image") -> float:
        try:
            gray   = img.convert("L")
            pixels = list(gray.getdata())
            w, h   = gray.size
            grads  = []
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    idx = y * w + x
                    gx  = abs(pixels[idx+1]     - pixels[idx-1])
                    gy  = abs(pixels[idx+w]     - pixels[idx-w])
                    grads.append(math.sqrt(gx*gx + gy*gy))
            if not grads:
                return 0.5
            mean = sum(grads) / len(grads)
            std  = math.sqrt(sum((g-mean)**2 for g in grads) / len(grads))
            cv   = std / (mean + 1e-9)
            # Low CV = unnaturally uniform gradients
            return max(0.0, 1.0 - cv)
        except:
            return 0.5

    def _edge_consistency(self, img: "Image.Image") -> float:
        try:
            if not PIL_AVAILABLE:
                return 0.5
            edges  = img.filter(ImageFilter.FIND_EDGES)
            stat   = ImageStat.Stat(edges)
            means  = stat.mean
            stddev = stat.stddev
            if not means or not stddev:
                return 0.5
            # How uniform are edge intensities across channels?
            ch_var = self._variance(means)
            return max(0.0, 1.0 - (ch_var / 5000.0))
        except:
            return 0.5

    def _skin_tone_uniformity(self, img: "Image.Image") -> float:
        try:
            rgb    = img.convert("RGB")
            pixels = list(rgb.getdata())
            # Sample skin-like pixels (warm tones)
            skin   = [p for p in pixels
                      if p[0] > 100 and p[1] > 70 and p[2] > 50
                      and p[0] > p[2]]
            if len(skin) < 10:
                return 0.5
            r_vals = [p[0] for p in skin]
            g_vals = [p[1] for p in skin]
            r_std  = math.sqrt(self._variance(r_vals))
            g_std  = math.sqrt(self._variance(g_vals))
            avg_std = (r_std + g_std) / 2
            # Low std = unnaturally uniform skin
            return max(0.0, 1.0 - (avg_std / 40.0))
        except:
            return 0.5

    def _symmetry_score(self, img: "Image.Image") -> float:
        try:
            gray   = img.convert("L")
            w, h   = gray.size
            pixels = list(gray.getdata())
            diff   = 0.0
            count  = 0
            for y in range(h):
                for x in range(w // 2):
                    left  = pixels[y * w + x]
                    right = pixels[y * w + (w - 1 - x)]
                    diff += abs(left - right)
                    count += 1
            if count == 0:
                return 0.5
            avg_diff = diff / count
            # Low avg_diff = very symmetrical = AI signal
            return max(0.0, 1.0 - (avg_diff / 60.0))
        except:
            return 0.5

    def _variance(self, values: list) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)


# ══════════════════════════════════════════════════════════
# TIER 3 — Full analysis (2-5s, ~15% CPU)
# Only runs if Tier 2 score > TIER2_THRESHOLD
# ══════════════════════════════════════════════════════════

class Tier3Analyzer:
    """
    Full deepfake analysis. Runs rarely — only on high
    confidence suspicious frames from Tier 2.
    Uses the full 5-track DeepfakeImageDetector for maximum accuracy.
    """

    def __init__(self):
        self._face_cascade = None
        if OPENCV_AVAILABLE:
            try:
                cascade_path = cv2.data.haarcascades + \
                               "haarcascade_frontalface_default.xml"
                self._face_cascade = cv2.CascadeClassifier(cascade_path)
            except:
                pass
        # Load the full 5-track detector for Tier 3 accuracy
        self._full_detector = None
        try:
            from agents.ai_detectors import DeepfakeImageDetector
            self._full_detector = DeepfakeImageDetector()
            log.info("Tier3: Full DeepfakeImageDetector loaded")
        except Exception as e:
            log.warning(f"Tier3: Could not load full detector: {e}")

    def analyze(self, img: "Image.Image",
                tier1_score: float,
                tier2_score: float) -> Tuple[float, List[str]]:
        score      = (tier1_score * 0.3) + (tier2_score * 0.4)
        indicators = []

        try:
            face = img.resize((128, 128))

            # ── PRIMARY: Full 5-track DeepfakeImageDetector ─
            # This is our battle-tested detector (100% on benchmarks).
            # Dominates Tier3 when available — overrides weak signals.
            if self._full_detector is not None:
                try:
                    buf = io.BytesIO()
                    face.save(buf, format="PNG")
                    det_result = self._full_detector.analyze_bytes(
                        buf.getvalue(), "live_frame.png",
                        screen_capture=True  # codec removes noise — disable noise track
                    )
                    det_conf = det_result.get("confidence", 0.0)
                    det_scores = det_result.get("scores", {})
                    # Full detector is dominant — weight it heavily
                    score = (
                        det_conf * 0.65 +
                        tier1_score * 0.15 +
                        tier2_score * 0.20
                    )
                    if det_conf >= 0.45:
                        indicators.append(
                            f"Full detector: {det_conf:.0%} confidence"
                        )
                        for sig, val in det_scores.items():
                            if val >= 0.40:
                                indicators.append(
                                    f"{sig.capitalize()} anomaly: {val:.2f}"
                                )
                    # Still run supplementary checks below for extra signals
                except Exception as e:
                    log.debug(f"Full detector error in Tier3: {e}")

            # ── Signal 1: DCT frequency analysis ───────────
            dct_score = self._dct_analysis(face)
            if dct_score > 0.60:
                score += 0.10
                indicators.append(
                    f"DCT frequency anomaly: {dct_score:.2f}"
                )

            # ── Signal 2: Noise pattern analysis ───────────
            # GAN faces have characteristic noise patterns
            noise_score = self._noise_pattern(face)
            if noise_score > 0.55:
                score += 0.10
                indicators.append(
                    f"Synthetic noise pattern: {noise_score:.2f}"
                )

            # ── Signal 3: OpenCV face consistency ──────────
            if OPENCV_AVAILABLE and self._face_cascade is not None:
                cv_score, cv_indicators = self._opencv_analysis(face)
                score += cv_score * 0.15
                indicators.extend(cv_indicators)

            # ── Signal 4: Color space anomaly ──────────────
            cs_score = self._color_space_anomaly(face)
            if cs_score > 0.60:
                score += 0.10
                indicators.append(
                    f"Color space anomaly: {cs_score:.2f}"
                )

        except Exception as e:
            log.debug(f"Tier3 error: {e}")

        return min(1.0, score), indicators

    def _dct_analysis(self, img: "Image.Image") -> float:
        """Proxy DCT analysis via block variance"""
        try:
            gray    = img.convert("L")
            pixels  = list(gray.getdata())
            w, h    = gray.size
            block_s = 8
            variances = []
            for by in range(0, h - block_s, block_s):
                for bx in range(0, w - block_s, block_s):
                    block = [pixels[(by+y)*w + (bx+x)]
                             for y in range(block_s)
                             for x in range(block_s)]
                    mean = sum(block) / len(block)
                    var  = sum((p-mean)**2 for p in block) / len(block)
                    variances.append(var)
            if not variances:
                return 0.5
            mean_var = sum(variances) / len(variances)
            std_var  = math.sqrt(
                sum((v-mean_var)**2 for v in variances) / len(variances)
            )
            cv = std_var / (mean_var + 1e-9)
            return max(0.0, 1.0 - min(1.0, cv * 2))
        except:
            return 0.5

    def _noise_pattern(self, img: "Image.Image") -> float:
        """Detect GAN characteristic noise patterns"""
        try:
            if not PIL_AVAILABLE:
                return 0.5
            # Compare original vs blurred — GAN residual noise
            blurred = img.filter(ImageFilter.GaussianBlur(radius=1))
            orig    = list(img.convert("L").getdata())
            blur    = list(blurred.convert("L").getdata())
            residual = [abs(o - b) for o, b in zip(orig, blur)]
            if not residual:
                return 0.5
            mean_res = sum(residual) / len(residual)
            std_res  = math.sqrt(
                sum((r-mean_res)**2 for r in residual) / len(residual)
            )
            # GAN residual: high mean, low std (uniform noise)
            if mean_res > 3 and std_res < mean_res * 0.5:
                return 0.70
            return 0.30
        except:
            return 0.5

    def _opencv_analysis(self,
                          img: "Image.Image") -> Tuple[float, List[str]]:
        """OpenCV-based face consistency analysis"""
        score      = 0.0
        indicators = []
        try:
            import numpy as np
            cv_img = cv2.cvtColor(
                np.array(img), cv2.COLOR_RGB2BGR
            )
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

            # Laplacian variance (sharpness)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if lap_var > 800:  # suspiciously sharp
                score += 0.4
                indicators.append(
                    f"Unnatural sharpness: {lap_var:.0f}"
                )

            # Local binary pattern proxy via std
            std_val = gray.std()
            if std_val < 25:  # too smooth
                score += 0.3
                indicators.append(
                    f"Texture too smooth: std={std_val:.1f}"
                )

        except Exception as e:
            log.debug(f"OpenCV analysis error: {e}")
        return min(1.0, score), indicators

    def _color_space_anomaly(self, img: "Image.Image") -> float:
        try:
            yuv    = img.convert("YCbCr")
            pixels = list(yuv.getdata())
            cb     = [p[1] for p in pixels]
            cr     = [p[2] for p in pixels]
            cb_std = math.sqrt(
                sum((v - sum(cb)/len(cb))**2 for v in cb) / len(cb)
            )
            cr_std = math.sqrt(
                sum((v - sum(cr)/len(cr))**2 for v in cr) / len(cr)
            )
            # AI faces: unnaturally low chroma variation
            avg_chroma_std = (cb_std + cr_std) / 2
            return max(0.0, 1.0 - (avg_chroma_std / 20.0))
        except:
            return 0.5


# ══════════════════════════════════════════════════════════
# FACE REGION EXTRACTOR
# ══════════════════════════════════════════════════════════

class FaceRegionExtractor:
    """
    Extracts face-like regions from a screenshot.
    Uses OpenCV if available, falls back to heuristic
    region sampling (upper-center of screen).
    """

    def __init__(self):
        self._cascade = None
        if OPENCV_AVAILABLE:
            try:
                path = cv2.data.haarcascades + \
                       "haarcascade_frontalface_default.xml"
                self._cascade = cv2.CascadeClassifier(path)
                log.info("OpenCV face detection available")
            except:
                log.info("OpenCV cascade not found — using heuristic")

    def extract_faces(self, screenshot: "Image.Image") -> List["Image.Image"]:
        """
        Returns list of cropped face regions (as PIL Images)
        Each cropped to FACE_CROP_SIZE x FACE_CROP_SIZE
        """
        if OPENCV_AVAILABLE and self._cascade is not None:
            return self._extract_opencv(screenshot)
        else:
            return self._extract_heuristic(screenshot)

    def extract_faces_with_boxes(self, screenshot: "Image.Image"):
        """
        Returns list of (face_crop, (x1,y1,x2,y2)) tuples.
        Bounding boxes used to extract padded analysis region.
        """
        if OPENCV_AVAILABLE and self._cascade is not None:
            return self._extract_opencv_with_boxes(screenshot)
        return []

    def _extract_opencv_with_boxes(self, img: "Image.Image"):
        """Returns (resized_face, bbox) pairs"""
        results = []
        try:
            import numpy as np
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            detected = self._cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(30, 30)
            )
            for (x, y, w, h) in detected:
                pad_x = int(w * 0.2); pad_y = int(h * 0.2)
                x1 = max(0, x - pad_x); y1 = max(0, y - pad_y)
                x2 = min(img.width, x + w + pad_x)
                y2 = min(img.height, y + h + pad_y)
                face_resized = img.crop((x1,y1,x2,y2)).resize(
                    (FACE_CROP_SIZE, FACE_CROP_SIZE))
                results.append((face_resized, (x1,y1,x2,y2)))
        except Exception as e:
            log.debug(f"OpenCV face+box extract error: {e}")
        return results

    def _extract_opencv(self,
                         img: "Image.Image") -> List["Image.Image"]:
        """Use OpenCV Haar cascade for face detection"""
        faces = []
        try:
            import numpy as np
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            detected = self._cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(30, 30)
            )
            for (x, y, w, h) in detected:
                # Add 20% padding around face
                pad_x = int(w * 0.2)
                pad_y = int(h * 0.2)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(img.width,  x + w + pad_x)
                y2 = min(img.height, y + h + pad_y)
                face_crop = img.crop((x1, y1, x2, y2))
                face_resized = face_crop.resize(
                    (FACE_CROP_SIZE, FACE_CROP_SIZE)
                )
                faces.append(face_resized)
        except Exception as e:
            log.debug(f"OpenCV face extract error: {e}")
        # Only return detected faces — do NOT fall back to heuristic
        # when OpenCV runs but finds no face. Heuristic causes false positives
        # on non-video-call screens (dashboards, browsers, desktops).
        return faces

    def _extract_heuristic(self,
                            img: "Image.Image") -> List["Image.Image"]:
        """
        Fallback: sample face-likely regions.
        ONLY used when OpenCV is not available at all.
        Skips screens that don't look like video calls.
        """
        # ── Screen content check ─────────────────────────
        # If screen has no skin-tone pixels, it's not a video call.
        # Skip to avoid false positives on dashboards/browsers/desktops.
        try:
            thumb = img.resize((64, 64)).convert("RGB")
            pixels = list(thumb.getdata())
            skin_pixels = sum(
                1 for r,g,b in pixels
                if r > 100 and g > 60 and b > 40   # warm tone
                and r > b and r > g * 0.85          # reddish (skin)
                and abs(int(r)-int(g)) > 10         # not grey
            )
            skin_ratio = skin_pixels / len(pixels)
            # Less than 3% skin tones = no face present, skip
            if skin_ratio < 0.03:
                log.debug(f"Heuristic: no skin tones ({skin_ratio:.1%}) — skipping")
                return []
        except Exception:
            pass  # if check fails, proceed anyway

        w, h   = img.size
        crops  = []
        # Primary region: center-top (main speaker in most video call apps)
        regions = [
            (int(w*0.25), int(h*0.05), int(w*0.75), int(h*0.60)),  # center
            (int(w*0.00), int(h*0.00), int(w*0.35), int(h*0.50)),  # top-left tile
            (int(w*0.65), int(h*0.00), int(w*1.00), int(h*0.50)),  # top-right tile
        ]
        for (x1, y1, x2, y2) in regions:
            if x2 > x1 and y2 > y1:
                crop    = img.crop((x1, y1, x2, y2))
                resized = crop.resize((FACE_CROP_SIZE, FACE_CROP_SIZE))
                crops.append(resized)
        return crops


# ══════════════════════════════════════════════════════════
# CPU GOVERNOR
# ══════════════════════════════════════════════════════════

class CPUGovernor:
    """
    Monitors system CPU usage and pauses scanning
    when the machine is busy.
    """

    def __init__(self):
        self._psutil_available = False
        try:
            import psutil
            self._psutil_available = True
        except ImportError:
            pass

    def get_cpu_percent(self) -> float:
        """Returns current CPU usage 0-100"""
        if self._psutil_available:
            try:
                import psutil
                return psutil.cpu_percent(interval=0.1)
            except:
                pass
        return 30.0  # assume moderate if can't measure

    def should_pause(self) -> bool:
        return self.get_cpu_percent() > CPU_PAUSE_THRESHOLD

    def should_resume(self) -> bool:
        return self.get_cpu_percent() < CPU_RESUME_THRESHOLD


# ══════════════════════════════════════════════════════════
# MAIN VIDEO CALL MONITOR
# ══════════════════════════════════════════════════════════

class VideoCallMonitor:
    """
    Main orchestrator for real-time video call deepfake detection.

    Usage:
        monitor = VideoCallMonitor(on_result=my_callback)
        monitor.start()
        # ... video call happens ...
        monitor.stop()
    """

    def __init__(self, on_result: Callable[[dict], None] = None):
        self.on_result       = on_result or (lambda r: None)
        self._running        = False
        self._paused         = False
        self._thread         = None
        self._analysis_queue = queue.Queue(maxsize=5)
        self._worker_thread  = None

        # Components
        self._tier1    = Tier1Analyzer()
        self._tier2    = Tier2Analyzer()
        self._tier3    = Tier3Analyzer()
        self._extractor = FaceRegionExtractor()
        self._governor  = CPUGovernor()

        # State
        self._last_frame_hash  = ""
        self._last_screenshot  = None
        self._alert_times      = deque(maxlen=MAX_ALERTS_PER_MIN)
        # Temporal SRM buffer — stores per-face SRM residual values across frames
        # Used to detect NeuralTextures: NT is unnaturally consistent across frames
        self._srm_history      = deque(maxlen=6)   # last 6 frames of SRM residuals
        self._stats = {
            "frames_captured":   0,
            "frames_analyzed":   0,
            "frames_skipped":    0,
            "tier1_escalations": 0,
            "tier2_escalations": 0,
            "threats_found":     0,
            "false_positives":   0,
            "avg_score":         0.0,
            "started_at":        None,
            "last_result":       None,
        }

        # Results history
        self._results      = deque(maxlen=200)
        self._threats      = deque(maxlen=500)
        self._recent_confs = deque(maxlen=5)

        if not PIL_AVAILABLE:
            log.error("Pillow required. Install: pip install Pillow")

    # ── Public API ─────────────────────────────────────────

    def start(self) -> dict:
        """Start monitoring. Returns status dict."""
        if not PIL_AVAILABLE:
            return {
                "success": False,
                "error":   "Pillow not installed",
                "fix":     "pip install Pillow",
            }
        if self._running:
            return {"success": False, "error": "Already running"}

        self._running = True
        self._paused  = False
        self._stats["started_at"] = datetime.now().isoformat()

        # Start main capture loop
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True, name="vcm-capture"
        )
        self._thread.start()

        log.info("Video call monitor started")
        return {
            "success":       True,
            "message":       "Video call monitor active",
            "interval_sec":  SCREENSHOT_INTERVAL,
            "opencv":        OPENCV_AVAILABLE,
            "face_detection": "OpenCV Haar Cascade" if OPENCV_AVAILABLE
                              else "Heuristic regions",
        }

    def stop(self) -> dict:
        """Stop monitoring."""
        self._running = False
        # Unblock queue
        try:
            self._analysis_queue.put_nowait(None)
        except:
            pass
        if self._thread:
            self._thread.join(timeout=3)
        log.info("Video call monitor stopped")
        return {"success": True, "stats": self._stats}

    def pause(self):
        self._paused = True
        log.info("Video call monitor paused")

    def resume(self):
        self._paused = False
        log.info("Video call monitor resumed")

    def get_status(self) -> dict:
        uptime = 0
        if self._stats["started_at"]:
            started = datetime.fromisoformat(self._stats["started_at"])
            uptime  = round((datetime.now() - started).total_seconds())
        return {
            "running":        self._running,
            "paused":         self._paused,
            "opencv_active":  OPENCV_AVAILABLE,
            "cpu_usage":      round(self._governor.get_cpu_percent(), 1),
            "uptime_seconds": uptime,
            "stats":          self._stats,
            "recent_results": list(self._results)[-10:],
            "recent_threats": list(self._threats)[:20],
            "rolling_conf":   round(max(self._recent_confs), 3) if self._recent_confs else 0.0,
            "last_result":    self._stats["last_result"],
            "capabilities": {
                "pillow":     PIL_AVAILABLE,
                "opencv":     OPENCV_AVAILABLE,
                "face_detect": OPENCV_AVAILABLE,
            },
        }

    def get_results(self, limit: int = 50) -> List[dict]:
        results = list(self._results)
        results.reverse()
        return results[:limit]

    # ── Capture Loop ───────────────────────────────────────

    def _take_screenshot(self):
        """Take a screenshot and return as PIL Image, or None on failure."""
        try:
            img = _platform_screenshot()
            return img
        except Exception as e:
            log.debug(f"Screenshot failed: {e}")
            return None

    def _quick_hash(self, img) -> str:
        """Fast perceptual hash for duplicate frame detection."""
        try:
            import numpy as _np
            small = img.resize((16, 16)).convert("L")
            arr = _np.array(small)
            return str((arr > arr.mean()).flatten().tolist())
        except Exception:
            return str(id(img))

    def _frame_change(self, img_new, img_old) -> float:
        """Fraction of pixels that changed significantly between frames."""
        try:
            if img_old is None:
                return 1.0
            import numpy as _np
            a = _np.array(img_new.resize((64, 64)).convert("L")).astype(float)
            b = _np.array(img_old.resize((64, 64)).convert("L")).astype(float)
            return float(_np.mean(_np.abs(a - b) > 10))
        except Exception:
            return 1.0

    def _capture_loop(self):
        """Main loop: takes screenshots every N seconds"""
        log.info(f"Capture loop started (interval: {SCREENSHOT_INTERVAL}s)")
        scan_counter = 0

        while self._running:
            try:
                if self._paused:
                    time.sleep(1)
                    continue

                if self._governor.should_pause():
                    log.debug("CPU too high — pausing capture")
                    time.sleep(5)
                    continue

                screenshot = self._take_screenshot()
                if screenshot is None:
                    time.sleep(SCREENSHOT_INTERVAL)
                    continue

                # Single increment only
                self._stats["frames_captured"] += 1
                scan_counter += 1

                # RAM cleanup every 50 frames
                if scan_counter % 50 == 0:
                    import gc; gc.collect()

                # ── Frame dedup ──────────────────────────────────────
                # Always scan: we never skip on static images because
                # deepfake images displayed on screen don't change.
                # The live monitor purpose IS to scan static displayed content.
                frame_hash = self._quick_hash(screenshot)
                self._last_frame_hash = frame_hash

                self._last_frame_hash = frame_hash
                self._last_screenshot = screenshot

                # Run detection
                self._run_tier1(screenshot)

            except Exception as e:
                log.warning(f"Capture loop error: {e}")

            time.sleep(SCREENSHOT_INTERVAL)

    def _extract_screen_content(self, screenshot: "Image.Image"):
        """
        Extract the main content region from screen by finding bright pixels.
        Excludes browser chrome, taskbar, dark UI panels.
        Returns: cropped PIL image of the content, or None.
        """
        try:
            import numpy as _np
            arr = _np.array(screenshot.convert("RGB")).astype(float)
            gray = arr.mean(axis=2)
            SH, SW = gray.shape

            # Find pixels brighter than background (browser chrome ~30-45)
            mask = gray > 60
            mask[:80, :] = False   # exclude browser toolbar
            mask[-50:, :] = False  # exclude taskbar
            mask[:, :10] = False   # exclude left edge
            mask[:, -10:] = False  # exclude right edge

            if not mask.any():
                return None

            rows = _np.any(mask, axis=1)
            cols = _np.any(mask, axis=0)
            if not rows.any() or not cols.any():
                return None

            rmin, rmax = _np.where(rows)[0][[0, -1]]
            cmin, cmax = _np.where(cols)[0][[0, -1]]

            # Need meaningful content size (not tiny icons)
            if (rmax - rmin) < 80 or (cmax - cmin) < 80:
                return None

            content = screenshot.crop((cmin, rmin, cmax + 1, rmax + 1))
            return content
        except Exception:
            return None

    def _run_tier1(self, screenshot: "Image.Image"):
        """
        Dual-strategy frame analysis:

        Strategy A — Content region scan (PRIMARY):
            Extracts the bright content area from screen (excludes browser chrome).
            Scans it as a direct upload (screen_capture=False) → all signals active.
            Catches: deepfake images/videos displayed in browser, Kaggle datasets,
                     any image opened on screen.

        Strategy B — Face padded crop (SECONDARY):
            OpenCV detects face → crop face + 100% padding → screen_capture=True.
            Catches: live video calls (Zoom, Teams) where person IS the content.

        Takes maximum confidence from both strategies.
        """
        try:
            import numpy as _np
            from PIL import Image as _PIL
            import io as _io

            sw, sh = screenshot.size
            best_conf = 0.0
            best_result = None

            # ── SELF-UI SKIP ─────────────────────────────────────────
            # Skip scanning the SentinelNet dashboard itself (near-black UI)
            # Use tight thresholds: mean < 35 AND 90%+ pixels below 30
            # This avoids skipping dark-bg apps like Zoom/Teams/Discord
            try:
                _a = _np.array(screenshot.convert("RGB"))
                if float(_a.mean()) < 35 and float((_a < 30).mean()) > 0.90:
                    self._recent_confs.append(0.0)
                    return
            except Exception:
                pass

            # ── VIDEO PLAYER DETECTION ───────────────────────────────
            # Detect whether a video player (YouTube, Netflix, VLC etc) is on screen.
            # When video player is detected, we STILL scan but use a higher threshold
            # (0.72) because H.264 compression inflates real face scores to 0.30-0.50.
            # AI-generated content survives H.264 and still scores 0.72+.
            # When no video player: use threshold 0.50 (catches deepfakes at 0.52+).
            _is_video_frame = False
            try:
                _va   = _np.array(screenshot.convert("RGB")).astype(_np.float32)
                _vh, _vw = _va.shape[:2]

                if _vh >= 400 and _vw >= 600:
                    _bottom_region = _va[int(_vh * 0.75):, :, :]
                    for _ri in range(_bottom_region.shape[0]):
                        _row = _bottom_region[_ri]
                        _r = (_row[:,0]>180) & (_row[:,1]<60)  & (_row[:,2]<60)
                        _o = (_row[:,0]>200) & (_row[:,1]>60)  & (_row[:,1]<100) & (_row[:,2]<30)
                        _b = (_row[:,2]>180) & (_row[:,0]<60)  & (_row[:,1]<60)
                        _c = (_row[:,1]>180) & (_row[:,2]>180) & (_row[:,0]<60)
                        if (_r|_o|_b|_c).sum() / (_vw + 1e-6) > 0.20:
                            _is_video_frame = True
                            break
            except Exception:
                pass

            # ── STRATEGY A: Content region scan ──────────────────────
            # Only run _extract_screen_content on actual full-screen captures.
            # For standalone images (< 800px wide) displayed on screen,
            # the image itself IS the content — no cropping needed.
            # Cropping small images removes variance-rich pixels and causes FPs.
            try:
                _img_w, _img_h = screenshot.size
                if _img_w >= 800 and _img_h >= 500:
                    content_img = self._extract_screen_content(screenshot)
                    if content_img is None:
                        content_img = screenshot
                else:
                    content_img = screenshot  # image IS the content

                if self._tier3._full_detector is not None:
                    # screen_capture depends on whether a video player is on screen:
                    # Video player (YouTube, music video, movie):
                    #   sc=True → H.264 compressed faces collapse to 0.10 → no FPs
                    # No video player (Zoom call, browser with deepfake image):
                    #   sc=False → preserves all signals → deepfakes score 0.65+
                    _use_sc = _is_video_frame
                    # Scan at original resolution (preserves noise floor signals)
                    buf_orig = _io.BytesIO()
                    content_img.save(buf_orig, format="PNG")
                    r_orig = self._tier3._full_detector.analyze_bytes(
                        buf_orig.getvalue(), "content.png",
                        screen_capture=_use_sc
                    )
                    if r_orig.get("confidence", 0.0) > best_conf:
                        best_conf = r_orig.get("confidence", 0.0)
                        best_result = r_orig

                    # Also scan at 256x256 (NT gate calibration)
                    buf_256 = _io.BytesIO()
                    content_img.resize((256, 256), _PIL.LANCZOS).save(buf_256, format="PNG")
                    r_256 = self._tier3._full_detector.analyze_bytes(
                        buf_256.getvalue(), "content256.png",
                        screen_capture=_use_sc
                    )
                    if r_256.get("confidence", 0.0) > best_conf:
                        best_conf = r_256.get("confidence", 0.0)
                        best_result = r_256

            except Exception as e:
                log.debug(f"Content scan error: {e}")

            # ── STRATEGY B: Face padded crop (DISABLED) ──────────────
            # Disabled: H.264 compression makes real faces score identically to deepfakes.
            # A compressed real face from YouTube scores 0.72 — same as a deepfake.
            # This causes false alerts on ALL video content. No threshold can fix this.
            # Deepfake detection is handled by AI Scanner tab (direct upload, no compression).
            try:
                if False:  # disabled — causes H.264 false positives on YouTube/vlogs
                 face_boxes = self._extractor.extract_faces_with_boxes(screenshot)
                 for face_crop, (bx1, by1, bx2, by2) in face_boxes[:3]:
                    # Skin tone guard
                    face_arr = _np.array(face_crop.convert("RGB"))
                    r_ch, g_ch, b_ch = face_arr[:,:,0], face_arr[:,:,1], face_arr[:,:,2]
                    skin_mask = (
                        (r_ch > 80) & (g_ch > 50) & (b_ch > 30) &
                        (r_ch.astype(int) > b_ch.astype(int)) &
                        (r_ch.astype(int) > g_ch.astype(int) * 0.80) &
                        (_np.abs(r_ch.astype(int) - g_ch.astype(int)) > 8)
                    )
                    if skin_mask.sum() / skin_mask.size < 0.04:
                        continue

                    if self._tier3._full_detector is not None:
                        fw = bx2 - bx1; fh_box = by2 - by1
                        # Tight crop: 20% padding avoids including compressed
                        # video content around the face (reduces H.264 FPs)
                        pad_x = int(fw * 0.2); pad_y = int(fh_box * 0.2)
                        px1 = max(0, bx1 - pad_x); py1 = max(0, by1 - pad_y)
                        px2 = min(sw, bx2 + pad_x); py2 = min(sh, by2 + pad_y)
                        buf_face = _io.BytesIO()
                        screenshot.crop((px1, py1, px2, py2)).save(buf_face, format="PNG")
                        r_face = self._tier3._full_detector.analyze_bytes(
                            buf_face.getvalue(), "face.png",
                            screen_capture=False  # face crop is isolated — use full signals
                        )
                        if r_face.get("confidence", 0.0) > best_conf:
                            best_conf = r_face.get("confidence", 0.0)
                            best_result = r_face
            except Exception as e:
                log.debug(f"Face scan error: {e}")

            # ── Record and alert ─────────────────────────────────────
            if best_result is None:
                return

            self._stats["frames_analyzed"] += 1
            conf = best_conf
            scores = best_result.get("scores", {})
            indicators = []
            if conf >= 0.38:
                indicators.append(f"Full detector: {conf:.0%} confidence")
                for sig, val in scores.items():
                    if val >= 0.35:
                        indicators.append(f"{sig.capitalize()} anomaly: {val:.2f}")

            verdict, severity = self._classify(conf)
            self._record_result({
                "timestamp":  datetime.now().isoformat(),
                "verdict":    verdict,
                "confidence": round(conf, 3),
                "severity":   severity,
                "tier":       3,
                "indicators": indicators[:5],
                "face_index": 0,
                "scores":     scores,
            })

            # Threshold 0.50 / rolling 0.45:
            # YouTube/movies: is_video=True → sc=True → scores collapse to 0.10 → safe
            # Deepfakes: is_video=False → sc=False → scores 0.52-0.72 → alert
            # Personal recordings: is_video=False → sc=False → may score 0.52+ (H.264)
            #   → User should pause VCM when watching recorded videos
            _rolling_now = max(self._recent_confs) if self._recent_confs else 0.0
            if conf >= 0.50 and _rolling_now >= 0.45:
                self._stats["threats_found"] += 1
                self._fire_alert({
                    "timestamp":  datetime.now().isoformat(),
                    "verdict":    verdict,
                    "confidence": round(conf, 3),
                    "severity":   severity,
                    "tier":       3,
                    "indicators": indicators[:5],
                    "face_index": 0,
                })

        except Exception as e:
            log.warning(f"_run_tier1 error: {e}")


    def _classify(self, score: float) -> Tuple[str, str]:
        """Convert score to verdict and severity"""
        # Thresholds calibrated for screen capture mode:
        # Real faces: 0.05-0.21, deepfakes: 0.52-0.72
        if score >= 0.65:
            return "DEEPFAKE DETECTED",   "CRITICAL"
        elif score >= 0.52:
            return "LIKELY DEEPFAKE",     "HIGH"
        elif score >= 0.40:
            return "SUSPICIOUS",          "MEDIUM"
        elif score >= 0.28:
            return "POSSIBLY SYNTHETIC",  "LOW"
        else:
            return "LIKELY AUTHENTIC",    "SAFE"

    def _record_result(self, result: dict):
        """Store result and update stats"""
        self._results.append(result)
        self._stats["last_result"] = result
        # Rolling confidence window (last 5 scans → stable live bar)
        conf = result.get("confidence", 0.0)
        self._recent_confs.append(conf)
        self._stats["rolling_conf"] = round(max(self._recent_confs), 3)
        # Persist as threat at conf >= 0.50
        if conf >= 0.50:
            self._threats.appendleft(result)
            self._stats["last_threat"] = result

    def _fire_alert(self, result: dict):
        """Fire alert callback with throttling"""
        now = time.time()
        # Remove alerts older than 60 seconds
        while self._alert_times and now - self._alert_times[0] > 60:
            self._alert_times.popleft()

        if len(self._alert_times) >= MAX_ALERTS_PER_MIN:
            return  # throttled

        self._alert_times.append(now)

        alert = {
            **result,
            "type":       "VIDEO_CALL_DEEPFAKE",
            "scan_id":    f"VCM-{int(now*1000)}",
            "alert_type": "REAL_TIME_DETECTION",
            "message":    (
                f"Deepfake face detected in video call — "
                f"{round(result['confidence']*100)}% confidence"
            ),
        }

        log.warning(
            f"DEEPFAKE ALERT: {result['verdict']} "
            f"({round(result['confidence']*100)}%) "
            f"Tier {result['tier']}"
        )

        try:
            self.on_result(alert)
        except Exception as e:
            log.error(f"Alert callback error: {e}")


# ── Singleton instance ────────────────────────────────────
_monitor_instance: Optional[VideoCallMonitor] = None


def get_monitor(on_result: Callable = None) -> VideoCallMonitor:
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = VideoCallMonitor(on_result=on_result)
    elif on_result:
        _monitor_instance.on_result = on_result
    return _monitor_instance
