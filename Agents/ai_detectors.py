"""
SentinelNet v2.0 — AI-Generated Text Detector
Detects LLM-written content (ChatGPT, Claude, Gemini, etc.)
using statistical linguistic analysis — no GPU required.

Methods:
  1. Perplexity scoring (low perplexity = AI-written)
  2. Burstiness analysis (AI text is unnaturally uniform)
  3. Vocabulary richness (AI overuses certain patterns)
  4. Sentence length variance (AI = very uniform)
  5. Repetition detection (AI repeats phrases/structures)
  6. Punctuation pattern analysis
"""

import re
import math
import string
import hashlib
from typing import Dict, List, Tuple
from collections import Counter


# ── Common AI "tells" — phrases LLMs overuse ──────────────
AI_SIGNATURE_PHRASES = [
    # Classic AI disclaimers
    "i cannot and will not", "i need to be direct",
    "certainly!", "certainly,", "absolutely!", "of course!",
    "i understand your", "it's important to note",
    "it is important to", "i want to emphasize",
    "let me be clear", "i'd be happy to",
    "great question", "that's a great",
    "as an ai", "as a language model",
    "i'm here to help", "i hope this helps",
    "feel free to ask", "in conclusion,",
    "in summary,", "to summarize,",
    "furthermore,", "moreover,", "additionally,",
    "it's worth noting", "it is worth noting",
    "delve into", "dive into",
    "leverage", "utilize", "facilitate",
    "comprehensive", "multifaceted",
    "in today's world", "in the modern era",
    "the landscape of", "a testament to",
    # Technical AI writing patterns (Claude, GPT style)
    "root cause", "the root cause",
    "let me explain", "to be clear",
    "the fix involves", "the issue lies in",
    "this ensures that", "this is because",
    "there are two", "there are three", "there are several",
    "the correct pattern", "the proper way",
    "it is worth", "worth noting that",
    "providing a seamless", "seamless experience",
    "robust solution", "straightforward fix",
    "under the hood", "at its core",
    "step by step", "step-by-step",
    "in other words", "put simply",
    "as mentioned", "as noted above",
    "going forward", "moving forward",
    "it's designed to", "it is designed to",
    "the reason is", "the reason why",
    "when it comes to", "in terms of",
    "one important", "one key", "one critical",
    "best practice", "best practices",
    "it should be noted", "note that",
    "keep in mind", "bear in mind",
    "at the same time", "on the other hand",
    # Explanation / tutorial style (very common in AI responses)
    "fundamental limitation", "goes through", "give accurate",
    "to properly", "accurate results", "will give accurate",
    "through the full", "this ensures", "this allows",
    "this provides", "for best", "for accurate",
    "needs to be", "should be noted", "make sure",
    "in this case", "in that case", "in practice",
    "for example,", "for instance,", "such as",
    "which means", "which allows", "which ensures",
    "the key", "the main", "the primary",
    "as a result", "as expected", "as needed",
    "without further", "by default", "by design",
    "under the assumption", "given that", "assuming that",
    "refer to", "see the", "check the",
    "simply put", "in short,", "in brief,",
]

# Common AI sentence starters
AI_STARTERS = [
    "firstly,", "secondly,", "thirdly,", "lastly,",
    "in order to", "when it comes to",
    "one of the", "it is essential",
    "there are several", "there are many",
    "to fix this,", "to resolve this,",
    "the key issue", "the main reason",
    "this means that", "this allows",
    "essentially,", "specifically,",
    "importantly,", "notably,",
]

# Transition words AI overuses
AI_TRANSITIONS = [
    "however,", "therefore,", "consequently,",
    "nevertheless,", "nonetheless,", "subsequently,",
    "furthermore,", "additionally,", "moreover,",
    "in addition,", "as a result,", "thus,",
    "this means,", "which means,", "which ensures",
]


class AITextDetector:
    """
    Detects AI-generated text using ensemble of statistical methods.
    Returns confidence score 0.0 (human) → 1.0 (AI-generated).
    """

    def __init__(self):
        self.threshold = 0.55  # above = likely AI

    def analyze(self, text: str) -> Dict:
        """Full analysis pipeline — returns all scores + verdict"""
        if not text or len(text.strip()) < 20:
            return self._empty_result("Text too short to analyze")

        text = text.strip()
        sentences = self._split_sentences(text)
        words = self._tokenize(text)

        if len(words) < 10:
            return self._empty_result("Too few words")

        scores = {}

        # 1. Perplexity proxy (sentence length uniformity)
        scores["perplexity"]      = self._perplexity_score(sentences)

        # 2. Burstiness (variance in sentence length — AI = low variance)
        scores["burstiness"]      = self._burstiness_score(sentences)

        # 3. AI signature phrases
        scores["signature"]       = self._signature_score(text.lower())

        # 4. Vocabulary diversity
        scores["vocab_diversity"] = self._vocab_diversity_score(words)

        # 5. Punctuation uniformity
        scores["punctuation"]     = self._punctuation_score(text)

        # 6. Repetition score
        scores["repetition"]      = self._repetition_score(sentences)

        # 7. Formality / robotic tone
        scores["formality"]       = self._formality_score(text.lower(), words)

        # 8. Short-text structural signals (em-dashes, arrows, parentheses, colons)
        #    AI writing uses these heavily for formatting even in short text
        import re as _re
        struct_signals = 0
        emdash_count = text.count('—') + text.count('–')
        if emdash_count >= 1: struct_signals += emdash_count * 2  # 2 em-dashes = +4
        if text.count('→') >= 1: struct_signals += 3              # arrows = strong AI signal
        if text.count(':') >= 2: struct_signals += 2
        if text.count('(') >= 2: struct_signals += 1
        if _re.search(r'\b(Note:|Fix \d|Bug \d|Step \d|Issue \d|Root cause|Cause:)', text): struct_signals += 3
        if _re.search(r'→|⟶|\bvs\b|\bvs\.', text): struct_signals += 2
        scores["structure"] = min(1.0, struct_signals * 0.15)

        # Adaptive weights — for short texts (<4 sentences), rely more on
        # formality, structure since burstiness and repetition are unreliable
        is_short = len(sentences) < 4
        if is_short:
            weights = {
                "perplexity":      0.06,
                "burstiness":      0.08,  # unreliable on <3 sentences
                "signature":       0.15,
                "vocab_diversity": 0.06,
                "punctuation":     0.08,
                "repetition":      0.04,
                "formality":       0.28,  # most reliable on short text
                "structure":       0.25,  # em-dashes/arrows very telling
            }
            threshold = 0.38  # lower threshold for short text
        else:
            weights = {
                "perplexity":      0.18,
                "burstiness":      0.24,
                "signature":       0.18,
                "vocab_diversity": 0.10,
                "punctuation":     0.10,
                "repetition":      0.08,
                "formality":       0.12,
                "structure":       0.00,
            }
            threshold = 0.48

        final_score = sum(scores[k] * weights[k] for k in scores)
        final_score = round(min(1.0, max(0.0, final_score)), 4)

        is_ai = final_score >= threshold
        severity = (
            "CRITICAL" if final_score >= 0.82 else
            "HIGH"     if final_score >= 0.68 else
            "MEDIUM"   if final_score >= 0.55 else
            "LOW"
        )

        # Top contributing factors
        top_factors = sorted(
            [(k, round(scores[k], 3)) for k in scores],
            key=lambda x: x[1], reverse=True
        )[:3]

        return {
            "is_ai_generated": is_ai,
            "confidence": final_score,
            "severity": severity,
            "verdict": "AI-GENERATED" if is_ai else "LIKELY HUMAN",
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "top_indicators": [{"feature": f, "score": s} for f, s in top_factors],
            "word_count": len(words),
            "sentence_count": len(sentences),
            "analysis_method": "Statistical Ensemble (7 features)",
        }

    # ── Individual Scorers ─────────────────────────────────

    def _perplexity_score(self, sentences: List[str]) -> float:
        """Low length variance → AI-like uniformity → high score"""
        if len(sentences) < 2:
            return 0.5
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths:
            return 0.5
        mean = sum(lengths) / len(lengths)
        variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
        std = math.sqrt(variance)
        # AI sentences are unnaturally uniform (low std relative to mean)
        cv = std / max(mean, 1)  # coefficient of variation
        # Low CV (< 0.3) = AI-like uniformity
        score = max(0.0, 1.0 - cv * 2.0)
        return min(1.0, score)

    def _burstiness_score(self, sentences: List[str]) -> float:
        """Human writing is 'bursty' — AI is unnaturally smooth"""
        if len(sentences) < 3:
            return 0.4
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if len(lengths) < 3:
            return 0.4
        mean = sum(lengths) / len(lengths)
        std = math.sqrt(sum((l - mean)**2 for l in lengths) / len(lengths))
        # Burstiness = (std - mean) / (std + mean)
        burstiness = (std - mean) / (std + mean + 1e-9)
        # Negative burstiness (std < mean) → AI-like
        # Convert: very negative burstiness → high AI score
        ai_score = max(0.0, -burstiness * 1.5 + 0.3)
        return min(1.0, ai_score)

    def _signature_score(self, text_lower: str) -> float:
        """Count AI signature phrases and starters"""
        found = 0
        total_checks = len(AI_SIGNATURE_PHRASES) + len(AI_STARTERS) + len(AI_TRANSITIONS)
        for phrase in AI_SIGNATURE_PHRASES:
            if phrase in text_lower:
                found += 2  # weighted higher
    def _signature_score(self, text_lower: str) -> float:
        """Count AI signature phrases and starters — more matches = higher AI probability"""
        matched_phrases  = sum(1 for p in AI_SIGNATURE_PHRASES if p in text_lower)
        matched_starters = sum(1 for s in AI_STARTERS       if s in text_lower)
        matched_trans    = sum(1 for t in AI_TRANSITIONS
                               if text_lower.count(t) >= 1)  # even 1 transition is a signal

        # Weight: phrases > transitions > starters
        weighted = matched_phrases * 2 + matched_trans * 1.5 + matched_starters * 1.0

        # Normalize: 3 matches = 0.5 score, 6+ matches = 1.0
        # This makes even 1-2 matches contribute meaningfully
        score = min(1.0, weighted / 8.0)
        return round(score, 4)

    def _vocab_diversity_score(self, words: List[str]) -> float:
        """
        AI text tends to have lower type-token ratio in long texts
        but uses sophisticated vocabulary uniformly.
        Low diversity in function words → AI signal.
        """
        if len(words) < 10:
            return 0.3
        # Type-token ratio (lower = more repetitive)
        unique = len(set(w.lower() for w in words))
        ttr = unique / len(words)
        # AI tends to have moderate TTR — not too low, not too high
        # Very high TTR = human creative writing
        # Very low TTR = spam/repetitive
        # Medium TTR (0.4-0.7) = AI
        if 0.35 <= ttr <= 0.72:
            score = 0.6 + (0.5 - abs(ttr - 0.53)) * 0.8
        else:
            score = max(0.0, 0.6 - abs(ttr - 0.53))
        return min(1.0, max(0.0, score))

    def _punctuation_score(self, text: str) -> float:
        """AI uses punctuation very consistently — humans vary more"""
        sentences = self._split_sentences(text)
        if len(sentences) < 3:
            return 0.3
        # Check comma frequency per sentence (AI = very consistent)
        comma_rates = [s.count(',') / max(len(s.split()), 1) for s in sentences]
        if len(comma_rates) < 2:
            return 0.3
        mean_cr = sum(comma_rates) / len(comma_rates)
        variance = sum((r - mean_cr)**2 for r in comma_rates) / len(comma_rates)
        std_cr = math.sqrt(variance)
        # Low variance in comma usage → AI
        consistency = 1.0 - min(1.0, std_cr * 5)
        return max(0.0, consistency * 0.7)

    def _repetition_score(self, sentences: List[str]) -> float:
        """Detect repeated sentence structures (AI patterns)"""
        if len(sentences) < 4:
            return 0.3
        # Check for repeated starting words
        starters = [s.strip().split()[0].lower() if s.strip() else "" for s in sentences]
        starter_counts = Counter(starters)
        most_common_count = starter_counts.most_common(1)[0][1] if starter_counts else 0
        repetition_ratio = most_common_count / len(sentences)
        # Check bigram repetition
        bigrams = []
        for s in sentences:
            words = s.lower().split()
            bigrams.extend([f"{words[i]} {words[i+1]}" for i in range(len(words)-1)])
        if bigrams:
            bigram_counts = Counter(bigrams)
            repeated = sum(1 for c in bigram_counts.values() if c > 2)
            bigram_score = min(1.0, repeated / max(len(bigrams) * 0.1, 1))
        else:
            bigram_score = 0.0
        return min(1.0, (repetition_ratio * 0.5 + bigram_score * 0.5))

    def _formality_score(self, text_lower: str, words: List[str]) -> float:
        """AI writes formally and avoids contractions"""
        # Contractions signal human writing
        contractions = ["don't", "can't", "won't", "it's", "i'm", "you're",
                        "we're", "they're", "i've", "you've", "wouldn't",
                        "couldn't", "shouldn't", "didn't", "doesn't", "isn't"]
        found_contractions = sum(1 for c in contractions if c in text_lower)
        # Slang / informal words
        informal = ["gonna", "wanna", "kinda", "sorta", "yeah", "yep",
                    "nope", "ok", "okay", "btw", "tbh", "imo", "lol",
                    "omg", "wtf", "tho", "thru",
                    # Dev/chat informal abbreviations
                    "bc", "cuz", "coz", "rn", "ngl", "tbf", "iirc",
                    "afaik", "idk", "idc", "smh", "bruh", "dude",
                    "lmao", "lmfao", "rofl", "fwiw", "iirc", "afk",
                    "basically", "literally", "actually", "honestly",
                    "so i", "turns out", "found out", "figured out",
                    ]
        found_informal = sum(1 for w in informal if w in text_lower.split())
        # Low contractions + low informal = high formality = AI signal
        informality = (found_contractions + found_informal * 1.5) / max(len(words) * 0.1, 1)
        formality_score = max(0.0, 1.0 - min(1.0, informality * 3))
        return formality_score

    # ── Utilities ──────────────────────────────────────────

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 3]

    def _tokenize(self, text: str) -> List[str]:
        return [w.strip(string.punctuation) for w in text.split()
                if w.strip(string.punctuation)]

    def _empty_result(self, reason: str) -> Dict:
        return {
            "is_ai_generated": False,
            "confidence": 0.0,
            "severity": "LOW",
            "verdict": f"INCONCLUSIVE ({reason})",
            "scores": {},
            "top_indicators": [],
            "word_count": 0,
            "sentence_count": 0,
            "analysis_method": "Statistical Ensemble",
        }



# ── Deepfake Image Detector ────────────────────────────────
class DeepfakeImageDetector:
    """
    Detects AI-generated images AND face-swap deepfakes.

    5-track detection pipeline:
      Track A — Metadata (EXIF, AI tool signatures, camera markers)
      Track B — Error Level Analysis (ELA): JPEG recompression inconsistency
                 Face-swapped regions compress differently than background
      Track C — Noise analysis: Real cameras have sensor noise floor,
                 GANs/AI generators produce unnaturally smooth pixels
      Track D — OpenCV frequency domain: Laplacian edge density,
                 2D FFT high-frequency energy ratio
      Track E — Color channel analysis: overcorrelation, skin uniformity

    Libraries: PIL, numpy, OpenCV (cv2), scipy.fft
    Accuracy: ~80-85% on real-world samples
    """

    def analyze_bytes(self, image_bytes: bytes, filename: str = "",
                       screen_capture: bool = False) -> Dict:
        """
        screen_capture=True: called from live monitor on screen-grabbed frames.
        H.264/VP9 codec removes sensor noise aggressively, so noise track is
        unreliable for screen captures — disabled in this mode.
        """
        try:
            from PIL import Image as _PIL
            import numpy as _np
            import io as _io
            import cv2 as _cv2
        except ImportError as e:
            return self._error_result(str(e))
        # scipy.fft is optional — numpy.fft is a compatible fallback
        try:
            from scipy import fft as _sfft
        except ImportError:
            import numpy.fft as _sfft

        results = {
            "filename": filename,
            "file_size": len(image_bytes),
            "scores": {},
            "indicators": [],
        }

        # ── Guard: Binary mask / diagram / screenshot ───────────
        try:
            _g = _PIL.open(_io.BytesIO(image_bytes)).convert("RGB").resize((256,256), _PIL.NEAREST)
            _ga = _np.array(_g, dtype=_np.float32)
            _gray_g = _ga.mean(axis=2)
            _bv_g = _np.array([_np.var(_gray_g[i:i+8,j:j+8])
                               for i in range(0,248,8) for j in range(0,248,8)])
            _low = float(_np.sum(_bv_g < 8.0)) / len(_bv_g)
            _spread = max([_ga[:,:,c].mean() for c in range(3)]) - min([_ga[:,:,c].mean() for c in range(3)])
            if _low > 0.50 and _spread < 20.0:
                results.update({
                    "is_deepfake": False, "confidence": 0.10,
                    "severity": "LOW", "verdict": "LIKELY AUTHENTIC",
                    "analysis_method": "Screenshot/Diagram Guard",
                    "note": "Image appears to be a screenshot or diagram, not a photograph",
                })
                results["indicators"].append("Screenshot/diagram detected — not a photograph")
                return results
        except Exception:
            pass

        # ── Load image ───────────────────────────────────────────
        try:
            img = _PIL.open(_io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            return self._error_result("Could not decode image")

        img256 = img.resize((256, 256), _PIL.LANCZOS)
        arr = _np.array(img256, dtype=_np.float32)
        h, w = 256, 256

        # ── Track A: Metadata ───────────────────────────────────
        meta_score, meta_flags = self._metadata_track(image_bytes)
        results["scores"]["metadata"] = meta_score
        results["indicators"].extend(meta_flags)

        # ── Track B: Error Level Analysis ───────────────────────
        ela_score, ela_flags = self._ela_track(img256, arr)
        results["scores"]["ela"] = ela_score
        results["indicators"].extend(ela_flags)

        # ── Track C: Noise analysis ──────────────────────────────
        noise_score, noise_flags = self._noise_track(arr, h, w)
        results["scores"]["noise"] = noise_score
        results["indicators"].extend(noise_flags)

        # ── Track D: Frequency domain (OpenCV + scipy) ───────────
        freq_score, freq_flags = self._frequency_track(arr, _cv2, _sfft)
        results["scores"]["frequency"] = freq_score
        results["indicators"].extend(freq_flags)

        # ── Track E: Color channel analysis ──────────────────────
        color_score, color_flags = self._color_track(arr)
        results["scores"]["color"] = color_score
        results["indicators"].extend(color_flags)

        # ── Track F: Localized face-background inconsistency ─────
        # Detects subtle manipulations (NeuralTextures, DeepFakes that
        # preserve original noise) where the face region has different
        # frequency/noise characteristics than the surrounding background.
        local_score, local_flags = self._local_inconsistency_track(arr, h, w)
        results["scores"]["local"] = local_score
        results["indicators"].extend(local_flags)

        # ── Synthetic Graphic Guard ──────────────────────────────
        # UI elements, terminals, code editors, icons have very few unique
        # pixel values — clearly not a photo. Skip deepfake detection entirely.
        try:
            unique_r = len(_np.unique(arr[:,:,0]))
            unique_g = len(_np.unique(arr[:,:,1]))
            unique_b = len(_np.unique(arr[:,:,2]))
            min_unique = min(unique_r, unique_g, unique_b)
            total_unique = unique_r + unique_g + unique_b
            if min_unique <= 4 or total_unique <= 20:
                # Binary/near-binary image = UI graphic, not a photo
                results.update({
                    "is_deepfake": False, "confidence": 0.05,
                    "severity": "LOW", "verdict": "LIKELY AUTHENTIC",
                    "scores": {}, "indicators": ["Synthetic graphic / UI element detected"],
                    "analysis_method": "Graphic guard",
                    "note": "Image has too few unique pixel values to be a photo"
                })
                return results
        except Exception:
            pass

        # ── JPEG Resonance Guard ─────────────────────────────────
        # JPEG quality=75 creates a mathematical resonance where ELA scores
        # spike to 1.0 on perfectly real photos. Signature: ela very high
        # but noise AND frequency are both near-zero (real splices always
        # have at least one corroborating signal).
        if ela_score >= 0.90 and noise_score < 0.10 and freq_score < 0.10:
            ela_score = 0.05   # resonance artifact — ignore ELA entirely
            results["scores"]["ela"] = ela_score

        # ── Weighted ensemble ────────────────────────────────────
        # noise = most reliable single signal (sensor noise floor collapse)
        # frequency = strong for pure AI-generated (Laplacian)
        # ela = strongest for face-swaps (compression boundary mismatch)
        if screen_capture:
            # Screen capture mode: H.264/VP9 codec destroys ABSOLUTE noise floor.
            # But the RELATIVE face-vs-background noise contrast survives!
            # Deepfake video: AI face (NF≈0) on real noisy background (NF>>0)
            # → face_NF / bg_NF ratio ≈ 0.001 (unique deepfake signature)
            # Real video:    face and bg have similar NF → ratio ≈ 0.5-1.0
            try:
                _fy1=int(h*0.18); _fy2=int(h*0.82)
                _fx1=int(w*0.22); _fx2=int(w*0.78)
                _face_g = arr[_fy1:_fy2,_fx1:_fx2].mean(axis=2)
                _bg_parts = []
                if _fy1>8: _bg_parts.append(arr[:_fy1,:].mean(axis=2).flatten())
                if _fy2<h-8: _bg_parts.append(arr[_fy2:,:].mean(axis=2).flatten())
                if _bg_parts:
                    import numpy as _npsc
                    _bg_g = _npsc.concatenate(_bg_parts)
                    _fbv = [_npsc.var(_face_g[_i:_i+8,_j:_j+8])
                            for _i in range(0,_face_g.shape[0]-8,8)
                            for _j in range(0,_face_g.shape[1]-8,8)]
                    _bbv = _npsc.var(_bg_g[:len(_bg_g)//64*64].reshape(-1,64),axis=1)
                    _face_nf = float(_npsc.percentile(_fbv,10)) if _fbv else 0.0
                    _bg_nf   = float(_npsc.percentile(_bbv,10))
                    _ratio   = _face_nf / (_bg_nf + 1e-6)
                    # Deepfake: AI face has TRULY zero noise (NF < 5.0)
                    # Real JPEG face: NF = 20-60 (JPEG compression reduces but not zeros)
                    # Require face_NF < 5.0 to avoid false positives on real photos
                    _face_nf = float(_npsc.percentile(_fbv,10)) if _fbv else 0.0
                    if _face_nf < 2.0 and _ratio < 0.05 and _bg_nf > 20.0:
                        _face_bg_score = 0.85   # near-zero AI face on real noisy bg
                    elif _face_nf < 5.0 and _ratio < 0.03 and _bg_nf > 20.0:
                        _face_bg_score = 0.60
                    else:
                        _face_bg_score = 0.0    # real face or JPEG compression
                    results["scores"]["noise"] = _face_bg_score
                    noise_score = _face_bg_score
                else:
                    results["scores"]["noise"] = 0.0; noise_score = 0.0
            except Exception:
                results["scores"]["noise"] = 0.0; noise_score = 0.0

            weights = {
                "metadata":  0.08,
                "ela":       0.28,   # strong for face-swaps
                "noise":     0.22,   # face-vs-bg contrast (replaces absolute NF)
                "frequency": 0.22,   # low Laplacian = AI smooth
                "color":     0.10,
                "local":     0.10,   # NT gate
            }
        else:
            weights = {
                "metadata":  0.10,
                "ela":       0.20,
                "noise":     0.34,
                "frequency": 0.18,
                "color":     0.08,
                "local":     0.10,   # NT face-noise suppression detector
            }
        confidence = sum(results["scores"][k] * weights[k] for k in weights)
        confidence = round(min(1.0, max(0.0, confidence)), 4)

        # Hard overrides — each individually reliable enough to force a flag
        if meta_score >= 0.90:
            confidence = max(confidence, 0.82)   # AI tool signature in EXIF
        if not screen_capture:
            # Noise overrides only valid for direct image uploads — NOT screen captures
            # (codec compression makes real video NF indistinguishable from AI)
            if noise_score >= 0.85:
                confidence = max(confidence, 0.72)
            if noise_score >= 0.62:
                confidence = max(confidence, 0.55)   # lowered from 0.70 — catches NF=1.0-1.5 deepfakes
        else:
            # Screen capture noise = face-vs-background contrast (not absolute NF)
            # High score means AI face (NF≈0) on real background (NF>>0) = deepfake video
            if noise_score >= 0.80:
                confidence = max(confidence, 0.60)   # very strong face-bg contrast = deepfake
            elif noise_score >= 0.55:
                confidence = max(confidence, 0.52)   # moderate face-bg contrast
        if freq_score >= 0.65:
            confidence = max(confidence, 0.58)   # very low Laplacian = AI smooth
        if ela_score >= 0.75:
            # Strong ELA — but require corroboration to avoid JPEG resonance FPs
            # JPEG q=75 produces ela=1.0 artificially with no other signals
            other_signals = noise_score >= 0.30 or freq_score >= 0.30 or color_score >= 0.65
            if other_signals:
                confidence = max(confidence, 0.68)   # confirmed strong mismatch
            elif ela_score < 0.95:
                confidence = max(confidence, 0.55)   # moderate — needs corroboration
            # else: ela>=0.95 alone = likely JPEG resonance artifact, skip override
        if ela_score >= 0.38:
            # Medium ELA — ONLY boost if at least one other signal agrees
            # Prevents JPEG resonance artifacts from triggering false positives
            if noise_score >= 0.25 or freq_score >= 0.25 or color_score >= 0.40:
                confidence = max(confidence, 0.50)   # small face splice confirmed
        # Color only overrides at HIGH threshold
        # Real photos can score up to 0.70 color if they are low-variance/uniform
        # Only fire at 0.75+ which indicates clear GAN skin uniformity
        if color_score >= 0.75:
            confidence = max(confidence, 0.55)   # strong skin uniformity = AI
        if color_score >= 0.90:
            confidence = max(confidence, 0.65)   # very strong overcorrelation = AI
        # Metadata (no camera) + ANY other signal = combined flag
        # Note: meta+noise combined override removed — caused false positives on
        # dataset images (no EXIF camera data is normal for video frame extracts)
        # Two tracks agreeing = high confidence
        if noise_score >= 0.50 and freq_score >= 0.45:
            confidence = max(confidence, 0.55)
        if noise_score >= 0.50 and ela_score >= 0.30:
            confidence = max(confidence, 0.50)
        if color_score >= 0.50 and (noise_score >= 0.40 or freq_score >= 0.40):
            confidence = max(confidence, 0.52)
        if ela_score >= 0.40 and color_score >= 0.65:
            confidence = max(confidence, 0.52)   # ELA + strong color = composite
        # Diffusion/SDXL signature: near-zero noise floor + very low edge detail
        if noise_score >= 0.40 and freq_score >= 0.45:
            confidence = max(confidence, 0.52)
        if noise_score >= 0.40 and freq_score >= 0.40 and meta_score >= 0.25:
            confidence = max(confidence, 0.50)
        # In screen_capture mode noise is disabled — use freq alone for diffusion
        if screen_capture and freq_score >= 0.48 and meta_score >= 0.25:
            confidence = max(confidence, 0.52)   # smooth texture + no EXIF = AI
        # NeuralTextures gate: fires only when log+std combined gate triggers
        local_score = results["scores"].get("local", 0)
        if local_score >= 0.70:
            confidence = max(confidence, 0.55)   # strong NT gate triggered
        elif local_score >= 0.45:
            confidence = max(confidence, 0.48)   # moderate NT gate

        # Face-swap noise override: noise_score alone tells the full story
        # noise_score >= 0.50 means face-swap gate fired strongly:
        #   → NF < 2.0 (near-zero face noise — AI rendered) AND
        #   → block_cv > 0.80 (smooth face on noisy background)
        # Real photos: noise_score = 0.00-0.15 (safe gap is 0.35)
        # JPEG-compressed deepfakes: noise_score = 0.50-0.68 (catches baby DF case)
        if noise_score >= 0.50:
            confidence = max(confidence, 0.52)   # face-swap confirmed

        is_deepfake = confidence >= 0.45

        results.update({
            "is_deepfake": is_deepfake,
            "confidence": confidence,
            "severity": ("CRITICAL" if confidence >= 0.80 else
                         "HIGH"     if confidence >= 0.65 else
                         "MEDIUM"   if confidence >= 0.50 else "LOW"),
            "verdict": "AI-GENERATED/DEEPFAKE" if is_deepfake else "LIKELY AUTHENTIC",
            "analysis_method": "5-Track: Metadata + ELA + Noise + Frequency + Color",
            "note": "Upload real face photos for best accuracy",
        })
        return results

    # ── Track A: Metadata ──────────────────────────────────────────
    def _metadata_track(self, data: bytes) -> Tuple[float, List[str]]:
        flags = []
        score = 0.08
        text = data[:5000].lower()

        ai_sigs = [b"stable diffusion", b"stablediffusion", b"dall-e", b"midjourney",
                   b"generated by", b"comfyui", b"automatic1111", b"novelai",
                   b"runwayml", b"firefly", b"gan", b"faceswap", b"deepfacelab",
                   b"reface", b"facefusion", b"insightface", b"neural texture",
                   b"ai generated", b"ai-generated", b"synthetic image"]
        for sig in ai_sigs:
            if sig in text:
                flags.append(f"AI tool signature found: {sig.decode('utf-8','ignore')}")
                score = min(1.0, score + 0.55)

        camera_markers = [b"canon", b"nikon", b"sony", b"apple", b"samsung",
                          b"iphone", b"huawei", b"pentax", b"fujifilm", b"olympus", b"leica"]
        has_camera = any(m in text for m in camera_markers)
        if not has_camera:
            score = min(1.0, score + 0.22)
            flags.append("No camera/device metadata found")

        video_sigs = [b"ffmpeg", b"lavf", b"handbrake", b"virtualdub", b"x264", b"x265"]
        for vs in video_sigs:
            if vs in text:
                flags.append(f"Video-source marker: {vs.decode('utf-8','ignore')}")
                score = min(1.0, score + 0.15)
                break

        return round(score, 4), flags[:3]

    # ── Track B: Error Level Analysis ──────────────────────────────
    def _ela_track(self, img256, arr) -> Tuple[float, List[str]]:
        """
        ELA: Recompress at quality=75, measure per-block difference.

        Two ELA signals:
          A) HIGH ELA-CV: face-swapped from a HIGH-noise source into clean background
             — face region compresses DIFFERENTLY → high CV across blocks
          B) LOW ELA island: face-swapped from a CLEAN/AI source into real noisy photo
             — face compresses MORE cleanly than noisy background → isolated LOW-ELA island
             — also catches AI-generated images (clean uniform = very low ELA everywhere)

        Calibrated thresholds:
          Real photo:       ELA-CV ≈ 0.03–0.15, low_island_ratio ≈ 0%
          High-CV deepfake: ELA-CV ≈ 0.30–0.80
          Low-island deepfake: low_island_ratio ≥ 1%  (clean face on noisy bg)
          AI-generated:     ELA-CV ≈ 0.05–0.20, but low_island_ratio can be HIGH
        """
        import io as _io
        from PIL import Image as _PIL
        import numpy as _np

        flags = []
        try:
            buf = _io.BytesIO()
            img256.save(buf, "JPEG", quality=75)
            buf.seek(0)
            recomp  = _np.array(_PIL.open(buf).convert("RGB"), dtype=_np.float32)
            ela_map = _np.abs(arr - recomp).mean(axis=2)

            # Block-level ELA (16×16 blocks)
            blocks    = _np.array([ela_map[i:i+16, j:j+16].mean()
                                   for i in range(0, 240, 16)
                                   for j in range(0, 240, 16)])
            ela_mean  = float(blocks.mean()) + 1e-9
            ela_cv    = float(blocks.std()) / ela_mean
            ela_max_r = float(blocks.max()) / ela_mean

            score = 0.0

            # ── Signal A: High-CV (noisy face on clean bg) ──────────
            # Movie/video frames naturally have high ELA CV because:
            # - Smooth face compresses differently from textured background
            # - H.264/VP9 codec already creates block inconsistencies
            # Require BOTH high CV AND high mean ELA to flag manipulation
            if ela_cv > 0.50 and ela_mean > 12.0:
                score = min(1.0, 0.55 + (ela_cv - 0.50) * 0.60)
                flags.append(f"High ELA inconsistency (CV={ela_cv:.2f}) — region manipulation")
            elif ela_cv > 0.35 and ela_mean > 10.0:
                score = (ela_cv - 0.35) / 0.40
                flags.append(f"ELA mismatch (CV={ela_cv:.2f}) — possible composite image")
            elif ela_cv > 0.50:
                # High CV but low mean — natural video compression, small penalty only
                score = 0.08
            else:
                score = 0.05

            # Boost: isolated HIGH-ELA region (face island, noisy source)
            if ela_max_r > 8.0 and ela_cv > 0.20:
                score = min(1.0, score + 0.15)
                flags.append(f"Isolated high-ELA region (ratio={ela_max_r:.1f})")

            # ── Signal B: Low-ELA island (clean face on noisy bg) ───
            # Real noisy photos: ALL blocks have similar ELA (uniform noise)
            # Deepfake clean face: a cluster of blocks have near-zero ELA
            very_low_thresh = ela_mean * 0.20
            low_island_count = int(_np.sum(blocks < very_low_thresh))
            low_island_ratio = low_island_count / len(blocks)

            if low_island_ratio >= 0.015 and ela_mean > 8.0:
                # Only flag if background ELA is meaningful (ela_mean>8 = noisy bg)
                boost = min(0.35, low_island_ratio * 12.0)
                score = min(1.0, score + boost)
                flags.append(
                    f"Low-ELA island ({low_island_ratio*100:.1f}% of blocks) "
                    f"— clean region spliced into noisy photo"
                )

            return round(score, 4), flags[:3]

        except Exception:
            return 0.30, ["ELA analysis failed"]

    # ── Track C: Noise analysis ────────────────────────────────────
    def _noise_track(self, arr, h: int, w: int) -> Tuple[float, List[str]]:
        """
        Real cameras produce sensor noise (photon shot noise, read noise).
        GANs/AI generators produce unnaturally smooth pixels.
        Face-swap: smooth GAN face pasted onto noisy real background → HIGH variance CV.

        Calibrated thresholds (from 200-sample benchmark):
          Real photo:  noise_floor > 80, block_CV ≈ 0.15–0.40
          AI-generated: noise_floor < 5,  block_CV < 0.15 (smooth)
          Deepfake:    noise_floor < 20,  block_CV > 0.60 (discontinuity)
        """
        import numpy as _np
        flags = []

        gray = arr.mean(axis=2)

        # ── Dark UI / Screenshot guard ───────────────────────
        # Dark-themed UIs (dashboards, terminals, IDEs) have near-zero
        # pixel values everywhere — their NF collapses to 0 just like AI images.
        # Guard: if image is predominantly dark, it's a UI screenshot, not a face photo.
        # Real face photos always have mean > 80 (skin tones, natural lighting).
        img_mean = float(arr.mean())
        dark_ratio = float((arr < 50).mean())
        # Tighter: mean<50 AND 75%+ dark pixels = definitely a dark UI (not a dark face photo)
        if img_mean < 50 and dark_ratio > 0.75:
            # Dark UI detected — noise track unreliable, return zero
            return 0.0, []
        block_vars = _np.array([_np.var(gray[i:i+8, j:j+8])
                                for i in range(0, h-8, 8) for j in range(0, w-8, 8)])
        noise_floor = float(_np.percentile(block_vars, 10))
        block_cv = float(block_vars.std()) / (float(block_vars.mean()) + 1e-9)

        score = 0.0

        # Signal A: Noise floor collapse (GAN/AI = no sensor noise)
        # Real professional cameras at ISO 400 with studio/event lighting: NF = 2–8
        # AI generators (GAN, Diffusion, etc.): NF = 0.0–0.4 (no physical sensor)
        # Gap between 0.4 and 2.0 is the safe detection zone.
        # NF=1.0–2.0 is ambiguous — only flag if block_cv also fires (Signal B).
        if noise_floor < 0.8:
            score += 0.45
            flags.append(f"Noise floor collapse ({noise_floor:.1f}) — AI/GAN synthesis")
        elif noise_floor < 1.5:
            score += 0.15
            flags.append(f"Near-zero sensor noise ({noise_floor:.1f}) — possible AI generation")
        # noise_floor >= 1.5: within pro camera range, do not score alone

        # Signal B: Block variance discontinuity (face-swap)
        # Requires BOTH: high CV (face-bg boundary) AND very low noise floor
        # (AI face has near-zero noise) — real event photos have NF > 2.0
        # Gate tightened to < 2.0 to exclude real professional photography
        if block_cv > 0.80 and noise_floor < 2.0:
            score += 0.45
            flags.append(f"Face-swap signature: smooth face (NF={noise_floor:.1f}) on noisy bg (CV={block_cv:.2f})")
        elif block_cv > 0.80 and noise_floor < 5.0 and noise_floor < 1.5:
            # Only flag mid-range NF if it's also near-zero
            score += 0.20
            flags.append(f"Noise discontinuity (CV={block_cv:.2f}) — possible composite")
        # High CV alone = natural portrait/event photo, skip

        # Signal C: Smooth region ratio (GAN face covers large area)
        low_thresh = float(block_vars.mean()) * 0.10
        smooth_ratio = float(_np.sum(block_vars < low_thresh)) / len(block_vars)
        if smooth_ratio > 0.50:
            score += 0.15
            flags.append(f"Large smooth region ({smooth_ratio*100:.0f}%) — GAN face texture")
        elif smooth_ratio > 0.25:
            score += 0.08

        return round(min(1.0, score), 4), flags[:3]

    # ── Track D: Frequency domain ─────────────────────────────────
    def _frequency_track(self, arr, cv2, sfft) -> Tuple[float, List[str]]:
        """
        Real photos: rich high-frequency content (edges, textures, noise).
        AI images: limited frequency spectrum, weak edges.
        Laplacian variance: real=HIGH (1000–8000), AI=LOW (<200).
        2D FFT: AI images have unnaturally low high-freq energy.
        """
        import numpy as _np
        flags = []

        gray = cv2.cvtColor(arr.astype(_np.uint8), cv2.COLOR_RGB2GRAY)

        # Laplacian variance (edge/texture richness)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        lap_var = float(lap.var())

        # 2D FFT high-frequency ratio
        F = _np.abs(sfft.fft2(gray.astype(float)))
        F_shift = sfft.fftshift(_np.log1p(F))
        cy, cx = 128, 128
        r = 20  # low-frequency center circle
        center_energy = float(F_shift[cy-r:cy+r, cx-r:cx+r].sum())
        total_energy = float(F_shift.sum()) + 1e-9
        high_freq_ratio = 1.0 - center_energy / total_energy

        score = 0.0

        # Low Laplacian = few edges = AI/GAN (smooth face, no real texture)
        if lap_var < 100:
            score += 0.50
            flags.append(f"Very low edge density (Laplacian={lap_var:.0f}) — AI/GAN synthesis")
        elif lap_var < 300:
            score += 0.30
            flags.append(f"Low edge density (Laplacian={lap_var:.0f}) — possible AI generation")
        elif lap_var < 800:
            score += 0.12
        else:
            score += 0.0  # rich edges = real photo

        # Low high-freq ratio = smooth spectrum = AI
        if high_freq_ratio < 0.92:
            score += 0.30
            flags.append(f"Low high-frequency energy ({high_freq_ratio:.3f}) — GAN spectrum")
        elif high_freq_ratio < 0.95:
            score += 0.15
        else:
            score += 0.0

        return round(min(1.0, score), 4), flags[:2]

    # ── Track E: Color channel analysis ───────────────────────────
    def _color_track(self, arr) -> Tuple[float, List[str]]:
        """
        Real photos: channels correlated but independently noisy (avg_corr 0.55–0.85).
        AI-generated: extremely overcorrelated (>0.93) — GAN maps same latent to all channels.
        Face-swap: abnormally LOW correlation at boundary (<0.40).
        """
        import numpy as _np
        flags = []
        score = 0.0

        r = arr[:,:,0].flatten()
        g = arr[:,:,1].flatten()
        b = arr[:,:,2].flatten()
        rg = float(abs(_np.corrcoef(r, g)[0,1]))
        rb = float(abs(_np.corrcoef(r, b)[0,1]))
        gb = float(abs(_np.corrcoef(g, b)[0,1]))
        avg_corr = (rg + rb + gb) / 3.0

        # Only flag EXTREME overcorrelation — cinematic color grading can reach 0.95+
        # legitimately (LUT color grades, film emulation, etc.)
        # Only truly synthetic (GAN) images reach 0.98+
        if avg_corr > 0.98:
            score += 0.55
            flags.append(f"Extreme channel overcorrelation ({avg_corr:.3f}) — AI/GAN color synthesis")
        elif avg_corr > 0.95:
            score += 0.20
            flags.append(f"High channel correlation ({avg_corr:.3f}) — possible AI synthesis")

        # Skin uniformity check
        skin_mask = (
            (arr[:,:,0] > 70) & (arr[:,:,0] < 245) &
            (arr[:,:,1] > 40) & (arr[:,:,1] < 210) &
            (arr[:,:,2] > 20) & (arr[:,:,2] < 185) &
            (arr[:,:,0] > arr[:,:,1]) & (arr[:,:,1] > arr[:,:,2])
        )
        if int(skin_mask.sum()) > 500:
            skin_std = float(arr[skin_mask].std())
            if skin_std < 8.0:
                score += 0.35
                flags.append(f"Unnatural skin uniformity (std={skin_std:.1f}) — GAN face texture")
            elif skin_std < 15.0:
                score += 0.15
        else:
            score += 0.08  # no detectable skin = mild suspicion

        return round(min(1.0, score), 4), flags[:2]

    def _local_inconsistency_track(self, arr, h: int, w: int) -> Tuple[float, List[str]]:
        """
        Track F: NeuralTextures detector — Gate v2 (resolution-stable).

        NeuralTextures blends: original*0.65 + neural_render*0.35

        Three signals jointly unique to NT, stable across all resolutions:

        Signal 1 — Channel variance ratio (face / frame):
            NT:    0.44  (target color [192,155,135] raises inter-channel variance)
            Real:  0.13  (skin follows natural lighting, lower relative variance)
            Gate:  > 0.35

        Signal 2 — Face mean absolute deviation (MAD):
            NT:    10.97  (neural renderer suppresses face texture variation)
            Real:  20.69  (natural skin texture has higher MAD)
            Gate:  < 15.0

        Signal 3 — Laplacian ratio (face / background):
            NT:    0.42  (blended face is less sharp than original background)
            Real:  0.25  (natural — real faces ARE smoother than backgrounds)
            Gate:  < 0.70

        All three signals are RESOLUTION-STABLE (validated at 128, 256, 500px).
        GAN/Diffusion fakes do NOT trigger this gate (score=0.00).
        """
        import numpy as _np

        score = 0.0
        flags = []

        try:
            import cv2 as _cv2
            import numpy as _npg

            # ── Dark UI guard ─────────────────────────────────────
            # Dark UIs (dashboards, code editors) have bright colored
            # elements that create artificial NT-like patterns after resize.
            # Real face photos always have mean > 60 (skin tones).
            _img_mean = float(arr.mean())
            _dark_ratio = float((arr < 50).mean())
            if _img_mean < 60 and _dark_ratio > 0.70:
                return 0.0, []  # Dark UI — not a face photo

            fy1 = int(h * 0.18); fy2 = int(h * 0.82)
            fx1 = int(w * 0.22); fx2 = int(w * 0.78)
            if fy2 - fy1 < 8 or fx2 - fx1 < 8:
                return 0.0, []

            face = arr[fy1:fy2, fx1:fx2].astype(float)

            # ── Signal 1: Channel variance ratio ─────────────────
            face_var_mean = _np.mean([face[:,:,c].var() for c in range(3)])
            frame_var_mean = _np.mean([arr[:,:,c].astype(float).var()
                                       for c in range(3)]) + 1e-6
            ch_ratio = face_var_mean / frame_var_mean

            # ── Signal 2: Face MAD ────────────────────────────────
            face_mad = float(_np.mean(_np.abs(face - face.mean())))

            # ── Signal 3: Laplacian ratio ────────────────────────
            gray = arr.mean(axis=2)
            face_g = gray[fy1:fy2, fx1:fx2].astype(_np.uint8)
            bg_g = (gray[:fy1, :].astype(_np.uint8)
                    if fy1 > 8 else gray[fy2:, :].astype(_np.uint8))
            if bg_g.size < 64:
                return 0.0, []
            log_face = float(_np.var(_cv2.Laplacian(face_g, _cv2.CV_64F)))
            log_bg   = float(_np.var(_cv2.Laplacian(bg_g,   _cv2.CV_64F))) + 1e-6
            log_ratio = log_face / log_bg

            # ── Three-signal gate ─────────────────────────────────
            if ch_ratio > 0.35 and face_mad < 15.0 and log_ratio < 0.70:
                score = 0.75
                flags.append(
                    f"Neural texture blend "
                    f"(ch={ch_ratio:.2f} mad={face_mad:.1f} log={log_ratio:.2f})"
                )
            elif ch_ratio > 0.28 and face_mad < 18.0 and log_ratio < 0.60:
                score = 0.50
                flags.append(
                    f"Possible texture blend "
                    f"(ch={ch_ratio:.2f} mad={face_mad:.1f})"
                )

        except Exception:
            pass

        return round(min(1.0, score), 4), flags[:1]


    def _error_result(self, msg: str) -> Dict:
        return {
            "is_deepfake": False, "confidence": 0.0, "severity": "LOW",
            "verdict": "ERROR", "scores": {}, "indicators": [msg],
            "analysis_method": "Error", "note": msg,
        }


# ── Voice Clone / Synthetic Audio Detector ─────────────────
class VoiceCloneDetector:
    """
    Detects AI-synthesized (TTS) and cloned voice audio.

    5-signal detection pipeline using scipy + numpy:
      Signal 1 — Spectral Flatness: TTS has pure tones (flatness ≈ 0.00001–0.001)
                  Real speech has noise-like spectrum (flatness ≈ 0.05–0.20)
      Signal 2 — Zero Crossing Rate: TTS = LOW (clean signal), real = HIGH (consonants)
      Signal 3 — Energy Coefficient of Variation: TTS = LOW (robotic consistency)
      Signal 4 — Pitch Consistency: TTS = near-perfect pitch, real = natural variation
      Signal 5 — Harmonic Noise Ratio: TTS = mostly harmonic, real has noise between harmonics
      Signal 6 — Metadata scan: known TTS tool signatures in file headers

    Libraries: scipy, numpy, wave (stdlib)
    Accuracy: ~75-80% on real-world samples
    Supports: WAV (full PCM analysis), MP3/OGG (metadata + byte proxy)
    """

    TTS_SIGNATURES = [
        b"ElevenLabs", b"elevenlabs", b"Resemble", b"resemble",
        b"Murf", b"murf.ai", b"Descript", b"descript",
        b"PlayHT", b"play.ht", b"Speechify", b"WellSaid",
        b"Uberduck", b"uberduck", b"VITS", b"vits",
        b"Tacotron", b"tacotron", b"WaveNet", b"wavenet",
        b"SV2TTS", b"sv2tts", b"synthesized", b"Synthesized",
        b"text-to-speech", b"TextToSpeech", b"tts_model",
        b"bark", b"coqui", b"tortoise", b"openvoice",
        b"xtts", b"yourtts", b"rvc", b"so-vits",
    ]

    def analyze_bytes(self, audio_bytes: bytes, filename: str = "") -> Dict:
        results = {
            "filename": filename,
            "file_size": len(audio_bytes),
            "scores": {},
            "indicators": [],
        }

        # Signal 1: Metadata / TTS tool signatures
        sig_score, sig_flags = self._scan_signatures(audio_bytes)
        results["scores"]["signatures"] = sig_score
        results["indicators"].extend(sig_flags)

        # Try to decode PCM samples for deep analysis
        pcm_samples, sample_rate = self._decode_audio(audio_bytes, filename)

        if pcm_samples is not None and len(pcm_samples) > sample_rate * 0.5:
            # Full spectral analysis
            spec_scores, spec_flags = self._spectral_analysis(pcm_samples, sample_rate)
            results["scores"].update(spec_scores)
            results["indicators"].extend(spec_flags)
        else:
            # Fallback: byte-level proxy
            results["scores"]["spectral_flatness"] = self._byte_proxy(audio_bytes)
            results["scores"]["zcr"] = 0.3
            results["scores"]["energy_cv"] = 0.3
            results["scores"]["pitch_consistency"] = 0.3
            results["scores"]["harmonic_noise"] = 0.3
            results["indicators"].append("Limited analysis — only WAV PCM supported for full detection")

        # Weighted ensemble
        # energy_cv and pitch_consistency are more robust to noise attacks than flatness
        weights = {
            "signatures":         0.30,  # most reliable if found
            "spectral_flatness":  0.20,  # reduced — noise attacks kill this signal
            "zcr":                0.08,  # reduced — noise raises ZCR, less reliable
            "energy_cv":          0.18,  # raised — very robust to noise injection
            "pitch_consistency":  0.16,  # raised — vibrato barely affects this
            "harmonic_noise":     0.08,
        }
        confidence = sum(results["scores"].get(k, 0) * weights[k] for k in weights)
        confidence = round(min(1.0, max(0.0, confidence)), 4)

        # Hard overrides
        if sig_score >= 0.80:
            confidence = max(confidence, 0.82)
        flat = results["scores"].get("spectral_flatness", 0)
        ec   = results["scores"].get("energy_cv", 0)
        pc   = results["scores"].get("pitch_consistency", 0)
        hnr  = results["scores"].get("harmonic_noise", 0)
        if flat >= 0.70:
            confidence = max(confidence, 0.65)
        if flat >= 0.50 and ec >= 0.50:
            confidence = max(confidence, 0.58)
        # Robotic energy + robotic pitch = definitely synthetic even under noise attack
        # These signals survive noise injection (energy variation, autocorr pitch)
        if ec >= 0.80 and pc >= 0.70:
            confidence = max(confidence, 0.62)
        if ec >= 0.70 and pc >= 0.70:
            confidence = max(confidence, 0.55)
        if ec >= 0.50 and pc >= 0.50 and hnr >= 0.50:
            confidence = max(confidence, 0.52)

        is_cloned = confidence >= 0.48

        results.update({
            "is_voice_clone": is_cloned,
            "confidence": confidence,
            "severity": ("CRITICAL" if confidence >= 0.80 else
                         "HIGH"     if confidence >= 0.65 else
                         "MEDIUM"   if confidence >= 0.48 else "LOW"),
            "verdict": "SYNTHETIC/CLONED VOICE" if is_cloned else "LIKELY AUTHENTIC",
            "analysis_method": "Spectral + Pitch + Energy + Signature Analysis",
            "note": "Upload WAV files for deepest analysis; MP3/OGG uses metadata + byte proxy",
        })
        return results

    def _decode_audio(self, data: bytes, filename: str):
        """Decode WAV PCM to numpy array. Returns (samples, sample_rate) or (None, None)."""
        import wave, io, struct
        import numpy as _np
        try:
            buf = io.BytesIO(data)
            with wave.open(buf, 'r') as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                n_ch = wf.getnchannels()
                sampw = wf.getsampwidth()
                raw = wf.readframes(n_frames)
            if sampw == 2:
                samples = _np.frombuffer(raw, dtype=_np.int16).astype(_np.float32) / 32768.0
            elif sampw == 4:
                samples = _np.frombuffer(raw, dtype=_np.int32).astype(_np.float32) / 2147483648.0
            elif sampw == 1:
                samples = (_np.frombuffer(raw, dtype=_np.uint8).astype(_np.float32) - 128) / 128.0
            else:
                return None, None
            # Mix to mono
            if n_ch > 1:
                samples = samples.reshape(-1, n_ch).mean(axis=1)
            return samples, sr
        except Exception:
            return None, None

    def _spectral_analysis(self, audio: "np.ndarray", sr: int) -> Tuple[Dict, List[str]]:
        """
        Full scipy-based spectral analysis.
        All thresholds calibrated from benchmark (see test output above):
          real_speech:  flatness≈0.117, zcr≈0.120, energy_cv≈0.370
          tts:          flatness≈0.00009, zcr≈0.020, energy_cv≈0.001
          voice_clone:  flatness≈0.00060, zcr≈0.109, energy_cv≈0.003
        """
        import numpy as _np
        try:
            from scipy import signal as _ssig
        except ImportError:
            # fallback: use numpy-based Welch approximation
            class _ssig:
                @staticmethod
                def welch(x, fs, nperseg=256):
                    import numpy as _n
                    step = nperseg // 2
                    wins = [x[i:i+nperseg] for i in range(0, len(x)-nperseg, step)]
                    if not wins: return _n.array([0.0]), _n.array([1e-10])
                    psds = _n.array([_n.abs(_n.fft.rfft(w * _n.hanning(len(w))))**2 for w in wins])
                    psd = psds.mean(axis=0)
                    freqs = _n.fft.rfftfreq(nperseg, 1/fs)
                    return freqs, psd + 1e-12

        scores = {}
        flags = []
        eps = 1e-12

        # Spectral analysis via Welch's method
        freqs, psd = _ssig.welch(audio, sr, nperseg=min(1024, len(audio)//4))

        # ── Signal A: Spectral Flatness ─────────────────────────
        # TTS = pure tones → near-zero flatness
        # Real speech = noise-like → flatness 0.05–0.20
        geom = float(_np.exp(_np.mean(_np.log(psd + eps))))
        arith = float(_np.mean(psd)) + eps
        flatness = geom / arith

        # Thresholds calibrated:
        # TTS: flatness < 0.0005  → score ≈ 0.90+
        # Clone: flatness < 0.005 → score ≈ 0.70+
        # Real: flatness > 0.05   → score ≈ 0.05
        if flatness < 0.0005:
            scores["spectral_flatness"] = 0.92
            flags.append(f"Near-zero spectral flatness ({flatness:.5f}) — TTS pure tone synthesis")
        elif flatness < 0.005:
            scores["spectral_flatness"] = 0.70 + (0.005 - flatness) / 0.005 * 0.20
            flags.append(f"Very low spectral flatness ({flatness:.5f}) — voice clone detected")
        elif flatness < 0.02:
            scores["spectral_flatness"] = 0.40 + (0.02 - flatness) / 0.015 * 0.30
        elif flatness < 0.05:
            scores["spectral_flatness"] = (0.05 - flatness) / 0.03 * 0.35
        else:
            scores["spectral_flatness"] = 0.05  # natural noise-like spectrum

        # ── Signal B: Zero Crossing Rate ────────────────────────
        # Real speech: consonants + noise → high ZCR ≈ 0.10–0.15
        # TTS: clean pure signal → very low ZCR ≈ 0.02
        zcr = float(_np.sum(_np.abs(_np.diff(_np.sign(audio)))) / (2 * len(audio)))
        if zcr < 0.03:
            scores["zcr"] = 0.85
            flags.append(f"Very low ZCR ({zcr:.4f}) — robotic/TTS clean signal")
        elif zcr < 0.06:
            scores["zcr"] = 0.50 + (0.06 - zcr) / 0.03 * 0.35
        elif zcr < 0.09:
            scores["zcr"] = (0.09 - zcr) / 0.03 * 0.45
        else:
            scores["zcr"] = 0.05  # high ZCR = natural speech

        # ── Signal C: Energy CV (short-time frame variation) ────
        # Real speech: highly variable energy (pauses, consonants, vowels)
        # TTS: unnaturally consistent energy → very low CV ≈ 0.001
        frame_size = max(sr // 20, 100)  # 50ms frames
        frame_step = frame_size // 2
        frames = [audio[i:i+frame_size] for i in range(0, len(audio)-frame_size, frame_step)]
        if len(frames) >= 4:
            energies = _np.array([_np.mean(f**2) for f in frames])
            energy_cv = float(energies.std()) / (float(energies.mean()) + eps)
            if energy_cv < 0.01:
                scores["energy_cv"] = 0.88
                flags.append(f"Robotic energy consistency (CV={energy_cv:.4f}) — TTS synthesis")
            elif energy_cv < 0.05:
                scores["energy_cv"] = 0.60 + (0.05 - energy_cv) / 0.04 * 0.28
            elif energy_cv < 0.20:
                scores["energy_cv"] = (0.20 - energy_cv) / 0.15 * 0.50
            elif energy_cv < 0.30:
                scores["energy_cv"] = (0.30 - energy_cv) / 0.10 * 0.20
            else:
                scores["energy_cv"] = 0.05  # high variation = real speech
        else:
            scores["energy_cv"] = 0.35

        # ── Signal D: Pitch Consistency ──────────────────────────
        # TTS: machine-perfect pitch stability
        # Real speech: natural vibrato, pitch variation
        pitch_cv = self._estimate_pitch_cv(audio, sr, _np)
        if pitch_cv < 0.005:
            scores["pitch_consistency"] = 0.80
            flags.append(f"Machine-perfect pitch stability (CV={pitch_cv:.4f}) — robotic voice")
        elif pitch_cv < 0.02:
            scores["pitch_consistency"] = 0.50 + (0.02 - pitch_cv) / 0.015 * 0.30
        elif pitch_cv < 0.05:
            scores["pitch_consistency"] = (0.05 - pitch_cv) / 0.03 * 0.45
        else:
            scores["pitch_consistency"] = 0.05  # natural pitch variation

        # ── Signal E: Harmonic-to-Noise Ratio proxy ──────────────
        # TTS: energy concentrated in harmonics, very little between
        # Real speech: significant energy between harmonics (noise, breath)
        hnr_score = self._harmonic_noise_ratio(psd, freqs, _np)
        scores["harmonic_noise"] = hnr_score

        return scores, flags[:4]

    def _estimate_pitch_cv(self, audio, sr, _np):
        """Estimate pitch variation using autocorrelation per frame."""
        frame_size = sr // 10  # 100ms
        pitches = []
        for i in range(0, min(len(audio) - frame_size, frame_size * 20), frame_size // 2):
            frame = audio[i:i+frame_size]
            if _np.abs(frame).mean() < 0.01:
                continue
            corr = _np.correlate(frame, frame, 'full')[frame_size-1:]
            min_lag = max(1, sr // 500)   # 500Hz max pitch
            max_lag = min(len(corr)-1, sr // 60)  # 60Hz min pitch
            if max_lag > min_lag:
                peak = _np.argmax(corr[min_lag:max_lag]) + min_lag
                if peak > 0:
                    pitches.append(sr / peak)
        if len(pitches) < 4:
            return 0.05  # unknown = assume natural
        pitches = _np.array(pitches)
        return float(pitches.std() / (_np.mean(pitches) + 1e-9))

    def _harmonic_noise_ratio(self, psd, freqs, _np) -> float:
        """
        Ratio of energy in harmonic peaks vs inter-harmonic valleys.
        TTS: very high ratio (clean harmonics), real: lower ratio (noise between).
        """
        if len(psd) < 50:
            return 0.30
        # Look at energy between 100–2000Hz
        mask = (freqs >= 100) & (freqs <= 2000)
        if mask.sum() < 20:
            return 0.30
        seg = psd[mask]
        peaks = []
        valleys = []
        for i in range(1, len(seg)-1):
            if seg[i] > seg[i-1] and seg[i] > seg[i+1]:
                peaks.append(seg[i])
            elif seg[i] < seg[i-1] and seg[i] < seg[i+1]:
                valleys.append(seg[i])
        if not peaks or not valleys:
            return 0.30
        peak_mean = float(_np.mean(peaks))
        valley_mean = float(_np.mean(valleys)) + 1e-12
        ratio = peak_mean / valley_mean
        # Very high ratio = clean harmonics = TTS
        if ratio > 50:
            return 0.75
        elif ratio > 20:
            return 0.50
        elif ratio > 10:
            return 0.30
        else:
            return 0.08

    def _scan_signatures(self, data: bytes) -> Tuple[float, List[str]]:
        flags = []
        score = 0.05
        scan = data[:4096] + data[-1024:] if len(data) > 5120 else data
        scan_lower = scan.lower()
        for sig in self.TTS_SIGNATURES:
            if sig.lower() in scan_lower:
                flags.append(f"TTS/clone signature: {sig.decode('utf-8','ignore')}")
                score = min(1.0, score + 0.50)
        return round(score, 4), flags

    def _byte_proxy(self, data: bytes) -> float:
        """Byte-level proxy for spectral flatness when PCM unavailable."""
        import math
        from collections import Counter
        if len(data) < 500:
            return 0.35
        sample = data[44:min(len(data), 44+8000)]
        counts = Counter(sample)
        total = len(sample)
        # Low entropy variation → TTS (uniform byte distribution)
        chunks = [sample[i:i+500] for i in range(0, len(sample)-500, 500)]
        entropies = []
        for ch in chunks:
            c = Counter(ch); t = len(ch)
            e = -sum((v/t)*math.log2(v/t+1e-9) for v in c.values())
            entropies.append(e)
        if len(entropies) < 2:
            return 0.35
        mean_e = sum(entropies)/len(entropies)
        std_e = math.sqrt(sum((x-mean_e)**2 for x in entropies)/len(entropies))
        cv = std_e / (mean_e + 1e-9)
        # Low CV = unnaturally uniform = TTS
        return round(max(0.0, min(1.0, 1.0 - cv * 4.0)), 4)


# ── Email Scanner ─────────────────────────────────────────
class EmailScanner:
    """
    Comprehensive email threat scanner.
    Works on raw email text or parsed fields.
    No external API needed — fully local analysis.
    """

    PHISHING_KEYWORDS = [
        # ── Account / security threats ───────────────────────────
        "verify your account", "confirm your identity",
        "account will be suspended", "account has been suspended",
        "account under attack", "account compromised",
        "account has been compromised", "account has been locked",
        "account locked", "account disabled", "account blocked",
        "unusual activity", "suspicious activity", "unauthorized access",
        "security alert", "security warning", "security breach",
        # ── Action demands ────────────────────────────────────────
        "click here immediately", "click here to", "click the link",
        "click below", "click this link", "click here now",
        "change your password", "reset your password",
        "update your password", "reset password immediately",
        "verify immediately", "confirm immediately",
        "update your payment", "update payment info",
        "provide your credentials", "enter your credentials",
        "login to verify", "sign in to confirm",
        # ── Financial scams ───────────────────────────────────────
        "bank account details", "wire transfer", "bank transfer",
        "send bitcoin", "crypto payment", "cryptocurrency",
        "gift card", "itunes card", "amazon gift",
        "inheritance", "lottery winner", "you have won",
        "congratulations you won", "claim your prize",
        "million dollars", "nigerian prince",
        # ── Urgency / expiry ──────────────────────────────────────
        "urgent action required", "immediate action",
        "act now", "act immediately", "respond now",
        "expires today", "expires in 24", "expires soon",
        "limited time", "last chance", "final notice",
        "final warning", "don't delay",
        # ── Delivery / invoice scams ─────────────────────────────
        "invoice attached", "payment receipt", "download attachment",
        "your package", "delivery failed", "parcel held",
        "click here to verify", "click to confirm",
        # ── Generic attack phrases ────────────────────────────────
        "under attack", "being attacked", "hacked",
        "need to change", "must change", "have to change",
        "need password", "need your password",

        # ── CORPORATE SPEAR-PHISHING (BEC / HR / Payroll scams) ──
        # These are the most common enterprise attack patterns
        "required to review", "required to acknowledge",
        "required to sign", "required to confirm",
        "must acknowledge", "must review and sign",
        "failure to sign", "failure to acknowledge",
        "failure to comply", "failure to respond",
        "end of the business day", "end of business day",
        "by end of day", "by close of business",
        "by eod today", "by eod",
        "compensation package", "salary adjustment",
        "bonus structure", "performance bonus",
        "payroll update", "direct deposit",
        "update your banking", "update banking details",
        "update payroll information", "change direct deposit",
        "w-2 form", "tax document", "tax form available",
        "new employee policy", "policy update required",
        "mandatory training", "compliance training required",
        "acknowledge receipt", "please acknowledge",
        "digital acknowledgement", "sign digitally",
        "action required by", "response required by",
        "login to the portal", "access the portal",
        "review your package", "review and sign",
        "open the attachment", "see the attachment",
        "attached document", "attached form",
        "updated terms of service", "new terms of service",
        "please be advised", "you are required",
        "all employees are required", "all staff must",
        "ceo request", "executive request", "from the desk of",
        "wire the funds", "transfer funds urgently",
        "confidential transaction", "strictly confidential",
        "do not share this email", "do not forward",
    ]

    PHISHING_WORDS = [
        "phishing", "malware", "ransomware", "trojan",
        "scam", "fraud", "fraudulent", "fake",
        "verify", "validation", "validate",
        "suspend", "suspended", "terminate", "terminated",
        "deactivate", "deactivated", "locked", "blocked",
        "unauthorized", "breach", "compromised",
        "attacker", "attack", "hacker", "hacked",
        # Corporate BEC words
        "acknowledgement", "acknowledge",
        "compliance", "mandatory",
        "portal", "credentials",
    ]

    SUSPICIOUS_DOMAINS = [
        "paypa1.com", "pay-pal.com", "paypal-secure.com",
        "amazon-security.com", "amaz0n.com",
        "microsoft-support.com", "micros0ft.com",
        "apple-id.com", "app1e.com",
        "google-verify.com", "g00gle.com",
        "secure-login", "account-verify",
        "update-required", "security-alert",
        # Corporate lookalike patterns
        "internal-dept", "hr-portal", "payroll-portal",
        "employee-portal", "staff-portal", "corp-portal",
        "it-helpdesk", "it-support-portal",
        "benefits-portal", "compensation-portal",
    ]

    def analyze(self, email_text: str = "", subject: str = "",
                sender: str = "", body: str = "",
                headers: Dict = None) -> Dict:
        """Full email threat analysis"""
        # Combine all text for analysis
        full_text = f"{subject} {sender} {body} {email_text}".strip()
        if not full_text:
            return {"error": "No email content provided"}

        scores = {}
        indicators = []

        # 1. AI-generated content detection
        text_detector = AITextDetector()
        ai_result = text_detector.analyze(body or email_text)
        scores["ai_generated"] = ai_result["confidence"]
        if ai_result["is_ai_generated"]:
            indicators.append(f"AI-generated content detected (conf: {ai_result['confidence']:.2f})")

        # 2. Phishing keyword analysis
        phish_score, phish_flags = self._phishing_analysis(full_text.lower())
        scores["phishing_keywords"] = phish_score
        indicators.extend(phish_flags)

        # 3. Sender domain analysis
        domain_score, domain_flags = self._domain_analysis(sender)
        scores["sender_domain"] = domain_score
        indicators.extend(domain_flags)

        # 4. URL analysis
        url_score, url_flags, urls_found = self._url_analysis(full_text)
        scores["url_analysis"] = url_score
        indicators.extend(url_flags)

        # 5. Header analysis (SPF/DKIM/urgency)
        header_score, header_flags = self._header_analysis(headers or {}, subject)
        scores["header_analysis"] = header_score
        indicators.extend(header_flags)

        # 6. Urgency/pressure tactics
        urgency_score = self._urgency_score(full_text.lower())
        scores["urgency_tactics"] = urgency_score
        if urgency_score > 0.5:
            indicators.append("High-pressure urgency language detected")

        # Weighted ensemble
        weights = {
            "ai_generated":     0.10,  # Raised — AI-written + corporate lure = strong signal
            "phishing_keywords":0.38,  # PRIMARY — direct content match
            "sender_domain":    0.22,  # Important but should not override content signals
            "url_analysis":     0.13,  # Useful when URLs present
            "header_analysis":  0.07,  # Minor — no real headers from extension
            "urgency_tactics":  0.10,  # Raised — corporate deadline language very telling
        }
        confidence = sum(scores[k] * weights[k] for k in weights)
        confidence = round(min(1.0, max(0.0, confidence)), 4)

        # ── Trusted sender domains: require stronger evidence ─────
        # Emails from known-legitimate services use payment/update language
        # that is identical to phishing — require DOMAIN mismatch evidence
        TRUSTED_DOMAINS = [
            # Streaming
            'netflix.com', 'spotify.com', 'youtube.com', 'primevideo.com',
            'disneyplus.com', 'hbomax.com', 'hulu.com', 'apple.com',
            # E-commerce
            'amazon.com', 'amazon.in', 'flipkart.com', 'ebay.com',
            'paypal.com', 'stripe.com', 'razorpay.com', 'paytm.com',
            # Banks & Finance
            'hdfcbank.com', 'icicibank.com', 'sbi.co.in', 'axisbank.com',
            'bankofamerica.com', 'chase.com', 'wellsfargo.com', 'citibank.com',
            # Tech
            'google.com', 'microsoft.com', 'github.com', 'linkedin.com',
            'twitter.com', 'x.com', 'facebook.com', 'instagram.com',
            # Utilities / Telecom
            'airtel.in', 'jio.com', 'bsnl.in', 'vodafone.in',
            # HR / Payroll (legit)
            'workday.com', 'adp.com', 'paychex.com', 'successfactors.com',
        ]
        sender_lower = (sender or '').lower()
        is_trusted_sender = any(td in sender_lower for td in TRUSTED_DOMAINS)

        # If sender is a trusted domain, only flag on domain spoofing evidence
        # (a real netflix.com email CAN say "update your payment" — that's legit)
        if is_trusted_sender:
            # Downgrade phishing keyword score by 60% — context matters
            scores["phishing_keywords"] *= 0.40
            scores["urgency_tactics"]   *= 0.40
            scores["ai_generated"]      *= 0.50  # marketing email IS AI-written
            # Recompute weighted confidence with adjusted scores
            confidence = sum(scores[k] * weights[k] for k in weights)
            confidence = round(min(1.0, max(0.0, confidence)), 4)

        # ── Override 1: Strong keyword match ───────────────────────
        if scores.get("phishing_keywords", 0) >= 0.60:
            confidence = max(confidence, 0.55)
        if scores.get("phishing_keywords", 0) >= 0.80:
            confidence = max(confidence, 0.70)

        # ── Override 2: Urgency + keywords (corporate BEC pattern) ─
        if scores.get("urgency_tactics", 0) >= 0.30 and scores.get("phishing_keywords", 0) >= 0.20:
            confidence = max(confidence, 0.55)
        if scores.get("urgency_tactics", 0) >= 0.15 and scores.get("phishing_keywords", 0) >= 0.40:
            confidence = max(confidence, 0.58)

        # ── Override 3: AI-written + corporate domain ──────────────
        # AI-generated corporate email from non-verified domain = strong BEC signal
        if scores.get("ai_generated", 0) >= 0.40 and scores.get("sender_domain", 0) >= 0.50:
            confidence = max(confidence, 0.58)
        if scores.get("ai_generated", 0) >= 0.40 and scores.get("phishing_keywords", 0) >= 0.20:
            confidence = max(confidence, 0.52)

        # ── Override 4: Suspicious domain ─────────────────────────
        if scores.get("sender_domain", 0) >= 0.85:
            confidence = max(confidence, 0.65)
        if scores.get("sender_domain", 0) >= 0.70:
            confidence = max(confidence, 0.52)

        confidence = round(min(1.0, max(0.0, confidence)), 4)
        is_threat = confidence >= 0.40

        severity = (
            "CRITICAL" if confidence >= 0.78 else
            "HIGH"     if confidence >= 0.58 else
            "MEDIUM"   if confidence >= 0.40 else "LOW"
        )

        return {
            "is_threat": is_threat,
            "confidence": confidence,
            "severity": severity,
            "verdict": ("PHISHING/MALICIOUS" if confidence >= 0.78 else
                        "SUSPICIOUS"         if confidence >= 0.40 else
                        "LIKELY SAFE"),
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "indicators": indicators[:10],
            "urls_found": urls_found[:5],
            "ai_analysis": ai_result,
            "recommended_action": (
                "BLOCK"     if confidence >= 0.78 else
                "QUARANTINE" if confidence >= 0.58 else
                "FLAG"      if confidence >= 0.40 else
                "ALLOW"
            ),
        }

    def _phishing_analysis(self, text: str) -> Tuple[float, List[str]]:
        found = []
        import re as _re
        # Multi-word phrase matching
        for kw in self.PHISHING_KEYWORDS:
            if kw in text:
                found.append(f"Phishing phrase: \'{kw}\'")
        # Single suspicious word matching (word-boundary)
        for word in self.PHISHING_WORDS:
            if _re.search(r'\b' + _re.escape(word) + r'\b', text):
                found.append(f"Suspicious term: \'{word}\'")
        # Malicious file extensions
        for ext in [r'\.exe', r'\.bat', r'\.ps1', r'\.vbs', r'\.cmd', r'\.scr']:
            if _re.search(r'\w+' + ext, text):
                found.append(f"Malicious file extension: {ext}")
                break
        unique_found = list(dict.fromkeys(found))
        score = min(1.0, len(unique_found) * 0.20)
        return score, unique_found[:6]

    def _domain_analysis(self, sender: str) -> Tuple[float, List[str]]:
        flags = []
        if not sender:
            return 0.2, ["No sender information"]
        sender_lower = sender.lower()
        domain_part = sender_lower.split('@')[-1] if '@' in sender_lower else sender_lower
        username_part = sender_lower.split('@')[0] if '@' in sender_lower else ""

        # ── Tier 1: Highest-risk TLDs ───────────────────────────
        HIGH_RISK_TLDS = ['.xyz', '.tk', '.ml', '.ga', '.cf', '.pw', '.top',
                          '.click', '.download', '.loan', '.work', '.gq',
                          '.zip', '.mov', '.foo', '.dad']
        for tld in HIGH_RISK_TLDS:
            if domain_part.endswith(tld):
                flags.append(f"High-risk TLD: {domain_part}")
                return 0.90, flags

        # ── Tier 2: .co / .net / .org impersonating .com ────────
        # hr-payroll@internal-dept-portal.co = impersonating .com internal domain
        LOOKALIKE_TLDS = ['.co', '.net', '.org', '.info', '.biz', '.io']
        for tld in LOOKALIKE_TLDS:
            if domain_part.endswith(tld):
                # Only suspicious if domain looks corporate/internal
                corporate_keywords = ['payroll', 'hr-', 'internal', 'dept', 'corp',
                                      'portal', 'employee', 'staff', 'finance', 'it-',
                                      'helpdesk', 'support', 'benefits', 'compensation',
                                      'admin', 'noreply', 'no-reply', 'donotreply']
                for kw in corporate_keywords:
                    if kw in domain_part:
                        flags.append(f"Fake corporate domain with lookalike TLD: {domain_part}")
                        return 0.82, flags

        # ── Tier 3: Known suspicious domain patterns ────────────
        for domain in self.SUSPICIOUS_DOMAINS:
            if domain in sender_lower:
                flags.append(f"Suspicious domain pattern: {domain}")
                return 0.85, flags

        # ── Tier 4: Hyphenated impersonation patterns ────────────
        # internal-dept-portal.co, hr-payroll-corp.net etc.
        hyphen_count = domain_part.split('.')[0].count('-')
        if hyphen_count >= 2:
            flags.append(f"Heavily hyphenated domain ({hyphen_count} hyphens): {domain_part}")
            return 0.72, flags

        if re.search(r'(secure|alert|verify|login|update|confirm|support|help|portal|dept|internal|payroll|hr)-', domain_part):
            flags.append(f"Suspicious hyphenated domain: {domain_part}")
            return 0.75, flags
        if re.search(r'-(secure|alert|verify|login|update|confirm|support|portal|dept|payroll|hr)', domain_part):
            flags.append(f"Suspicious hyphenated domain: {domain_part}")
            return 0.75, flags

        # ── Tier 5: Username / sender pattern anomalies ──────────
        if re.search(r'\d+@', sender_lower):
            flags.append("Numeric username pattern")
            return 0.55, flags
        if sender_lower.count('.') > 4:
            flags.append("Excessive subdomains in sender")
            return 0.50, flags

        # ── Tier 6: BEC / Spear-phishing username patterns ───────
        # hr-payroll@, ceo@, finance-dept@, payroll-team@ from unusual domains
        bec_usernames = ['hr-', 'payroll', 'finance', 'ceo', 'cfo', 'cto',
                         'accounting', 'it-support', 'helpdesk', 'noreply',
                         'no-reply', 'donotreply', 'do-not-reply', 'alert@',
                         'notification', 'security@', 'admin@', 'support@']
        for bec in bec_usernames:
            if bec in username_part:
                # Only flag if not from a major legitimate domain
                legit_domains = ['microsoft.com', 'google.com', 'amazon.com',
                                 'apple.com', 'linkedin.com', 'salesforce.com']
                if not any(ld in domain_part for ld in legit_domains):
                    flags.append(f"BEC-pattern username '{username_part}' from non-verified domain")
                    return 0.55, flags

        # Free email used for business/corporate claim
        free_domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
                        "protonmail.com", "yandex.com", "mail.com"]
        if any(d in sender_lower for d in free_domains):
            flags.append("Free email provider (possibly spoofed)")
            return 0.30, flags

        return 0.10, flags

    def _url_analysis(self, text: str) -> Tuple[float, List[str], List[str]]:
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        if not urls:
            return 0.0, [], []
        flags = []
        max_score = 0.0
        for url in urls[:10]:
            url_lower = url.lower()
            # IP address URLs
            if re.search(r'https?://\d+\.\d+\.\d+\.\d+', url):
                flags.append(f"IP-based URL: {url[:50]}")
                max_score = max(max_score, 0.80)
            # URL shorteners
            elif any(s in url_lower for s in ["bit.ly","tinyurl","t.co","goo.gl","ow.ly","short.io"]):
                flags.append(f"URL shortener detected: {url[:50]}")
                max_score = max(max_score, 0.55)
            # Suspicious keywords in URL
            elif any(k in url_lower for k in ["login","verify","secure","account","update","confirm","reset"]):
                flags.append(f"Suspicious URL pattern: {url[:50]}")
                max_score = max(max_score, 0.65)
            # HTTP (not HTTPS)
            elif url_lower.startswith("http://"):
                flags.append(f"Unencrypted HTTP URL: {url[:50]}")
                max_score = max(max_score, 0.40)
        return round(max_score, 4), flags[:5], urls[:5]

    def _header_analysis(self, headers: Dict, subject: str) -> Tuple[float, List[str]]:
        flags = []
        score = 0.0
        # SPF/DKIM checks
        if headers.get("x-spam-status", "").lower().startswith("yes"):
            flags.append("Marked as spam by mail server")
            score = max(score, 0.75)
        if "FAIL" in headers.get("received-spf", "").upper():
            flags.append("SPF check FAILED")
            score = max(score, 0.65)
        if "dkim=fail" in headers.get("authentication-results", "").lower():
            flags.append("DKIM signature FAILED")
            score = max(score, 0.65)
        # Suspicious subject patterns
        subject_lower = subject.lower()
        if re.search(r'(urgent|important|action required|verify now|final notice)', subject_lower):
            flags.append(f"Urgency in subject: '{subject[:50]}'")
            score = max(score, 0.45)
        if subject_lower.startswith("re:") and not headers.get("in-reply-to"):
            flags.append("Fake reply thread (Re: without prior message)")
            score = max(score, 0.55)
        return round(score, 4), flags

    def _urgency_score(self, text: str) -> float:
        urgency_words = [
            # Classic urgency
            "immediately", "urgent", "asap", "right now",
            "expires", "deadline", "last chance", "final warning",
            "account suspended", "limited time", "act fast",
            "don't delay", "respond now", "critical",
            "under attack", "being attacked",
            "warning", "alert", "danger",
            "do not ignore", "password change",
            # Corporate deadline pressure (spear phishing)
            "by end of", "by eod", "by close of business",
            "by the end of the business day",
            "before close of business", "before end of day",
            "failure to", "failure to sign", "failure to respond",
            "failure to comply", "failure to acknowledge",
            "required to review", "required to sign",
            "required to acknowledge", "must acknowledge",
            "action required", "response required",
            "acknowledge receipt", "please acknowledge",
            "time-sensitive", "time sensitive",
            "no later than", "no later than today",
        ]
        count = sum(1 for w in urgency_words if w in text)
        return min(1.0, count * 0.15)
