"""
Discord Mosaic TTS — Shared Utilities
=======================================
Common functions used by multiple scripts.

Includes techniques ported from Nia's voice stack:
- Whisper hallucination detection  (voice-client.js)
- Emotional transcript enhancement (emotional-transcript-processor.js)
- Discord audio post-processing    (text-to-speech.js discordMode)
- Noise filtering                  (voice-client.js)
"""

import re
import os
import shutil
import subprocess
import numpy as np
from sentence_transformers import SentenceTransformer

import config

# ──────────────────────────────────────────────
#  Embedding model (loaded once, shared)
# ──────────────────────────────────────────────
_embed_model = None


def get_embed_model() -> SentenceTransformer:
    """Lazy-load the sentence-transformer model."""
    global _embed_model
    if _embed_model is None:
        print(f"Loading embedding model '{config.EMBEDDING_MODEL}'...")
        _embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embed_model


def encode_text(text: str) -> np.ndarray:
    """Encode a string into a semantic embedding vector."""
    return get_embed_model().encode(text)


# ──────────────────────────────────────────────
#  Cosine similarity
# ──────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ──────────────────────────────────────────────
#  Semantic beat splitting
# ──────────────────────────────────────────────
def split_semantic_beats(text: str) -> list[str]:
    """
    Split a message into semantic 'beats' — meaningful sub-phrases
    separated by punctuation or common conjunctions.
    """
    split_regex = r'[.!?;]|\s+but\s+|\s+and\s+|\s+so\s+|\s+then\s+'
    beats = [b.strip() for b in re.split(split_regex, text) if b.strip()]
    return beats if beats else [text.strip()]


# ──────────────────────────────────────────────
#  Discord log reader
# ──────────────────────────────────────────────
def read_new_messages(log_file: str, last_pos: int) -> tuple[list[str], int]:
    """
    Read new lines from the Discord export log since `last_pos`.
    Returns (list_of_messages, new_file_position).
    """
    messages = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            f.seek(last_pos)
            for line in f:
                match = re.match(r"\[\d{2}:\d{2}:\d{2}\]\s*(.+)", line)
                if match:
                    messages.append(match.group(1).strip())
            last_pos = f.tell()
    except FileNotFoundError:
        pass  # file hasn't been created yet — that's fine
    return messages, last_pos


# ──────────────────────────────────────────────
#  Clip matching
# ──────────────────────────────────────────────
def find_best_clips(
    phrase_text: str,
    clips: list[dict],
    top_n: int = 1,
    min_similarity: float = None,
) -> list[dict]:
    """
    Find the top-N clips most semantically similar to `phrase_text`.
    Each clip dict must have an 'embedding' key.
    """
    if min_similarity is None:
        min_similarity = config.SEMANTIC_THRESHOLD

    phrase_emb = encode_text(phrase_text)
    scored = []
    for clip in clips:
        sim = cosine_similarity(phrase_emb, clip["embedding"])
        scored.append((sim, clip))
    scored.sort(reverse=True, key=lambda x: x[0])

    return [clip for sim_val, clip in scored if sim_val >= min_similarity][:top_n]


# ══════════════════════════════════════════════
#  PORTED FROM NIA'S VOICE STACK
# ══════════════════════════════════════════════


# ──────────────────────────────────────────────
#  Whisper Hallucination Detection
#  (from Nia's voice-client.js → isHallucination)
# ──────────────────────────────────────────────

# Known Whisper hallucination patterns — these are actual glitches
# where Whisper invents text from noise/silence
_HALLUCINATION_PATTERNS = [
    re.compile(r"(\bha\s*){5,}", re.IGNORECASE),
    re.compile(r"(\bhaha\s*){4,}", re.IGNORECASE),
    re.compile(r"(\blol\s*){4,}", re.IGNORECASE),
    re.compile(r"(\bum\s*){5,}", re.IGNORECASE),
    re.compile(r"(\buh\s*){5,}", re.IGNORECASE),
    re.compile(r"thank you(\.|\s)*thank you(\.|\s)*thank you", re.IGNORECASE),
    re.compile(r"please subscribe", re.IGNORECASE),
    re.compile(r"like and subscribe", re.IGNORECASE),
    re.compile(r"see you in the next", re.IGNORECASE),
    re.compile(r"\[music\](\s*\[music\])+", re.IGNORECASE),
    re.compile(r"♪+"),
]


def is_whisper_hallucination(text: str) -> bool:
    """
    Detect if a Whisper transcript is a hallucination.
    Ported from Nia's voice-client.js isHallucination().

    Common hallucinations include:
    - Repeated filler words ("um um um um um")
    - YouTube outros ("please subscribe", "see you in the next")
    - Music note spam
    - Long repeated patterns
    """
    if not text or len(text) < 50:
        return False

    # Check known patterns
    for pattern in _HALLUCINATION_PATTERNS:
        if pattern.search(text):
            return True

    # Check for repeated LONG patterns (8+ chars repeated 3+ times)
    # Catches actual loops but not natural repetition like "no no no"
    for length in range(8, 31):
        for i in range(len(text) - (length * 3)):
            chunk = text[i : i + length]
            next1 = text[i + length : i + length * 2]
            next2 = text[i + length * 2 : i + length * 3]
            if chunk == next1 == next2:
                return True

    # Very long transcripts from short clips are suspicious
    if len(text) > 500:
        return True

    return False


def clean_hallucination(text: str) -> str:
    """
    Try to salvage useful text from a hallucinated transcript.
    Ported from Nia's voice-client.js cleanHallucination().
    """
    for length in range(2, 21):
        for i in range(len(text) - (length * 2)):
            chunk = text[i : i + length]
            next_chunk = text[i + length : i + length * 2]
            if chunk == next_chunk:
                cleaned = text[:i].strip()
                if len(cleaned) > 0:
                    return cleaned
    return text[:100].strip()


# ──────────────────────────────────────────────
#  Emotional Transcript Enhancement
#  (from Nia's emotional-transcript-processor.js)
# ──────────────────────────────────────────────

# Vocal sound patterns → replacement labels
_VOCAL_SOUNDS = {
    "laughter": {
        "patterns": [
            re.compile(r"\b(ha+h[ah]+|he+h[eh]+|hehe+|haha+|hah+)\b", re.IGNORECASE),
            re.compile(r"\b(ahah+|eheh+|ihih+|ohoh+)\b", re.IGNORECASE),
        ],
        "replacement": "*laughs*",
    },
    "giggle": {
        "patterns": [
            re.compile(r"\b(tehe+|teehee+|hihihi+)\b", re.IGNORECASE),
        ],
        "replacement": "*giggles*",
    },
    "sigh": {
        "patterns": [
            re.compile(r"\b(sigh+s?)\b", re.IGNORECASE),
        ],
        "replacement": "*sighs*",
    },
    "gasp": {
        "patterns": [
            re.compile(r"\b(oh+!|ah+!|whoa+!)\b", re.IGNORECASE),
        ],
        "replacement": "*gasps*",
    },
    "groan": {
        "patterns": [
            re.compile(r"\b(ugh+|urgh+|argh+)\b", re.IGNORECASE),
        ],
        "replacement": "*groans*",
    },
}

# Words to emphasize (from addExpressionEmphasis)
_EMPHASIS_WORDS = [
    "really", "very", "so", "totally", "absolutely",
    "never", "always", "definitely", "exactly",
]


def enhance_transcript(text: str) -> str:
    """
    Enhance a Whisper transcript with emotional markers.
    Ported from Nia's emotional-transcript-processor.js.

    - Detects laughter, sighs, gasps, groans
    - Preserves drawn-out words (Noooo, Niiaaaa)
    - Cleans up punctuation
    """
    if not text:
        return text

    enhanced = text

    # Detect vocal sounds
    for _sound_type, cfg in _VOCAL_SOUNDS.items():
        for pattern in cfg["patterns"]:
            enhanced = pattern.sub(cfg["replacement"], enhanced)

    # Clean up spacing/punctuation
    enhanced = re.sub(r"\s+", " ", enhanced)
    enhanced = re.sub(r"([.!?,])([A-Z])", r"\1 \2", enhanced)
    enhanced = re.sub(r"\.{2,}(?!\.\s)", "...", enhanced)
    enhanced = re.sub(r"!{2,}", "!", enhanced)
    enhanced = re.sub(r"\?{2,}", "?", enhanced)

    return enhanced.strip()


# ──────────────────────────────────────────────
#  Prosody-Aware Whisper Prompt
#  (from Nia's speech-to-text.js _buildPromptWithProsody)
# ──────────────────────────────────────────────

def build_whisper_prompt() -> str:
    """
    Build a Whisper prompt that helps capture drawn-out words,
    laughter, filler words, and emphasis.
    Ported from Nia's speech-to-text.js.
    """
    parts = [
        # Examples of drawn-out words and emphasis
        "Niiaaaa, pleeeease, nooo, yesss, I looove this, that's sooo cool!",
        # Filler words
        "Umm, ahh, hmm, uhh, like, you know...",
        # Non-speech sounds
        "*laughs* Hahaha, hehe, that's hilarious!",
        "*sighs* Ahh... umm... I don't know...",
        # Instruction
        "Preserve all drawn-out words, laughter, sighs, and emotional emphasis exactly as spoken.",
    ]
    return " ".join(parts)


# ──────────────────────────────────────────────
#  Discord Audio Post-Processing
#  (from Nia's text-to-speech.js discordMode)
# ──────────────────────────────────────────────

_ffmpeg_available = None


def is_ffmpeg_available() -> bool:
    """Check if FFmpeg is on PATH."""
    global _ffmpeg_available
    if _ffmpeg_available is None:
        _ffmpeg_available = shutil.which("ffmpeg") is not None
    return _ffmpeg_available


def apply_discord_audio_processing(input_path: str, output_path: str) -> bool:
    """
    Apply Discord-like audio processing to a clip via FFmpeg.
    Ported from Nia's text-to-speech.js discordMode.

    Makes clips sound less "studio perfect" — like they're coming
    through a Discord voice channel:
    - Gentle high-pass filter (removes sub-bass rumble)
    - Slight compression (evens out volume)
    - Bandwidth limiting (simulates Opus codec)
    - Tiny bit of noise (prevents uncanny-valley silence)

    Returns True if processing succeeded, False otherwise.
    """
    if not is_ffmpeg_available():
        return False

    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", ",".join([
                "highpass=f=80",                              # remove sub-bass
                "acompressor=threshold=-20dB:ratio=3:attack=5:release=50",  # gentle compression
                "lowpass=f=7500",                             # bandwidth limit (Opus-like)
                "anlmdn=s=0.0001",                            # barely perceptible noise
                "volume=0.95",                                # slight level reduction
            ]),
            "-ar", "48000",   # Discord's sample rate
            "-ac", "2",       # Stereo
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ──────────────────────────────────────────────
#  Noise / Quality Filtering
#  (from Nia's voice-client.js noise filtering)
# ──────────────────────────────────────────────

def is_clip_too_quiet(audio_segment) -> bool:
    """
    Check if a pydub AudioSegment is below the minimum energy threshold.
    Filters out near-silent clips that would just be dead air.
    """
    return audio_segment.dBFS < config.MIN_CLIP_ENERGY_DB


def is_transcript_too_short(text: str) -> bool:
    """
    Check if a transcript is too short to be meaningful.
    Ported from Nia's minTranscriptLength noise filter.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", text)
    return len(cleaned) < config.MIN_TRANSCRIPT_LENGTH
