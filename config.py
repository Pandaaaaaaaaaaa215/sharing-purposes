"""
Discord Mosaic TTS — Configuration
===================================
Edit this file to match your setup. All other scripts import from here.
"""

import os

# ──────────────────────────────────────────────
#  Paths  (change these to match your machine)
# ──────────────────────────────────────────────
BASE_DIR = r"C:\Users\Benja\Downloads\discord-mosaic-tts (1)\discord-mosaic-tts"

# Where raw audio clips live (mp3, wav, ogg, etc.)
RAW_CLIPS_DIR = r"C:\Users\Benja\Downloads\discord-mosaic-tts (1)\discord-mosaic-tts\raw_wavs"

# Generated micro-clips metadata (auto-created by clip_parser.py)
MICRO_CLIPS_JSON = r"C:\Users\Benja\Downloads\discord-mosaic-tts (1)\discord-mosaic-tts\clips\micro_clips.json"

# Directory for the sliced clip audio files
MICRO_CLIPS_DIR = r"C:\Users\Benja\Downloads\discord-mosaic-tts (1)\discord-mosaic-tts\clips"

# Discord message log (written by read_messages.py, read by main.py)
DISCORD_LOG_FILE = r"C:\Users\Benja\Downloads\discord-mosaic-tts (1)\discord-mosaic-tts\discord_export.txt"

# Semantic coverage report output
COVERAGE_REPORT = os.path.join(BASE_DIR, "semantic_coverage_report.json")

# ──────────────────────────────────────────────
#  Audio Devices
# ──────────────────────────────────────────────
VIRTUAL_CABLE_ID = 12       # Virtual cable device ID (for Discord input)
PC_OUTPUT_ID = None          # Default speakers (None = system default)

# ──────────────────────────────────────────────
#  Discord Bot
# ──────────────────────────────────────────────
BOT_TOKEN = ""                              # Your bot token (keep secret!)
MY_USER_ID = 537318854708428820             # Your Discord user ID
MY_CHANNEL_ID = 1467660677538775296         # Astro Lounge VC text chat

# ──────────────────────────────────────────────
#  Clip Parser (clip_parser.py)
# ──────────────────────────────────────────────
WHISPER_MODEL = "base"          # tiny | base | small | medium | large
MIN_CLIP_DURATION = 0.3         # seconds — ignore clips shorter than this
MAX_CLIP_DURATION = 8.0         # seconds — split segments longer than this

# Whisper transcription quality
WHISPER_TEMPERATURE = 0.2       # 0.2 captures emphasis/drawn-out words better
WHISPER_LANGUAGE = "en"

# Phrase-level slicing — creates tight word-group clips for better matching
# e.g. "oh yeah it's all coming together" → "oh yeah", "it's all", "coming together"
PHRASE_SLICING = True           # enable word-level phrase clips
MAX_PHRASE_WORDS = 4            # max words per phrase clip (1-4 word n-grams)

# Hallucination filtering (ported from Nia's voice-client.js)
FILTER_HALLUCINATIONS = True    # auto-detect and discard Whisper hallucinations

# Emotional transcript enhancement (ported from Nia's emotional-transcript-processor.js)
ENHANCE_TRANSCRIPTS = True      # detect laughter, sighs, emphasis in transcripts

# ──────────────────────────────────────────────
#  Semantic Engine
# ──────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SEMANTIC_THRESHOLD = 0.6        # minimum similarity to count as a match
MULTI_CLIP_MODE = True          # play multiple clips per beat?
MAX_CLIPS_PER_BEAT = 3          # max clips to chain per semantic beat

# ──────────────────────────────────────────────
#  Audio Processing (ported from Nia's text-to-speech.js discordMode)
# ──────────────────────────────────────────────
DISCORD_AUDIO_MODE = True       # FFmpeg post-processing for natural VC sound
# Makes clips sound less "studio perfect" — subtle compression,
# slight bandwidth limiting, gentle noise to mimic Discord codec.
# Requires FFmpeg on PATH.

# ──────────────────────────────────────────────
#  Noise Filtering (ported from Nia's voice-client.js)
# ──────────────────────────────────────────────
MIN_TRANSCRIPT_LENGTH = 4       # ignore clips with fewer characters of text
MIN_CLIP_ENERGY_DB = -45        # ignore clips quieter than this (dBFS)

# ──────────────────────────────────────────────
#  Timing
# ──────────────────────────────────────────────
CHECK_INTERVAL = 1.0            # seconds between polling for new messages
MAX_PLAYBACK_DURATION = 6.0     # seconds — cut clips longer than this during playback
FADE_OUT_MS = 300               # milliseconds — fade out when cutting a clip short

# ──────────────────────────────────────────────
#  Helper / Diagnostics
# ──────────────────────────────────────────────
UPDATE_GRAPH = True             # live matplotlib histogram in helper.py
HELPER_THRESHOLD = 0.75         # stricter threshold for coverage analysis
