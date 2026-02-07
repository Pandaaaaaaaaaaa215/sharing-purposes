"""
Discord Mosaic TTS — Main Engine
==================================
Polls the Discord message log, splits messages into semantic beats,
finds the best-matching audio clips, and plays them on both your
speakers and a virtual audio cable (so Discord can hear it).

Enhancements from Nia's voice stack:
- Discord audio post-processing for natural VC sound
- Enhanced semantic matching against emotionally-tagged transcripts

Usage:
    python main.py
"""

import os
import sys
import json
import time
import tempfile

import numpy as np
import sounddevice as sd
from pydub import AudioSegment

import config
import utils


# ──────────────────────────────────────────────
#  Load clip catalog + precompute embeddings
# ──────────────────────────────────────────────
def load_clips() -> list[dict]:
    """Load micro_clips.json and precompute embeddings for each clip."""
    if not os.path.exists(config.MICRO_CLIPS_JSON):
        print(f"⚠ Clip catalog not found: {config.MICRO_CLIPS_JSON}")
        print(f"  Run 'python clip_parser.py' first to generate it.")
        sys.exit(1)

    with open(config.MICRO_CLIPS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    clips = data.get("clips", [])
    if not clips:
        print("⚠ Clip catalog is empty. Add audio files and run clip_parser.py.")
        sys.exit(1)

    # Show catalog metadata
    filters = data.get("filters", {})
    if filters:
        active = [k for k, v in filters.items() if v is True]
        if active:
            print(f"Catalog filters: {', '.join(active)}")

    print(f"Loaded {len(clips)} clips from catalog")
    print("Precomputing embeddings...")
    for clip in clips:
        clip["embedding"] = utils.encode_text(clip["text"])
    print(f"Embeddings ready ✔")

    return clips


# ──────────────────────────────────────────────
#  Dual-output audio playback
# ──────────────────────────────────────────────
def play_clip_dual(clip_audio: AudioSegment):
    """
    Play a clip on both PC speakers and the virtual audio cable simultaneously.
    """
    samples = np.array(clip_audio.get_array_of_samples()).astype(np.float32)
    samples /= np.iinfo(clip_audio.array_type).max  # normalize to [-1.0, 1.0]

    if clip_audio.channels == 2:
        samples = samples.reshape((-1, 2))
    else:
        samples = samples.reshape((-1, 1))

    sr = clip_audio.frame_rate

    # Play on BOTH outputs simultaneously, then wait
    try:
        sd.play(samples, samplerate=sr, device=config.PC_OUTPUT_ID, blocking=False)

        if config.VIRTUAL_CABLE_ID is not None:
            with sd.OutputStream(
                samplerate=sr,
                channels=samples.shape[1],
                device=config.VIRTUAL_CABLE_ID,
            ) as stream:
                stream.write(samples)

        sd.wait()

    except sd.PortAudioError as e:
        print(f"    ⚠ Audio device error: {e}")
        print(f"    Falling back to default output only...")
        sd.play(samples, samplerate=sr, blocking=True)


# ──────────────────────────────────────────────
#  Process a single message
# ──────────────────────────────────────────────
def process_message(msg: str, clips: list[dict]):
    """Split a message into beats and play matching clips."""
    print(f"\n  Message: '{msg}'")

    beats = utils.split_semantic_beats(msg)
    for beat in beats:
        print(f"    Beat: '{beat}'")

        top_n = config.MAX_CLIPS_PER_BEAT if config.MULTI_CLIP_MODE else 1
        selected = utils.find_best_clips(beat, clips, top_n=top_n)

        # Fallback: if nothing matched, grab the single best regardless
        if not selected:
            print(f"      No match above threshold ({config.SEMANTIC_THRESHOLD})")
            selected = utils.find_best_clips(beat, clips, top_n=1, min_similarity=0)

        for clip in selected:
            clip_path = os.path.join(config.MICRO_CLIPS_DIR, clip["clip_file"])
            if not os.path.exists(clip_path):
                print(f"      ⚠ Missing file: {clip['clip_file']}")
                continue

            clip_audio = AudioSegment.from_file(clip_path)
            dur = clip_audio.duration_seconds
            print(f"      ▶ '{clip['clip_file']}' ({dur:.1f}s) — \"{clip['text'][:50]}\"")
            play_clip_dual(clip_audio)


# ──────────────────────────────────────────────
#  Main loop
# ──────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  Discord Mosaic TTS — Main Engine")
    print("=" * 50)

    # Check FFmpeg for discord audio mode
    if config.DISCORD_AUDIO_MODE:
        if utils.is_ffmpeg_available():
            print("Discord audio mode: ON (FFmpeg available)")
        else:
            print("⚠ FFmpeg not found — Discord audio mode disabled")

    clips = load_clips()

    print(f"\nPolling '{config.DISCORD_LOG_FILE}' every {config.CHECK_INTERVAL}s...")
    print("Press Ctrl+C to stop.\n")

    # Skip to end of existing log — only process NEW messages
    last_pos = 0
    try:
        if os.path.exists(config.DISCORD_LOG_FILE):
            last_pos = os.path.getsize(config.DISCORD_LOG_FILE)
            print(f"Skipped to end of log ({last_pos} bytes). Waiting for new messages...")
    except OSError:
        pass

    try:
        while True:
            messages, last_pos = utils.read_new_messages(config.DISCORD_LOG_FILE, last_pos)
            if messages:
                print(f"── {len(messages)} new message(s) ──")
            for msg in messages:
                process_message(msg, clips)
            time.sleep(config.CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nStopping Mosaic TTS. ✔")


if __name__ == "__main__":
    main()
