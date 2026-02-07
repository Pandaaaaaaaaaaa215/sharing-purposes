"""
Discord Mosaic TTS — Clip Parser
==================================
Scans a folder of audio files, transcribes them with OpenAI Whisper,
and slices them into micro-clips using Whisper's own segment/word
boundaries. Produces a micro_clips.json catalog.

Enhancements from Nia's voice stack:
- Prosody-aware Whisper prompting (better transcriptions)
- Hallucination detection & filtering
- Emotional transcript enhancement (laughter, sighs, emphasis)
- Noise filtering (skip too-quiet or too-short clips)

Usage:
    python clip_parser.py                          # process clips/raw → clips/sliced
    python clip_parser.py --input ./my_audio       # custom input folder
    python clip_parser.py --model small             # use a larger Whisper model
    python clip_parser.py --force                   # re-process everything

Requires: pip install openai-whisper pydub
          (ffmpeg must be on PATH)
"""

import os
import sys
import json
import argparse
import hashlib
from datetime import datetime

import whisper
from pydub import AudioSegment

import config
import utils


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────
def file_hash(path: str) -> str:
    """Quick MD5 of a file for change detection."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    return "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip()


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS.mmm."""
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:06.3f}"


# ──────────────────────────────────────────────
#  Split a long Whisper segment using word times
# ──────────────────────────────────────────────
def split_segment_by_words(segment: dict, max_dur: float) -> list[dict]:
    """
    If a Whisper segment exceeds max_dur, split it into sub-segments
    using word-level timestamps. Groups words into chunks that stay
    under max_dur each.
    """
    words = segment.get("words", [])
    if not words:
        return [segment]

    sub_segments = []
    current_words = []
    chunk_start = words[0]["start"]

    for w in words:
        if current_words and (w["end"] - chunk_start) > max_dur:
            sub_segments.append({
                "start": chunk_start,
                "end": current_words[-1]["end"],
                "text": " ".join(cw["word"].strip() for cw in current_words),
            })
            current_words = []
            chunk_start = w["start"]
        current_words.append(w)

    if current_words:
        sub_segments.append({
            "start": chunk_start,
            "end": current_words[-1]["end"],
            "text": " ".join(cw["word"].strip() for cw in current_words),
        })

    return sub_segments


def create_phrase_clips(segment: dict, phrase_size: int = 3) -> list[dict]:
    """
    Create overlapping phrase-level clips from a segment's word timestamps.
    For example with phrase_size=3, "oh yeah it's all coming together" becomes:
      - "oh yeah it's"
      - "yeah it's all"
      - "it's all coming"
      - "all coming together"
    Plus single words and pairs for short matches.

    This gives the semantic matcher much tighter clips to work with.
    """
    words = segment.get("words", [])
    if not words:
        return []

    phrases = []

    # Generate n-grams from 1 up to phrase_size
    for n in range(1, phrase_size + 1):
        for i in range(len(words) - n + 1):
            group = words[i:i + n]
            text = " ".join(w["word"].strip() for w in group)
            dur = group[-1]["end"] - group[0]["start"]

            # Skip single words shorter than 0.2s (just noise)
            if n == 1 and dur < 0.2:
                continue
            # Skip very short text
            if len(text.strip()) < 2:
                continue

            phrases.append({
                "start": group[0]["start"],
                "end": group[-1]["end"],
                "text": text,
            })

    return phrases


# ──────────────────────────────────────────────
#  Core: transcribe + slice a single audio file
# ──────────────────────────────────────────────
def process_audio_file(
    filepath: str,
    model: whisper.Whisper,
    output_dir: str,
    min_dur: float = config.MIN_CLIP_DURATION,
    max_dur: float = config.MAX_CLIP_DURATION,
) -> list[dict]:
    """
    Transcribe an audio file with Whisper and slice it into micro-clips.
    Uses Nia-ported enhancements for quality filtering.
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]
    print(f"\n{'─'*50}")
    print(f"  Processing: {os.path.basename(filepath)}")
    print(f"{'─'*50}")

    # Load full audio via pydub (for slicing + energy analysis)
    full_audio = AudioSegment.from_file(filepath)
    duration_s = len(full_audio) / 1000.0
    print(f"  Duration: {duration_s:.1f}s | Channels: {full_audio.channels} | "
          f"Sample rate: {full_audio.frame_rate}Hz")

    # Transcribe with Whisper (single pass)
    # Uses prosody-aware prompt from Nia's speech-to-text.js
    print(f"  Transcribing with Whisper ({config.WHISPER_MODEL})...")
    result = model.transcribe(
        filepath,
        word_timestamps=True,
        verbose=False,
        language=config.WHISPER_LANGUAGE,
        temperature=config.WHISPER_TEMPERATURE,
        initial_prompt=utils.build_whisper_prompt(),
    )
    segments = result.get("segments", [])
    print(f"  Whisper returned {len(segments)} segment(s)")

    if not segments:
        print("  ⚠ No speech detected, skipping.")
        return []

    clips = []
    clip_index = 0
    skipped = {"hallucination": 0, "too_short": 0, "too_quiet": 0, "too_brief": 0}

    for seg in segments:
        seg_text = seg["text"].strip()
        seg_dur = seg["end"] - seg["start"]

        if not seg_text or seg_dur < min_dur:
            skipped["too_brief"] += 1
            continue

        # ── Hallucination check (from Nia's voice-client.js) ──
        if config.FILTER_HALLUCINATIONS and utils.is_whisper_hallucination(seg_text):
            cleaned = utils.clean_hallucination(seg_text)
            if utils.is_transcript_too_short(cleaned):
                print(f"    ✖ Hallucination discarded: '{seg_text[:40]}...'")
                skipped["hallucination"] += 1
                continue
            else:
                print(f"    ⚠ Hallucination cleaned: '{seg_text[:30]}...' → '{cleaned}'")
                seg_text = cleaned

        # Split long segments using word-level timestamps
        if seg_dur > max_dur:
            print(f"    Segment too long ({seg_dur:.1f}s), splitting by word boundaries...")
            sub_segs = split_segment_by_words(seg, max_dur)
        else:
            sub_segs = [{
                "start": seg["start"],
                "end": seg["end"],
                "text": seg_text,
            }]

        # Also create phrase-level clips for tighter matching
        # These are small n-gram slices (1-4 words) from word timestamps
        if config.PHRASE_SLICING:
            phrase_clips = create_phrase_clips(seg, phrase_size=config.MAX_PHRASE_WORDS)
            if phrase_clips:
                # Only keep phrases that are different from the full segment
                full_texts = {s["text"].lower().strip() for s in sub_segs}
                phrase_clips = [p for p in phrase_clips if p["text"].lower().strip() not in full_texts]
                sub_segs.extend(phrase_clips)

        for sub in sub_segs:
            sub_dur = sub["end"] - sub["start"]
            sub_text = sub["text"].strip()

            if sub_dur < min_dur or not sub_text:
                skipped["too_brief"] += 1
                continue

            # ── Noise filter: transcript length (from Nia's voice-client.js) ──
            if utils.is_transcript_too_short(sub_text):
                print(f"    ✖ Too short: '{sub_text}'")
                skipped["too_short"] += 1
                continue

            # Slice audio
            start_ms = int(sub["start"] * 1000)
            end_ms = int(sub["end"] * 1000)
            clip_audio = full_audio[start_ms:end_ms]

            # ── Noise filter: energy level ──
            if utils.is_clip_too_quiet(clip_audio):
                print(f"    ✖ Too quiet ({clip_audio.dBFS:.1f} dBFS): '{sub_text[:30]}'")
                skipped["too_quiet"] += 1
                continue

            # ── Emotional enhancement (from Nia's emotional-transcript-processor.js) ──
            if config.ENHANCE_TRANSCRIPTS:
                enhanced_text = utils.enhance_transcript(sub_text)
                if enhanced_text != sub_text:
                    print(f"    ✨ Enhanced: '{sub_text[:30]}' → '{enhanced_text[:30]}'")
                    sub_text = enhanced_text

            # Export clip
            clip_filename = f"{safe_filename(basename)}_{clip_index:04d}.wav"
            clip_path = os.path.join(output_dir, clip_filename)
            clip_audio.export(clip_path, format="wav")

            # ── Optional: Discord audio processing ──
            if config.DISCORD_AUDIO_MODE:
                processed_path = clip_path.replace(".wav", "_dc.wav")
                if utils.apply_discord_audio_processing(clip_path, processed_path):
                    os.replace(processed_path, clip_path)

            clip_meta = {
                "clip_file": clip_filename,
                "source_file": os.path.basename(filepath),
                "text": sub_text,
                "text_original": sub["text"].strip(),  # keep original for debugging
                "start": round(sub["start"], 3),
                "end": round(sub["end"], 3),
                "duration": round(sub_dur, 3),
                "energy_db": round(clip_audio.dBFS, 1),
            }
            clips.append(clip_meta)
            print(f"    ✔ [{format_timestamp(sub['start'])} → {format_timestamp(sub['end'])}] "
                  f"'{sub_text[:60]}{'…' if len(sub_text) > 60 else ''}'")
            clip_index += 1

    # Report skipped
    total_skipped = sum(skipped.values())
    if total_skipped > 0:
        parts = [f"{v} {k}" for k, v in skipped.items() if v > 0]
        print(f"  Filtered out {total_skipped} segment(s): {', '.join(parts)}")

    return clips


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Parse audio clips with Whisper")
    parser.add_argument("--input", default=config.RAW_CLIPS_DIR,
                        help="Folder of raw audio files to process")
    parser.add_argument("--output", default=config.MICRO_CLIPS_DIR,
                        help="Folder to save sliced clip WAVs")
    parser.add_argument("--model", default=config.WHISPER_MODEL,
                        help="Whisper model size (tiny/base/small/medium/large)")
    parser.add_argument("--force", action="store_true",
                        help="Re-process all files even if unchanged")
    parser.add_argument("--dedupe", action="store_true",
                        help="Just deduplicate existing catalog without re-processing")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Dedupe-only mode
    if args.dedupe:
        if not os.path.exists(config.MICRO_CLIPS_JSON):
            print("No catalog found, nothing to deduplicate.")
            sys.exit(0)
        with open(config.MICRO_CLIPS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_clips = data.get("clips", [])
        print(f"Loaded {len(all_clips)} clips, deduplicating...")
    else:
        # Normal processing mode
        # Check for existing catalog (for incremental processing)
        existing_catalog = {}
        if os.path.exists(config.MICRO_CLIPS_JSON) and not args.force:
            with open(config.MICRO_CLIPS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                for clip in data.get("clips", []):
                    src = clip.get("source_file", "")
                    existing_catalog.setdefault(src, []).append(clip)
            print(f"Loaded existing catalog with {sum(len(v) for v in existing_catalog.values())} clips")

        # Scan input directory
        audio_exts = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".webm", ".mp4"}
        audio_files = sorted(
            f for f in os.listdir(input_dir)
            if os.path.splitext(f)[1].lower() in audio_exts
        )

        if not audio_files:
            print(f"\n⚠ No audio files found in: {input_dir}")
            print(f"  Supported formats: {', '.join(audio_exts)}")
            print(f"  Drop your audio files there and run this script again.")
            sys.exit(0)

        print(f"\nFound {len(audio_files)} audio file(s) in {input_dir}")
        print(f"Filters: hallucination={config.FILTER_HALLUCINATIONS}, "
              f"enhance={config.ENHANCE_TRANSCRIPTS}, "
              f"discord_audio={config.DISCORD_AUDIO_MODE}")

        # Load Whisper model (once)
        print(f"\nLoading Whisper model '{args.model}'...")
        model = whisper.load_model(args.model)
        print("Model loaded ✔")

        # Process each file
        all_clips = []

        for filename in audio_files:
            filepath = os.path.join(input_dir, filename)

            if not args.force and filename in existing_catalog:
                print(f"\n  Skipping (unchanged): {filename}")
                all_clips.extend(existing_catalog[filename])
                continue

            clips = process_audio_file(filepath, model, output_dir)
            all_clips.extend(clips)

    # ── Deduplicate clips ──
    # Remove exact text duplicates from same source, and near-overlapping time ranges
    before_dedup = len(all_clips)
    seen = set()
    deduped = []
    for clip in all_clips:
        # Key: source file + exact text (case-insensitive)
        key = (clip.get("source_file", ""), clip["text"].lower().strip())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(clip)

    all_clips = deduped
    if before_dedup != len(all_clips):
        print(f"\n  Deduplication: {before_dedup} → {len(all_clips)} clips ({before_dedup - len(all_clips)} duplicates removed)")

    # Clean up orphaned sliced files
    catalog_files = {c["clip_file"] for c in all_clips}
    orphaned = 0
    for f in os.listdir(output_dir):
        if f.endswith(".wav") and f not in catalog_files:
            os.remove(os.path.join(output_dir, f))
            orphaned += 1
    if orphaned:
        print(f"  Cleaned up {orphaned} orphaned clip file(s)")

    # Save catalog
    catalog = {
        "generated_at": datetime.now().isoformat(),
        "whisper_model": args.model,
        "whisper_prompt": utils.build_whisper_prompt()[:80] + "...",
        "filters": {
            "hallucination_detection": config.FILTER_HALLUCINATIONS,
            "emotional_enhancement": config.ENHANCE_TRANSCRIPTS,
            "discord_audio_mode": config.DISCORD_AUDIO_MODE,
            "min_energy_db": config.MIN_CLIP_ENERGY_DB,
            "min_transcript_length": config.MIN_TRANSCRIPT_LENGTH,
        },
        "total_clips": len(all_clips),
        "clips": all_clips,
    }

    with open(config.MICRO_CLIPS_JSON, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"\n{'═'*50}")
    print(f"  Done! {len(all_clips)} micro-clips cataloged.")
    print(f"  Catalog: {config.MICRO_CLIPS_JSON}")
    print(f"  Clips:   {output_dir}")
    print(f"{'═'*50}")


if __name__ == "__main__":
    main()
