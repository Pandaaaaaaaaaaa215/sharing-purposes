"""
Discord Mosaic TTS — Waveform Deduplicator
=============================================
Compares actual audio waveforms to find duplicate/near-duplicate clips,
even if they have different filenames or transcriptions.

Uses audio fingerprinting (chromagram-based) to detect clips that
sound the same regardless of minor volume/encoding differences.

Usage:
    python deduplicator.py                    # preview duplicates
    python deduplicator.py --delete           # delete duplicates and update catalog
    python deduplicator.py --threshold 0.92   # adjust similarity threshold

Requires: pip install numpy pydub scipy
"""

import os
import sys
import json
import argparse
import hashlib
import numpy as np
from datetime import datetime
from collections import defaultdict

from pydub import AudioSegment

import config


def log(icon, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {icon} {msg}")


# ──────────────────────────────────────────────
#  Audio fingerprinting
# ──────────────────────────────────────────────
def audio_to_fingerprint(audio: AudioSegment, n_bins=32, hop_ms=50):
    """
    Create a compact fingerprint from audio waveform.
    Uses spectral energy bands — fast and effective for near-duplicate detection.
    """
    # Normalize to mono 16kHz
    audio = audio.set_channels(1).set_frame_rate(16000)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    if len(samples) == 0:
        return np.array([])

    # Normalize amplitude
    peak = np.abs(samples).max()
    if peak > 0:
        samples = samples / peak

    # Simple spectral fingerprint using short-time energy in frequency bands
    hop = int(16000 * hop_ms / 1000)
    window = hop * 2
    n_frames = max(1, (len(samples) - window) // hop)

    fingerprint = []
    for i in range(n_frames):
        start = i * hop
        frame = samples[start:start + window]

        # Apply window function
        frame = frame * np.hanning(len(frame))

        # FFT
        spectrum = np.abs(np.fft.rfft(frame))

        # Bin into n_bins frequency bands
        bin_size = max(1, len(spectrum) // n_bins)
        bands = []
        for b in range(n_bins):
            band_start = b * bin_size
            band_end = min((b + 1) * bin_size, len(spectrum))
            if band_start < len(spectrum):
                bands.append(np.mean(spectrum[band_start:band_end]))
            else:
                bands.append(0.0)

        fingerprint.append(bands)

    return np.array(fingerprint)


def fingerprint_similarity(fp1, fp2):
    """
    Compare two fingerprints. Returns 0.0 - 1.0 similarity score.
    Handles different lengths by comparing the overlapping portion.
    """
    if len(fp1) == 0 or len(fp2) == 0:
        return 0.0

    # Use the shorter one as reference
    min_len = min(len(fp1), len(fp2))
    max_len = max(len(fp1), len(fp2))

    # Length ratio penalty — very different lengths are unlikely duplicates
    len_ratio = min_len / max_len
    if len_ratio < 0.5:
        return 0.0

    # Compare overlapping frames
    a = fp1[:min_len].flatten()
    b = fp2[:min_len].flatten()

    # Cosine similarity
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    cosine_sim = dot / (norm_a * norm_b)

    # Weight by length ratio
    return float(cosine_sim * len_ratio)


def raw_bytes_hash(filepath):
    """Quick MD5 of raw file bytes — catches exact file duplicates."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────
#  Find duplicates
# ──────────────────────────────────────────────
def find_duplicates(clips_dir, catalog_path, threshold=0.90):
    """
    Scan all clips and find duplicates using:
    1. Exact file hash (byte-identical)
    2. Waveform fingerprint similarity
    """
    if not os.path.exists(catalog_path):
        log("❌", "No catalog found. Run clip_parser.py first.")
        return [], {}

    with open(catalog_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clips = data.get("clips", [])
    if not clips:
        log("📭", "Catalog is empty.")
        return [], data

    log("📊", f"Scanning {len(clips)} clips...")

    # Phase 1: Exact file hash duplicates
    log("🔍", "Phase 1: Checking for byte-identical files...")
    hash_groups = defaultdict(list)
    missing = []

    for clip in clips:
        path = os.path.join(clips_dir, clip["clip_file"])
        if not os.path.exists(path):
            missing.append(clip["clip_file"])
            continue
        h = raw_bytes_hash(path)
        hash_groups[h].append(clip)

    exact_dupes = {h: group for h, group in hash_groups.items() if len(group) > 1}
    if exact_dupes:
        count = sum(len(g) - 1 for g in exact_dupes.values())
        log("🔴", f"Found {count} byte-identical duplicate(s)")
    else:
        log("✔", "No byte-identical duplicates")

    if missing:
        log("⚠", f"{len(missing)} clip file(s) missing from disk")

    # Phase 2: Waveform fingerprint comparison
    log("🔍", "Phase 2: Computing audio fingerprints...")
    fingerprints = []
    for i, clip in enumerate(clips):
        path = os.path.join(clips_dir, clip["clip_file"])
        if not os.path.exists(path):
            fingerprints.append(None)
            continue

        try:
            audio = AudioSegment.from_file(path)
            fp = audio_to_fingerprint(audio)
            fingerprints.append(fp)
        except Exception as e:
            log("⚠", f"  Failed to fingerprint {clip['clip_file']}: {e}")
            fingerprints.append(None)

        if (i + 1) % 50 == 0:
            log("  ", f"  Fingerprinted {i+1}/{len(clips)}...")

    log("✔", f"Fingerprinted {sum(1 for f in fingerprints if f is not None)}/{len(clips)} clips")

    # Compare all pairs (O(n^2) but fine for typical library sizes)
    log("🔍", f"Phase 3: Comparing pairs (threshold={threshold})...")
    wave_dupes = []
    seen_pairs = set()

    total_comparisons = len(clips) * (len(clips) - 1) // 2
    comparison_count = 0

    for i in range(len(clips)):
        if fingerprints[i] is None:
            continue
        for j in range(i + 1, len(clips)):
            if fingerprints[j] is None:
                continue

            comparison_count += 1
            if comparison_count % 5000 == 0:
                log("  ", f"  Compared {comparison_count}/{total_comparisons}...")

            # Skip if durations are very different (quick filter)
            dur_i = clips[i].get("duration", 0)
            dur_j = clips[j].get("duration", 0)
            if dur_i > 0 and dur_j > 0:
                dur_ratio = min(dur_i, dur_j) / max(dur_i, dur_j)
                if dur_ratio < 0.5:
                    continue

            sim = fingerprint_similarity(fingerprints[i], fingerprints[j])
            if sim >= threshold:
                pair_key = (min(i, j), max(i, j))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    wave_dupes.append({
                        "clip_a": clips[i],
                        "clip_b": clips[j],
                        "similarity": round(sim, 3),
                        "type": "waveform"
                    })

    if wave_dupes:
        log("🔴", f"Found {len(wave_dupes)} waveform-similar pair(s)")
    else:
        log("✔", "No waveform duplicates found")

    # Combine results
    all_dupes = []

    # Add exact hash dupes
    for h, group in exact_dupes.items():
        keep = group[0]
        for dupe in group[1:]:
            all_dupes.append({
                "keep": keep,
                "remove": dupe,
                "similarity": 1.0,
                "type": "exact"
            })

    # Add waveform dupes (keep the one with longer/better text)
    for pair in wave_dupes:
        a = pair["clip_a"]
        b = pair["clip_b"]
        # Keep whichever has more text (likely better transcription)
        if len(a.get("text", "")) >= len(b.get("text", "")):
            keep, remove = a, b
        else:
            keep, remove = b, a

        all_dupes.append({
            "keep": keep,
            "remove": remove,
            "similarity": pair["similarity"],
            "type": pair["type"]
        })

    return all_dupes, data


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Find and remove duplicate audio clips")
    parser.add_argument("--threshold", "-t", type=float, default=0.90,
                        help="Waveform similarity threshold (0.0-1.0, default 0.90)")
    parser.add_argument("--delete", "-d", action="store_true",
                        help="Actually delete duplicates (default is preview only)")
    args = parser.parse_args()

    print("=" * 55)
    print("  Discord Mosaic TTS — Waveform Deduplicator")
    print("=" * 55)
    print()

    clips_dir = config.MICRO_CLIPS_DIR
    catalog_path = config.MICRO_CLIPS_JSON

    dupes, catalog_data = find_duplicates(clips_dir, catalog_path, args.threshold)

    if not dupes:
        log("✅", "Library is clean — no duplicates found!")
        return

    # Display results
    print()
    log("📋", f"Found {len(dupes)} duplicate(s):")
    print()

    for i, d in enumerate(dupes, 1):
        keep = d["keep"]
        remove = d["remove"]
        sim = d["similarity"]
        dtype = d["type"]

        tag = "EXACT" if dtype == "exact" else f"{sim:.0%}"
        print(f"  {i}. [{tag}]")
        print(f"     KEEP:   {keep['clip_file']}  \"{keep.get('text', '')[:50]}\"")
        print(f"     REMOVE: {remove['clip_file']}  \"{remove.get('text', '')[:50]}\"")
        print()

    if not args.delete:
        log("💡", f"Preview only. Run with --delete to remove {len(dupes)} duplicate(s).")
        log("  ", f"  python deduplicator.py --delete --threshold {args.threshold}")
        return

    # Actually delete
    log("🗑️", f"Deleting {len(dupes)} duplicate(s)...")

    remove_files = set()
    remove_texts = set()
    for d in dupes:
        remove_files.add(d["remove"]["clip_file"])
        remove_texts.add((d["remove"].get("source_file", ""), d["remove"]["text"].lower()))

    # Remove from catalog
    original_count = len(catalog_data.get("clips", []))
    catalog_data["clips"] = [
        c for c in catalog_data["clips"]
        if c["clip_file"] not in remove_files
    ]
    new_count = len(catalog_data["clips"])
    catalog_data["total_clips"] = new_count

    # Save updated catalog
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog_data, f, indent=2, ensure_ascii=False)

    log("✔", f"Catalog updated: {original_count} → {new_count} clips")

    # Delete actual files
    deleted = 0
    for clip_file in remove_files:
        path = os.path.join(clips_dir, clip_file)
        if os.path.exists(path):
            os.remove(path)
            deleted += 1

    log("✔", f"Deleted {deleted} file(s) from disk")
    log("✅", "Deduplication complete!")


if __name__ == "__main__":
    main()
