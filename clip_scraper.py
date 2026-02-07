"""
Discord Mosaic TTS — Clip Scraper
====================================
Quickly build your clip library from various sources.

Modes:
  1. YouTube URLs     — download and extract audio segments
  2. Bulk folder      — import loose audio files
  3. Mic recording    — record yourself directly (coming soon)

The scraper downloads/copies audio into clips/raw/ and then
automatically runs clip_parser.py to transcribe and catalog them.

Usage:
    python clip_scraper.py url "https://youtube.com/watch?v=..."
    python clip_scraper.py url "https://youtube.com/watch?v=..." --start 0:30 --end 1:45
    python clip_scraper.py urls urls.txt              # one URL per line
    python clip_scraper.py folder "C:/my_clips/"      # bulk import
    python clip_scraper.py list                        # show current library stats

Requires: pip install yt-dlp
          (ffmpeg must be on PATH)
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from datetime import datetime

import config


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────
def log(icon, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {icon} {msg}")


def check_tool(name):
    """Check if a command-line tool is available."""
    return shutil.which(name) is not None


def parse_timestamp(ts_str):
    """Parse a timestamp like '1:30' or '0:05:20' into seconds."""
    if not ts_str:
        return None
    parts = ts_str.split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 1:
        return parts[0]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


def safe_filename(name):
    """Sanitize for filesystem."""
    return "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip()[:80]


# ──────────────────────────────────────────────
#  YouTube / URL Download
# ──────────────────────────────────────────────
def download_url(url, output_dir, start=None, end=None, label=None):
    """
    Download audio from a YouTube URL (or any yt-dlp supported URL).
    Optionally trim to a time range.
    Returns the path to the downloaded audio file.
    """
    if not check_tool("yt-dlp"):
        log("❌", "yt-dlp not found! Install it:")
        log("  ", "  pip install yt-dlp")
        return None

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Get video info for filename
    log("🔍", f"Fetching info: {url}")
    try:
        result = subprocess.run(
            ["yt-dlp", "--print", "title", "--no-download", url],
            capture_output=True, text=True, timeout=30
        )
        title = result.stdout.strip() or "untitled"
    except Exception:
        title = "untitled"

    filename = label or safe_filename(title)
    if start or end:
        tag = f"_{start or '0'}_{end or 'end'}".replace(":", "m")
        filename += tag

    output_path = os.path.join(output_dir, f"{filename}.wav")

    # Step 2: Download as audio
    log("⬇️", f"Downloading: '{title}'")

    cmd = [
        "yt-dlp",
        "--js-runtimes", "deno",
        "-x",                           # extract audio
        "--audio-format", "wav",         # output as WAV
        "--audio-quality", "0",          # best quality
        "-o", output_path,               # output path
        "--no-playlist",                 # single video only
    ]

    # yt-dlp can handle time ranges via --download-sections
    if start or end:
        start_sec = parse_timestamp(start) if start else 0
        end_sec = parse_timestamp(end) if end else None
        if end_sec:
            section = f"*{start_sec}-{end_sec}"
        else:
            section = f"*{start_sec}-inf"
        cmd.extend(["--download-sections", section])
        log("✂️", f"Trimming: {start or '0:00'} → {end or 'end'}")

    cmd.append(url)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            log("❌", f"yt-dlp failed: {proc.stderr[:200]}")

            # Fallback: download full then trim with ffmpeg
            if (start or end) and check_tool("ffmpeg"):
                log("🔄", "Trying fallback: download full + ffmpeg trim...")
                return download_and_trim_fallback(url, output_dir, filename, start, end)
            return None
    except subprocess.TimeoutExpired:
        log("❌", "Download timed out (5 min limit)")
        return None

    # yt-dlp sometimes adds extra extensions
    if not os.path.exists(output_path):
        # Look for the file with any extension
        for f in os.listdir(output_dir):
            if f.startswith(filename):
                actual_path = os.path.join(output_dir, f)
                log("📄", f"Found as: {f}")
                return actual_path
        log("❌", f"File not found after download: {output_path}")
        return None

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log("✔", f"Downloaded: {filename}.wav ({size_mb:.1f} MB)")
    return output_path


def download_and_trim_fallback(url, output_dir, filename, start, end):
    """Fallback: download full audio, then trim with ffmpeg."""
    full_path = os.path.join(output_dir, f"{filename}_full.wav")
    trimmed_path = os.path.join(output_dir, f"{filename}.wav")

    # Download without time range
    cmd = [
        "yt-dlp", "--js-runtimes", "deno", "-x", "--audio-format", "wav",
        "--audio-quality", "0", "-o", full_path,
        "--no-playlist", url
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            log("❌", f"Fallback download failed")
            return None
    except subprocess.TimeoutExpired:
        return None

    # Find the actual downloaded file
    if not os.path.exists(full_path):
        for f in os.listdir(output_dir):
            if f.startswith(f"{filename}_full"):
                full_path = os.path.join(output_dir, f)
                break

    if not os.path.exists(full_path):
        return None

    # Trim with ffmpeg
    ffmpeg_cmd = ["ffmpeg", "-y", "-i", full_path]
    if start:
        ffmpeg_cmd.extend(["-ss", str(parse_timestamp(start))])
    if end:
        ffmpeg_cmd.extend(["-to", str(parse_timestamp(end))])
    ffmpeg_cmd.extend(["-ar", "16000", "-ac", "1", trimmed_path])

    try:
        subprocess.run(ffmpeg_cmd, capture_output=True, check=True, timeout=60)
        os.remove(full_path)
        log("✔", f"Trimmed with ffmpeg: {filename}.wav")
        return trimmed_path
    except Exception as e:
        log("❌", f"FFmpeg trim failed: {e}")
        return None


# ──────────────────────────────────────────────
#  Bulk folder import
# ──────────────────────────────────────────────
def import_folder(source_dir, output_dir):
    """Copy all audio files from a folder into clips/raw/."""
    audio_exts = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".webm", ".mp4"}
    os.makedirs(output_dir, exist_ok=True)

    files = [
        f for f in os.listdir(source_dir)
        if os.path.splitext(f)[1].lower() in audio_exts
    ]

    if not files:
        log("⚠", f"No audio files found in: {source_dir}")
        return 0

    log("📂", f"Found {len(files)} audio file(s) in {source_dir}")

    copied = 0
    for f in files:
        src = os.path.join(source_dir, f)
        dst = os.path.join(output_dir, f)
        if os.path.exists(dst):
            log("  ", f"  Skipped (exists): {f}")
            continue
        shutil.copy2(src, dst)
        log("✔", f"  Copied: {f}")
        copied += 1

    return copied


# ──────────────────────────────────────────────
#  Library stats
# ──────────────────────────────────────────────
def show_library_stats():
    """Show current clip library statistics."""
    print()
    print("=" * 50)
    print("  Clip Library Stats")
    print("=" * 50)

    if not os.path.exists(config.MICRO_CLIPS_JSON):
        log("⚠", "No clip catalog found. Run clip_parser.py first.")
        return

    with open(config.MICRO_CLIPS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    clips = data.get("clips", [])
    total = len(clips)
    if total == 0:
        log("📭", "Catalog is empty.")
        return

    total_dur = sum(c.get("duration", 0) for c in clips)
    sources = set(c.get("source_file", "?") for c in clips)
    avg_dur = total_dur / total

    log("📊", f"Total clips:    {total}")
    log("📊", f"Total duration: {total_dur:.1f}s ({total_dur/60:.1f} min)")
    log("📊", f"Average clip:   {avg_dur:.1f}s")
    log("📊", f"Source files:   {len(sources)}")

    print()
    log("📁", "Source breakdown:")
    source_counts = {}
    for c in clips:
        src = c.get("source_file", "?")
        source_counts[src] = source_counts.get(src, 0) + 1
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        log("  ", f"  {count:4d} clips — {src}")

    # Show some sample texts
    print()
    log("📝", "Sample clip texts:")
    for c in clips[:10]:
        log("  ", f"  \"{c['text'][:60]}\"")
    if total > 10:
        log("  ", f"  ... and {total - 10} more")

    # Check raw folder for unprocessed files
    if os.path.exists(config.RAW_CLIPS_DIR):
        raw_files = [f for f in os.listdir(config.RAW_CLIPS_DIR)
                     if os.path.splitext(f)[1].lower() in {".mp3",".wav",".ogg",".flac",".m4a"}]
        processed_sources = sources
        unprocessed = [f for f in raw_files if f not in processed_sources]
        if unprocessed:
            print()
            log("⚠", f"{len(unprocessed)} unprocessed file(s) in raw/:")
            for f in unprocessed[:5]:
                log("  ", f"  {f}")
            log("  ", f"Run 'python clip_parser.py' to process them!")


# ──────────────────────────────────────────────
#  Auto-run clip parser after importing
# ──────────────────────────────────────────────
def run_clip_parser():
    """Run clip_parser.py to process any new raw files."""
    parser_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clip_parser.py")
    if not os.path.exists(parser_path):
        log("⚠", "clip_parser.py not found — run it manually")
        return

    log("🔄", "Running clip parser on new files...")
    print()
    subprocess.run([sys.executable, parser_path])


# ──────────────────────────────────────────────
#  Main CLI
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Build your clip library from various sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clip_scraper.py url "https://youtube.com/watch?v=dQw4w9WgXcQ"
  python clip_scraper.py url "https://youtube.com/watch?v=..." --start 0:30 --end 1:45
  python clip_scraper.py url "https://youtube.com/watch?v=..." --label "rick_roll"
  python clip_scraper.py urls my_links.txt
  python clip_scraper.py folder "C:/Downloads/meme_sounds/"
  python clip_scraper.py list
        """
    )

    sub = parser.add_subparsers(dest="command")

    # URL command
    url_cmd = sub.add_parser("url", help="Download from a YouTube/video URL")
    url_cmd.add_argument("link", help="Video URL")
    url_cmd.add_argument("--start", "-s", help="Start time (e.g. 0:30 or 1:05:20)")
    url_cmd.add_argument("--end", "-e", help="End time (e.g. 1:45)")
    url_cmd.add_argument("--label", "-l", help="Custom filename label")
    url_cmd.add_argument("--no-parse", action="store_true", help="Skip auto clip parsing")

    # Batch URLs command
    urls_cmd = sub.add_parser("urls", help="Download from a text file of URLs")
    urls_cmd.add_argument("file", help="Text file with one URL per line (supports # comments)")
    urls_cmd.add_argument("--no-parse", action="store_true", help="Skip auto clip parsing")

    # Folder command
    folder_cmd = sub.add_parser("folder", help="Import audio files from a folder")
    folder_cmd.add_argument("path", help="Source folder path")
    folder_cmd.add_argument("--no-parse", action="store_true", help="Skip auto clip parsing")

    # List command
    sub.add_parser("list", help="Show library stats")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    print("=" * 50)
    print("  Discord Mosaic TTS — Clip Scraper")
    print("=" * 50)
    print()

    raw_dir = config.RAW_CLIPS_DIR
    os.makedirs(raw_dir, exist_ok=True)

    if args.command == "url":
        path = download_url(args.link, raw_dir, args.start, args.end, args.label)
        if path and not args.no_parse:
            print()
            run_clip_parser()

    elif args.command == "urls":
        if not os.path.exists(args.file):
            log("❌", f"File not found: {args.file}")
            return

        with open(args.file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

        log("📋", f"Processing {len(lines)} URL(s)...")
        print()

        success = 0
        for i, line in enumerate(lines, 1):
            # Support formats:
            # URL
            # URL label
            # URL start end
            # URL start end label
            parts = line.split()
            url = parts[0]
            start = None
            end = None
            label = None

            if len(parts) == 2:
                # Could be a label or a start time
                if ":" in parts[1] or parts[1].replace(".", "").isdigit():
                    start = parts[1]
                else:
                    label = parts[1]
            elif len(parts) == 3:
                if ":" in parts[1] or parts[1].replace(".", "").isdigit():
                    start = parts[1]
                    # Third part could be end time or label
                    if ":" in parts[2] or parts[2].replace(".", "").isdigit():
                        end = parts[2]
                    else:
                        label = parts[2]
                else:
                    label = parts[1]
            elif len(parts) >= 4:
                start = parts[1]
                end = parts[2]
                label = parts[3]

            log("📥", f"[{i}/{len(lines)}] {url}")
            try:
                path = download_url(url, raw_dir, start, end, label)
                if path:
                    success += 1
            except Exception as e:
                log("❌", f"  Failed: {e}")
                log("⏭️", f"  Skipping and continuing...")
            print()

        log("📊", f"Downloaded {success}/{len(lines)} URLs")

        if success > 0 and not args.no_parse:
            print()
            run_clip_parser()

    elif args.command == "folder":
        copied = import_folder(args.path, raw_dir)
        log("📊", f"Imported {copied} new file(s)")

        if copied > 0 and not args.no_parse:
            print()
            run_clip_parser()

    elif args.command == "list":
        show_library_stats()


if __name__ == "__main__":
    main()
