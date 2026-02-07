"""
Discord Mosaic TTS — Clip Manager UI
======================================
A simple local web UI to add YouTube URLs and manage your clip library.

Usage:
    python clip_ui.py

Opens a browser at http://localhost:8642
"""

import os
import sys
import json
import subprocess
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from datetime import datetime

import config

PORT = 8642

# Shared state for progress
progress = {
    "running": False,
    "logs": [],
    "current_url": "",
    "done": 0,
    "total": 0
}


def get_library_stats():
    """Get current clip library info."""
    # Try loading from catalog JSON first
    if os.path.exists(config.MICRO_CLIPS_JSON):
        with open(config.MICRO_CLIPS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        clips = data.get("clips", [])
        total_dur = sum(c.get("duration", 0) for c in clips)
        sources = len(set(c.get("source_file", "?") for c in clips))

        return {
            "total": len(clips),
            "duration": round(total_dur, 1),
            "sources": sources,
            "clips": clips[-50:]  # last 50 for display
        }

    # Fallback: scan clips directory for wav files (no catalog yet)
    clips_dir = config.MICRO_CLIPS_DIR
    if os.path.exists(clips_dir):
        wav_files = [f for f in os.listdir(clips_dir)
                     if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]
        if wav_files:
            clips = []
            for f in sorted(wav_files)[-50:]:
                clips.append({
                    "clip_file": f,
                    "source_file": "uncataloged",
                    "text": "(not parsed yet — run Parse New)",
                    "duration": 0,
                })
            return {
                "total": len(wav_files),
                "duration": 0,
                "sources": 0,
                "clips": clips,
                "uncataloged": True
            }

    # Also check raw clips dir
    raw_dir = config.RAW_CLIPS_DIR
    if os.path.exists(raw_dir):
        raw_files = [f for f in os.listdir(raw_dir)
                     if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac', '.m4a', '.webm'))]
        if raw_files:
            clips = []
            for f in sorted(raw_files)[-50:]:
                clips.append({
                    "clip_file": f,
                    "source_file": "raw",
                    "text": "(raw — run Parse New to transcribe)",
                    "duration": 0,
                })
            return {
                "total": len(raw_files),
                "duration": 0,
                "sources": 0,
                "clips": clips,
                "uncataloged": True
            }

    return {"total": 0, "duration": 0, "sources": 0, "clips": []}


def download_and_parse(url, label=None, start=None, end=None):
    """Download a URL and parse it. Runs in background thread."""
    progress["running"] = True
    progress["logs"] = []
    progress["current_url"] = url
    progress["done"] = 0
    progress["total"] = 1

    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        progress["logs"].append(line)
        print(line)

    try:
        raw_dir = config.RAW_CLIPS_DIR
        os.makedirs(raw_dir, exist_ok=True)

        # Build yt-dlp command
        cmd = ["yt-dlp", "--js-runtimes", "deno", "-x", "--audio-format", "wav",
               "--audio-quality", "0", "--no-playlist"]

        if start or end:
            from clip_scraper import parse_timestamp
            start_sec = parse_timestamp(start) if start else 0
            end_sec = parse_timestamp(end) if end else None
            if end_sec:
                cmd.extend(["--download-sections", f"*{start_sec}-{end_sec}"])
            else:
                cmd.extend(["--download-sections", f"*{start_sec}-inf"])

        # Get title for filename
        log(f"Fetching info: {url}")
        try:
            result = subprocess.run(
                ["yt-dlp", "--js-runtimes", "deno", "--print", "title", "--no-download", url],
                capture_output=True, text=True, timeout=30
            )
            title = result.stdout.strip() or "untitled"
        except Exception:
            title = "untitled"

        safe_title = label or "".join(c if c.isalnum() or c in "-_ " else "_" for c in title).strip()[:80]
        output_path = os.path.join(raw_dir, f"{safe_title}.wav")
        cmd.extend(["-o", output_path, url])

        log(f"Downloading: '{title}'")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if proc.returncode != 0:
            log(f"Download failed: {proc.stderr[:200]}")
            progress["running"] = False
            return

        # Find the actual file (yt-dlp may add extensions)
        actual_path = output_path
        if not os.path.exists(actual_path):
            for f in os.listdir(raw_dir):
                if f.startswith(safe_title):
                    actual_path = os.path.join(raw_dir, f)
                    break

        if os.path.exists(actual_path):
            size_mb = os.path.getsize(actual_path) / (1024 * 1024)
            log(f"Downloaded: {os.path.basename(actual_path)} ({size_mb:.1f} MB)")
        else:
            log("File not found after download")
            progress["running"] = False
            return

        # Run clip parser
        log("Running Whisper transcription + slicing...")
        parser_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clip_parser.py")
        if os.path.exists(parser_path):
            proc = subprocess.run(
                [sys.executable, parser_path],
                capture_output=True, text=True, timeout=600
            )
            for line in proc.stdout.strip().split("\n")[-10:]:
                if line.strip():
                    log(line.strip())

        progress["done"] = 1
        log("Done! Clips added to library.")

    except Exception as e:
        log(f"Error: {e}")
    finally:
        progress["running"] = False


def run_command(cmd_name):
    """Run a management command in background thread."""
    progress["running"] = True
    progress["logs"] = []
    progress["current_url"] = cmd_name
    progress["done"] = 0
    progress["total"] = 1

    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        progress["logs"].append(line)
        print(line)

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))

        if cmd_name == "dedupe":
            log("Running deduplication...")
            parser_path = os.path.join(script_dir, "clip_parser.py")
            proc = subprocess.run(
                [sys.executable, parser_path, "--dedupe"],
                capture_output=True, text=True, timeout=120
            )

        elif cmd_name == "reparse":
            log("Re-parsing all clips with Whisper (this may take a while)...")
            parser_path = os.path.join(script_dir, "clip_parser.py")
            proc = subprocess.run(
                [sys.executable, parser_path, "--force"],
                capture_output=True, text=True, timeout=3600
            )

        elif cmd_name == "parse_new":
            log("Parsing new clips only...")
            parser_path = os.path.join(script_dir, "clip_parser.py")
            proc = subprocess.run(
                [sys.executable, parser_path],
                capture_output=True, text=True, timeout=3600
            )

        elif cmd_name == "clear_log":
            log_file = config.DISCORD_LOG_FILE
            if os.path.exists(log_file):
                open(log_file, "w").close()
                log(f"Cleared {log_file}")
            else:
                log("Log file not found")
            progress["done"] = 1
            log("Done!")
            progress["running"] = False
            return

        elif cmd_name == "wave_dedupe":
            log("Running waveform deduplication (this scans all audio)...")
            dedup_path = os.path.join(script_dir, "deduplicator.py")
            proc = subprocess.run(
                [sys.executable, dedup_path, "--delete"],
                capture_output=True, text=True, timeout=600
            )

        elif cmd_name == "list_devices":
            log("Listing audio devices...")
            proc = subprocess.run(
                [sys.executable, "-m", "sounddevice"],
                capture_output=True, text=True, timeout=10
            )

        else:
            log(f"Unknown command: {cmd_name}")
            progress["running"] = False
            return

        # Show output
        output = proc.stdout.strip() if proc.stdout else ""
        errors = proc.stderr.strip() if proc.stderr else ""

        for line in output.split("\n"):
            if line.strip():
                log(line.strip())
        if errors:
            for line in errors.split("\n")[-5:]:
                if line.strip():
                    log(f"stderr: {line.strip()}")

        progress["done"] = 1
        if proc.returncode == 0:
            log("Done!")
        else:
            log(f"Finished with exit code {proc.returncode}")

    except subprocess.TimeoutExpired:
        log("Command timed out!")
    except Exception as e:
        log(f"Error: {e}")
    finally:
        progress["running"] = False


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mosaic TTS — Clip Manager</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }

  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --border: #1e1e2e;
    --accent: #7c3aed;
    --accent2: #a855f7;
    --green: #22c55e;
    --red: #ef4444;
    --yellow: #eab308;
    --text: #e4e4e7;
    --dim: #71717a;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Space Grotesk', system-ui, sans-serif;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    padding: 2rem;
  }

  .container {
    max-width: 800px;
    margin: 0 auto;
  }

  h1 {
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .subtitle {
    color: var(--dim);
    font-size: 0.9rem;
    margin-bottom: 2rem;
  }

  /* Stats bar */
  .stats {
    display: flex;
    gap: 1.5rem;
    margin-bottom: 2rem;
    flex-wrap: wrap;
  }

  .stat {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.5rem;
    flex: 1;
    min-width: 120px;
  }

  .stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    font-family: var(--mono);
    color: var(--accent2);
  }

  .stat-label {
    font-size: 0.75rem;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.2rem;
  }

  /* Input section */
  .input-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }

  .input-row {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 0.75rem;
  }

  .input-row:last-child { margin-bottom: 0; }

  input[type="text"] {
    flex: 1;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: var(--text);
    font-family: var(--mono);
    font-size: 0.85rem;
    outline: none;
    transition: border-color 0.2s;
  }

  input[type="text"]:focus {
    border-color: var(--accent);
  }

  input[type="text"]::placeholder {
    color: var(--dim);
  }

  .small-input {
    max-width: 120px;
  }

  button {
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-family: var(--sans);
    font-weight: 600;
    font-size: 0.9rem;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
  }

  button:hover { background: var(--accent2); transform: translateY(-1px); }
  button:active { transform: translateY(0); }
  button:disabled {
    background: var(--border);
    color: var(--dim);
    cursor: not-allowed;
    transform: none;
  }

  /* Progress / Log */
  .log-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    display: none;
  }

  .log-section.active { display: block; }

  .log-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
  }

  .log-header h3 {
    font-size: 0.9rem;
    font-weight: 600;
  }

  .status-badge {
    font-family: var(--mono);
    font-size: 0.75rem;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-weight: 600;
  }

  .status-running {
    background: rgba(234, 179, 8, 0.15);
    color: var(--yellow);
    animation: pulse 1.5s infinite;
  }

  .status-done {
    background: rgba(34, 197, 94, 0.15);
    color: var(--green);
  }

  .status-error {
    background: rgba(239, 68, 68, 0.15);
    color: var(--red);
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
  }

  .log-output {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    font-family: var(--mono);
    font-size: 0.8rem;
    line-height: 1.6;
    max-height: 300px;
    overflow-y: auto;
    color: var(--dim);
  }

  .log-output .success { color: var(--green); }
  .log-output .error { color: var(--red); }
  .log-output .info { color: var(--yellow); }

  /* Clip list */
  .clips-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
  }

  .clips-section h3 {
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 1rem;
  }

  .clip-list {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    max-height: 400px;
    overflow-y: auto;
  }

  .clip-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0.75rem;
    background: var(--bg);
    border-radius: 6px;
    font-size: 0.8rem;
  }

  .clip-text {
    font-family: var(--mono);
    color: var(--text);
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    margin-right: 1rem;
  }

  .clip-dur {
    font-family: var(--mono);
    color: var(--dim);
    font-size: 0.75rem;
    flex-shrink: 0;
  }

  .empty-state {
    text-align: center;
    color: var(--dim);
    padding: 2rem;
    font-size: 0.85rem;
  }

  .tools-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 0.75rem;
  }

  .tool-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.3rem;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 0.75rem;
    cursor: pointer;
    transition: all 0.2s;
    text-align: center;
  }

  .tool-btn:hover {
    border-color: var(--accent);
    background: rgba(124, 58, 237, 0.08);
    transform: translateY(-2px);
  }

  .tool-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
    transform: none;
  }

  .tool-icon { font-size: 1.4rem; }

  .tool-label {
    font-weight: 600;
    font-size: 0.85rem;
    color: var(--text);
  }

  .tool-desc {
    font-size: 0.7rem;
    color: var(--dim);
    line-height: 1.3;
  }

  .refresh-btn {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--dim);
    padding: 0.4rem 0.8rem;
    font-size: 0.75rem;
  }

  .refresh-btn:hover { border-color: var(--accent); color: var(--text); }
</style>
</head>
<body>
<div class="container">
  <h1>Mosaic TTS</h1>
  <p class="subtitle">Clip Manager — paste a YouTube URL to add clips to your library</p>

  <div class="stats" id="stats">
    <div class="stat">
      <div class="stat-value" id="stat-clips">—</div>
      <div class="stat-label">Clips</div>
    </div>
    <div class="stat">
      <div class="stat-value" id="stat-duration">—</div>
      <div class="stat-label">Total Seconds</div>
    </div>
    <div class="stat">
      <div class="stat-value" id="stat-sources">—</div>
      <div class="stat-label">Sources</div>
    </div>
  </div>

  <div class="input-section">
    <div class="input-row">
      <input type="text" id="url-input" placeholder="Paste YouTube URL here..." autofocus>
      <button id="add-btn" onclick="addURL()">Add &amp; Parse</button>
    </div>
    <div class="input-row">
      <input type="text" id="label-input" class="small-input" placeholder="Label (optional)">
      <input type="text" id="start-input" class="small-input" placeholder="Start (0:30)">
      <input type="text" id="end-input" class="small-input" placeholder="End (1:45)">
    </div>
  </div>

  <div class="log-section" id="log-section">
    <div class="log-header">
      <h3>Progress</h3>
      <span class="status-badge" id="status-badge">IDLE</span>
    </div>
    <div class="log-output" id="log-output"></div>
  </div>

  <div class="clips-section">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
      <h3 style="margin:0;">Tools</h3>
    </div>
    <div class="tools-grid">
      <button class="tool-btn" onclick="runCmd('dedupe')">
        <span class="tool-icon">🧹</span>
        <span class="tool-label">Deduplicate</span>
        <span class="tool-desc">Remove duplicate clips from catalog</span>
      </button>
      <button class="tool-btn" onclick="runCmd('parse_new')">
        <span class="tool-icon">🆕</span>
        <span class="tool-label">Parse New</span>
        <span class="tool-desc">Transcribe only new raw audio files</span>
      </button>
      <button class="tool-btn" onclick="runCmd('reparse')">
        <span class="tool-icon">🔄</span>
        <span class="tool-label">Re-Parse All</span>
        <span class="tool-desc">Force re-transcribe everything (slow)</span>
      </button>
      <button class="tool-btn" onclick="runCmd('list_devices')">
        <span class="tool-icon">🔊</span>
        <span class="tool-label">Audio Devices</span>
        <span class="tool-desc">List available audio device IDs</span>
      </button>
      <button class="tool-btn" onclick="runCmd('clear_log')">
        <span class="tool-icon">📝</span>
        <span class="tool-label">Clear Chat Log</span>
        <span class="tool-desc">Empty discord_export.txt</span>
      </button>
      <button class="tool-btn" onclick="runCmd('wave_dedupe')">
        <span class="tool-icon">🔊</span>
        <span class="tool-label">Waveform Dedupe</span>
        <span class="tool-desc">Find &amp; remove audio-identical clips</span>
      </button>
    </div>
  </div>

  <div style="height:1.5rem;"></div>

  <div class="clips-section">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
      <h3 style="margin:0;">Recent Clips</h3>
      <button class="refresh-btn" onclick="loadStats()">Refresh</button>
    </div>
    <div class="clip-list" id="clip-list">
      <div class="empty-state">Loading...</div>
    </div>
  </div>
</div>

<script>
  const urlInput = document.getElementById('url-input');
  const labelInput = document.getElementById('label-input');
  const startInput = document.getElementById('start-input');
  const endInput = document.getElementById('end-input');
  const addBtn = document.getElementById('add-btn');
  const logSection = document.getElementById('log-section');
  const logOutput = document.getElementById('log-output');
  const statusBadge = document.getElementById('status-badge');

  // Enter key to submit
  urlInput.addEventListener('keydown', e => { if (e.key === 'Enter') addURL(); });

  async function runCmd(cmd) {
    // Disable all tool buttons
    document.querySelectorAll('.tool-btn').forEach(b => b.disabled = true);
    addBtn.disabled = true;
    logSection.classList.add('active');
    logOutput.innerHTML = '';
    statusBadge.textContent = 'RUNNING...';
    statusBadge.className = 'status-badge status-running';

    try {
      await fetch('/api/command?cmd=' + encodeURIComponent(cmd));
    } catch(e) {}

    pollProgress(true);
  }

  async function addURL() {
    const url = urlInput.value.trim();
    if (!url) return;

    addBtn.disabled = true;
    logSection.classList.add('active');
    logOutput.innerHTML = '';
    statusBadge.textContent = 'DOWNLOADING...';
    statusBadge.className = 'status-badge status-running';

    const params = new URLSearchParams({ url });
    if (labelInput.value.trim()) params.set('label', labelInput.value.trim());
    if (startInput.value.trim()) params.set('start', startInput.value.trim());
    if (endInput.value.trim()) params.set('end', endInput.value.trim());

    try {
      await fetch('/api/add?' + params.toString());
    } catch(e) {}

    // Poll for progress
    pollProgress(false);
  }

  async function pollProgress(toolMode) {
    try {
      const res = await fetch('/api/progress');
      const data = await res.json();

      // Update log
      logOutput.innerHTML = data.logs.map(line => {
        let cls = '';
        if (line.includes('Done!') || line.includes('Downloaded:')) cls = 'success';
        else if (line.includes('Error') || line.includes('failed') || line.includes('Failed')) cls = 'error';
        else if (line.includes('Downloading') || line.includes('Running') || line.includes('Fetching')) cls = 'info';
        return `<div class="${cls}">${escapeHtml(line)}</div>`;
      }).join('');

      logOutput.scrollTop = logOutput.scrollHeight;

      if (data.running) {
        statusBadge.textContent = 'PROCESSING...';
        statusBadge.className = 'status-badge status-running';
        setTimeout(() => pollProgress(toolMode), 1000);
      } else {
        const hasError = data.logs.some(l => l.includes('Error') || l.includes('failed'));
        if (hasError) {
          statusBadge.textContent = 'ERROR';
          statusBadge.className = 'status-badge status-error';
        } else {
          statusBadge.textContent = 'DONE';
          statusBadge.className = 'status-badge status-done';
        }
        addBtn.disabled = false;
        document.querySelectorAll('.tool-btn').forEach(b => b.disabled = false);
        if (!toolMode) {
          urlInput.value = '';
          labelInput.value = '';
          startInput.value = '';
          endInput.value = '';
          urlInput.focus();
        }
        loadStats();
      }
    } catch(e) {
      setTimeout(() => pollProgress(toolMode), 2000);
    }
  }

  async function loadStats() {
    try {
      const res = await fetch('/api/stats');
      const data = await res.json();

      document.getElementById('stat-clips').textContent = data.total;
      document.getElementById('stat-duration').textContent = data.duration;
      document.getElementById('stat-sources').textContent = data.sources;

      const list = document.getElementById('clip-list');
      if (data.clips.length === 0) {
        list.innerHTML = '<div class="empty-state">No clips yet. Add a YouTube URL above!</div>';
      } else {
        list.innerHTML = data.clips.reverse().map(c =>
          `<div class="clip-item">
            <span class="clip-text">${escapeHtml(c.text || '(no text)')}</span>
            <span class="clip-dur">${(c.duration || 0).toFixed(1)}s</span>
          </div>`
        ).join('');
      }
    } catch(e) {
      console.error('Failed to load stats:', e);
    }
  }

  function escapeHtml(str) {
    return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  // Load on start
  loadStats();
</script>
</body>
</html>"""


class ClipManagerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default HTTP logging

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif parsed.path == "/api/stats":
            stats = get_library_stats()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(stats).encode("utf-8"))

        elif parsed.path == "/api/progress":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(progress).encode("utf-8"))

        elif parsed.path == "/api/add":
            params = parse_qs(parsed.query)
            url = params.get("url", [""])[0]
            label = params.get("label", [None])[0]
            start = params.get("start", [None])[0]
            end = params.get("end", [None])[0]

            if url and not progress["running"]:
                thread = threading.Thread(
                    target=download_and_parse,
                    args=(url, label, start, end),
                    daemon=True
                )
                thread.start()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        elif parsed.path == "/api/command":
            params = parse_qs(parsed.query)
            cmd = params.get("cmd", [""])[0]

            if cmd and not progress["running"]:
                thread = threading.Thread(
                    target=run_command,
                    args=(cmd,),
                    daemon=True
                )
                thread.start()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    print("=" * 50)
    print("  Mosaic TTS — Clip Manager UI")
    print("=" * 50)
    print()
    print(f"  Open: http://localhost:{PORT}")
    print(f"  Press Ctrl+C to stop")
    print()

    server = HTTPServer(("127.0.0.1", PORT), ClipManagerHandler)

    # Auto-open browser
    webbrowser.open(f"http://localhost:{PORT}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()
