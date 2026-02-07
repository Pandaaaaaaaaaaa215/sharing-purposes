# Discord Mosaic TTS

A semantic soundboard that listens to your Discord messages in real-time and responds by playing the most contextually relevant audio clips — stitched together like a mosaic.

## How It Works

1. **You type** in a Discord voice-chat text channel
2. **Message Reader** (`read_messages.py`) captures your messages via a Discord bot
3. **Main Engine** (`main.py`) splits each message into semantic "beats", finds the best-matching audio clips using sentence embeddings, and plays them through both your speakers and a virtual audio cable (so Discord hears it too)
4. **Coverage Helper** (`helper.py`) shows a live histogram of how well your clip library covers incoming messages

## Setup

### Prerequisites

- **Python 3.10+**
- **FFmpeg** on your PATH ([download](https://ffmpeg.org/download.html))
- **VB-Cable** or similar virtual audio cable ([download](https://vb-audio.com/Cable/))

### Install

```bash
pip install -r requirements.txt
```

> **Note:** Whisper also needs `ffmpeg` installed system-wide.

### Configure

Edit **`config.py`**:

| Setting | What to change |
|---|---|
| `BOT_TOKEN` | Your Discord bot token |
| `MY_USER_ID` | Your Discord user ID (right-click → Copy User ID) |
| `MY_CHANNEL_ID` | The VC text channel ID |
| `VIRTUAL_CABLE_ID` | Your virtual cable's device index (run `python -m sounddevice` to list devices) |
| `WHISPER_MODEL` | `tiny` for speed, `small`/`medium` for accuracy |

### Prepare Your Clips

1. Drop raw audio files (mp3, wav, ogg, flac, m4a) into `Clip_creation/clips/raw/`
2. Run the clip parser:

```bash
python clip_parser.py
```

This will:
- Transcribe every file with OpenAI Whisper
- Split on silence boundaries into micro-clips
- Save sliced WAVs to `Clip_creation/clips/sliced/`
- Generate `Clip_creation/clips/micro_clips.json` with text + timing metadata

**Re-run with `--force`** to reprocess everything, or just add new files and run again (it skips unchanged files).

### Run

```bash
# Start everything:
python run.py

# Skip the coverage graph:
python run.py --no-helper
```

Or run scripts individually:
```bash
python read_messages.py    # Discord bot
python main.py             # TTS engine
python helper.py           # Coverage monitor
```

## Project Structure

```
C:\Users\Benja\OneDrive\Documents\Discord TTS\
├── config.py          # All settings in one place
├── utils.py           # Shared functions (embeddings, beat splitting, etc.)
├── clip_parser.py     # Whisper-based audio transcription + slicing
├── read_messages.py   # Discord bot — captures messages to log file
├── main.py            # Core TTS engine — matches & plays clips
├── helper.py          # Live semantic coverage monitor
├── run.py             # Launcher — starts all processes
├── requirements.txt
├── discord_export.txt              # Live message log
├── semantic_coverage_report.json   # Coverage stats
└── Clip_creation/
    └── clips/
        ├── raw/           # Drop your source audio files here
        ├── sliced/        # Auto-generated micro-clip WAVs
        └── micro_clips.json  # Auto-generated catalog
```

## Tips

- **Finding your audio device IDs:** Run `python -m sounddevice` to list all devices with their index numbers
- **Improving match quality:** Use `small` or `medium` Whisper models for better transcription, and add more clips to cover common phrases
- **Threshold tuning:** Lower `SEMANTIC_THRESHOLD` in config.py if too few clips match; raise it if irrelevant clips play
- **The coverage helper** is your best friend for finding gaps — look for clusters of red ✖ marks to know what phrases need more clips
