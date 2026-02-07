import os
import json
import re
import time
import numpy as np
import sounddevice as sd
from sentence_transformers import SentenceTransformer
from pydub import AudioSegment
from playsound import  playsound

# ---------------- CONFIG ----------------
VIRTUAL_CABLE_ID = 12       # Discord output
PC_OUTPUT_ID = None         # Default speakers
MICRO_CLIPS_JSON = r"C:\Users\Benja\OneDrive\Documents\Discord TTS\Clip_creation\clips\micro_clips.json"
MICRO_CLIPS_DIR = r"C:\Users\Benja\OneDrive\Documents\Discord TTS\Clip_creation\clips"
DISCORD_LOG_FILE = r"C:\Users\Benja\OneDrive\Documents\Discord TTS\discord_export.txt"
CHECK_INTERVAL = 1.0         # seconds between checking for new messages
SEMANTIC_THRESHOLD = 0.6      # similarity threshold for clips
MULTI_CLIP_MODE = True
MAX_CLIPS_PER_BEAT = 3
# ----------------------------------------

# --- Load micro-clips metadata ---
with open(MICRO_CLIPS_JSON, "r", encoding="utf-8") as f:
    clips_data = json.load(f)
clips = clips_data["clips"]

# --- Load embeddings ---
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
for clip in clips:
    clip["embedding"] = embed_model.encode(clip["text"])

def play_clip_dual(clip_audio: AudioSegment):
    samples = np.array(clip_audio.get_array_of_samples()).astype(np.float32)
    samples /= np.iinfo(clip_audio.array_type).max  # normalize to -1.0 → 1.0

    if clip_audio.channels == 2:
        samples = samples.reshape((-1, 2))
    else:
        samples = samples.reshape((-1, 1))

    print(f"Playing clip for {clip_audio.duration_seconds:.2f}s...")

    # Play on PC speakers asynchronously
    sd.play(samples, samplerate=clip_audio.frame_rate, device=PC_OUTPUT_ID, blocking=False)

    # Wait a tiny bit to avoid overlap (optional, can also just block the virtual cable)
    sd.wait()

    # Play on virtual cable synchronously
    sd.play(samples, samplerate=clip_audio.frame_rate, device=VIRTUAL_CABLE_ID, blocking=True)


# --- Helper: find top N clips for a beat ---
def find_best_clips(phrase_text, top_n=1, min_similarity=SEMANTIC_THRESHOLD):
    phrase_emb = embed_model.encode(phrase_text)
    sims = []
    for clip in clips:
        sim = np.dot(phrase_emb, clip["embedding"]) / (np.linalg.norm(phrase_emb) * np.linalg.norm(clip["embedding"]))
        sims.append((sim, clip))
    sims.sort(reverse=True, key=lambda x: x[0])
    top_clips = [clip for sim_val, clip in sims if sim_val >= min_similarity][:top_n]
    return top_clips

# --- Helper: split message into semantic beats ---
def split_semantic_beats(text):
    split_regex = r'[.!?]| but | and | so '
    beats = [b.strip() for b in re.split(split_regex, text) if b.strip()]
    if not beats:
        beats = [text.strip()]  # fallback
    return beats

# --- Helper: read new messages from discord_export.txt ---
def read_new_messages(log_file, last_pos):
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
        print(f"File {log_file} not found, retrying...")
    return messages, last_pos

# --- Main loop ---
print("Starting live Discord mosaic TTS...")
last_file_pos = 0

try:
    while True:
        messages, last_file_pos = read_new_messages(DISCORD_LOG_FILE, last_file_pos)
        if messages:
            print(f"Detected {len(messages)} new message(s).")
        for msg in messages:
            print(f"\nProcessing message: '{msg}'")
            beats = split_semantic_beats(msg)
            for beat in beats:
                print(f"  Beat: '{beat}'")
                top_n = MAX_CLIPS_PER_BEAT if MULTI_CLIP_MODE else 1
                selected_clips = find_best_clips(beat, top_n=top_n)
                if not selected_clips:
                    print(f"    No clip matched above threshold ({SEMANTIC_THRESHOLD})")
                    # fallback: pick the top clip anyway
                    selected_clips = find_best_clips(beat, top_n=1, min_similarity=0)
                for clip in selected_clips:
                    clip_path = os.path.join(MICRO_CLIPS_DIR, clip["clip_file"])
                    if not os.path.exists(clip_path):
                        print(f"    Clip file not found: {clip_path}")
                        continue
                    clip_audio = AudioSegment.from_file(clip_path)
                    print(f"    Playing clip: '{clip['clip_file']}' ({clip['text']})")
                    play_clip_dual(clip_audio)
        time.sleep(CHECK_INTERVAL)
except KeyboardInterrupt:
    print("Stopping live mosaic TTS.")

