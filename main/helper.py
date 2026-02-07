import os
import re
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
DISCORD_LOG_FILE = r"C:\Users\Benja\OneDrive\Documents\Discord TTS\discord_export.txt"
MICRO_CLIPS_JSON = r"C:\Users\Benja\OneDrive\Documents\Discord TTS\Clip_creation\clips\micro_clips.json"
SEMANTIC_THRESHOLD = 0.75
CHECK_INTERVAL = 2.0  # seconds between checking for new messages
UPDATE_GRAPH = True    # whether to update histogram live
# ----------------------------------------

# --- Load micro-clips ---
with open(MICRO_CLIPS_JSON, "r", encoding="utf-8") as f:
    clips_data = json.load(f)
clips = clips_data["clips"]

# --- Load embeddings ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
for clip in clips:
    clip["embedding"] = embed_model.encode(clip["text"])

# --- Helpers ---
def split_semantic_beats(text):
    split_regex = r'[.!?]| but | and | so '
    beats = [b.strip() for b in re.split(split_regex, text) if b.strip()]
    if not beats:
        beats = [text.strip()]
    return beats

def find_best_clip_similarity(phrase_text):
    phrase_emb = embed_model.encode(phrase_text)
    best_sim = -1
    for clip in clips:
        clip_emb = clip["embedding"]
        sim = np.dot(phrase_emb, clip_emb) / (np.linalg.norm(phrase_emb) * np.linalg.norm(clip_emb))
        if sim > best_sim:
            best_sim = sim
    return best_sim

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

# --- Real-time monitoring ---
print("Starting real-time semantic coverage helper...")
last_file_pos = 0
beat_sims = []

try:
    while True:
        messages, last_file_pos = read_new_messages(DISCORD_LOG_FILE, last_file_pos)
        if messages:
            print(f"\nDetected {len(messages)} new message(s).")
        for msg in messages:
            beats = split_semantic_beats(msg)
            for beat in beats:
                sim = find_best_clip_similarity(beat)
                beat_sims.append((beat, sim))
                status = "✔" if sim >= SEMANTIC_THRESHOLD else "✖"
                print(f"[{status} {sim:.2f}] {beat}")
        
        # --- Optional: update histogram live ---
        if UPDATE_GRAPH and beat_sims:
            sims = [s for b, s in beat_sims]
            plt.clf()
            plt.hist(sims, bins=20, color='skyblue', edgecolor='black')
            plt.axvline(SEMANTIC_THRESHOLD, color='red', linestyle='dashed', linewidth=2, label='Threshold')
            plt.title("Semantic Coverage (Similarity per Beat)")
            plt.xlabel("Similarity")
            plt.ylabel("Number of Beats")
            plt.legend()
            plt.pause(0.05)  # brief pause to render

        time.sleep(CHECK_INTERVAL)

except KeyboardInterrupt:
    print("\nStopping real-time semantic coverage helper.")
    # Save final report
    with open("semantic_coverage_report.json", "w", encoding="utf-8") as f:
        json.dump([{"beat": b, "similarity": s} for b, s in beat_sims], f, indent=2)
    print("Saved report to semantic_coverage_report.json")
