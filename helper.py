"""
Discord Mosaic TTS — Semantic Coverage Helper
================================================
Monitors incoming messages and shows how well your clip library covers
the semantic content. Displays a live histogram and saves a report on exit.

Usage:
    python helper.py
"""

import json
import time

import matplotlib.pyplot as plt

import config
import utils


def main():
    print("="*50)
    print("  Semantic Coverage Monitor")
    print("="*50)

    # Load clips + embeddings
    import os
    if not os.path.exists(config.MICRO_CLIPS_JSON):
        print(f"⚠ No clip catalog found. Run 'python clip_parser.py' first.")
        raise SystemExit(1)

    with open(config.MICRO_CLIPS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    clips = data.get("clips", [])

    print(f"Loaded {len(clips)} clips, computing embeddings...")
    for clip in clips:
        clip["embedding"] = utils.encode_text(clip["text"])
    print("Ready ✔\n")

    # Skip to end of existing log — only monitor NEW messages
    last_pos = 0
    try:
        if os.path.exists(config.DISCORD_LOG_FILE):
            last_pos = os.path.getsize(config.DISCORD_LOG_FILE)
            print(f"Skipped to end of log. Waiting for new messages...")
    except OSError:
        pass

    beat_sims: list[tuple[str, float]] = []
    threshold = config.HELPER_THRESHOLD

    try:
        while True:
            messages, last_pos = utils.read_new_messages(config.DISCORD_LOG_FILE, last_pos)
            if messages:
                print(f"\n── {len(messages)} new message(s) ──")

            for msg in messages:
                beats = utils.split_semantic_beats(msg)
                for beat in beats:
                    # Find best similarity
                    best_sim = 0.0
                    for clip in clips:
                        sim = utils.cosine_similarity(
                            utils.encode_text(beat),
                            clip["embedding"],
                        )
                        if sim > best_sim:
                            best_sim = sim

                    beat_sims.append((beat, best_sim))
                    icon = "✔" if best_sim >= threshold else "✖"
                    print(f"  [{icon} {best_sim:.2f}] {beat}")

            # Live histogram
            if config.UPDATE_GRAPH and beat_sims:
                sims = [s for _, s in beat_sims]
                plt.clf()
                plt.hist(sims, bins=20, color="skyblue", edgecolor="black")
                plt.axvline(threshold, color="red", linestyle="dashed",
                            linewidth=2, label=f"Threshold ({threshold})")
                plt.title("Semantic Coverage — Similarity per Beat")
                plt.xlabel("Best Cosine Similarity")
                plt.ylabel("Beat Count")
                plt.legend()
                plt.tight_layout()
                plt.pause(0.05)

            time.sleep(config.CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nStopping coverage monitor.")

        # Save report
        if beat_sims:
            report = [
                {"beat": b, "similarity": round(s, 4)}
                for b, s in beat_sims
            ]
            with open(config.COVERAGE_REPORT, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            covered = sum(1 for _, s in beat_sims if s >= threshold)
            total = len(beat_sims)
            pct = (covered / total * 100) if total else 0
            print(f"\n  Coverage: {covered}/{total} beats ({pct:.0f}%) above {threshold}")
            print(f"  Report saved: {config.COVERAGE_REPORT}")
        print("Done ✔")


if __name__ == "__main__":
    main()
