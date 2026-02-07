"""
Discord Mosaic TTS — Launcher
================================
Starts all three services (message reader, main engine, coverage helper)
as parallel subprocesses and manages their lifecycle.

Usage:
    python run.py                  # start all three
    python run.py --no-helper      # skip the coverage monitor
"""

import subprocess
import sys
import time
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Launch Discord Mosaic TTS")
    parser.add_argument("--no-helper", action="store_true",
                        help="Don't start the coverage helper")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    python = sys.executable  # use the same Python interpreter

    scripts = {
        "Message Reader": os.path.join(base_dir, "read_messages.py"),
        "Main Engine":    os.path.join(base_dir, "main.py"),
    }
    if not args.no_helper:
        scripts["Coverage Helper"] = os.path.join(base_dir, "helper.py")

    print("=" * 50)
    print("  Discord Mosaic TTS — Launcher")
    print("=" * 50)
    print()

    processes = {}
    for name, script in scripts.items():
        print(f"  Starting {name}...")
        p = subprocess.Popen([python, script], cwd=base_dir)
        processes[name] = p

    print(f"\n  {len(processes)} process(es) running. Press Ctrl+C to stop all.\n")

    try:
        while True:
            for name, p in list(processes.items()):
                ret = p.poll()
                if ret is not None:
                    print(f"\n⚠ {name} exited with code {ret}")
                    # Shut everything down if a critical process dies
                    raise KeyboardInterrupt
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nShutting down all processes...")
        for name, p in processes.items():
            if p.poll() is None:
                p.terminate()
                print(f"  ✔ Stopped {name}")

        # Give them a moment to clean up
        for p in processes.values():
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

        print("\nAll stopped. ✔")


if __name__ == "__main__":
    main()
