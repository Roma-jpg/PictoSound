#!/usr/bin/env python3
"""
wrapper.py
==========
Converts any audio file to a spectrogram PNG and then back to a WAV file.
Uses the specified virtual environment's Python interpreter.
Intermediate PNG is kept (not deleted).

Usage:
    python wrapper.py input.mp3 output.wav
"""

import sys
import subprocess
import os
from pathlib import Path

# Hardcoded path to your venv Python interpreter (Windows)
VENV_PYTHON = r"C:\Users\Ромэо\PycharmProjects\Scripts\.venv2\Scripts\python.exe"

def run_script(script_name, *args):
    """Run a Python script with the venv interpreter, forcing UTF-8 output."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [VENV_PYTHON, script_name] + list(args)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",       # force UTF-8 decoding of stdout/stderr
            env=env
        )
        # Print outputs (they may still contain non‑printable chars, so handle errors)
        if result.stdout:
            print(result.stdout.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
        if result.stderr:
            print(result.stderr.encode('utf-8', errors='replace').decode('utf-8', errors='replace'), file=sys.stderr)
        return result.returncode
    except Exception as e:
        print(f"Failed to run {script_name}: {e}", file=sys.stderr)
        return 1

def main():
    if len(sys.argv) != 3:
        print("Usage: python wrapper.py input_audio output.wav")
        sys.exit(1)

    input_audio = sys.argv[1]
    output_wav = sys.argv[2]

    # Create intermediate PNG name based on input filename
    input_stem = Path(input_audio).stem
    png_path = f"{input_stem}_spectrogram.png"

    # Step 1: audio -> PNG spectrogram
    print(f"Step 1: Encoding {input_audio} to {png_path}")
    ret = run_script("audio_to_spectrogram.py", input_audio, png_path)
    if ret != 0:
        print("Encoding failed.")
        sys.exit(ret)

    # Step 2: PNG spectrogram -> WAV audio (and possibly MP3 + cover)
    print(f"\nStep 2: Decoding {png_path} to {output_wav}")
    ret = run_script("spectrogram_to_audio.py", png_path, output_wav)
    if ret != 0:
        print("Decoding failed.")
        sys.exit(ret)

    print(f"\nDone! PNG spectrogram retained at: {png_path}")
    print(f"Output WAV: {output_wav}")

if __name__ == "__main__":
    main()