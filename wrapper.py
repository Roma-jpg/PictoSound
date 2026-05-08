# ===== wrapper.py =====
#!/usr/bin/env python3
"""
wrapper.py - end-to-end conversion: audio -> spectrogram PNG -> audio.
Keeps the intermediate PNG for inspection.
"""

import os
import subprocess
import sys
from pathlib import Path

# ----- configuration (adjust to your environment) -----
VENV_PYTHON = sys.executable
SCRIPT_DIR = Path(__file__).resolve().parent
ENCODER = SCRIPT_DIR / "audio_to_spectrogram.py"
DECODER = SCRIPT_DIR / "spectrogram_to_audio.py"
# -------------------------------------------------------


def run_script(script_path: Path, *args: str) -> int:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [str(VENV_PYTHON), str(script_path), *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
        if proc.stdout:
            sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        return proc.returncode
    except FileNotFoundError:
        print(f"ERROR: Python not found at {VENV_PYTHON}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: {script_path.name} failed: {e}", file=sys.stderr)
        return 1


def main():
    if len(sys.argv) != 3:
        print("Usage: python wrapper.py input_audio output_audio", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    png_file = f"{Path(input_file).stem}_spectrogram.png"

    # step 1 - encode to png
    print(f"Encoding {input_file} -> {png_file}")
    ret = run_script(ENCODER, input_file, png_file, "--metadata-format", "json")
    if ret != 0:
        sys.exit("Encoding failed.")

    # step 2 - decode to final audio
    print(f"\nDecoding {png_file} -> {output_file}")
    ret = run_script(DECODER, png_file, output_file, "--output-format", "auto")
    if ret != 0:
        sys.exit("Decoding failed.")

    print(f"\nDone. Intermediate PNG: {png_file}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()