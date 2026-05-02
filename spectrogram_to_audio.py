"""
spectrogram_to_audio.py
========================
Decodes a PNG produced by audio_to_spectrogram.py back into:
  - A stereo audio file (WAV or MP3, default MP3)
  - Printed metadata (artist, album, title)
  - Optionally saved album cover (cover_out.png)
  - Threaded reconstruction (left/right channels in parallel)
  - Cleans up temporary WAV file when final output is MP3

Usage:
    python spectrogram_to_audio.py input.png [output_audio] [--save-cover]
    If output_audio is omitted, the script uses the input PNG's basename + '.mp3'
"""

import numpy as np
import librosa
import cv2
import soundfile as sf
import sys
import struct
import os
import subprocess
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor

from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC, TIT2, TPE1, TALB

MAGIC = b'\xDE\xAD\xBE\xEF\x53\x50\x45\x43'


def read_uint32_row(canvas, row, col_start=0):
    """Read a big‑endian uint32 from a horizontal row of pixels."""
    bytes_ = bytes(int(canvas[row, col_start + i, 0]) for i in range(4))
    return struct.unpack('>I', bytes_)[0]


def spectrogram_to_audio(spec_norm, sr, n_fft, hop_length, min_db):
    norm = spec_norm.astype(np.float32) / 255.0
    mag_db = norm * (-min_db) + min_db
    magnitude = librosa.db_to_amplitude(mag_db)
    audio = librosa.griffinlim(
        magnitude,
        n_iter=64,
        hop_length=hop_length,
        win_length=n_fft,
    )
    return audio


def read_bytes_from_zone(canvas, x_start, y_start, zone_width, byte_count):
    result = bytearray()
    for i in range(byte_count):
        col = x_start + (i % zone_width)
        row = y_start + (i // zone_width)
        if row >= canvas.shape[0] or col >= canvas.shape[1]:
            break
        result.append(int(canvas[row, col, 0]))
    return bytes(result)


def decode_metadata(raw_bytes):
    if len(raw_bytes) < 4:
        return '', '', ''
    payload_len = struct.unpack('>I', raw_bytes[:4])[0]
    payload = raw_bytes[4:4 + payload_len].decode('utf-8', errors='replace')
    parts = payload.split('\x00')
    while len(parts) < 3:
        parts.append('')
    return parts[0], parts[1], parts[2]


def convert_wav_to_mp3_with_tags(wav_path, mp3_path, artist, album, title,
                                 cover_rgb=None, bitrate='192k'):
    """
    Convert WAV to MP3 using ffmpeg.
    Embeds cover art (max 500px) and ID3v2.3 tags directly with ffmpeg.
    """
    if not shutil.which('ffmpeg'):
        print("  ffmpeg not found – skipping MP3 conversion.")
        return False

    cover_temp = None
    try:
        # Start command: input WAV
        cmd = ['ffmpeg', '-y', '-i', wav_path]

        # --- Optional cover art ---
        if cover_rgb is not None:
            h, w = cover_rgb.shape[:2]
            max_dim = 500
            if h > max_dim or w > max_dim:
                scale = max_dim / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                cover_rgb = cv2.resize(cover_rgb, (new_w, new_h),
                                       interpolation=cv2.INTER_AREA)
                print(f"  Cover resized to {new_w}x{new_h} for thumbnails")

            # Encode to JPEG and write to a temporary file
            cover_bgr = cv2.cvtColor(cover_rgb, cv2.COLOR_RGB2BGR)
            success, jpeg_bytes = cv2.imencode('.jpg', cover_bgr,
                                               [cv2.IMWRITE_JPEG_QUALITY, 90])
            if success:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                    f.write(jpeg_bytes.tobytes())
                    cover_temp = f.name
                cmd += ['-i', cover_temp]
                # Map audio (first input) and image (second input)
                cmd += ['-map', '0:a', '-map', '1:v',
                        '-c:v', 'copy',
                        '-disposition:v', 'attached_pic']
                print("  Cover image prepared for embedding")
            else:
                print("  Warning: Could not encode cover as JPEG – skipping cover art.")

        # Audio encoding
        cmd += ['-codec:a', 'libmp3lame', '-b:a', bitrate]

        # Force ID3v2.3 (more compatible than ffmpeg's default v2.4)
        cmd += ['-id3v2_version', '3']

        # Metadata tags (only added if the value is not empty)
        if title:
            cmd += ['-metadata', f'title={title}']
        if artist:
            cmd += ['-metadata', f'artist={artist}']
        if album:
            cmd += ['-metadata', f'album={album}']

        cmd.append(mp3_path)

        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("  MP3 created with embedded tags and cover (if provided)")
        return True

    except subprocess.CalledProcessError:
        print("  ffmpeg encoding/tagging failed – MP3 not created.")
        return False
    except Exception as e:
        print(f"  Error during MP3 creation: {e}")
        return False
    finally:
        # Clean up temporary cover file
        if cover_temp and os.path.exists(cover_temp):
            try:
                os.unlink(cover_temp)
            except OSError:
                pass


def main(input_image, output_audio=None, save_cover=False):
    print(f"Loading image: {input_image}")
    raw = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Cannot open image: {input_image}")

    # Convert BGRA → RGBA (consistent with encoder)
    if raw.shape[2] == 4:
        canvas = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA)
    else:
        canvas = cv2.cvtColor(raw, cv2.COLOR_BGR2RGBA)

    # Verify magic
    magic_found = bytes(int(canvas[i, 0, 0]) for i in range(8))
    if magic_found != MAGIC:
        raise ValueError(
            f"Magic mismatch – not a valid spectrogram PNG.\n"
            f"Expected: {MAGIC.hex()}\nFound   : {magic_found.hex()}"
        )

    # Read manifest
    sr            = read_uint32_row(canvas,  8)
    n_fft         = read_uint32_row(canvas,  9)
    hop_length    = read_uint32_row(canvas, 10)
    freq_bins     = read_uint32_row(canvas, 11)
    time_frames   = read_uint32_row(canvas, 12)
    spec_x        = read_uint32_row(canvas, 13)
    right_y       = read_uint32_row(canvas, 14)
    meta_x        = read_uint32_row(canvas, 15)
    meta_zone_w   = read_uint32_row(canvas, 16)
    meta_text_rows= read_uint32_row(canvas, 17)
    cover_y_offset= read_uint32_row(canvas, 18)
    cover_w       = read_uint32_row(canvas, 19)
    cover_h       = read_uint32_row(canvas, 20)
    cover_present = read_uint32_row(canvas, 21)
    min_db_raw    = read_uint32_row(canvas, 22)
    meta_byte_count= read_uint32_row(canvas, 23)

    # Convert min_db back to signed
    if min_db_raw & 0x80000000:
        min_db = -((0x100000000 - min_db_raw) & 0xFFFFFFFF)
    else:
        min_db = min_db_raw

    print(f"\n── Manifest ─────────────────────────────")
    print(f"  Sample rate   : {sr} Hz")
    print(f"  n_fft         : {n_fft}")
    print(f"  hop_length    : {hop_length}")
    print(f"  Freq bins     : {freq_bins}")
    print(f"  Time frames   : {time_frames}")
    print(f"  Min dB        : {min_db}")
    print(f"  Spec X start  : {spec_x}")
    print(f"  Meta X start  : {meta_x}")
    print(f"  Meta zone W   : {meta_zone_w}")
    print(f"  Cover present : {cover_present}")
    if cover_present:
        print(f"  Cover offset  : row {cover_y_offset}, size {cover_w} x {cover_h}")

    # Extract spectrograms
    left_slice  = canvas[0:freq_bins, spec_x:spec_x+time_frames, 0]
    right_slice = canvas[right_y:right_y+freq_bins, spec_x:spec_x+time_frames, 0]

    # Flip vertical axis (encoder stored low frequencies at bottom)
    left_slice  = left_slice[::-1]
    right_slice = right_slice[::-1]

    print("\nReconstructing channels in parallel (threading)...")
    # Threaded reconstruction
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_left = executor.submit(
            spectrogram_to_audio, left_slice, sr, n_fft, hop_length, min_db
        )
        future_right = executor.submit(
            spectrogram_to_audio, right_slice, sr, n_fft, hop_length, min_db
        )
        audio_left = future_left.result()
        audio_right = future_right.result()

    # Trim to equal length
    min_len = min(len(audio_left), len(audio_right))
    audio_left  = audio_left[:min_len]
    audio_right = audio_right[:min_len]

    # Normalise
    peak = max(np.max(np.abs(audio_left)), np.max(np.abs(audio_right)), 1e-9)
    audio_left  = audio_left  / peak * 0.95
    audio_right = audio_right / peak * 0.95

    stereo = np.stack([audio_left, audio_right], axis=1)

    # Determine final output filename
    if output_audio is None:
        base = os.path.splitext(input_image)[0]
        output_audio = base + '.mp3'          # default MP3
        print(f"Output audio not specified, using: {output_audio}")

    # Decide target format and create a temporary WAV file
    target_ext = os.path.splitext(output_audio)[1].lower()
    if target_ext not in ('.wav', '.mp3'):
        print(f"Warning: unknown extension '{target_ext}', defaulting to .mp3")
        target_ext = '.mp3'
        output_audio = os.path.splitext(output_audio)[0] + '.mp3'

    # Create a temporary WAV file (will be deleted after conversion if needed)
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_wav_path = temp_wav.name
    temp_wav.close()
    sf.write(temp_wav_path, stereo, sr)
    print(f"\nTemporary WAV written: {temp_wav_path}")

    # Decode metadata text
    raw_meta = read_bytes_from_zone(canvas, meta_x, 0, meta_zone_w, meta_byte_count)
    artist, album, title = decode_metadata(raw_meta)

    print(f"\n── Metadata from PNG ─────────────────────")
    print(f"  Artist : {artist or '(none)'}")
    print(f"  Album  : {album or '(none)'}")
    print(f"  Title  : {title or '(none)'}")

    # Extract cover image (if present)
    cover_rgb = None
    if cover_present and cover_w > 0 and cover_h > 0:
        canvas_h, canvas_w = canvas.shape[:2]
        if (cover_y_offset + cover_h <= canvas_h and
            meta_x + cover_w <= canvas_w):
            cover_rgba = canvas[cover_y_offset:cover_y_offset+cover_h,
                                meta_x:meta_x+cover_w, :]
            cover_rgb = cv2.cvtColor(cover_rgba, cv2.COLOR_RGBA2RGB)
            print(f"  Cover extracted: {cover_w} x {cover_h} pixels")
        else:
            print(f"  ERROR: Cover coordinates out of bounds")

    # Optionally save cover as PNG
    if save_cover and cover_rgb is not None:
        cover_out = os.path.splitext(output_audio)[0] + '_cover.png'
        cv2.imwrite(cover_out, cv2.cvtColor(cover_rgb, cv2.COLOR_RGB2BGR))
        print(f"  Cover saved  : {cover_out}")

    # Final output generation
    if target_ext == '.wav':
        # Move temp WAV to final destination
        shutil.move(temp_wav_path, output_audio)
        print(f"\nSaved WAV : {output_audio}")
        print(f"  Duration : {min_len/sr:.2f} s")
        print(f"  Channels : stereo")
    else:  # target_ext == '.mp3'
        print("\n── Converting to MP3 ─────────────────────")
        ok = convert_wav_to_mp3_with_tags(
            wav_path=temp_wav_path,
            mp3_path=output_audio,
            artist=artist,
            album=album,
            title=title,
            cover_rgb=cover_rgb,
            bitrate='192k'
        )
        if ok:
            print(f"  MP3 created : {output_audio}")
            # Delete the temporary WAV file (cleanup)
            os.unlink(temp_wav_path)
            print("  Temporary WAV deleted (kept PNG spectrogram).")
        else:
            print("  MP3 conversion failed – keeping temporary WAV as fallback.")
            fallback_wav = os.path.splitext(output_audio)[0] + '.wav'
            shutil.move(temp_wav_path, fallback_wav)
            print(f"  Fallback WAV saved as: {fallback_wav}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python spectrogram_to_audio.py input.png [output_audio] [--save-cover]")
        sys.exit(1)

    _input = sys.argv[1]
    _output = None
    _save_cover = False

    # Parse arguments: optional output and flag
    for arg in sys.argv[2:]:
        if arg == '--save-cover':
            _save_cover = True
        elif _output is None:
            _output = arg
        else:
            print(f"Ignoring extra argument: {arg}")

    main(_input, _output, _save_cover)