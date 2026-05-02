"""
audio_to_spectrogram.py
========================
Encodes a stereo audio file into a single PNG spectrogram image with:
  - A 64-pixel-wide header strip (magic bytes + manifest)
  - Left channel spectrogram (top half)
  - Right channel spectrogram (bottom half)
  - Metadata zone (right side): artist, album, title as raw bytes
  - Optional album cover (below metadata, max 1024x1024)

Image format: 8-bit RGBA PNG

Usage:
    python audio_to_spectrogram.py input.mp3 [output.png]
    If output.png is omitted, the script tries to use the song title (from metadata)
    or falls back to the input file's basename.
"""

import numpy as np
import librosa
import cv2
import sys
import struct
import os
import re
from mutagen import File as MutagenFile
from mutagen.id3 import ID3, APIC
from mutagen.flac import FLAC, Picture
from mutagen.mp4 import MP4, MP4Cover
import io

# ── Constants ────────────────────────────────────────────────────────────────
HEADER_WIDTH   = 64
N_FFT          = 2048
HOP_LENGTH     = 512
MIN_DB         = -80
MAX_COVER_SIZE = 1024
MAGIC          = b'\xDE\xAD\xBE\xEF\x53\x50\x45\x43'


# ── Metadata extraction from audio file ──────────────────────────────────────

def extract_cover_from_mutagen(audio_file):
    """
    Extract cover art as a CV2 RGB image (numpy array) from any Mutagen object.
    Returns None if no cover found.
    """
    cover_data = None

    # 1) ID3 (MP3, AIFF, etc.)
    if hasattr(audio_file, 'tags') and audio_file.tags is not None:
        if hasattr(audio_file.tags, 'getall'):
            for apic in audio_file.tags.getall('APIC'):
                cover_data = apic.data
                break

    # 2) FLAC / Ogg FLAC (direct 'pictures' attribute)
    if cover_data is None and hasattr(audio_file, 'pictures'):
        pics = audio_file.pictures
        if pics:
            cover_data = pics[0].data

    # 3) MP4 / M4A ('covr' atom)
    if cover_data is None and hasattr(audio_file, 'get'):
        covr = audio_file.get('covr')
        if covr:
            cover_data = covr[0]

    # 4) Fallback: some containers store cover under 'APIC:' key
    if cover_data is None and hasattr(audio_file, '__contains__'):
        if 'APIC:' in audio_file:
            cover_data = audio_file['APIC:'].data

    if cover_data is None:
        return None

    # Decode bytes to OpenCV image
    nparr = np.frombuffer(cover_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize if too large
    h, w = img.shape[:2]
    if h > MAX_COVER_SIZE or w > MAX_COVER_SIZE:
        scale = MAX_COVER_SIZE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def get_metadata(filepath):
    """
    Returns (artist, album, title, cover_image) extracted from the audio file.
    cover_image is a numpy RGB array or None.
    """
    artist = album = title = ''
    cover_img = None

    try:
        # Open the file WITH full metadata (needed for cover art)
        full_meta = MutagenFile(filepath, easy=False)
        if full_meta is None:
            print("Warning: No metadata tags found.")
            return artist, album, title, cover_img

        # ----- Text metadata extraction -----
        # Try EasyID3 first if it's an MP3 (most convenient)
        if hasattr(full_meta, 'info') and full_meta.mime[0] == 'audio/mpeg':
            try:
                from mutagen.easyid3 import EasyID3
                easy = EasyID3(filepath)
                artist = easy.get('artist', [''])[0]
                album  = easy.get('album', [''])[0]
                title  = easy.get('title', [''])[0]
            except Exception:
                pass

        # Fallback for other formats (FLAC, MP4, OGG, etc.)
        if not artist and hasattr(full_meta, 'get'):
            artist = full_meta.get('artist', [''])[0] if full_meta.get('artist') else ''
            album  = full_meta.get('album', [''])[0] if full_meta.get('album') else ''
            title  = full_meta.get('title', [''])[0] if full_meta.get('title') else ''

        # Second fallback: try ID3 frames directly (for files with ID3 but not EasyID3)
        if not artist and hasattr(full_meta, 'tags') and full_meta.tags:
            tags = full_meta.tags
            tpe1 = tags.get('TPE1')
            if tpe1 and hasattr(tpe1, 'text'):
                artist = str(tpe1.text[0])
            talb = tags.get('TALB')
            if talb and hasattr(talb, 'text'):
                album = str(talb.text[0])
            tit2 = tags.get('TIT2')
            if tit2 and hasattr(tit2, 'text'):
                title = str(tit2.text[0])

        # ----- Cover extraction (always use full_meta, not the easy one) -----
        cover_img = extract_cover_from_mutagen(full_meta)

        if cover_img is not None:
            print(f"  Cover found: {cover_img.shape[1]}x{cover_img.shape[0]} px")
        else:
            print("  No cover art found.")

    except Exception as e:
        print(f"Warning: Could not read metadata: {e}")

    return artist, album, title, cover_img

# ── Spectrogram and canvas helpers ───────────────────────────────────────────

def encode_uint32_row(canvas, row, value, col_start=0):
    b = struct.pack('>I', value)
    for i, byte in enumerate(b):
        canvas[row, col_start + i] = [byte, byte, byte, 255]


def audio_to_spectrogram(y):
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    mag_db = np.clip(mag_db, MIN_DB, 0)
    norm = (mag_db - MIN_DB) / (-MIN_DB)
    return (norm * 255).astype(np.uint8)


def encode_metadata_bytes(artist='', album='', title=''):
    payload = f"{artist}\x00{album}\x00{title}\x00".encode('utf-8')
    return struct.pack('>I', len(payload)) + payload


def write_bytes_to_zone(canvas, data_bytes, x_start, y_start, zone_width):
    for i, b in enumerate(data_bytes):
        col = x_start + (i % zone_width)
        row = y_start + (i // zone_width)
        if row >= canvas.shape[0]:
            print(f"Warning: metadata overflow at byte {i}")
            break
        canvas[row, col] = [b, b, b, 255]

def sanitize_filename(name):
    """Remove invalid characters for a file name."""
    return re.sub(r'[\\/*?:"<>|]', "", name)[:200]

# ── Main ─────────────────────────────────────────────────────────────────────

def main(input_audio, output_image=None):
    print(f"Loading audio: {input_audio}")

    # Extract metadata and cover from file
    artist, album, title, cover = get_metadata(input_audio)

    # If output_image not provided, generate from song title or input basename
    if output_image is None:
        base = ""
        if title:
            base = sanitize_filename(title)
        else:
            base = os.path.splitext(os.path.basename(input_audio))[0]
        output_image = f"{base}.png"
        print(f"Output PNG not specified, using: {output_image}")

    # Load audio
    y_stereo, sr = librosa.load(input_audio, sr=None, mono=False)

    # Ensure stereo
    if y_stereo.ndim == 1 or y_stereo.shape[0] == 1:
        print("Mono source detected — duplicating channel.")
        if y_stereo.ndim == 1:
            y_stereo = np.stack([y_stereo, y_stereo])
        else:
            y_stereo = np.stack([y_stereo[0], y_stereo[0]])
    elif y_stereo.shape[0] > 2:
        print(f"Warning: {y_stereo.shape[0]} channels, using first two.")
        y_stereo = y_stereo[:2]

    y_left, y_right = y_stereo[0], y_stereo[1]

    print("Computing spectrograms…")
    spec_left = audio_to_spectrogram(y_left)
    spec_right = audio_to_spectrogram(y_right)

    freq_bins = spec_left.shape[0]
    time_frames = spec_left.shape[1]

    # Prepare metadata bytes
    meta_bytes = encode_metadata_bytes(artist, album, title)

    # Canvas dimensions
    img_height = freq_bins * 2
    cover_h, cover_w = (cover.shape[:2] if cover is not None else (0, 0))
    meta_zone_w = max(cover_w, 256, 64)
    meta_text_rows = (len(meta_bytes) + meta_zone_w - 1) // meta_zone_w
    cover_y_in_meta = meta_text_rows
    img_width = HEADER_WIDTH + time_frames + meta_zone_w

    print(f"Canvas: {img_width} x {img_height} px")
    canvas = np.zeros((img_height, img_width, 4), dtype=np.uint8)
    canvas[:, :, 3] = 255

    # Spectrograms
    spec_x = HEADER_WIDTH
    left_rgb = np.stack([spec_left[::-1]] * 3, axis=-1)
    right_rgb = np.stack([spec_right[::-1]] * 3, axis=-1)
    canvas[0:freq_bins, spec_x:spec_x+time_frames, :3] = left_rgb
    canvas[freq_bins:freq_bins*2, spec_x:spec_x+time_frames, :3] = right_rgb

    # Metadata text zone
    meta_x = HEADER_WIDTH + time_frames
    write_bytes_to_zone(canvas, meta_bytes, meta_x, 0, meta_zone_w)

    # Cover image
    if cover is not None:
        cy = cover_y_in_meta
        canvas[cy:cy+cover_h, meta_x:meta_x+cover_w, :3] = cover
        canvas[cy:cy+cover_h, meta_x:meta_x+cover_w, 3] = 255

    # Header magic and manifest
    for i, b in enumerate(MAGIC):
        canvas[i, 0] = [b, b, b, 255]

    manifest = [
        sr, N_FFT, HOP_LENGTH, freq_bins, time_frames,
        HEADER_WIDTH, freq_bins, meta_x, meta_zone_w,
        meta_text_rows, cover_y_in_meta, cover_w, cover_h,
        1 if cover is not None else 0,
        MIN_DB & 0xFFFFFFFF,
        len(meta_bytes)
    ]
    for row_offset, value in enumerate(manifest):
        encode_uint32_row(canvas, 8 + row_offset, value, col_start=0)

    # Save
    out_bgra = cv2.cvtColor(canvas, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(output_image, out_bgra)
    print(f"Saved spectrogram: {output_image}")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Freq bins: {freq_bins}, Time frames: {time_frames}")
    print(f"  Artist: {artist or '(none)'}")
    print(f"  Album : {album or '(none)'}")
    print(f"  Title : {title or '(none)'}")
    if cover is not None:
        print(f"  Cover: {cover_w}x{cover_h}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python audio_to_spectrogram.py input.mp3 [output.png]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) == 3 else None
    main(input_path, output_path)