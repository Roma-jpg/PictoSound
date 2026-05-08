# ===== audio_to_spectrogram.py =====
"""
Usage:
  python audio_to_spectrogram.py input.wav [options]

Encodes any audio file into a PNG spectrogram with embedded metadata and cover art.
"""

import argparse
import json
import logging
import re
import struct
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import librosa
import numpy as np
from mutagen import File as MutagenFile
from mutagen.id3 import ID3

# ----------------------------------------------------------------------
# constants & configuration
# ----------------------------------------------------------------------
HEADER_WIDTH = 64  # pixels reserved for magic + manifest (col 0..63)
N_FFT = 2048
HOP_LENGTH = 512
MIN_DB = -80
MAX_COVER_SIZE = 1024
MAX_CANVAS_DIM = 20_000

# default magic string (8 ASCII characters or 8 bytes)
DEFAULT_MAGIC_HEX = "524f4d454f353538"

# layout indices for manifest rows (stored as uint32 big-endian)
(
    IDX_SR,
    IDX_NFFT,
    IDX_HOP_LEN,
    IDX_FREQ_BINS,
    IDX_TIME_FRAMES,
    IDX_SPEC_X,
    IDX_RIGHT_Y,
    IDX_META_X,
    IDX_META_ZONE_W,
    IDX_META_TEXT_ROWS,
    IDX_COVER_Y_OFF,
    IDX_COVER_W,
    IDX_COVER_H,
    IDX_COVER_PRESENT,
    IDX_MIN_DB_RAW,
    IDX_META_BYTE_COUNT,
    IDX_PREEMPHASIS_GAMMA,
    IDX_NUM_CHANNELS,
) = range(18)

VERSION_ROW, VERSION_COL = 7, 8

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# helper functions
# ----------------------------------------------------------------------
def sanitize_filename(name: str) -> str:
    """Remove illegal filesystem characters."""
    return re.sub(r'[\\/*?:"<>|]', "", name)[:200]


def encode_uint32_row(canvas: np.ndarray, row: int, value: int, col_start: int = 0) -> None:
    # write a big-endian uint32 as 4 grey pixels (R=G=B=byte)
    b = struct.pack('>I', value)
    for i, byte in enumerate(b):
        canvas[row, col_start + i] = [byte, byte, byte, 255]


def write_bytes_to_zone(canvas: np.ndarray, data_bytes: bytes,
                        x_start: int, y_start: int, zone_width: int) -> None:
    """write bytes row-major into a pixel zone (R=G=B=byte, alpha=255)"""
    for i, b in enumerate(data_bytes):
        col = x_start + (i % zone_width)
        row = y_start + (i // zone_width)
        if row >= canvas.shape[0]:
            log.warning("Metadata zone overflow at byte %d - truncating.", i)
            break
        canvas[row, col] = [b, b, b, 255]


def extract_cover_from_mutagen(audio_file) -> Optional[np.ndarray]:
    # extract cover art as rgb numpy array, resized if needed.
    cover_data = None
    try:
        if hasattr(audio_file, 'tags') and audio_file.tags is not None:
            if hasattr(audio_file.tags, 'getall'):
                for apic in audio_file.tags.getall('APIC'):
                    cover_data = apic.data
                    break
        if cover_data is None and hasattr(audio_file, 'pictures'):
            pics = audio_file.pictures
            if pics:
                cover_data = pics[0].data
        if cover_data is None and hasattr(audio_file, 'get'):
            covr = audio_file.get('covr')
            if covr:
                cover_data = covr[0]
        if cover_data is None and hasattr(audio_file, '__contains__'):
            if 'APIC:' in audio_file:
                cover_data = audio_file['APIC:'].data
    except Exception:
        return None

    if cover_data is None:
        return None

    nparr = np.frombuffer(cover_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if h > MAX_COVER_SIZE or w > MAX_COVER_SIZE:
        scale = MAX_COVER_SIZE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def _coerce_tag_value(value):
    # convert tag values to json-friendly scalars or string lists.
    if value is None:
        return ''
    if isinstance(value, list):
        return [str(v) for v in value]
    if hasattr(value, 'text'):
        text = value.text
        if isinstance(text, list):
            return [str(v) for v in text]
        return str(text)
    if hasattr(value, 'data'):
        try:
            return f"<binary length {len(value.data)}>"
        except Exception:
            return '<binary>'
    if isinstance(value, (bytes, bytearray)):
        return f"<binary length {len(value)}>"
    return str(value)


def _first_text(value) -> str:
    # pick the first scalar text value from a tag value.
    if isinstance(value, list):
        return str(value[0]) if value else ''
    if value is None:
        return ''
    return str(value)


def get_all_metadata(filepath: str) -> Tuple[str, str, str, dict, str]:
    # extract artist, album, title, full metadata dict, and original extension.
    metadata = {}
    artist = album = title = ''
    ext = Path(filepath).suffix.lower()

    try:
        audio = MutagenFile(filepath, easy=False)
        if audio is None:
            return artist, album, title, metadata, ext

        tags = getattr(audio, 'tags', None)
        if isinstance(tags, ID3):
            for key, frame in tags.items():
                if key == 'APIC':
                    continue
                value = _coerce_tag_value(frame)
                metadata[f'id3_{key}'] = value
                if key == 'TPE1' and not artist:
                    artist = _first_text(value)
                elif key == 'TALB' and not album:
                    album = _first_text(value)
                elif key == 'TIT2' and not title:
                    title = _first_text(value)
        elif tags is not None and hasattr(tags, 'items'):
            for key, value in tags.items():
                metadata[key] = _coerce_tag_value(value)
                lower_key = key.lower()
                if lower_key == 'artist' and not artist:
                    artist = _first_text(metadata[key])
                elif lower_key == 'album' and not album:
                    album = _first_text(metadata[key])
                elif lower_key == 'title' and not title:
                    title = _first_text(metadata[key])

        # mp4 / other tag containers may expose keys() + get()
        if hasattr(audio, 'keys') and hasattr(audio, 'get') and tags is not None and not isinstance(tags, ID3):
            for key in audio.keys():
                try:
                    if key in metadata:
                        continue
                    value = audio.get(key)
                    if value is not None:
                        metadata[key] = _coerce_tag_value(value)
                except Exception:
                    pass

        # technical
        if hasattr(audio, 'info') and audio.info is not None:
            if hasattr(audio.info, 'length'):
                metadata['length_seconds'] = float(audio.info.length)
            if hasattr(audio.info, 'bitrate'):
                metadata['bitrate_bps'] = int(audio.info.bitrate)

        # convenience fields - actually mutate the variables
        artist = _first_text(metadata.get('artist', metadata.get('TPE1', metadata.get('id3_TPE1', artist))))
        album = _first_text(metadata.get('album', metadata.get('TALB', metadata.get('id3_TALB', album))))
        title = _first_text(metadata.get('title', metadata.get('TIT2', metadata.get('id3_TIT2', title))))

    except Exception as e:
        log.warning("Error reading metadata: %s", e)

    # remove overly long strings
    for k, v in list(metadata.items()):
        if isinstance(v, str) and len(v) > 10000:
            log.warning("Skipping field '%s' (too long)", k)
            del metadata[k]

    return artist, album, title, metadata, ext


def audio_to_spectrogram(y: np.ndarray, preemphasis_gamma: float = 0.0) -> np.ndarray:
    # convert audio to normalized magnitude spectrogram (0..255)
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann')
    mag = np.abs(stft)
    if preemphasis_gamma > 0:
        mag = mag ** preemphasis_gamma
    mag_db = librosa.amplitude_to_db(mag, ref=np.max)
    mag_db = np.clip(mag_db, MIN_DB, 0)
    norm = (mag_db - MIN_DB) / (-MIN_DB)
    return (norm * 255).astype(np.uint8)


def encode_metadata_json(artist: str, album: str, title: str,
                         full_metadata: dict, original_ext: str) -> bytes:
    # version 1 metadata: json with all tags + original extension.
    payload = json.dumps({
        'artist': artist,
        'album': album,
        'title': title,
        'original_extension': original_ext,
        'full_metadata': full_metadata,
    }, ensure_ascii=False).encode('utf-8')
    return struct.pack('>I', len(payload)) + payload


def encode_metadata_old(artist: str, album: str, title: str) -> bytes:
    # legacy null-separated format (version 0)
    payload = f"{artist}\x00{album}\x00{title}\x00".encode('utf-8')
    return struct.pack('>I', len(payload)) + payload


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Encode audio to spectrogram PNG")
    parser.add_argument("input_audio", help="Input audio file")
    parser.add_argument("output_image", nargs="?", default=None,
                        help="Output PNG filename (default: derived from title)")
    parser.add_argument("--preemphasis", type=float, default=0.0,
                        help="Spectrogram power law gamma (e.g. 0.33); enables version 1")
    parser.add_argument("--metadata-format", choices=['old', 'json'], default='json',
                        help="Metadata encoding (default: json, version 1)")
    parser.add_argument("--magic", default=DEFAULT_MAGIC_HEX,
                        help=f"Magic signature as hex string (default: {DEFAULT_MAGIC_HEX})")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # convert hex magic to bytes (must be 8 bytes)
    try:
        magic_bytes = bytes.fromhex(args.magic)
        if len(magic_bytes) != 8:
            raise ValueError
    except Exception:
        sys.exit(f"Error: --magic must be a 16-character hex string (8 bytes), got '{args.magic}'")
    log.info("Using magic: %s", magic_bytes.hex())

    # load audio
    log.info("Loading: %s", args.input_audio)
    artist, album, title, full_meta, orig_ext = get_all_metadata(args.input_audio)
    log.info("Artist: %s | Album: %s | Title: %s | Ext: %s",
             artist or '-', album or '-', title or '-', orig_ext)

    y_stereo, sr = librosa.load(args.input_audio, sr=None, mono=False)
    if y_stereo.ndim == 1:
        log.info("Mono source - duplicating channel")
        y_stereo = np.stack([y_stereo, y_stereo])
    elif y_stereo.shape[0] > 2:
        log.warning("%d channels found - using first two", y_stereo.shape[0])
        y_stereo = y_stereo[:2]
    num_channels = y_stereo.shape[0]
    y_left, y_right = y_stereo[0], y_stereo[1]

    # version logic
    preemphasis = args.preemphasis if args.preemphasis > 0 else 0.0
    version = 1 if (preemphasis > 0 or args.metadata_format == 'json') else 0

    # spectrograms
    log.info("Computing spectrograms (gamma=%.3f)...", preemphasis)
    spec_left = audio_to_spectrogram(y_left, preemphasis)
    spec_right = audio_to_spectrogram(y_right, preemphasis)
    freq_bins, time_frames = spec_left.shape

    # metadata block
    if version >= 1 and args.metadata_format == 'json':
        meta_bytes = encode_metadata_json(artist, album, title, full_meta, orig_ext)
    else:
        meta_bytes = encode_metadata_old(artist, album, title)

    # cover
    cover = None
    try:
        audio_file = MutagenFile(args.input_audio, easy=False)
        if audio_file:
            cover = extract_cover_from_mutagen(audio_file)
    except Exception as e:
        log.warning("Cover extraction failed: %s", e)

    cover_h, cover_w = (cover.shape[:2] if cover is not None else (0, 0))

    # metadata zone layout
    meta_zone_w = max(cover_w, 256, 64)
    meta_text_rows = (len(meta_bytes) + meta_zone_w - 1) // meta_zone_w
    cover_y_offset = meta_text_rows  # placed right after text zone

    # canvas dimensions
    img_width = HEADER_WIDTH + time_frames + meta_zone_w
    img_height = freq_bins * 2
    cover_bottom = cover_y_offset + cover_h
    if cover_bottom > img_height:
        img_height = cover_bottom

    if img_height > MAX_CANVAS_DIM or img_width > MAX_CANVAS_DIM:
        sys.exit("Canvas too large - aborting.")

    log.info("Canvas %dx%d px", img_width, img_height)
    canvas = np.zeros((img_height, img_width, 4), dtype=np.uint8)
    canvas[:, :, 3] = 255

    # ---- magic signature (rows 0..7, col 0 only) ----
    for row, byte in enumerate(magic_bytes):
        canvas[row, 0] = [byte, byte, byte, 255]

    # ---- version byte at row 7, col 8 ----
    canvas[VERSION_ROW, VERSION_COL] = [version, version, version, 255]

    # ---- manifest (rows 8..25, cols 0..3) ----
    min_db_raw = struct.unpack('>I', struct.pack('>i', MIN_DB))[0]
    manifest = [
        sr, N_FFT, HOP_LENGTH, freq_bins, time_frames,
        HEADER_WIDTH, freq_bins, HEADER_WIDTH + time_frames, meta_zone_w,
        meta_text_rows, cover_y_offset, cover_w, cover_h,
        1 if cover is not None else 0,
        min_db_raw,
        len(meta_bytes),
        int(preemphasis * 1000) if version >= 1 else 0,
        num_channels,
    ]
    for idx, val in enumerate(manifest):
        encode_uint32_row(canvas, 8 + idx, val, col_start=0)

    # ---- spectrograms ----
    spec_x = HEADER_WIDTH
    left_rgb = np.stack([spec_left[::-1]] * 3, axis=-1)
    right_rgb = np.stack([spec_right[::-1]] * 3, axis=-1)
    canvas[0:freq_bins, spec_x:spec_x + time_frames, :3] = left_rgb
    canvas[freq_bins:freq_bins * 2, spec_x:spec_x + time_frames, :3] = right_rgb

    # ---- metadata text zone (starts at row 0, col meta_x) ----
    meta_x = HEADER_WIDTH + time_frames
    write_bytes_to_zone(canvas, meta_bytes, meta_x, 0, meta_zone_w)

    # ---- cover art (if any) ----
    if cover is not None:
        cy = cover_y_offset
        canvas[cy:cy + cover_h, meta_x:meta_x + cover_w, :3] = cover
        canvas[cy:cy + cover_h, meta_x:meta_x + cover_w, 3] = 255
        log.info("Cover placed at row %d, %dx%d px", cy, cover_w, cover_h)

    # ---- save PNG ----
    out_path = args.output_image
    if out_path is None:
        base = sanitize_filename(title) if title else Path(args.input_audio).stem
        out_path = f"{base}.png"
        log.info("Output image: %s", out_path)

    # convert RGBA to BGRA for OpenCV
    out_bgra = cv2.cvtColor(canvas, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(out_path, out_bgra)
    log.info("Saved: %s", out_path)
    log.info("Sample rate: %d, bins: %d, frames: %d, channels: %d",
             sr, freq_bins, time_frames, num_channels)


if __name__ == "__main__":
    main()