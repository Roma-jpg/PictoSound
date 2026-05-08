# ===== spectrogram_to_audio.py =====
"""
Usage:
  python spectrogram_to_audio.py input.png [options]

Reconstructs audio, metadata and cover art from a spectrogram PNG.
"""

import argparse
import json
import logging
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

import cv2
import librosa
import mutagen
import numpy as np
import soundfile as sf
from mutagen.id3 import (
    COMM,
    ID3,
    ID3NoHeaderError,
    TALB,
    TCON,
    TDRC,
    TIT2,
    TPE1,
    TRCK,
    TXXX,
)
from mutagen.oggvorbis import OggVorbis

# ----------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------
DEFAULT_MAGIC_HEX = "524f4d454f353538"
MAGIC_ROW_START = 0
MAGIC_ROW_END = 7
VERSION_ROW, VERSION_COL = 7, 8
MANIFEST_ROW_START = 8

IDX_SR = 0
IDX_NFFT = 1
IDX_HOP_LEN = 2
IDX_FREQ_BINS = 3
IDX_TIME_FRAMES = 4
IDX_SPEC_X = 5
IDX_RIGHT_Y = 6
IDX_META_X = 7
IDX_META_ZONE_W = 8
IDX_META_TEXT_ROWS = 9
IDX_COVER_Y_OFF = 10
IDX_COVER_W = 11
IDX_COVER_H = 12
IDX_COVER_PRESENT = 13
IDX_MIN_DB_RAW = 14
IDX_META_BYTE_COUNT = 15
IDX_PREEMPHASIS_GAMMA = 16
IDX_NUM_CHANNELS = 17

MAX_CANVAS_DIM = 20_000

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# low-level helpers
# ----------------------------------------------------------------------
def read_uint32_row(canvas: np.ndarray, row: int, col_start: int = 0) -> int:
    # read big-endian uint32 from a row of pixels.
    if row < 0 or row >= canvas.shape[0] or col_start + 3 >= canvas.shape[1]:
        raise ValueError(f"uint32 read out of bounds: row {row}, col {col_start}")
    b = bytes(int(canvas[row, col_start + i, 0]) for i in range(4))
    return struct.unpack('>I', b)[0]


def read_bytes_from_zone(canvas: np.ndarray, x_start: int, y_start: int,
                         zone_width: int, byte_count: int) -> bytes:
    # read bytes from a pixel zone (row-major)
    result = bytearray()
    for i in range(byte_count):
        col = x_start + (i % zone_width)
        row = y_start + (i // zone_width)
        if row >= canvas.shape[0] or col >= canvas.shape[1]:
            log.warning("Metadata zone truncated at byte %d/%d", i, byte_count)
            break
        result.append(int(canvas[row, col, 0]))
    return bytes(result)


# ----------------------------------------------------------------------
# audio reconstruction
# ----------------------------------------------------------------------
def spectrogram_to_audio(
        spec_norm: np.ndarray,
        sr: int,
        n_fft: int,
        hop_length: int,
        min_db: float,
        n_iter: int = 32,
        momentum: Optional[float] = None,
        preemphasis_gamma: float = 0.0,
) -> np.ndarray:
    # convert normalized spectrogram (0-255) back to audio using Griffin-Lim
    norm = spec_norm.astype(np.float32) / 255.0
    mag_db = norm * (-min_db) + min_db
    magnitude = librosa.db_to_amplitude(mag_db)
    if preemphasis_gamma > 0:
        magnitude = magnitude ** (1.0 / preemphasis_gamma)
        np.clip(magnitude, 0, None, out=magnitude)

    kwargs = {
        'n_iter': n_iter,
        'hop_length': hop_length,
        'win_length': n_fft,
        'window': 'hann',
    }
    if momentum is not None:
        try:
            kwargs['momentum'] = momentum
        except TypeError:
            log.warning("librosa version does not support momentum - ignoring")
    return librosa.griffinlim(magnitude, **kwargs)


# ----------------------------------------------------------------------
# metadata decoding
# ----------------------------------------------------------------------
def _scalar_text(value) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ''
    if value is None:
        return ''
    return str(value)


def decode_metadata(raw_bytes: bytes, version: int) -> Tuple[str, str, str, dict, str]:
    # returns (artist, album, title, full_metadata_dict, original_extension)
    if len(raw_bytes) < 4:
        return '', '', '', {}, ''
    payload_len = struct.unpack('>I', raw_bytes[:4])[0]
    payload = raw_bytes[4:4 + payload_len].decode('utf-8', errors='replace')

    if version >= 1:
        try:
            data = json.loads(payload)
            artist = _scalar_text(data.get('artist', ''))
            album = _scalar_text(data.get('album', ''))
            title = _scalar_text(data.get('title', ''))
            ext = _scalar_text(data.get('original_extension', ''))
            full = data.get('full_metadata', {})
            return artist, album, title, full, ext
        except Exception:
            log.warning("JSON metadata failed - falling back to old format")

    # legacy null-separated format
    parts = payload.split('\x00')
    while len(parts) < 3:
        parts.append('')
    return parts[0], parts[1], parts[2], {}, ''


# ----------------------------------------------------------------------
# tag embedding helpers
# ----------------------------------------------------------------------
def _textify(value) -> str:
    if value is None:
        return ''
    if isinstance(value, list):
        if not value:
            return ''
        return _textify(value[0])
    if hasattr(value, 'text'):
        return _textify(value.text)
    if hasattr(value, 'data'):
        try:
            return f"<binary length {len(value.data)}>"
        except Exception:
            return '<binary>'
    if isinstance(value, (bytes, bytearray)):
        return f"<binary length {len(value)}>"
    return str(value)


def _write_standard_id3(audio: ID3, key: str, text: str) -> bool:
    key = key.lower()
    if not text:
        return True
    try:
        if key == 'title':
            audio['TIT2'] = TIT2(encoding=3, text=[text])
        elif key == 'artist':
            audio['TPE1'] = TPE1(encoding=3, text=[text])
        elif key == 'album':
            audio['TALB'] = TALB(encoding=3, text=[text])
        elif key in ('date', 'year'):
            audio['TDRC'] = TDRC(encoding=3, text=[text])
        elif key == 'tracknumber':
            audio['TRCK'] = TRCK(encoding=3, text=[text])
        elif key == 'genre':
            audio['TCON'] = TCON(encoding=3, text=[text])
        elif key == 'comment':
            audio['COMM::eng'] = COMM(encoding=3, lang='eng', desc='', text=[text])
        else:
            return False
        return True
    except Exception as e:
        log.debug("Could not set standard ID3 tag '%s': %s", key, e)
        return False


def embed_full_metadata(mp3_path: str, metadata_dict: dict) -> None:
    # write metadata into mp3 using raw id3 frames plus txxx fallback.
    try:
        audio = ID3(mp3_path)
    except ID3NoHeaderError:
        audio = ID3()

    for key, value in metadata_dict.items():
        if isinstance(value, str) and len(value) > 1000:
            continue
        text = _textify(value)
        if not text:
            continue

        clean_key = key.lower().replace('id3_', '')
        if _write_standard_id3(audio, clean_key, text):
            continue

        desc = str(key)[:255]
        try:
            audio[f'TXXX:{desc}'] = TXXX(encoding=3, desc=desc, text=[text])
        except Exception as e:
            log.debug("Could not set custom tag '%s': %s", key, e)

    audio.save(mp3_path, v2_version=3)
    log.info("Full metadata embedded into MP3")


def write_ogg_tags(output_path: str, extra_tags: dict) -> None:
    # write vorbis comment tags into ogg output.
    if not extra_tags:
        return

    tag_map = {
        'title': ('title', 'TIT2'),
        'artist': ('artist', 'TPE1'),
        'album': ('album', 'TALB'),
        'date': ('date', 'TDRC', 'year'),
        'tracknumber': ('tracknumber', 'TRCK'),
        'genre': ('genre', 'TCON'),
        'comment': ('comment', 'COMM'),
    }

    try:
        ogg = OggVorbis(output_path)
        written = set()
        for vorbis_key, source_keys in tag_map.items():
            for sk in source_keys:
                val = extra_tags.get(sk)
                if val is None:
                    val = extra_tags.get(f'id3_{sk}')
                if val is None:
                    continue
                text = _textify(val)
                if text:
                    ogg[vorbis_key] = [text]
                    written.add(vorbis_key)
                    break

        # keep a few extra safe keys if they look like plain comments.
        for key, value in extra_tags.items():
            key_lower = str(key).lower()
            if key_lower in written:
                continue
            if not key_lower.isidentifier() and not key_lower.replace('_', '').isalnum():
                continue
            text = _textify(value)
            if text:
                ogg[key_lower] = [text]
        ogg.save()
        log.info("Full metadata embedded into OGG")
    except Exception as e:
        log.warning("OGG tag writing failed: %s", e)


def convert_wav_to_output(
        wav_path: str,
        output_path: str,
        artist: str,
        album: str,
        title: str,
        cover_rgb: Optional[np.ndarray] = None,
        bitrate: str = '192k',
        extra_tags: dict = None,
) -> bool:
    # convert wav to target format (wav, mp3, ogg). returns true on success.
    out_ext = Path(output_path).suffix.lower()

    # --- wav: just move ---
    if out_ext == '.wav':
        shutil.move(wav_path, output_path)
        log.info("WAV saved: %s", output_path)
        return True

    # --- mp3 (using ffmpeg + mutagen for full tags) ---
    if out_ext == '.mp3':
        if not shutil.which('ffmpeg'):
            log.error("ffmpeg not found - cannot create MP3")
            return False

        cover_temp = None
        try:
            cmd = ['ffmpeg', '-y', '-i', wav_path]
            if cover_rgb is not None:
                # resize cover for thumbnailing
                h, w = cover_rgb.shape[:2]
                max_dim = 500
                if h > max_dim or w > max_dim:
                    scale = max_dim / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    cover_rgb = cv2.resize(cover_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                cover_bgr = cv2.cvtColor(cover_rgb, cv2.COLOR_RGB2BGR)
                success, jpeg_bytes = cv2.imencode('.jpg', cover_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if success:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                        f.write(jpeg_bytes.tobytes())
                        cover_temp = f.name
                    cmd += ['-i', cover_temp, '-map', '0:a', '-map', '1:v', '-c:v', 'copy', '-disposition:v',
                            'attached_pic']
                else:
                    log.warning("Could not encode cover - skipping")
            cmd += ['-codec:a', 'libmp3lame', '-b:a', bitrate, '-id3v2_version', '3']
            if title:
                cmd += ['-metadata', f'title={title}']
            if artist:
                cmd += ['-metadata', f'artist={artist}']
            if album:
                cmd += ['-metadata', f'album={album}']
            cmd.append(output_path)

            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log.info("MP3 created: %s", output_path)

            if extra_tags:
                embed_full_metadata(output_path, extra_tags)
            return True
        except Exception as e:
            log.error("MP3 conversion failed: %s", e)
            return False
        finally:
            if cover_temp and os.path.exists(cover_temp):
                os.unlink(cover_temp)

    # --- ogg (ogg vorbis) using ffmpeg ---
    if out_ext == '.ogg':
        if not shutil.which('ffmpeg'):
            log.error("ffmpeg not found - cannot create OGG")
            return False

        if cover_rgb is not None:
            log.warning("Cover art cannot be embedded into OGG - skipping")

        try:
            cmd = [
                'ffmpeg', '-y', '-i', wav_path,
                '-c:a', 'libvorbis', '-q:a', '4',
            ]
            if title:
                cmd += ['-metadata', f'title={title}']
            if artist:
                cmd += ['-metadata', f'artist={artist}']
            if album:
                cmd += ['-metadata', f'album={album}']
            cmd.append(output_path)

            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log.info("OGG created: %s", output_path)

            if extra_tags:
                write_ogg_tags(output_path, extra_tags)
            return True
        except Exception as e:
            log.error("OGG conversion failed: %s", e)
            return False

    # --- any other extension: fallback to wav ---
    log.warning("Output format '%s' not directly supported - saving as WAV instead", out_ext)
    fallback = Path(output_path).with_suffix('.wav')
    shutil.move(wav_path, str(fallback))
    log.info("WAV saved as: %s", fallback)
    return True


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Decode spectrogram PNG to audio")
    parser.add_argument("input_image", help="Input PNG spectrogram")
    parser.add_argument("output_audio", nargs="?", default=None,
                        help="Output audio file (default: from metadata or input name)")
    parser.add_argument("--magic", default=DEFAULT_MAGIC_HEX,
                        help="Expected magic signature (hex string)")
    parser.add_argument("--output-format", choices=['wav', 'mp3', 'auto'], default='auto',
                        help="Output format (auto = use original extension if stored)")
    parser.add_argument("--save-cover", action="store_true",
                        help="Save extracted cover art as PNG")
    parser.add_argument("--iterations", type=int, default=32,
                        help="Griffin-Lim iterations")
    parser.add_argument("--momentum", type=float, default=None,
                        help="Griffin-Lim momentum (librosa >=0.10)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # load and prepare image
    log.info("Loading: %s", args.input_image)
    raw = cv2.imread(args.input_image, cv2.IMREAD_UNCHANGED)
    if raw is None:
        sys.exit("Cannot open image")
    if raw.shape[0] > MAX_CANVAS_DIM or raw.shape[1] > MAX_CANVAS_DIM:
        sys.exit("Image too large - aborting")
    canvas = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA) if raw.shape[2] == 4 else cv2.cvtColor(raw, cv2.COLOR_BGR2RGBA)

    # verify magic
    try:
        expected_magic = bytes.fromhex(args.magic)
        if len(expected_magic) != 8:
            raise ValueError
    except Exception:
        sys.exit(f"Invalid --magic hex string: {args.magic}")
    magic_found = bytes(int(canvas[i, 0, 0]) for i in range(MAGIC_ROW_START, MAGIC_ROW_END + 1))
    if magic_found != expected_magic:
        sys.exit(f"Magic mismatch.\nExpected: {expected_magic.hex()}\nFound:   {magic_found.hex()}")

    # version
    version = int(canvas[VERSION_ROW, VERSION_COL, 0]) if VERSION_ROW < canvas.shape[0] and VERSION_COL < canvas.shape[
        1] else 0
    log.info("Version: %d", version)

    # read manifest (up to 18 fields)
    max_fields = 18 if version >= 1 else 16
    manifest = []
    for i in range(max_fields):
        try:
            manifest.append(read_uint32_row(canvas, MANIFEST_ROW_START + i, 0))
        except ValueError:
            manifest.append(0)
    while len(manifest) < max_fields:
        manifest.append(0)

    # unpack
    sr = manifest[IDX_SR]
    n_fft = manifest[IDX_NFFT]
    hop_length = manifest[IDX_HOP_LEN]
    freq_bins = manifest[IDX_FREQ_BINS]
    time_frames = manifest[IDX_TIME_FRAMES]
    spec_x = manifest[IDX_SPEC_X]
    right_y = manifest[IDX_RIGHT_Y]
    meta_x = manifest[IDX_META_X]
    meta_zone_w = manifest[IDX_META_ZONE_W]
    meta_text_rows = manifest[IDX_META_TEXT_ROWS]
    cover_y_off = manifest[IDX_COVER_Y_OFF]
    cover_w = manifest[IDX_COVER_W]
    cover_h = manifest[IDX_COVER_H]
    cover_present = manifest[IDX_COVER_PRESENT]
    min_db_raw = manifest[IDX_MIN_DB_RAW]
    meta_byte_count = manifest[IDX_META_BYTE_COUNT]
    preemphasis_gamma = (manifest[IDX_PREEMPHASIS_GAMMA] / 1000.0) if version >= 1 else 0.0
    num_channels = manifest[IDX_NUM_CHANNELS] if version >= 1 else 2

    # signed min_db
    min_db = struct.unpack('>i', struct.pack('>I', min_db_raw))[0]

    # geometry validation
    if spec_x + time_frames > canvas.shape[1]:
        sys.exit("Image too narrow for spectrogram data")
    if meta_x + meta_zone_w > canvas.shape[1]:
        sys.exit("Metadata zone out of bounds")
    if cover_present and (cover_y_off + cover_h > canvas.shape[0] or meta_x + cover_w > canvas.shape[1]):
        sys.exit("Cover coordinates out of bounds")

    log.info("Sample rate: %d, bins: %d, frames: %d, pre-emphasis: %.3f, orig channels: %d",
             sr, freq_bins, time_frames, preemphasis_gamma, num_channels)

    # extract spectrograms (flip vertical)
    left_slice = canvas[0:freq_bins, spec_x:spec_x + time_frames, 0][::-1]
    right_slice = canvas[right_y:right_y + freq_bins, spec_x:spec_x + time_frames, 0][::-1]

    # reconstruct audio
    log.info("Reconstructing audio (Griffin-Lim, iter=%d, momentum=%s)...", args.iterations, args.momentum)
    if num_channels == 1:
        audio = spectrogram_to_audio(left_slice, sr, n_fft, hop_length, min_db, args.iterations, args.momentum,
                                     preemphasis_gamma)
    else:
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_l = ex.submit(spectrogram_to_audio, left_slice, sr, n_fft, hop_length, min_db, args.iterations,
                              args.momentum, preemphasis_gamma)
            fut_r = ex.submit(spectrogram_to_audio, right_slice, sr, n_fft, hop_length, min_db, args.iterations,
                              args.momentum, preemphasis_gamma)
            audio_left = fut_l.result()
            audio_right = fut_r.result()

            # trim to equal length
            min_len = min(len(audio_left), len(audio_right))
            audio_left, audio_right = audio_left[:min_len], audio_right[:min_len]

            # normalize
            peak = max(np.max(np.abs(audio_left)), np.max(np.abs(audio_right)), 1e-9)
            scale = 0.95 / peak
            audio_left *= scale
            audio_right *= scale

            audio = np.stack([audio_left, audio_right], axis=1)

    # for mono, normalize the single reconstructed channel.
    if num_channels == 1:
        peak = max(np.max(np.abs(audio)), 1e-9)
        audio = audio * (0.95 / peak)

    # read metadata
    raw_meta = read_bytes_from_zone(canvas, meta_x, 0, meta_zone_w, meta_byte_count)
    artist, album, title, full_tags, orig_ext = decode_metadata(raw_meta, version)
    log.info("Metadata: artist='%s' album='%s' title='%s' orig_ext='%s'", artist, album, title, orig_ext)

    # cover extraction
    cover_rgb = None
    if cover_present and cover_w > 0 and cover_h > 0:
        cover_rgba = canvas[cover_y_off:cover_y_off + cover_h, meta_x:meta_x + cover_w, :]
        cover_rgb = cv2.cvtColor(cover_rgba, cv2.COLOR_RGBA2RGB)
        log.info("Cover extracted: %dx%d px", cover_w, cover_h)
        if args.save_cover:
            cover_out = Path(args.input_image).stem + '_cover.png'
            cv2.imwrite(cover_out, cv2.cvtColor(cover_rgb, cv2.COLOR_RGB2BGR))
            log.info("Cover saved: %s", cover_out)

    # determine output filename and format
    out_path = args.output_audio
    if out_path is None:
        base = title if title else Path(args.input_image).stem
        out_path = f"{base}.wav"

    # respect --output-format or original extension
    original_suffix = Path(out_path).suffix.lower()
    if args.output_format == 'auto' and orig_ext:
        if original_suffix != orig_ext:
            log.info("Auto output format is overriding %s to original extension %s", original_suffix or '(none)',
                     orig_ext)
        out_path = str(Path(out_path).with_suffix(orig_ext))
    elif args.output_format == 'wav':
        out_path = str(Path(out_path).with_suffix('.wav'))
    elif args.output_format == 'mp3':
        out_path = str(Path(out_path).with_suffix('.mp3'))

    # write temporary wav
    tmp_wav = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp_wav = tmp.name
        sf.write(tmp_wav, audio, sr)
        log.info("Temporary WAV created: %s", tmp_wav)

        success = convert_wav_to_output(
            tmp_wav, out_path, artist, album, title,
            cover_rgb=cover_rgb, bitrate='192k', extra_tags=full_tags,
        )

        if success:
            if tmp_wav and os.path.exists(tmp_wav):
                os.unlink(tmp_wav)
            log.info("Final output: %s", out_path)
        else:
            fallback = str(Path(out_path).with_suffix('.wav'))
            if tmp_wav and os.path.exists(tmp_wav):
                shutil.move(tmp_wav, fallback)
                log.warning("Conversion failed - WAV kept as %s", fallback)
            elif os.path.exists(fallback):
                log.warning("Conversion failed - WAV already present as %s", fallback)
            else:
                log.warning("Conversion failed - no output produced")
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.unlink(tmp_wav)

    log.info("Done.")


if __name__ == "__main__":
    main()