# PictoSound

This tool allows you to convert any audio file to a single PNG spectrogram image, embedding full metadata and album art. You can reconstruct the audio back from that PNG. The pipeline is lossy - the reconstructed audio is not bit-perfect, but it retains essential stereo information. When saved as MP3 or Ogg Vorbis, the output includes all original tags and cover art.
You **can** edit the spectrogram image, but only within the region corresponding to actual audio data. This allows you to hide any graphic by converting the spectrogram back into an audio file. However, be aware that this will affect the resulting audio, and the alteration will be visible even in the simplest spectrogram analyzer.

## Example:
![edited_audio_example.jpg](https://github.com/Roma-jpg/PictoSound/blob/03d1d0a568ad072b0466d288612d66daf317774e/edited_audio_example.jpg)

## How it works

`audio_to_spectrogram.py` reads an audio file. The script extracts all metadata: artist, album, title, and other tags. The script also extracts cover art. The script computes left and right channel spectrograms. The script packs everything into one RGBA PNG.

- A small header in the upper left corner. Contains magic signature, version byte, manifest table.
- The two spectrograms side by side.
- A metadata text zone. JSON with all tags and original file extension.
- The cover image below the metadata zone. (The metadata section is usually 1-2 pixels in height, so it may appear that the cover image is just there, at the top, but it's not)

![output_example.jpg](https://github.com/Roma-jpg/PictoSound/blob/ac28d5c84cfebb20101b8595bcda108d42144aef/output_example.jpg)
<sup><sub>(Don't try to decode it back, it's a screenshot, you just won't be able to)</sub></sup>

`spectrogram_to_audio.py` reverses the process. Reads the PNG. Verifies the magic signature. Recovers audio parameters from the manifest. Reconstructs the two channels from the spectrograms. Uses a fast threaded Griffin-Lim algorithm. Writes a WAV, MP3, or Ogg Vorbis file. When creating MP3 or OGG, the script embeds the original metadata. For MP3, the script also embeds cover art. The decoder uses the original file extension stored in the PNG. You do not need to remember if the source was .ogg or .flac.

`wrapper.py` calls the two scripts in sequence. Handy for batch conversion or quick testing.

<details>
<summary>Click to expand usage example</summary>

```
Usage: python wrapper.py input_audio output_audio

Encoding .\dudeontheguitar - Barajatr.mp3 -> dudeontheguitar - Barajatr_spectrogram.png
INFO: Using magic: 524f4d454f353538
INFO: Loading: .\dudeontheguitar - Barajatr.mp3
INFO: Artist: dudeontheguitar/dododowson | Album: Barajatr | Title: Barajatr | Ext: .mp3
INFO: Computing spectrograms (gamma=0.000)...
INFO: Canvas 14405x2050 px
INFO: Cover placed at row 2, 300x300 px
INFO: Saved: dudeontheguitar - Barajatr_spectrogram.png
INFO: Sample rate: 44100, bins: 1025, frames: 14041, channels: 2

Decoding dudeontheguitar - Barajatr_spectrogram.png -> dude.mp3
INFO: Loading: dudeontheguitar - Barajatr_spectrogram.png
INFO: Version: 1
INFO: Sample rate: 44100, bins: 1025, frames: 14041, pre-emphasis: 0.000, orig channels: 2
INFO: Reconstructing audio (Griffin-Lim, iter=32, momentum=None)...
INFO: Metadata: artist='dudeontheguitar/dododowson' album='Barajatr' title='Barajatr' orig_ext='.mp3'
INFO: Cover extracted: 300x300 px
INFO: Temporary WAV created: C:\Users\BF17~1\AppData\Local\Temp\tmp7me4h4gw.wav
INFO: MP3 created: dude.mp3
INFO: Full metadata embedded into MP3
INFO: Final output: dude.mp3
INFO: Done.

Done. Intermediate PNG: dudeontheguitar - Barajatr_spectrogram.png
Output: dudeontheguitar - Barajatr_reconstructed.mp3
```
</details>

The PNG spectrogram always remains after decoding. The script cleans up only the temporary WAV file used during MP3/OGG creation.

Volume data is not stored. The output file is sometimes quieter than the original. Overall information is preserved.

## Usage

### Basic conversion from audio to PNG

```
python audio_to_spectrogram.py input.mp3
```

If you omit the output name, the script uses the song title from metadata as the filename. Otherwise it falls back to the input file's basename.

### Convert PNG back to audio

```
python spectrogram_to_audio.py spectrogram.png output.wav
```

The output extension decides the final format: .mp3, .ogg, or .wav. If you specify no output file, the script defaults to the PNG's name with the original extension from the stored metadata.

To save the extracted album cover as a separate PNG:

```
python spectrogram_to_audio.py spectrogram.png --save-cover
```

### Full roundtrip with the wrapper

```
python wrapper.py input.mp3 restored.wav
```

The wrapper uses a `VENV_PYTHON = sys.executable`, so it matches your environment. It produces an intermediate PNG (input_stem_spectrogram.png) and finally the WAV file.

## Customising the magic signature

```
python audio_to_spectrogram.py input.wav --magic 524F4D454F353538
python spectrogram_to_audio.py input.png --magic 524F4D454F353538
```

The default magic is `524F4D454F353538` (I wonder what that says in ASCII). Any 16-character hex string (8 bytes) is allowed. Even zeros or non-printable values.

## Detailed file structure

The spectrogram PNG is a structured container.

**Header strip**: first 64 columns, rows 0 to 63.

- Magic signature: rows 0 to 7, column 0.
- Version byte: row 7, column 1. Value 1 means JSON metadata with pre-emphasis support.
- Manifest: rows 8 to 25, columns 0 to 3. Each row stores one 32-bit big-endian integer. Pixel intensity equals the byte value.

The manifest includes sample rate, N_FFT, hop length, frequency bins, time frames, spectrogram coordinates, metadata zone dimensions, cover size, MIN_DB, metadata byte count, pre-emphasis gamma times 1000, and original channel count.

**Spectrogram area**: columns HEADER_WIDTH (64) to HEADER_WIDTH + time_frames - 1. Rows 0 to 2*freq_bins - 1.
- Top half: left channel. Low frequencies at the bottom.
- Bottom half: right channel.

Pixel intensity: linear mapping from dB (MIN_DB to 0) to 0 to 255.

**Metadata text zone**: columns meta_x ( = HEADER_WIDTH + time_frames ) to meta_x + meta_zone_w - 1. Starts at row 0. Stores a JSON object. The JSON contains artist, album, title, original_extension (e.g., .ogg), and full_metadata (all tags from source). The JSON is length-prefixed with 4 bytes big-endian. Written row-major into pixel red channels.

**Cover image**: placed directly below the metadata text zone. Y offset recorded in manifest. Stored as RGB pixels. Resized to max 1024x1024.

## Reconstruction details

The script will:

- Load the PNG, check magic and version, and read manifest values.
- Use the manifest to locate spectrograms, metadata, and cover art.
- Extract left and right channel magnitude spectrograms and flip them vertically.
- Run threaded reconstruction: both channels perform Griffin-Lim phase estimation in parallel with 32 iterations (default).
- Trim the two waveforms to equal length and normalize to 0.95 peak.
- Interleave for stereo, or mix to mono if the original was mono.
- Write a temporary WAV file.
- If the target output is MP3, use ffmpeg to convert the WAV to MP3 with libmp3lame, embedding metadata and cover art as ID3v2.3 tags.
- If the target output is OGG, use ffmpeg with the libvorbis encoder (quality `-q:a 4`), add basic tags (title, artist, album), and skip cover art because Ogg Vorbis does not support attached pictures.
- If the output is WAV or any other extension, rename or move the temporary WAV to the output path.

## Compression options and size reduction

The spectrogram PNG can become pretty large. A 3-minute song produces roughly a 30 MB image. Reduce size by tuning constants in `audio_to_spectrogram.py`.

- Lower N_FFT (default 2048) to reduce frequency resolution. Fewer bins mean shorter PNG height (2 times bins).
- Increase HOP_LENGTH (default 512) to reduce time frames. Narrower PNG.
- Change MIN_DB from default -80 dB to -60 dB. Discards quieter parts.
- Set PNG compression level to 9 in `cv2.imwrite`.
- Lower MAX_COVER_SIZE (default 1024) or remove cover art.

All parameters are read from the PNG header. The decoder works automatically.

## Dependencies

`numpy`, `librosa`, `opencv-python`, `soundfile`, `mutagen`, `ffmpeg`

Install them all with:

```
pip install numpy librosa opencv-python soundfile mutagen
```

Then ensure `ffmpeg` is in your PATH.

## Honest disclaimer

This tool has no practical use. I built this tool to learn how spectrograms work. I also learned how to pack metadata into images and reverse the process. No real-world use case exists for storing audio as a PNG. A three-minute song becomes a 30 megabyte image, while original can fit in just under 4MB. This is massively inefficient. The reconstruction is lossy and full of Griffin-Lim artefacts. The roundtrip takes longer than keeping the original MP3. You might misuse this as a bizarre form of encryption tho. Hide audio in a picture that looks like a spectrogram. The result is absurdly bloated and easily detectable. Do not use this for anything serious. This is a learning experiment. However hiding pictures of cats inside an audio is kind of fun.

## Notes

- Please, don't convert the output `.png` to `.jpg` or any format with a lossy compression, this WILL corrupt the data needed to restore it back to audio.
- Decoding is not real-time. A 3-minute song takes 10-15 seconds on a modern CPU.
- Griffin-Lim introduces artefacts. The result is still recognisable. Preserves stereo separation.
- Do not crop or edit the PNG outside the designated zones. The header and manifest positions are fixed.
- Old-format metadata (null-separated) is supported for version 0 PNGs. New files always use JSON.
