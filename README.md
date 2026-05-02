# PictoSound
A set of Python scripts that convert any audio file into a single PNG spectrogram image (with metadata and album art embedded) and then reconstruct the audio back from that PNG. The whole pipeline is lossy - the reconstructed audio is not bit-perfect but retains the essential stereo information and can include ID3 tags and cover art when saved as MP3.

## TL;DR - How it works  
audio_to_spectrogram.py reads an audio file, extracts its metadata (artist, album, title) and cover art, computes the left and right channel spectrograms, and packs everything into one RGBA PNG: a small header, the two spectrograms side by side, a metadata text zone, and the cover image at the top right.

spectrogram_to_audio.py does the reverse: it reads the PNG, parses the header to recover the audio parameters, reconstructs the two channels from the spectrograms (using a fast Griffin-Lim algorithm, now threaded), and writes either a WAV or an MP3 file. When creating MP3, it embeds the original metadata and cover art.

wrapper.py simply calls the two scripts in sequence - handy for batch conversion or just testing.

The PNG spectrogram is always kept after decoding; only the temporary WAV file (used during MP3 creation) is cleaned up.
Notice, volume data is not stored, so the output file might be a bit quieter than the original, but overall data is preserved.

## Usage

### Basic conversion - audio to PNG
```bash
python audio_to_spectrogram.py input.mp3
```

If you omit the output name, the script tries to use the song title (from metadata) as the filename, falling back to the input file’s basename.

### Convert PNG back to audio
```bash
python spectrogram_to_audio.py spectrogram.png output.mp3
```

The output extension decides the final format: .mp3 or .wav. If you don’t specify an output file, it defaults to the PNG’s name with .mp3 appended.

To also save the extracted album cover as a separate PNG:
```bash
python spectrogram_to_audio.py spectrogram.png --save-cover
```

### Full roundtrip with the wrapper
```bash
python wrapper.py input.mp3 restored.wav
```

The wrapper uses a hard-coded virtual environment python path (you may need to edit it). It produces an intermediate PNG (input_stem_spectrogram.png) and finally the WAV file.

## Detailed explanation

### File structure and blocks

The spectrogram PNG is not just a picture - it’s a structured container.

**Header strip (first 64 columns, rows 0..63)**

- Magic bytes (8 bytes) to identify the file.  
- 23 manifest values stored as 32-bit big-endian integers, one per row (rows 8..30). These values describe the audio parameters (sample rate, FFT size, hop length, frequency bins, time frames, min dB, and the exact coordinates of the spectrograms, metadata zone, and cover image inside the canvas).

**Spectrogram area (columns 64..(64+time_frames), rows 0..2*freq_bins)**

- Top half (rows 0..freq_bins-1): left channel, displayed with low frequencies at the bottom.  
- Bottom half (rows freq_bins..2*freq_bins-1): right channel.  

The pixel intensity is a linear mapping from dB to 0..255 (MIN_DB = -80 dB to 0 dB).

**Metadata text zone (right-most columns)**

A rectangular block where each pixel’s red channel stores one byte of a concatenated string:  
artist\0album\0title\0  

The block width is at least 256 pixels (or wider to fit the cover art).

- The first 4 bytes of the zone are the length of the UTF-8 payload.

**Cover image (below the metadata text zone, if present)**

- Stored as raw RGB pixels (max 1024×1024, resized automatically).  
- The manifest records its width, height, and Y offset.

All pixel values are stored in the red channel only (green and blue are copies of red, alpha is 255). This makes the spectrogram look like a grayscale image, but it also carries the extra information.

## Reconstruction (spectrogram to audio)

- The PNG is loaded and the magic bytes are verified.  
- The manifest is read from the header rows.  
- The left and right channel spectrograms are extracted (rows and columns according to the manifest) and flipped vertically.  

**Threaded reconstruction**

Both channels are processed in parallel using ThreadPoolExecutor. Each channel runs a Griffin-Lim phase estimation algorithm (64 iterations) to synthesise a waveform from the magnitude spectrogram.

- The two waveforms are trimmed to the same length, normalised to 0.95 peak, and interleaved into a stereo signal.  
- A temporary WAV file is written.  

If the requested output is MP3:
- ffmpeg is called to convert the temporary WAV to MP3.  
- Metadata (title, artist, album) and cover art are embedded as ID3v2.3 tags and an attached picture.  
- The temporary WAV is then deleted.  

If the output is WAV:
- The temporary file is simply renamed or moved to the final location.

## Compression options and size reduction

The spectrogram PNG can become quite large (e.g., ~70 MB for a 3-minute song), so here are ways to compress the whole pipeline:

- **Reduce the frequency resolution** - Modify N_FFT (default 2048). Lower values (e.g., 1024) give fewer frequency bins, making the PNG shorter (height = 2 × bins).  
- **Increase the hop length** - HOP_LENGTH (default 512). Larger hops produce fewer time frames, reducing the PNG width.  
- **Change MIN_DB** - Default -80 dB. Making this less negative (e.g., -60 dB) discards very quiet parts.  
- **PNG compression level** - Use:
```python
cv2.imwrite(..., [cv2.IMWRITE_PNG_COMPRESSION, 9])
```
- **Resize or remove cover art** - Lower MAX_COVER_SIZE (e.g., 300 pixels).  
- **Output format** - MP3 (e.g., 128k bitrate) is much smaller than WAV.

All these parameters are defined as constants at the top of audio_to_spectrogram.py. The decoder reads them from the PNG header automatically.

## Dependencies

- numpy  
- librosa  
- opencv-python  
- soundfile  
- mutagen  
- ffmpeg (only required for MP3 output)

Install with:
```bash
pip install numpy librosa opencv-python soundfile mutagen
```

Then make sure ffmpeg is in your PATH.

## Honest disclaimer: 
This tool is pretty useless for any practical task. I built it solely to understand how spectrograms work, how to pack metadata into images, and how to reverse the process. There's no real world use case for storing audio as a PNG -- it's massively inefficient (a three-minute song turns into a several-hundred-megabyte image), the reconstruction is lossy and full of Griffin‑Lim artefacts, and the whole roundtrip takes way longer than just keeping the original MP3. You could misuse it as a bizarre form of encryption (hide audio in a picture that looks like a spectrogram), but that would be an absurdly bloated and easily detectable way to hide data. Don't use this for anything serious. It's a learning experiment.

## Notes

- The decoding process is not real-time; a 3-minute song takes about 10-15 seconds on a modern CPU.  
- The Griffin-Lim algorithm introduces some artefacts, but the result is recognisable and keeps stereo separation.  
- The PNG header format is fixed - do not crop or edit the image outside the designated zones.  
- If you want to use the wrapper, edit the VENV_PYTHON variable inside wrapper.py.
