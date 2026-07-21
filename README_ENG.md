# anime_vfr

[Documentazione italiana](README.md)

`anime_vfr` converts hybrid MKV sources, typically NTSC anime DVDs, to progressive video while applying the appropriate treatment to each part of the content. A single source may contain telecined film, progressive material, and genuinely interlaced motion, so applying one global IVTC or bob operation is not always correct.

The pipeline produces VFR output using three operational strategies:

- `match_keep_pts`: reconstruct frames with TFM and preserve every source timestamp;
- `match_decimate`: use TFM and remove only redundancy confirmed by the TDecimate mapping;
- `bob_expand`: reconstruct individual fields for genuinely interlaced material.

The 24/30/60 fps labels in reports describe observed cadence; they do not select a strategy by themselves.

## Requirements and configuration

The project is configured for macOS/Homebrew. Before use, check [config.py](config.py), especially:

- paths for `mkvmerge`, `mkvextract`, `vspipe`, Python, MediaInfo, FFmpeg, and FFprobe;
- `ENCODER_BIN` and `ENCODER_PARAMS`.

The included configuration uses `ffmpeg-full`, `libx265`, the `fast` preset, and CRF 20. The encoder must accept Y4M through standard input and produce a Matroska-compatible video file.

The following are also required:

- Python with `numpy` and access to the same VapourSynth environment used by VSPipe;
- the `vsdeinterlace`, `vsaa`, `vstools`, and `vskernels` Python modules and their dependencies;
- the BestSource, TIVTC, fmtconv, and Sneedif/NNEDI3 VapourSynth plugins, plus the plugins required by `QTempGaussMC`;
- MKVToolNix, MediaInfo, and FFmpeg.

The source video must use Matroska track ID `0`, which is the track from which the pipeline extracts timestamps.

Minimal environment check:

```bash
/opt/homebrew/bin/python3.14 -c "import vapoursynth as vs; import numpy; print(vs.__version__)"
/opt/homebrew/bin/vspipe --version
/opt/homebrew/bin/mkvmerge --version
```

## Quick start

Running a complete analysis before encoding a new source is recommended:

```bash
python3.14 anime_vfr.py "/path/episode.mkv" --analyze-only --keep-work
python3.14 anime_vfr.py "/path/episode.mkv" --output "/path/encoded"
```

A directory can also be supplied; MKV files directly inside it are processed.

```bash
python3.14 anime_vfr.py "/path/series" --output "/path/encoded"
```

## Options

```text
anime_vfr.py source [options]
```

| Option | Behavior |
| --- | --- |
| `source` | An MKV file or a directory containing MKV files. |
| `--analyze-only` | Run analysis, classification, requested dedup, timecode, and VPY generation without encode or mux. |
| `--report` | Measure an existing MKV; it must be used without other options. |
| `--output PATH` | Destination directory for final MKV files. |
| `--work-dir PATH` | Root directory for per-run work directories. |
| `--keep-work` | Preserve intermediate artifacts. |
| `--bob [NUM/DEN]` | Force CFR bob over the whole title; default `60000/1001`. |
| `--bob-chapters LIST` | Force bob over one-based chapter numbers, for example `4,5`. |
| `--bob-range LIST` | Force bob over time ranges, for example `22:30-23:50`. |
| `--field-order tff\|bff` | Field order used by TFM/QTGMC; default `tff`. |
| `--progressive-dedup [N]` | Deduplicate an already progressive source without TIVTC or classification. |
| `--dedup [N]` | Enable optional dedup on matched/decimated segments. |
| `--crop L:R:T:B` | Remove pixels from all four edges before resizing. |
| `--resize WIDTHxHEIGHT` | Set the exact output resolution. |
| `--yuv444` | Produce 10-bit YUV 4:4:4 instead of 10-bit YUV 4:2:0. |
| `--threads N` | Set VapourSynth and prefetch threads; default `os.cpu_count()`. |
| `--analysis-workers N` | Set NumPy metric workers; defaults to `--threads`. |
| `--additional-vpy PATH` | Append a VPY snippet to the final clip. |
| `--frames N` | Encode the first `N` output frames. |
| `--frames A-B` | Encode the half-open output range `[A,B)`. |
| `--strip-audio` | Exclude audio tracks from the mux. |
| `--strip-sub` | Exclude subtitle tracks from the mux. |

### Analysis and reports

`--analyze-only` runs the same decision process used for encoding. Its report contains:

- a strategy table weighted by source-frame count;
- a strategy table weighted by PTS duration;
- every decision interval with source/output frames, times, strategy, observed cadence, and available reasons;
- structural-drop and optional-dedup statistics.

With `--keep-work`, the per-frame and per-run diagnostic files, final VPY, and timecodes remain available. Without `--keep-work`, the run directory is removed after the report is printed.

```bash
python3.14 anime_vfr.py "/path/episode.mkv" --analyze-only --keep-work
```

`--report` does not run the classifier. It extracts timestamps from Matroska track ID `0` of an existing MKV and summarizes time windows as 24, 30, or 60 fps, `other`, `VFR`, or `unknown`. It prints both an output-frame-count distribution and a duration distribution. This measures the muxed result; it does not explain the pipeline's earlier decisions.

```bash
python3.14 anime_vfr.py "/path/encoded/episode.mkv" --report
python3.14 anime_vfr.py "/path/encoded" --report
```

### Bob overrides

`--bob` skips TFM, TDecimate, classification, and dedup, reconstructs the entire title with bob, and uses CFR muxing. Its optional ratio must use `NUM/DEN` form; for example, use `--bob 50/1` for a 25i source converted to 50p.

`--bob-chapters` and `--bob-range` affect only the selected portions and leave the rest to automatic classification. Ranges accept seconds, `MM:SS`, or `HH:MM:SS` values and are comma-separated. Local overrides are incompatible with `--bob`, `--progressive-dedup`, and `--report`.

```bash
python3.14 anime_vfr.py "/path/episode.mkv" --bob-chapters 4
python3.14 anime_vfr.py "/path/episode.mkv" --bob-range 22:30-23:50,10:00-10:20
```

### Dedup

`--dedup [N]` compacts visual holds or duplicates on matched/decimated branches after the primary strategy has been selected. It does not replace the structural drops performed by `match_decimate`. `N` limits the maximum number of consecutive frames compacted into one output frame and defaults to 2 when omitted.

`--progressive-dedup [N]` applies only dedup and timing to an already progressive source. It must not be used on material that is still interlaced or telecined.

### Crop, resize, format, and additional filtering

`--crop` uses `LEFT:RIGHT:TOP:BOTTOM` form and removes the specified number of pixels from each edge. Cropping runs after matched/decimated/bob processing and always before resizing. Without `--resize`, the cropped resolution becomes the final resolution and retains the source SAR; with `--resize`, the order is `crop → resize` and the requested resolution is treated as square-pixel output.

For YUV 4:2:0 output, all four margins must be even to respect chroma subsampling. With `--yuv444`, odd margins are also accepted: each branch is converted to 4:4:4 before cropping.

```bash
python3.14 anime_vfr.py "/path/episode.mkv" --crop 8:8:0:0
python3.14 anime_vfr.py "/path/episode.mkv" --crop 7:9:1:3 --resize 768x576 --yuv444
```

The pipeline does not transform subtitle geometry. Positioned ASS, PGS, and VobSub tracks may therefore require separate adjustment after cropping or resizing.

`--additional-vpy` runs after assembly, resize, and conversion to the requested format. The snippet receives `clip`, must reassign it, and must not change frame count or order because timecodes have already been calculated. It must not call `set_output()`.

Minimal example:

```python
clip = core.std.SetFrameProp(clip, prop="_FieldBased", intval=0)
```

### Frame selection

`--frames` selects frames from the output timeline, after classification, decimation, and optional dedup. The pipeline still analyzes the complete source before applying the range. The final bound is exclusive.

Audio and subtitles are stream-copied over the same time bounds used for video. In this mode, Matroska chapters and attachments are not preserved; this includes attached fonts that may be required by ASS subtitles.

## Timing and transition safety

The pipeline separates matchability, redundancy, and observed cadence. TDecimate is used only in matchable runs with a validated redundancy mapping; other reconstructable runs preserve every PTS.

Transitions between matched and bob output are closed using the field dependencies expressed by TFM matches, preventing a reconstructed frame from using a field that belongs to the bob side of the boundary.

Final timecodes derive from the source timeline. The V2 file contains one timestamp per output frame plus a terminal timestamp, which preserves the exact final-frame duration after bob, decimation, dedup, or `--frames`. An unquantizable duration can be preserved by matched paths, but prevents safe bob processing of that frame.

## Output and work files

Without `--output`, `episode.mkv` produces `episode_1.mkv` next to the source. With `--output`, the original file name is preserved in the selected directory; the pipeline refuses to overwrite the source directly.

Each operation uses a unique subdirectory under `--work-dir`, or under `work` in the output directory by default. This prevents collisions between parallel runs. A complete mux reuses source audio, subtitles, chapters, and attachments unless explicitly excluded.

## Project structure

```text
anime_vfr.py   CLI entrypoint
pipeline.py    analysis, classification, and orchestration
branches.py    TFM/TDecimate branch construction
segments.py    operational mapping and segmentation
dedup.py       dedup on selected branches
timecodes.py   PTS validation and final timecodes
report.py      pre-encode and post-encode reports
media.py       source metadata and timestamps
encode.py      encoding and muxing
contracts.py   runtime contracts and validation
config.py      configurable binaries and parameters
```

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).

## Buy me a coffee!

[![Buy Me A Coffee](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/simonefil)
